"""Structured context-pruning seam for Hermes.

Pure data models and helpers that expose prompt/message/tool contributions as
structured, policy-aware records before they are joined or sent.  The core
Hermes seam is intentionally policy-free: it assembles contributions, invokes a
registered policy hook (e.g. from a plugin), validates the returned effective
view, and feeds that view into pressure estimation and provider dispatch.  When
no hook is registered, the config is disabled/invalid, or the hook fails, the
seam fails open and returns the unpruned view unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, TypeGuard

from agent.model_metadata import estimate_messages_tokens_rough
from agent.system_prompt import build_system_prompt_parts


logger = logging.getLogger(__name__)


_REQUIRED_PRUNING_CONFIG_KEYS: frozenset[str] = frozenset({
    "tokenjuice_prompt_pruning_enabled",
    "tokenjuice_prompt_pruning_threshold_tokens",
    "tokenjuice_prompt_pruning_trigger_ratio",
    "tokenjuice_prompt_pruning_target_ratio",
    "tokenjuice_prompt_pruning_soft_target_ratio",
    "tokenjuice_prompt_pruning_hard_target_ratio",
    "tokenjuice_prompt_pruning_min_saved_tokens",
    "tokenjuice_prompt_pruning_cache_ttl_seconds",
    "tokenjuice_prompt_pruning_protect_recent_messages",
    "tokenjuice_prompt_pruning_protect_recent_tool_interactions",
    "tokenjuice_prompt_pruning_classes",
    "tokenjuice_prompt_pruning_accounting_enabled",
})


KNOWN_CONTRIBUTION_CLASSES: frozenset[str] = frozenset({
    "identity",
    "system_instruction",
    "project_context",
    "memory_context",
    "skill_context",
    "developer_instruction",
    "user_message",
    "assistant_message",
    "terminal_tool_output",
    "exact_file_read",
    "diagnostic",
    "tool_schema",
    "unknown",
})


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_estimate(text: str) -> int:
    # Rough char-to-token estimate used elsewhere in Hermes for fast pressure checks.
    return (len(text) + 3) // 4


def _new_contribution(
    *,
    contribution_id: str,
    kind: str,
    provenance: str,
    class_: str,
    stability: str,
    content: str,
    cache_scope: str,
    token_estimate: int,
    atomic_group_id: Optional[str],
    prune_policy: str,
    protected_reason: str,
    created_at_epoch_ms: Optional[int] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    provider_message: Optional[Dict[str, Any]] = None,
    provider_tool: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a single structured contribution record with the required metadata.

    All callers must provide the metadata fields the Hermes tests and the
    TokenJuice policy plugin consume.  ``content_hash`` is a SHA-256 of the
    canonical text bytes so stable content can be matched across turns.
    """
    contribution: Dict[str, Any] = {
        "id": contribution_id,
        "kind": kind,
        "provenance": provenance,
        "class": class_,
        "stability": stability,
        "content_hash": _sha256_hex(content),
        "content": content,
        "cache_scope": cache_scope,
        "token_estimate": token_estimate,
        "char_count": len(content),
        "atomic_group_id": atomic_group_id,
        "prune_policy": prune_policy,
        "protected_reason": protected_reason,
    }
    if created_at_epoch_ms is not None:
        contribution["created_at_epoch_ms"] = created_at_epoch_ms
    if tool_calls is not None:
        contribution["tool_calls"] = tool_calls
    if provider_message is not None:
        contribution["provider_message"] = provider_message
    if provider_tool is not None:
        contribution["provider_tool"] = provider_tool
    return contribution


_DEFAULT_PROTECTED_REASON = "protected by default policy"
_RECENT_PROTECTED_REASON = "protected: recent message/tool interaction"
_ORPHAN_PROTECTED_REASON = "protected: orphan tool result without matching call"
_UNKNOWN_ROLE_PROTECTED_REASON = "protected: unknown message role"


def _prune_policy_for_class(class_: str) -> str:
    """Default prune policy for a contribution class.

    Only the explicitly disposable terminal output class is marked as hard-clear
    allowed.  Everything else is protected by default; future reviewed policy can
    extend the set via configuration, not magic numbers in this module.
    """
    if class_ == "terminal_tool_output":
        return "hard_clear_allowed"
    return "never"


def _tool_name_for_message(
    messages: List[Dict[str, Any]],
    tool_call_id: str,
) -> Optional[str]:
    """Find the function name of the assistant tool_call that produced *tool_call_id*."""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls") or []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            if isinstance(tc_id, str) and tc_id == tool_call_id:
                fn = tc.get("function") or {}
                if isinstance(fn, dict):
                    name = fn.get("name")
                    if isinstance(name, str) and name:
                        return name
                break
    return None


def _message_class(
    msg: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> str:
    role = msg.get("role") if isinstance(msg, dict) else None
    if role == "user":
        return "user_message"
    if role == "assistant":
        return "assistant_message"
    if role == "tool":
        tool_call_id = msg.get("tool_call_id") if isinstance(msg, dict) else None
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return "unknown"
        name = _tool_name_for_message(messages, tool_call_id)
        if name == "terminal":
            return "terminal_tool_output"
        if name == "read_file":
            return "exact_file_read"
        if name is not None:
            return "diagnostic"
        return "unknown"
    return "unknown"


def _message_content(msg: Dict[str, Any]) -> str:
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal: use a stable JSON representation for hashing/estimation.
        try:
            import json

            return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(content)
    return str(content) if content is not None else ""


def _atomic_group_id_for_message(
    msg: Dict[str, Any],
    messages: List[Dict[str, Any]],
    index: int,
) -> Optional[str]:
    """Return the tool_call_id for messages that belong to a tool interaction.

    Assistant messages that carry tool_calls are grouped with their matching tool
    result messages by sharing the same atomic_group_id.  Multiple tool calls in
    one assistant message form one batch group so pruning cannot orphan one call
    or one result from the same provider turn.
    """
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    if role == "assistant":
        tcs = msg.get("tool_calls") or []
        if tcs and isinstance(tcs, list):
            ids = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and tc_id:
                    ids.append(tc_id)
            if len(ids) == 1:
                return ids[0]
            if len(ids) > 1:
                return "tool-batch:" + ",".join(ids)
        return None
    if role == "tool":
        tool_call_id = msg.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id and _tool_name_for_message(messages, tool_call_id) is not None:
            for candidate in messages:
                if not isinstance(candidate, dict) or candidate.get("role") != "assistant":
                    continue
                tcs = candidate.get("tool_calls") or []
                if not isinstance(tcs, list):
                    continue
                ids = []
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        ids.append(tc_id)
                if tool_call_id in ids and len(ids) > 1:
                    return "tool-batch:" + ",".join(ids)
            return tool_call_id
        return None
    return None


def build_message_contributions(
    messages: List[Dict[str, Any]],
    *,
    phase: str = "preflight",
    protect_recent_messages: int = 0,
    protect_recent_tool_interactions: int = 0,
    deterministic_now_epoch_ms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Model conversation messages as structured contributions.

    Tool interactions are grouped by tool_call_id.  Recent messages and recent
    tool interactions are marked as protected.  Unknown roles and orphan tool
    results are protected with a descriptive reason.
    """
    now = deterministic_now_epoch_ms
    if now is None:
        now = int(time.time() * 1000)

    contributions: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        class_ = _message_class(msg, messages)
        content = _message_content(msg)
        atomic_group_id = _atomic_group_id_for_message(msg, messages, idx)
        protected_reason = _DEFAULT_PROTECTED_REASON
        if class_ == "unknown":
            if msg.get("role") == "tool":
                protected_reason = _ORPHAN_PROTECTED_REASON
            else:
                protected_reason = _UNKNOWN_ROLE_PROTECTED_REASON
        prune_policy = _prune_policy_for_class(class_)

        distance_from_end = len(messages) - 1 - idx
        if class_ in {"user_message", "assistant_message"} and protect_recent_messages > 0:
            if distance_from_end < protect_recent_messages:
                prune_policy = "never"
                protected_reason = _RECENT_PROTECTED_REASON
        if class_ == "terminal_tool_output" and protect_recent_tool_interactions > 0:
            if distance_from_end < protect_recent_tool_interactions:
                prune_policy = "never"
                protected_reason = _RECENT_PROTECTED_REASON

        _tool_calls = None
        if class_ == "assistant_message":
            _maybe_tool_calls = msg.get("tool_calls")
            if isinstance(_maybe_tool_calls, list) and _maybe_tool_calls:
                _tool_calls = _maybe_tool_calls

        contributions.append(
            _new_contribution(
                contribution_id=f"msg-{phase}-{idx}",
                kind="message" if class_ in {"user_message", "assistant_message"} else "tool_interaction",
                provenance="conversation_history",
                class_=class_,
                stability="session_stable" if class_ == "user_message" else "turn_ephemeral",
                content=content,
                cache_scope="body",
                token_estimate=_token_estimate(content),
                atomic_group_id=atomic_group_id,
                prune_policy=prune_policy,
                protected_reason=protected_reason,
                created_at_epoch_ms=now - (idx * 1000),
                tool_calls=_tool_calls,
                provider_message=copy.deepcopy(msg),
            )
        )
    return contributions


def build_tool_schema_contributions(
    tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Model tool schemas as atomic, protected schema contributions.

    Tool schemas are never pruned by default; this module only accounts for them
    so a policy hook can size the request correctly.
    """
    contributions: List[Dict[str, Any]] = []
    if not tools:
        return contributions
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        try:
            import json

            canonical = json.dumps(tool, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except Exception:
            canonical = str(tool)
        contributions.append(
            _new_contribution(
                contribution_id=f"tool-schema-{idx}",
                kind="tool_schema",
                provenance="tool_schema",
                class_="tool_schema",
                stability="stable_prefix",
                content=canonical,
                cache_scope="schema",
                token_estimate=_token_estimate(canonical),
                atomic_group_id=f"tool-schema-{idx}",
                prune_policy="never",
                protected_reason="protected: tool schema shape must be preserved",
                provider_tool=copy.deepcopy(tool),
            )
        )
    return contributions


def build_system_prompt_contributions(
    agent: Any,
    *,
    system_message: Optional[str] = None,
    structured_pruning_enabled: bool = False,
    test_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Expose system-prompt parts as structured contributions before join.

    When structured pruning is disabled, this returns a minimal list of content
    dicts that join to the exact same string as :func:`build_system_prompt`, so
    default behavior is byte-identical.  When enabled, it returns full metadata
    records for the stable identity, context files, and volatile per-session
    additions.
    """
    # Surface system-prompt parts as structured contributions.  If the agent
    # cannot render parts (minimal test fakes, partially-initialised agents),
    # fall back to a single system_instruction contribution carrying the provided
    # system prompt so the seam stays fail-open and can still invoke policy.
    try:
        parts = build_system_prompt_parts(agent, system_message=system_message)
    except Exception:
        logger.debug("build_system_prompt_parts failed for structured contributions; falling back", exc_info=True)
        parts = {"stable": "", "context": system_message or "", "volatile": ""}
    ordered = [
        ("stable", parts.get("stable") or "", "system_prompt", "identity"),
        ("context", parts.get("context") or "", "project_context", "project_context"),
        ("volatile", parts.get("volatile") or "", "system_prompt", "system_instruction"),
    ]
    if not structured_pruning_enabled:
        return [
            {"content": text}
            for _, text, _, _ in ordered
            if text.strip()
        ]

    contributions: List[Dict[str, Any]] = []
    for _tier, text, provenance, class_ in ordered:
        if not text.strip():
            continue
        contributions.append(
            _new_contribution(
                contribution_id=f"system-prompt-{_tier}",
                kind="system_part",
                provenance=provenance,
                class_=class_,
                stability="stable_prefix" if _tier == "stable" else "session_stable" if _tier == "context" else "volatile",
                content=text,
                cache_scope="prefix" if _tier == "stable" else "body" if _tier == "context" else "tail",
                token_estimate=_token_estimate(text),
                atomic_group_id=None,
                prune_policy="never",
                protected_reason=f"protected: {_tier} system-prompt tier",
            )
        )
    return contributions


def _is_valid_pruning_config(config: Any) -> TypeGuard[Dict[str, Any]]:
    """Fail-open validation: accept only a well-formed pruning configuration dict."""
    if not isinstance(config, dict):
        return False
    if not config.get("tokenjuice_prompt_pruning_enabled"):
        return False
    for key in _REQUIRED_PRUNING_CONFIG_KEYS:
        if key not in config:
            return False
    try:
        float(config["tokenjuice_prompt_pruning_trigger_ratio"])
        float(config["tokenjuice_prompt_pruning_target_ratio"])
        float(config["tokenjuice_prompt_pruning_soft_target_ratio"])
        float(config["tokenjuice_prompt_pruning_hard_target_ratio"])
        int(config["tokenjuice_prompt_pruning_threshold_tokens"])
        int(config["tokenjuice_prompt_pruning_min_saved_tokens"])
        int(config["tokenjuice_prompt_pruning_cache_ttl_seconds"])
        int(config["tokenjuice_prompt_pruning_protect_recent_messages"])
        int(config["tokenjuice_prompt_pruning_protect_recent_tool_interactions"])
    except (TypeError, ValueError):
        return False
    return True


def _build_structured_contributions(
    agent: Any,
    *,
    phase: str,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Assemble all contributions for a structured pruning call."""
    has_provider_system_message = bool(messages and messages[0].get("role") == "system")
    system_contributions = []
    if not has_provider_system_message:
        system_contributions = build_system_prompt_contributions(
            agent,
            system_message=system_prompt,
            structured_pruning_enabled=True,
            test_config=config,
        )
    message_contributions = build_message_contributions(
        messages,
        phase=phase,
        protect_recent_messages=int(config.get("tokenjuice_prompt_pruning_protect_recent_messages", 0)),
        protect_recent_tool_interactions=int(config.get("tokenjuice_prompt_pruning_protect_recent_tool_interactions", 0)),
    )
    tool_contributions = build_tool_schema_contributions(tools)
    return system_contributions + message_contributions + tool_contributions


def _resolve_structured_prune_hook(agent: Any) -> Optional[Any]:
    """Return the first available structured-pruning hook callable.

    Prefer an agent-bound callable (used by tests and direct callers); fall
    back to the plugin ``structured_context_prune`` hook so TokenJuice or any
    other plugin can register policy without mutating agent internals.
    """
    hook = getattr(agent, "_structured_context_prune", None)
    if callable(hook):
        return hook

    try:
        from hermes_cli.plugins import has_hook, invoke_hook

        if has_hook("structured_context_prune"):
            return lambda contributions, **kwargs: _first_result(
                invoke_hook("structured_context_prune", contributions=contributions, **kwargs)
            )
    except Exception:
        pass
    return None


def _first_result(results: Any) -> Optional[Dict[str, Any]]:
    """Return the first dict result from a list of plugin-hook results, or None."""
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            return first
    return None


def _validate_prune_result(result: Any) -> bool:
    """A hook result must be a dict with at least an effective_messages list."""
    if not isinstance(result, dict):
        return False
    effective_messages = result.get("effective_messages")
    if not isinstance(effective_messages, list):
        return False
    return True


def apply_structured_context_prune(
    agent: Any,
    *,
    phase: str,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    api_request_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Invoke the registered structured pruning hook and return the effective view.

    Returns ``None`` when pruning is disabled, the configuration is malformed,
    no hook is registered, the hook raises, or the hook returns a malformed
    response.  This is the fail-open contract: callers must fall back to the
    unpruned view unchanged.
    """
    config = getattr(agent, "_structured_pruning_config", None)
    if not _is_valid_pruning_config(config):
        return None

    hook = _resolve_structured_prune_hook(agent)
    if hook is None:
        return None

    compressor = getattr(agent, "context_compressor", None)
    threshold_tokens = int(config.get("tokenjuice_prompt_pruning_threshold_tokens", 0))
    context_length = int(getattr(compressor, "context_length", 0) or 0)
    trigger_ratio = float(config["tokenjuice_prompt_pruning_trigger_ratio"]) / 100.0
    trigger_tokens = int(threshold_tokens * trigger_ratio)

    tool_schema_tokens = 0
    if tools:
        from agent.model_metadata import _estimate_tools_tokens_rough

        tool_schema_tokens = _estimate_tools_tokens_rough(tools)

    contributions = _build_structured_contributions(
        agent,
        phase=phase,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        config=config,
    )

    try:
        result = hook(
            contributions,
            phase=phase,
            session_id=getattr(agent, "session_id", None) or "",
            turn_id=turn_id or "",
            api_request_id=api_request_id or "",
            task_id=task_id or "",
            model=getattr(agent, "model", None) or "",
            provider=getattr(agent, "provider", None) or "",
            api_mode=getattr(agent, "api_mode", None) or "",
            compression_enabled=bool(getattr(agent, "compression_enabled", False)),
            context_length=context_length,
            threshold_tokens=threshold_tokens,
            trigger_tokens=trigger_tokens,
            tool_schema_tokens=tool_schema_tokens,
            now_epoch_ms=int(time.time() * 1000),
        )
    except Exception:
        return None

    if not _validate_prune_result(result):
        return None

    return result


__all__ = [
    "build_message_contributions",
    "build_tool_schema_contributions",
    "build_system_prompt_contributions",
    "apply_structured_context_prune",
]
