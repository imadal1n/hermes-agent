"""Unit tests for the extracted turn prologue (``agent/turn_context.py``).

These exercise ``build_turn_context`` against a lightweight fake agent to
confirm the prologue produces the right ``TurnContext`` and applies the
``agent`` side effects the loop relies on — without spinning up a real
``AIAgent`` or hitting any provider.
"""

from __future__ import annotations

import copy
import pathlib
import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_request_tokens_rough
from agent.turn_context import TurnContext, build_turn_context
from hermes_state import SessionDB


def _load_tokenjuice_register():
    try:
        from tokenjuice_hermes.plugin import register

        return register
    except ModuleNotFoundError:
        sibling = pathlib.Path(__file__).resolve().parents[3] / "tokenjuice-hermes"
        if sibling.exists():
            sys.path.insert(0, str(sibling))
            from tokenjuice_hermes.plugin import register

            return register
        pytest.skip("tokenjuice-hermes checkout or package is required for cross-repo tests")


class _TokenJuiceHost:
    def __init__(self, config):
        self.config = config
        self.callbacks = {}

    def register_hook(self, name, callback):
        self.callbacks[name] = callback

    def register_middleware(self, _name, _callback):
        pass


class _FakeTodoStore:
    def has_items(self):
        return True

    def _hydrate(self, *_a, **_k):
        pass


class _FakeGuardrails:
    def __init__(self):
        self.reset_called = False

    def reset_for_turn(self):
        self.reset_called = True


class _FakeAgent:
    """Minimal stand-in covering only what the prologue touches."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = False
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2
        )
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        # Attributes the prologue assigns; recorded for assertions.
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_calls = 0
        self._session_messages = []
        self._pending_cli_user_message = None
        self._session_persist_lock = threading.RLock()
        # Records _cached_system_prompt at the moment _ensure_db_session()
        # is called (regression guard for #45499 turn-setup ordering).
        self._ensure_db_prompt_at_call = "<unset>"

    # --- methods the prologue calls ---
    def _ensure_db_session(self):
        self._ensure_db_prompt_at_call = self._cached_system_prompt

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, *_a, **_k):
        self._persist_calls += 1


def _make_agent_with_cooldown(db_path, session_id, *, cooldown_until=None):
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._emit_status = MagicMock()
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (messages, "SYSTEM")
    )

    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="cli")
    if cooldown_until is not None:
        db.record_compression_failure_cooldown(session_id, cooldown_until, "timeout")

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, session_id)
    agent.context_compressor = compressor
    agent._session_db = db
    return agent


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    """``build_turn_context`` calls ``auxiliary_client.set_runtime_main`` as a
    production side effect (telling aux tools the live main provider/model).
    That writes a module-level global these unit tests don't care about and
    which would otherwise leak into sibling tests (e.g. provider-parity
    resolution) when the per-test process isolation plugin is disabled. Stub
    it out so the prologue tests stay hermetic.
    """
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


def test_returns_turn_context_with_user_message_appended():
    agent = _FakeAgent()
    ctx = _build(agent)
    assert isinstance(ctx, TurnContext)
    assert ctx.user_message == "hello"
    # The user turn was appended and indexed.
    assert ctx.messages[-1] == {"role": "user", "content": "hello"}
    assert ctx.current_turn_user_idx == len(ctx.messages) - 1
    assert ctx.active_system_prompt == "SYSTEM"


def test_applies_agent_side_effects():
    agent = _FakeAgent()
    _build(agent)
    # Retry counters reset, guardrails reset, vision re-armed, turn counted.
    assert agent._invalid_tool_retries == 0
    assert agent._tool_guardrails.reset_called is True
    assert agent._vision_supported is True
    assert agent._user_turn_count == 1
    # Crash-resilience persistence fired once.
    assert agent._persist_calls == 1
    # task/turn ids assigned on the agent.
    assert agent._current_task_id
    assert agent._current_turn_id


def test_task_id_passthrough():
    agent = _FakeAgent()
    ctx = _build(agent, task_id="fixed-task")
    assert ctx.effective_task_id == "fixed-task"
    assert agent._current_task_id == "fixed-task"


def test_persist_user_message_becomes_original():
    agent = _FakeAgent()
    ctx = _build(agent, user_message="api-prefixed", persist_user_message="clean")
    # original_user_message tracks the clean persist override.
    assert ctx.original_user_message == "clean"
    # but the appended user turn carries the full (sanitized) message.
    assert ctx.messages[-1]["content"] == "api-prefixed"


def test_pending_cli_message_carries_durable_marker_to_new_turn_dict():
    """A close-persisted CLI input must not be written again by turn start."""
    agent = _FakeAgent()
    staged = {"role": "user", "content": "already durable", "_db_persisted": True}
    agent._pending_cli_user_message = staged

    ctx = _build(agent, user_message="already durable")

    assert ctx.messages[-1] is staged
    assert ctx.messages[-1]["content"] == "already durable"
    assert ctx.messages[-1]["_db_persisted"] is True
    assert agent._pending_cli_user_message is None


def test_stale_pending_cli_message_does_not_replace_new_turn_input():
    """A failed prior persistence handoff cannot substitute later user input."""
    agent = _FakeAgent()
    agent._pending_cli_user_message = {"role": "user", "content": "old prompt"}

    stale = agent._pending_cli_user_message
    ctx = _build(
        agent,
        user_message="new prompt",
        conversation_history=[{"role": "assistant", "content": "old answer"}],
    )

    assert ctx.messages[-1]["content"] == "new prompt"
    assert ctx.messages[-1] is not stale
    assert agent._pending_cli_user_message is None


def test_pending_cli_message_uses_clean_override_for_api_local_note():
    """A noted API message reuses the clean staged dict and its DB marker."""
    agent = _FakeAgent()
    staged = {"role": "user", "content": "clean prompt", "_db_persisted": True}
    agent._pending_cli_user_message = staged

    ctx = _build(
        agent,
        user_message="[MODEL NOTE]\n\nclean prompt",
        persist_user_message="clean prompt",
    )

    assert ctx.messages[-1] is staged
    assert ctx.messages[-1]["content"] == "[MODEL NOTE]\n\nclean prompt"
    assert ctx.messages[-1]["_db_persisted"] is True
    assert agent._pending_cli_user_message is None


def test_runtime_main_sync_happens_after_restore():
    agent = _FakeAgent()
    agent.model = "stale-fallback-model"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_key = "fallback-key"
    agent.api_mode = "codex_responses"

    def restore_primary():
        agent.model = "primary-model"
        agent.provider = "anthropic"
        agent.base_url = "https://api.anthropic.com"
        agent.api_key = "primary-key"
        agent.api_mode = "anthropic_messages"

    agent._restore_primary_runtime = restore_primary
    calls = []
    with patch(
        "agent.auxiliary_client.set_runtime_main",
        side_effect=lambda *args, **kwargs: calls.append((args, kwargs)),
    ):
        _build(agent)

    assert calls == [(
        ("anthropic", "primary-model"),
        {
            "base_url": "https://api.anthropic.com",
            "api_key": "primary-key",
            "api_mode": "anthropic_messages",
            "auth_mode": "",
        },
    )]


def test_memory_nudge_fires_at_interval():
    agent = _FakeAgent()
    agent._memory_nudge_interval = 1
    agent.valid_tool_names = {"memory"}
    agent._memory_store = object()
    ctx = _build(agent)
    assert ctx.should_review_memory is True
    assert agent._turns_since_memory == 0  # reset after firing


def test_no_review_when_memory_disabled():
    agent = _FakeAgent()
    ctx = _build(agent)
    assert ctx.should_review_memory is False


def test_ensure_db_session_runs_after_system_prompt_restore():
    """Regression for #45499.

    On a fresh API/gateway agent (``_cached_system_prompt is None``) the DB
    session row must be created AFTER the system prompt is restored/built, so
    the persisted snapshot is written non-NULL. If ``_ensure_db_session()``
    ran first it would insert ``system_prompt=NULL`` and trip the misleading
    "stored system prompt is null; rebuilding" warning plus a first-turn
    prefix cache miss.
    """
    agent = _FakeAgent()
    agent._cached_system_prompt = None  # fresh agent, no cached prompt yet

    def _restore(_agent, _system_message, _history):
        _agent._cached_system_prompt = "REBUILT-SYSTEM"

    _build(agent, restore_or_build_system_prompt=_restore)

    # The prompt was populated before the DB row was created.
    assert agent._ensure_db_prompt_at_call == "REBUILT-SYSTEM"
    assert agent._cached_system_prompt == "REBUILT-SYSTEM"


# ── Between-turns MCP refresh (cache-safe late-binding) ──────────────────────
#
# A slow MCP server that connects after the agent's build-time tool snapshot
# must become callable by the user's NEXT turn — without mutating an in-flight
# turn's cached request prefix. The prologue is exactly that boundary, so the
# refresh hook lives here. These assert the contract (R1/R2/R6 in the spec),
# not timing permutations.


def test_between_turns_refresh_adds_late_tool_when_servers_registered():
    """R1: a tool that registered since build lands in this turn's snapshot."""
    agent = _FakeAgent()

    new_def = {"type": "function", "function": {"name": "mcp_x_tool", "description": "", "parameters": {}}}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(model_tools, "get_tool_definitions", return_value=[new_def]):
        _build(agent)

    assert "mcp_x_tool" in agent.valid_tool_names
    assert any(t["function"]["name"] == "mcp_x_tool" for t in agent.tools)


def test_between_turns_refresh_skipped_when_no_servers():
    """R6: the common case (no MCP servers) never walks the registry."""
    agent = _FakeAgent()
    import model_tools

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=False), \
         patch.object(model_tools, "get_tool_definitions") as gtd:
        _build(agent)

    gtd.assert_not_called()


def test_between_turns_refresh_skipped_when_skip_flag_set():
    """Internal forks (background_review) set _skip_mcp_refresh to keep tools[]
    byte-identical to the parent for cache parity — the hook must honor it even
    when MCP servers are registered."""
    agent = _FakeAgent()
    agent._skip_mcp_refresh = True
    import model_tools

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(model_tools, "get_tool_definitions") as gtd:
        _build(agent)

    gtd.assert_not_called()


def test_between_turns_refresh_no_churn_when_unchanged():
    """R2: an unchanged tool set leaves the snapshot object identity intact
    (no needless swap → nothing for the next request prefix to diff against)."""
    agent = _FakeAgent()
    same = [{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}]
    agent.tools = same
    agent.valid_tool_names = {"a"}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(
             model_tools, "get_tool_definitions",
             return_value=[{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}],
         ):
        _build(agent)

    assert agent.tools is same  # not replaced → no churn


def test_preflight_skips_when_persisted_cooldown_survives_restart(tmp_path):
    agent = _make_agent_with_cooldown(
        tmp_path / "state.db",
        "sess-1",
        cooldown_until=4_000_000_000.0,
    )

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch("agent.turn_context.estimate_request_tokens_rough", return_value=999_999):
        ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._emit_status.assert_not_called()
    agent._compress_context.assert_not_called()


def test_preflight_still_runs_for_other_session_with_same_db(tmp_path):
    db_path = tmp_path / "state.db"
    _make_agent_with_cooldown(
        db_path,
        "sess-1",
        cooldown_until=4_000_000_000.0,
    )
    agent = _make_agent_with_cooldown(db_path, "sess-2")

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch("agent.turn_context.estimate_request_tokens_rough", return_value=999_999):
        ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._emit_status.assert_called_once()
    agent._compress_context.assert_called()


def test_expired_cooldown_allows_preflight(tmp_path):
    agent = _make_agent_with_cooldown(
        tmp_path / "state.db",
        "sess-1",
        cooldown_until=1.0,
    )

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch("agent.turn_context.estimate_request_tokens_rough", return_value=999_999):
        ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._emit_status.assert_called_once()
    agent._compress_context.assert_called()


_STRUCTURED_PRUNING_TEST_CONFIG: dict[str, bool | int | str] = {
    "tokenjuice_prompt_pruning_enabled": True,
    "tokenjuice_prompt_pruning_threshold_tokens": 8500,
    "tokenjuice_prompt_pruning_trigger_ratio": 80,
    "tokenjuice_prompt_pruning_target_ratio": 75,
    "tokenjuice_prompt_pruning_soft_target_ratio": 75,
    "tokenjuice_prompt_pruning_hard_target_ratio": 65,
    "tokenjuice_prompt_pruning_min_saved_tokens": 256,
    "tokenjuice_prompt_pruning_cache_ttl_seconds": 3600,
    "tokenjuice_prompt_pruning_protect_recent_messages": 8,
    "tokenjuice_prompt_pruning_protect_recent_tool_interactions": 2,
    "tokenjuice_prompt_pruning_classes": "terminal_tool_output",
    "tokenjuice_prompt_pruning_accounting_enabled": True,
}


def _make_over_threshold_history_with_old_terminal(cfg: dict[str, bool | int | str]):
    """Return a canonical history whose rough token pressure exceeds ``cfg`` threshold.

    The pressure comes from one old, disposable terminal tool-result message. The
    payload size is derived from the configured threshold so the fixture stays valid
    under different test configs without hard-coding a character count.
    """
    threshold = cfg["tokenjuice_prompt_pruning_threshold_tokens"]
    min_saved = cfg["tokenjuice_prompt_pruning_min_saved_tokens"]
    terminal_chars = (threshold + min_saved) * 4
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "I'll run a terminal command",
            "tool_calls": [
                {
                    "id": "call-terminal-1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-terminal-1", "content": "x" * terminal_chars},
    ]


def _make_agent_for_structured_pruning_test(
    tmp_path: pathlib.Path,
    session_id: str,
    cfg: dict[str, bool | int | str] | None = None,
):
    """Build a fake agent with compression enabled and an optional pruning config."""
    agent = _make_agent_with_cooldown(tmp_path / "state.db", session_id)
    agent.compression_enabled = True
    base_cfg = _STRUCTURED_PRUNING_TEST_CONFIG
    agent.context_compressor.context_length = int(
        base_cfg["tokenjuice_prompt_pruning_threshold_tokens"]
        / (base_cfg["tokenjuice_prompt_pruning_trigger_ratio"] / 100)
    )
    agent.context_compressor.threshold_tokens = base_cfg["tokenjuice_prompt_pruning_threshold_tokens"]
    agent.context_compressor.threshold_percent = base_cfg["tokenjuice_prompt_pruning_trigger_ratio"] / 100
    agent._cached_system_prompt = "SYSTEM"
    if cfg is not None:
        agent._structured_pruning_config = cfg
    return agent


def test_structured_pruning_preflight_lowers_pressure_and_skips_compression(tmp_path):
    """Given an over-threshold history with old terminal output and structured
    pruning enabled via a test hook, the preflight gate must invoke the hook,
    use the returned effective view for pressure estimation, skip compression,
    and leave the canonical history unchanged.
    """
    cfg = _STRUCTURED_PRUNING_TEST_CONFIG
    full_history = _make_over_threshold_history_with_old_terminal(cfg)
    canonical_history_snapshot = copy.deepcopy(full_history)
    system_prompt = "SYSTEM"
    pruned_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "[earlier turn]"},
        {"role": "user", "content": "hello"},
    ]

    def _prune_stub(_contributions, **kwargs):
        return {
            "effective_messages": pruned_history,
            "effective_system_prompt": system_prompt,
            "accounting": {
                "pruned_count": 1,
                "saved_tokens": cfg["tokenjuice_prompt_pruning_threshold_tokens"] * 4,
                "redacted": [{"id": "call-terminal-1", "class": "terminal_tool_output"}],
            },
        }

    agent = _make_agent_for_structured_pruning_test(tmp_path, "sess-prune", cfg)
    agent._structured_context_prune = MagicMock(side_effect=_prune_stub)

    ctx = _build(
        agent,
        user_message="hello",
        conversation_history=full_history,
        restore_or_build_system_prompt=lambda _a, _sm, _h: system_prompt,
    )

    agent._structured_context_prune.assert_called_once()
    call_kwargs = agent._structured_context_prune.call_args.kwargs
    assert call_kwargs.get("phase") == "preflight"
    assert call_kwargs.get("session_id") == agent.session_id

    pruned_pressure = estimate_request_tokens_rough(
        pruned_history,
        system_prompt=system_prompt,
        tools=agent.tools or None,
    )
    assert pruned_pressure < cfg["tokenjuice_prompt_pruning_threshold_tokens"]

    assert ctx.messages == canonical_history_snapshot + [{"role": "user", "content": "hello"}]
    assert full_history == canonical_history_snapshot
    agent._compress_context.assert_not_called()


def test_structured_pruning_preflight_disabled_keeps_existing_compression(tmp_path):
    """When structured pruning is disabled, the existing preflight compression
    path must run unchanged and the pruning hook must not be invoked.
    """
    cfg = dict(_STRUCTURED_PRUNING_TEST_CONFIG)
    cfg["tokenjuice_prompt_pruning_enabled"] = False
    full_history = _make_over_threshold_history_with_old_terminal(cfg)

    agent = _make_agent_for_structured_pruning_test(tmp_path, "sess-prune-disabled", cfg)
    agent._structured_context_prune = MagicMock()

    _build(
        agent,
        user_message="hello",
        conversation_history=full_history,
        restore_or_build_system_prompt=lambda _a, _sm, _h: "SYSTEM",
    )

    agent._structured_context_prune.assert_not_called()
    agent._compress_context.assert_called()


def test_structured_pruning_preflight_invalid_config_fails_open(tmp_path):
    """An invalid structured-pruning config must fail open: no pruning hook
    runs and the existing compression path remains available.
    """
    valid_cfg = _STRUCTURED_PRUNING_TEST_CONFIG
    invalid_cfg = dict(valid_cfg)
    invalid_cfg["tokenjuice_prompt_pruning_enabled"] = True
    invalid_cfg["tokenjuice_prompt_pruning_trigger_ratio"] = "not-a-number"
    full_history = _make_over_threshold_history_with_old_terminal(valid_cfg)

    agent = _make_agent_for_structured_pruning_test(tmp_path, "sess-prune-invalid", None)
    agent._structured_pruning_config = invalid_cfg
    agent._structured_context_prune = MagicMock()

    _build(
        agent,
        user_message="hello",
        conversation_history=full_history,
        restore_or_build_system_prompt=lambda _a, _sm, _h: "SYSTEM",
    )

    agent._structured_context_prune.assert_not_called()
    agent._compress_context.assert_called()


def test_structured_pruning_preflight_missing_config_keeps_existing_compression(tmp_path):
    """When no structured pruning config is present, the existing preflight
    compression path must run unchanged and the pruning hook must not be invoked.
    """
    cfg = _STRUCTURED_PRUNING_TEST_CONFIG
    full_history = _make_over_threshold_history_with_old_terminal(cfg)

    agent = _make_agent_for_structured_pruning_test(tmp_path, "sess-prune-missing")
    agent._structured_context_prune = MagicMock()

    _build(
        agent,
        user_message="hello",
        conversation_history=full_history,
        restore_or_build_system_prompt=lambda _a, _sm, _h: "SYSTEM",
    )

    agent._structured_context_prune.assert_not_called()
    agent._compress_context.assert_called()


def test_structured_pruning_preflight_discovers_plugin_before_native_compression(
    tmp_path,
    monkeypatch,
):
    events: list[str] = []
    cfg = dict(_STRUCTURED_PRUNING_TEST_CONFIG)
    cfg["tokenjuice_prompt_pruning_threshold_tokens"] = 316_200
    cfg["tokenjuice_prompt_pruning_trigger_ratio"] = 80
    full_history = [{"role": "user", "content": "pressure"}]
    pruned_history = [{"role": "user", "content": "small"}]

    plugins = types.ModuleType("hermes_cli.plugins")
    discovered = False

    def discover_plugins() -> None:
        nonlocal discovered
        events.append("discover")
        discovered = True

    def has_hook(name: str) -> bool:
        events.append(f"has:{name}:{discovered}")
        return discovered and name == "structured_context_prune"

    def invoke_hook(name: str, **kwargs):
        if name != "structured_context_prune":
            return []
        events.append(f"invoke:{name}")
        assert kwargs["current_pressure_tokens"] == 391_733
        assert kwargs["threshold_tokens"] == 316_200
        assert kwargs["trigger_tokens"] == 252_960
        return [{"effective_messages": pruned_history}]

    setattr(plugins, "discover_plugins", discover_plugins)
    setattr(plugins, "has_hook", has_hook)
    setattr(plugins, "invoke_hook", invoke_hook)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins)

    agent = _make_agent_for_structured_pruning_test(tmp_path, "sess-prune-plugin", cfg)
    agent.context_compressor.context_length = 372_000
    agent.context_compressor.threshold_tokens = 316_200
    agent.context_compressor.threshold_percent = 0.85

    def estimate(messages, **_kwargs):
        if messages == pruned_history:
            events.append("estimate:pruned")
            return 237_000
        events.append("estimate:canonical")
        return 391_733

    def should_compress(tokens: int) -> bool:
        events.append(f"should:{tokens}")
        return tokens >= agent.context_compressor.threshold_tokens

    agent.context_compressor.should_compress = MagicMock(side_effect=should_compress)

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch("agent.turn_context.estimate_request_tokens_rough", side_effect=estimate), \
         patch("agent.context_pruning.time.time", return_value=1_000.0):
        ctx = _build(
            agent,
            user_message="hello",
            conversation_history=full_history,
            restore_or_build_system_prompt=lambda _a, _sm, _h: "SYSTEM",
        )

    assert ctx.messages == full_history + [{"role": "user", "content": "hello"}]
    assert events[0] == "estimate:canonical"
    assert events.index("discover") < events.index("has:structured_context_prune:True")
    assert events.index("has:structured_context_prune:True") < events.index(
        "invoke:structured_context_prune"
    )
    assert events.index("invoke:structured_context_prune") < events.index("should:237000")
    assert "should:391733" not in events
    agent._compress_context.assert_not_called()


class TestStructuredContributionModel:
    """Red tests for the planned Hermes structured contribution model.

    These exercise helpers that will live in ``agent/context_pruning.py`` and
    model messages, tool interactions, and tool schemas as structured,
    policy-aware contributions before any pruning decision.
    """

    def test_structured_contribution_message_fields(self):
        from agent.context_pruning import build_message_contributions

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        contributions = build_message_contributions(messages, phase="preflight")
        assert isinstance(contributions, list) and len(contributions) == 2

        required = {
            "id",
            "kind",
            "provenance",
            "class",
            "stability",
            "content_hash",
            "cache_scope",
            "token_estimate",
            "char_count",
            "atomic_group_id",
            "prune_policy",
            "protected_reason",
            "content",
            "provider_message",
        }
        for contribution in contributions:
            missing = required - contribution.keys()
            assert not missing, f"contribution missing keys: {missing}"
            assert contribution["kind"] == "message"
            assert contribution["provenance"] == "conversation_history"
            assert contribution["class"] in {"user_message", "assistant_message"}
            assert contribution["provider_message"] in messages

    def test_structured_contribution_preserves_multimodal_provider_message(self):
        from agent.context_pruning import build_message_contributions

        content = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        messages = [{"role": "user", "content": content, "metadata": {"kept": True}}]

        contributions = build_message_contributions(messages, phase="pre_api")

        assert contributions[0]["content"] != str(content)
        assert contributions[0]["provider_message"] == messages[0]

    def test_structured_contribution_tool_interaction_atomic_group(self):
        from agent.context_pruning import build_message_contributions

        messages = [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": "I'll run a terminal command",
                "tool_calls": [
                    {
                        "id": "call-terminal-1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-terminal-1", "content": "output"},
        ]
        contributions = build_message_contributions(messages, phase="pre_api")

        tool_call_idx = next(
            i
            for i, c in enumerate(contributions)
            if c["class"] == "assistant_message" and c.get("tool_calls")
        )
        result_idx = next(
            i
            for i, c in enumerate(contributions)
            if c["class"] == "terminal_tool_output"
        )
        assert (
            contributions[tool_call_idx]["atomic_group_id"]
            == contributions[result_idx]["atomic_group_id"]
        )
        assert contributions[result_idx]["atomic_group_id"] == "call-terminal-1"

    def test_structured_contribution_multi_tool_batch_is_one_atomic_group(self):
        from agent.context_pruning import build_message_contributions

        messages = [
            {"role": "user", "content": "run both"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-a", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
                    {"id": "call-b", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call-a", "name": "terminal", "content": "a"},
            {"role": "tool", "tool_call_id": "call-b", "name": "terminal", "content": "b"},
        ]

        contributions = build_message_contributions(messages, phase="pre_api")

        grouped = [
            contribution["atomic_group_id"]
            for contribution in contributions
            if contribution["class"] in {"assistant_message", "terminal_tool_output"}
        ]
        assert grouped == ["tool-batch:call-a,call-b"] * 3

    def test_structured_contribution_orphan_tool_results_protected(self):
        from agent.context_pruning import build_message_contributions

        messages = [
            {"role": "tool", "tool_call_id": "missing-call", "content": "orphan"},
        ]
        contributions = build_message_contributions(messages, phase="pre_api")

        assert len(contributions) == 1
        contribution = contributions[0]
        assert contribution["class"] == "unknown"
        assert contribution["prune_policy"] == "never"
        assert contribution["atomic_group_id"] is None
        assert "orphan tool result" in contribution["protected_reason"].lower()

    def test_structured_contribution_tool_schema_atomic_groups(self):
        from agent.context_pruning import build_tool_schema_contributions

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run a shell command.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        contributions = build_tool_schema_contributions(tools)

        assert len(contributions) == 2
        for contribution in contributions:
            assert contribution["kind"] == "tool_schema"
            assert contribution["class"] == "tool_schema"
            assert contribution["provenance"] == "tool_schema"
            assert contribution["prune_policy"] == "never"
            assert contribution["atomic_group_id"] is not None
            assert isinstance(contribution["content_hash"], str)
            assert len(contribution["content_hash"]) == 64
            assert contribution["provider_tool"] in tools

    def test_structured_prune_hook_kwargs_leave_tokenjuice_target_ratios_authoritative(self):
        from types import SimpleNamespace

        from agent.context_pruning import apply_structured_context_prune

        cfg = dict(_STRUCTURED_PRUNING_TEST_CONFIG)
        seen = {}

        def _hook(_contributions, **kwargs):
            seen.update(kwargs)
            return {"effective_messages": [{"role": "user", "content": "kept"}]}

        agent = SimpleNamespace(
            _structured_pruning_config=cfg,
            _structured_context_prune=_hook,
            context_compressor=SimpleNamespace(context_length=100_000),
            compression_enabled=True,
            session_id="session-a",
            model="model-a",
            provider="provider-a",
            api_mode="chat_completions",
        )

        result = apply_structured_context_prune(
            agent,
            phase="pre_api",
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="SYSTEM",
            tools=None,
        )

        assert result is not None
        assert "target_tokens" not in seen
        assert seen["trigger_tokens"] == int(
            cfg["tokenjuice_prompt_pruning_threshold_tokens"]
            * cfg["tokenjuice_prompt_pruning_trigger_ratio"]
            / 100
        )

    def test_structured_contribution_recent_messages_protected(self):
        from agent.context_pruning import build_message_contributions

        protect_recent = _STRUCTURED_PRUNING_TEST_CONFIG[
            "tokenjuice_prompt_pruning_protect_recent_messages"
        ]
        messages = [
            {"role": "user", "content": f"msg-{i:02d}"}
            for i in range(protect_recent + 2)
        ]
        contributions = build_message_contributions(
            messages,
            phase="pre_api",
            protect_recent_messages=protect_recent,
        )

        protected_ids = {
            c["id"]
            for c in contributions[-protect_recent:]
            if c["class"] == "user_message"
        }
        for contribution in contributions[-protect_recent:]:
            if contribution["id"] in protected_ids:
                assert contribution["prune_policy"] == "never"
                assert "recent" in contribution["protected_reason"].lower()

    def test_structured_contribution_unknown_class_protected(self):
        from agent.context_pruning import build_message_contributions

        messages = [{"role": "foobar", "content": "unrecognized role"}]
        contributions = build_message_contributions(messages, phase="pre_api")

        assert len(contributions) == 1
        contribution = contributions[0]
        assert contribution["class"] == "unknown"
        assert contribution["prune_policy"] == "never"
        assert "unknown" in contribution["protected_reason"].lower()

    def test_structured_contribution_cache_scope_and_age_metadata(self):
        from agent.context_pruning import build_message_contributions

        messages = [{"role": "user", "content": "hello"}]
        contributions = build_message_contributions(
            messages,
            phase="pre_api",
            deterministic_now_epoch_ms=1_000_000,
        )

        assert len(contributions) == 1
        contribution = contributions[0]
        assert contribution["cache_scope"] in {"prefix", "body", "tail", "schema", "none"}
        assert contribution.get("created_at_epoch_ms") is not None
        assert isinstance(contribution["created_at_epoch_ms"], int)


class TestStructuredPruningTokenJuiceIntegration:
    def test_actual_hook_disabled_is_noop(self):
        from agent.context_pruning import apply_structured_context_prune

        cfg = dict(_STRUCTURED_PRUNING_TEST_CONFIG)
        cfg["tokenjuice_prompt_pruning_enabled"] = False
        host = _TokenJuiceHost(cfg)
        _load_tokenjuice_register()(host)

        agent = types.SimpleNamespace(
            _structured_pruning_config=cfg,
            _structured_context_prune=host.callbacks.get("structured_context_prune"),
            context_compressor=types.SimpleNamespace(context_length=100_000),
            compression_enabled=True,
            session_id="session-a",
            model="model-a",
            provider="provider-a",
            api_mode="chat_completions",
            tools=None,
        )

        result = apply_structured_context_prune(
            agent,
            phase="pre_api",
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="SYSTEM",
            tools=None,
        )

        assert result is None

    def test_actual_hook_preserves_shape_system_once_and_multicall_atomicity(self):
        from agent.context_pruning import apply_structured_context_prune

        cfg = dict(_STRUCTURED_PRUNING_TEST_CONFIG)
        cfg["tokenjuice_prompt_pruning_threshold_tokens"] = 400
        cfg["tokenjuice_prompt_pruning_trigger_ratio"] = 50
        cfg["tokenjuice_prompt_pruning_soft_target_ratio"] = 40
        cfg["tokenjuice_prompt_pruning_hard_target_ratio"] = 40
        cfg["tokenjuice_prompt_pruning_min_saved_tokens"] = 1
        cfg["tokenjuice_prompt_pruning_cache_ttl_seconds"] = 0
        cfg["tokenjuice_prompt_pruning_protect_recent_messages"] = 0
        cfg["tokenjuice_prompt_pruning_protect_recent_tool_interactions"] = 0
        host = _TokenJuiceHost(cfg)
        _load_tokenjuice_register()(host)

        multimodal_content = [
            {"type": "text", "text": "inspect image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": multimodal_content, "metadata": {"kept": True}},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-a", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
                    {"id": "call-b", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call-a", "name": "terminal", "content": "a" * 1200},
            {"role": "tool", "tool_call_id": "call-b", "name": "terminal", "content": "b" * 1200},
        ]
        agent = types.SimpleNamespace(
            _structured_pruning_config=cfg,
            _structured_context_prune=host.callbacks["structured_context_prune"],
            context_compressor=types.SimpleNamespace(context_length=100_000),
            compression_enabled=True,
            session_id="session-a",
            model="model-a",
            provider="provider-a",
            api_mode="chat_completions",
            tools=None,
        )

        result = apply_structured_context_prune(
            agent,
            phase="pre_api",
            messages=messages,
            system_prompt="SYSTEM",
            tools=None,
        )

        assert result is not None
        assert result["effective_messages"] == messages[:2]
        assert result["effective_system_prompt"] == ""

    def test_actual_hook_uses_target_ratio_when_phase_targets_are_omitted(self):
        from agent.context_pruning import apply_structured_context_prune

        cfg = {
            key: value
            for key, value in _STRUCTURED_PRUNING_TEST_CONFIG.items()
            if key
            not in {
                "tokenjuice_prompt_pruning_soft_target_ratio",
                "tokenjuice_prompt_pruning_hard_target_ratio",
            }
        }
        cfg["tokenjuice_prompt_pruning_threshold_tokens"] = 400
        cfg["tokenjuice_prompt_pruning_trigger_ratio"] = 50
        cfg["tokenjuice_prompt_pruning_target_ratio"] = 90
        cfg["tokenjuice_prompt_pruning_min_saved_tokens"] = 1
        cfg["tokenjuice_prompt_pruning_cache_ttl_seconds"] = 0
        cfg["tokenjuice_prompt_pruning_protect_recent_messages"] = 0
        cfg["tokenjuice_prompt_pruning_protect_recent_tool_interactions"] = 0
        host = _TokenJuiceHost(cfg)
        _load_tokenjuice_register()(host)

        messages = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "u" * 1400},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-a", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-a", "name": "terminal", "content": "a" * 480},
        ]
        agent = types.SimpleNamespace(
            _structured_pruning_config=cfg,
            _structured_context_prune=host.callbacks["structured_context_prune"],
            context_compressor=types.SimpleNamespace(context_length=100_000),
            compression_enabled=True,
            session_id="session-a",
            model="model-a",
            provider="provider-a",
            api_mode="chat_completions",
            tools=None,
        )

        result = apply_structured_context_prune(
            agent,
            phase="pre_api",
            messages=messages,
            system_prompt="SYSTEM",
            tools=None,
        )

        assert result is not None
        assert result["effective_messages"] == messages[:2]

    def test_actual_hook_ttl_protects_real_turn_ephemeral_terminal_candidate(self):
        from agent.context_pruning import build_message_contributions

        cfg = dict(_STRUCTURED_PRUNING_TEST_CONFIG)
        cfg["tokenjuice_prompt_pruning_threshold_tokens"] = 400
        cfg["tokenjuice_prompt_pruning_trigger_ratio"] = 50
        cfg["tokenjuice_prompt_pruning_min_saved_tokens"] = 1
        cfg["tokenjuice_prompt_pruning_protect_recent_messages"] = 0
        cfg["tokenjuice_prompt_pruning_protect_recent_tool_interactions"] = 0
        host = _TokenJuiceHost(cfg)
        _load_tokenjuice_register()(host)
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-a", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-a", "name": "terminal", "content": "a" * 1200},
        ]

        contributions = build_message_contributions(
            messages,
            phase="pre_api",
            deterministic_now_epoch_ms=10_000_000,
        )
        result = host.callbacks["structured_context_prune"](
            contributions,
            phase="pre_api",
            threshold_tokens=cfg["tokenjuice_prompt_pruning_threshold_tokens"],
            trigger_tokens=200,
            now_epoch_ms=10_000_000,
        )

        assert result is not None
        assert result["accounting"]["pruned_count"] == 0
        assert result["effective_messages"] == messages
