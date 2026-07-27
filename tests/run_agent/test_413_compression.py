"""Tests for payload/context-length → compression retry logic in AIAgent.

Verifies that:
- HTTP 413 errors trigger history compression and retry
- HTTP 400 context-length errors trigger compression (not generic 4xx abort)
- Preflight compression proactively compresses oversized sessions before API calls
- Structured context pruning lowers pressure at both gates before compression fires
"""

import copy
import pytest
#pytestmark = pytest.mark.skip(reason="Hangs in non-interactive environments")



from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from agent.context_compressor import SUMMARY_PREFIX
from agent.model_metadata import (
    _estimate_tools_tokens_rough,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)
from run_agent import AIAgent
import run_agent


# ---------------------------------------------------------------------------
# Fast backoff for compression retry tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    """Short-circuit the 2s time.sleep between compression retries.

    Production code has ``time.sleep(2)`` in multiple places after a 413/context
    compression, for rate-limit smoothing. Tests assert behavior, not timing.
    """
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp


def _make_413_error(*, use_status_code=True, message="Request entity too large"):
    """Create an exception that mimics a 413 HTTP error."""
    err = Exception(message)
    if use_status_code:
        err.status_code = 413
    return err


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        # Default matches production (`compression.enabled` defaults to True).
        # Overflow-recovery tests below verify that 413 / context-overflow
        # errors DO trigger compression; the disabled-path behavior is
        # covered explicitly by TestOverflowWithCompactionDisabled.
        a.compression_enabled = True
        a.save_trajectories = False
        return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_current_user_turn_is_persisted_before_provider_call(agent):
    """The inbound user turn is flushed before provider/tool work can crash."""
    observed = []

    def _record_persist(messages, conversation_history):
        observed.append(("persist", list(messages), list(conversation_history or [])))

    def _provider_crash(*_args, **_kwargs):
        observed.append(("provider", [], []))
        raise RuntimeError("provider died after turn-start persistence")

    agent.client.chat.completions.create.side_effect = _provider_crash

    with (
        patch.object(agent, "_persist_session", side_effect=_record_persist),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "new message that must survive a crash",
            conversation_history=[{"role": "user", "content": "old message"}],
        )

    assert result.get("failed") is True
    assert observed[0][0] == "persist"
    assert observed[1][0] == "provider"
    persisted_messages = observed[0][1]
    assert persisted_messages[-1] == {
        "role": "user",
        "content": "new message that must survive a crash",
    }


class TestHTTP413Compression:
    """413 errors should trigger compression, not abort as generic 4xx."""

    def test_413_triggers_compression(self, agent):
        """A 413 error should call _compress_context and retry, not abort."""
        # First call raises 413; second call succeeds after compression.
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Success after compression", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        # Prefill so there are multiple messages for compression to reduce
        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Compression reduces 3 messages down to 1
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "Success after compression"

    def test_413_not_treated_as_generic_4xx(self, agent):
        """413 must NOT hit the generic 4xx abort path; it should attempt compression."""
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        # If 413 were treated as generic 4xx, result would have "failed": True
        assert result.get("failed") is not True
        assert result["completed"] is True

    def test_413_error_message_detection(self, agent):
        """413 detected via error message string (no status_code attr)."""
        err = _make_413_error(use_status_code=False, message="error code: 413")
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_413_strips_vision_payloads_when_compression_cannot_reduce_messages(self, agent):
        """If compression leaves image payloads behind, strip them and retry.

        Browser vision tool results can contain base64 image parts. A 413 can
        persist even after summarisation when the remaining recent tool result
        still carries binary data; Hermes should evict the image payload and
        keep the text/placeholder context instead of failing immediately.
        """
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Recovered after image eviction", finish_reason="stop")
        request_payloads = []

        def _side_effect(**kwargs):
            request_payloads.append(kwargs)
            if len(request_payloads) == 1:
                raise err_413
            return ok_resp

        agent.client.chat.completions.create.side_effect = _side_effect

        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + ("a" * 2000)},
        }
        prefill = [
            {"role": "user", "content": "please inspect this page"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_vision",
                        "type": "function",
                        "function": {"name": "browser_vision", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_vision",
                "name": "browser_vision",
                "content": [
                    {"type": "text", "text": "Screenshot of the dashboard"},
                    image_part,
                ],
            },
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Simulate the bad production case: compression ran, but the
            # recent vision tool message survived so message count did not drop.
            mock_compress.side_effect = lambda msgs, *_a, **_k: (msgs, "compressed prompt")
            result = agent.run_conversation("continue", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after image eviction"
        assert len(request_payloads) == 2
        first_tool = next(m for m in request_payloads[0]["messages"] if m.get("role") == "tool")
        retried_tool = next(m for m in request_payloads[1]["messages"] if m.get("role") == "tool")
        assert "Screenshot of the dashboard" in str(first_tool["content"])
        assert "data:image" not in str(retried_tool["content"])
        assert "Screenshot of the dashboard" in str(retried_tool["content"])
        assert not getattr(agent, "_no_list_tool_content_models", set())

    def test_413_clears_conversation_history_on_persist(self, agent):
        """After 413-triggered compression, _persist_session must receive None history.

        Bug: _compress_context() creates a new session and resets _last_flushed_db_idx=0,
        but if conversation_history still holds the original (pre-compression) list,
        _flush_messages_to_session_db computes flush_from = max(len(history), 0) which
        exceeds len(compressed_messages), so messages[flush_from:] is empty and nothing
        is written to the new session → "Session found but has no messages" on resume.
        """
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(200)
        ]

        persist_calls = []

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(
                agent, "_persist_session",
                side_effect=lambda msgs, hist: persist_calls.append((list(msgs), hist)),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=big_history)

        assert any(hist is None for _msgs, hist in persist_calls), (
            "Expected at least one post-compression _persist_session call "
            "with conversation_history=None"
        )

    def test_context_overflow_clears_conversation_history_on_persist(self, agent):
        """After context-overflow compression, _persist_session must receive None history."""
        err_400 = Exception(
            "Error code: 400 - This endpoint's maximum context length is 128000 tokens. "
            "However, you requested about 270460 tokens."
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(200)
        ]

        persist_calls = []

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(
                agent, "_persist_session",
                side_effect=lambda msgs, hist: persist_calls.append((list(msgs), hist)),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=big_history)

        assert any(hist is None for _msgs, hist in persist_calls)

    def test_400_context_length_triggers_compression(self, agent):
        """A 400 with 'maximum context length' should trigger compression, not abort as generic 4xx.

        OpenRouter returns HTTP 400 (not 413) for context-length errors. Before
        the fix, this was caught by the generic 4xx handler which aborted
        immediately — now it correctly triggers compression+retry.
        """
        err_400 = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This endpoint's maximum context length is 204800 tokens. "
            "However, you requested about 270460 tokens.\", 'code': 400}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered after compression", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        # Must NOT have "failed": True (which would mean the generic 4xx handler caught it)
        assert result.get("failed") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after compression"

    def test_400_reduce_length_triggers_compression(self, agent):
        """A 400 with 'reduce the length' should trigger compression."""
        err_400 = Exception(
            "Error code: 400 - Please reduce the length of the messages"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_context_length_retry_rebuilds_request_after_compression(self, agent):
        """Retry must send the compressed transcript, not the stale oversized payload."""
        err_400 = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This endpoint's maximum context length is 128000 tokens. "
            "Please reduce the length of the messages.\"}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered after real compression", finish_reason="stop")

        request_payloads = []

        def _side_effect(**kwargs):
            request_payloads.append(kwargs)
            if len(request_payloads) == 1:
                raise err_400
            return ok_resp

        agent.client.chat.completions.create.side_effect = _side_effect

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        assert result["completed"] is True
        assert len(request_payloads) == 2
        assert len(request_payloads[1]["messages"]) < len(request_payloads[0]["messages"])
        assert request_payloads[1]["messages"][0] == {
            "role": "system",
            "content": "compressed prompt",
        }
        assert request_payloads[1]["messages"][1] == {
            "role": "user",
            "content": "compressed summary",
        }

    def test_413_cannot_compress_further(self, agent):
        """When compression can't reduce messages, return partial result."""
        err_413 = _make_413_error()
        agent.client.chat.completions.create.side_effect = [err_413]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Compression returns same number of messages → can't compress further
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "same prompt",
            )
            result = agent.run_conversation("hello")

        assert result["completed"] is False
        assert result.get("partial") is True
        assert "413" in result["error"]

    def test_413_retries_on_token_only_compression(self, agent):
        """Same message COUNT but fewer TOKENS must count as progress and retry.

        Regression for #39550/#23767: tool-result pruning / in-place
        summarization can shrink request size without dropping the message
        count. The old gate (len(messages) < original_len) treated that as
        'cannot compress further' and aborted; the fix re-estimates tokens and
        retries when they drop materially.
        """
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="OK after token-only compaction", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        # 3 large messages in, 3 much smaller messages out (same count, far
        # fewer tokens) — exactly the token-only-progress case.
        prefill = [
            {"role": "user", "content": "x" * 4000},
            {"role": "assistant", "content": "y" * 4000},
            {"role": "user", "content": "z" * 4000},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Same message count (3) but ~10x smaller content → token drop.
            mock_compress.return_value = (
                [
                    {"role": "user", "content": "x" * 300},
                    {"role": "assistant", "content": "y" * 300},
                    {"role": "user", "content": "z" * 300},
                ],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "OK after token-only compaction"


class TestPreflightCompression:
    """Preflight compression should compress history before the first API call."""

    def test_compress_context_emits_lifecycle_status_before_work(self, agent):
        """Direct context compression should tell gateway users why the turn paused."""
        # This test calls _compress_context directly and asserts the FIRST
        # status event is the lifecycle "Compacting context" message. With
        # compaction enabled the lazy feasibility probe would emit an
        # aux-provider warning first (no aux key in the hermetic test env),
        # displacing events[0]. The flag value is irrelevant to what this
        # test asserts, so disable it to suppress the probe.
        agent.compression_enabled = False
        events = []
        agent.status_callback = lambda ev, msg: events.append((ev, msg))

        def _fake_compress(
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            events.append(("compress", "started"))
            return [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}]

        with (
            patch.object(agent.context_compressor, "compress", side_effect=_fake_compress),
            patch.object(agent, "_build_system_prompt", return_value="new system prompt") as build_prompt,
            patch("run_agent.estimate_request_tokens_rough", return_value=42),
        ):
            compressed, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        # The compressor returned only the user-role summary; a summary
        # message no longer satisfies the human-anchor check, so the real
        # user turn is restored after it (repair merges the pair
        # summary-first before the next API call).
        assert compressed == [
            {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
            {"role": "user", "content": "hello"},
        ]
        assert new_system_prompt == "You are helpful."
        build_prompt.assert_not_called()
        assert events[0][0] == "lifecycle"
        assert "Compacting context" in events[0][1]
        assert events[1] == ("compress", "started")

    def test_compression_reuses_cached_prompt_when_memory_snapshot_is_unchanged(self, agent):
        """A memory reload without new injected text must keep the cache prefix."""
        agent.compression_enabled = False
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._cached_system_prompt = (
            "cached system prompt\n\n<memory>same facts</memory>"
        )
        memory_store = MagicMock()
        memory_store.format_for_system_prompt.return_value = "<memory>same facts</memory>"
        agent._memory_store = memory_store

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=[{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
            ),
            patch.object(agent, "_build_system_prompt") as build_prompt,
        ):
            _, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert new_system_prompt is agent._cached_system_prompt
        assert new_system_prompt == "cached system prompt\n\n<memory>same facts</memory>"
        build_prompt.assert_not_called()
        memory_store.load_from_disk.assert_called_once()

    def test_compression_rebuilds_prompt_when_memory_snapshot_changes(self, agent):
        """A changed memory block must be reflected in the next model request."""
        agent.compression_enabled = False
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._cached_system_prompt = (
            "cached system prompt\n\n<memory>old facts</memory>"
        )
        memory_store = MagicMock()
        memory_store.format_for_system_prompt.return_value = "<memory>new facts</memory>"
        agent._memory_store = memory_store

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=[{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
            ),
            patch.object(agent, "_build_system_prompt", return_value="rebuilt system prompt") as build_prompt,
        ):
            _, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert new_system_prompt == "rebuilt system prompt"
        build_prompt.assert_called_once_with("system prompt")
        memory_store.load_from_disk.assert_called_once()

    def test_compression_rebuilds_when_restored_prompt_predates_memory_write(self, agent):
        """Gateway fresh-agent path: a session-DB-restored prompt built with OLD
        memory must be rebuilt even though the in-memory snapshot is identical
        before and after the disk reload (the fresh MemoryStore already
        absorbed the mid-session write at init). Guards the containment check
        against regressing to before/after snapshot equality."""
        agent.compression_enabled = False
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_manager = None
        # Restored from SessionDB in an earlier process — built with fact A only.
        agent._cached_system_prompt = "system prompt\n\n<memory>fact A</memory>"
        memory_store = MagicMock()
        # Fresh store loaded fact A + fact B at agent init; stable across reload.
        memory_store.format_for_system_prompt.return_value = "<memory>fact A\nfact B</memory>"
        agent._memory_store = memory_store

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=[{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
            ),
            patch.object(agent, "_build_system_prompt", return_value="rebuilt with fact B") as build_prompt,
        ):
            _, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert new_system_prompt == "rebuilt with fact B"
        build_prompt.assert_called_once_with("system prompt")

    def test_compression_rebuilds_when_prompt_has_leftover_block_for_emptied_memory(self, agent):
        """A prompt still carrying a memory block after all entries were
        removed must be rebuilt — empty current blocks are vacuously
        'contained', so the leftover-header check has to catch this."""
        agent.compression_enabled = False
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._cached_system_prompt = (
            "system prompt\n\nMEMORY (your personal notes) [1% — 10/2,200 chars]\nold fact"
        )
        memory_store = MagicMock()
        memory_store.format_for_system_prompt.return_value = None  # emptied
        agent._memory_store = memory_store

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=[{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
            ),
            patch.object(agent, "_build_system_prompt", return_value="rebuilt without memory") as build_prompt,
        ):
            _, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert new_system_prompt == "rebuilt without memory"
        build_prompt.assert_called_once_with("system prompt")

    def test_preflight_compresses_oversized_history(self, agent):
        """When loaded history exceeds the model's context threshold, compress before API call."""
        agent.compression_enabled = True
        # Set a small context so the history is "oversized", but large enough
        # that the compressed result (2 short messages) fits in a single pass.
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 200

        # Build a history that will be large enough to trigger preflight
        # (each message ~50 chars ≈ 13 tokens, 40 messages ≈ 520 tokens > 200 threshold)
        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message number {i} with some extra text padding"})
            big_history.append({"role": "assistant", "content": f"Response number {i} with extra padding here"})

        ok_resp = _mock_response(content="After preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Simulate compression reducing messages to a small set that fits
            mock_compress.return_value = (
                [
                    {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
                    {"role": "user", "content": "hello"},
                ],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        # Preflight compression is a multi-pass loop (up to 3 passes for very
        # large sessions, breaking when no further reduction is possible).
        # First pass must have received the full oversized history.
        assert mock_compress.call_count >= 1, "Preflight compression never ran"
        first_call_messages = mock_compress.call_args_list[0].args[0]
        assert len(first_call_messages) >= 40, (
            f"First preflight pass should see the full history, got "
            f"{len(first_call_messages)} messages"
        )
        assert result["completed"] is True
        assert result["final_response"] == "After preflight"
        assert any(
            ev == "lifecycle" and "Preflight compression" in msg
            for ev, msg in status_messages
        )

    def test_preflight_defers_when_recent_real_usage_fit(self, agent):
        """A noisy rough estimate should not re-compact a recently fitting request."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000
        agent.context_compressor.last_prompt_tokens = 58_000
        agent.context_compressor.last_real_prompt_tokens = 58_000
        agent.context_compressor.last_rough_tokens_when_real_prompt_fit = 113_000

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(
            content="Used real fit",
            finish_reason="stop",
            usage={"prompt_tokens": 59_000, "completion_tokens": 100, "total_tokens": 59_100},
        )
        agent.client.chat.completions.create.side_effect = [ok_resp]
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=114_000),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=114_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_not_called()
        assert result["completed"] is True
        assert result["final_response"] == "Used real fit"
        assert not any(
            ev == "lifecycle" and "Preflight compression" in msg
            for ev, msg in status_messages
        )

    def test_preflight_compresses_when_rough_growth_after_fit_is_large(self, agent):
        """Large rough growth after a fitting request still triggers preflight."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000
        agent.context_compressor.last_prompt_tokens = 58_000
        agent.context_compressor.last_real_prompt_tokens = 58_000
        agent.context_compressor.last_rough_tokens_when_real_prompt_fit = 113_000

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(
            content="Compressed after growth",
            finish_reason="stop",
            usage={"prompt_tokens": 50_000, "completion_tokens": 100, "total_tokens": 50_100},
        )
        agent.client.chat.completions.create.side_effect = [ok_resp]

        # First rough estimate must clear the threshold so preflight fires
        # (rough growth since the last fitting request is large, so the
        # deferral path is NOT taken). Every estimate after compaction is
        # sub-threshold. Use a callable side_effect rather than a fixed list
        # so we don't have to predict how many times the loop re-estimates —
        # the post-response real-token estimate is an extra call that a
        # 2-element list would exhaust (StopIteration).
        _rough_calls = {"n": 0}

        def _rough_estimate(*_args, **_kwargs):
            _rough_calls["n"] += 1
            return 125_000 if _rough_calls["n"] == 1 else 40_000

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", side_effect=_rough_estimate),
            patch("agent.conversation_loop.estimate_request_tokens_rough", side_effect=_rough_estimate),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_no_preflight_when_under_threshold(self, agent):
        """When history fits within context, no preflight compression needed."""
        agent.compression_enabled = True
        # Large context — history easily fits
        agent.context_compressor.context_length = 1000000
        agent.context_compressor.threshold_tokens = 850000

        small_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        ok_resp = _mock_response(content="No compression needed", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=small_history)

        mock_compress.assert_not_called()
        assert result["completed"] is True

    def test_no_preflight_when_compression_disabled(self, agent):
        """Preflight should not run when compression is disabled."""
        agent.compression_enabled = False
        agent.context_compressor.context_length = 100
        agent.context_compressor.threshold_tokens = 85

        big_history = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
        ] * 10

        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_not_called()

    def test_preflight_respects_anti_thrash(self, agent):
        """Preflight must call ``should_compress()`` so anti-thrash applies.

        Regression for #29335 — preflight used to bypass ``should_compress()``
        and re-trigger every turn even when the prior two passes each saved
        <10% (the canonical infinite-compression-loop signal).
        """
        agent.compression_enabled = True
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 200

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(content="No preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent.context_compressor, "should_compress", return_value=False) as mock_should,
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        # The gate consulted should_compress — anti-thrash had a chance to vote.
        mock_should.assert_called()
        # And vetoed: even though tokens >= threshold, no compression ran.
        mock_compress.assert_not_called()
        assert result["completed"] is True

    def test_preflight_seeds_display_tokens_when_compression_aborts(self, agent):
        """Display must reflect the real context size even when compression no-ops.

        Regression: the CLI status bar reads ``last_prompt_tokens``, which only
        updated from a *successful* API response. When the loaded history was
        oversized but compression failed to reduce it (e.g. the auxiliary
        summary model timed out), the bar stayed stuck at the old, smaller
        value while the preflight estimate reported a much larger number —
        looking permanently out of sync.
        """
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000
        # Simulate a stale display value from an earlier, smaller turn.
        agent.context_compressor.last_prompt_tokens = 74_400

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded text"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded text"})

        ok_resp = _mock_response(content="After preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=144_669),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=144_669),
            # Compression no-ops (returns input unchanged) — mirrors an aux
            # summary-model timeout where the messages can't be reduced.
            patch.object(agent, "_compress_context", side_effect=lambda msgs, *a, **k: (msgs, agent._cached_system_prompt)),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        assert result["completed"] is True
        # The display token count was revised up to the fresh preflight estimate,
        # not left at the stale 74_400.
        assert agent.context_compressor.last_prompt_tokens == 144_669

    def test_preflight_seed_only_revises_upward(self, agent):
        """A larger tracked value must not be clobbered by a smaller estimate."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000
        # A real, larger usage figure is already tracked.
        agent.context_compressor.last_prompt_tokens = 160_000

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded text"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded text"})

        ok_resp = _mock_response(content="After preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=144_669),
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=144_669),
            patch.object(agent, "_compress_context", side_effect=lambda msgs, *a, **k: (msgs, agent._cached_system_prompt)),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("hello", conversation_history=big_history)

        # Smaller estimate must not overwrite the larger tracked value.
        assert agent.context_compressor.last_prompt_tokens == 160_000


class TestToolResultPreflightCompression:
    """Compression should trigger when tool results push context past the threshold."""

    def test_large_tool_results_trigger_compression(self, agent):
        """When tool results push estimated tokens past threshold, compress before next call."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000  # below the 135k reported usage
        agent.context_compressor.last_prompt_tokens = 130_000
        agent.context_compressor.last_completion_tokens = 5_000

        tc = SimpleNamespace(
            id="tc1", type="function",
            function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'),
        )
        tool_resp = _mock_response(
            content=None, finish_reason="stop", tool_calls=[tc],
            usage={"prompt_tokens": 130_000, "completion_tokens": 5_000, "total_tokens": 135_000},
        )
        ok_resp = _mock_response(
            content="Done after compression", finish_reason="stop",
            usage={"prompt_tokens": 50_000, "completion_tokens": 100, "total_tokens": 50_100},
        )
        agent.client.chat.completions.create.side_effect = [tool_resp, ok_resp]
        large_result = "x" * 100_000

        with (
            patch("run_agent.handle_function_call", return_value=large_result),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}], "compressed prompt",
            )
            result = agent.run_conversation("hello")

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_anthropic_prompt_too_long_safety_net(self, agent):
        """Anthropic 'prompt is too long' error triggers compression as safety net."""
        err_400 = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'prompt is too long: 233153 tokens > 200000 maximum'}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]
        prefill = [
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}], "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True


# ---------------------------------------------------------------------------
# Disabled auto-compaction on overflow (port of anomalyco/opencode#30749)
# ---------------------------------------------------------------------------

class TestOverflowWithCompactionDisabled:
    """When ``compression.enabled`` is False, NO automatic compaction may
    fire — including the provider/request-size overflow recovery paths.

    Ported from anomalyco/opencode#30749: the proactive token-threshold
    path already honoured the setting, but provider overflow errors
    (413 payload-too-large, context-overflow, long-context-tier 429) still
    silently compressed + rotated the session. The fix surfaces a terminal
    error so the user can compact manually, start fresh, or switch models.
    """

    @staticmethod
    def _prefill():
        return [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

    def test_413_does_not_compress_when_disabled(self, agent):
        """413 must NOT call _compress_context when compaction is disabled."""
        agent.compression_enabled = False
        err_413 = _make_413_error()
        # If the guard fails, a second (success) response would be consumed.
        agent.client.chat.completions.create.side_effect = [err_413, _mock_response()]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session") as mock_persist,
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=self._prefill())

        mock_compress.assert_not_called()
        mock_persist.assert_called()
        assert result.get("failed") is True
        assert result.get("compaction_disabled") is True
        assert "auto-compaction is disabled" in result["error"]

    def test_context_overflow_does_not_compress_when_disabled(self, agent):
        """400 'prompt is too long' must NOT compress when compaction disabled."""
        agent.compression_enabled = False
        err_400 = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': 'prompt is too long: "
            "233153 tokens > 200000 maximum'}}"
        )
        err_400.status_code = 400
        agent.client.chat.completions.create.side_effect = [err_400, _mock_response()]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=self._prefill())

        mock_compress.assert_not_called()
        assert result.get("compaction_disabled") is True

    def test_413_still_compresses_when_enabled(self, agent):
        """Control: with compaction enabled, 413 still triggers compression.

        Guards against the disabled-path guard accidentally swallowing the
        enabled path.
        """
        agent.compression_enabled = True
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}], "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=self._prefill())

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result.get("compaction_disabled") is not True


# ---------------------------------------------------------------------------
# Structured context pruning delays/avoid compaction (Todo 7)
# ---------------------------------------------------------------------------

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


def _setup_structured_pruning_agent(agent, cfg: dict[str, bool | int | str]) -> None:
    """Configure an AIAgent fixture for structured-pruning tests."""
    agent.compression_enabled = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.context_compressor.context_length = int(
        cfg["tokenjuice_prompt_pruning_threshold_tokens"]
        / (cfg["tokenjuice_prompt_pruning_trigger_ratio"] / 100)
    )
    agent.context_compressor.threshold_tokens = cfg["tokenjuice_prompt_pruning_threshold_tokens"]
    agent.context_compressor.threshold_percent = cfg["tokenjuice_prompt_pruning_trigger_ratio"] / 100
    agent.context_compressor.protect_first_n = 2
    agent.context_compressor.protect_last_n = 2
    agent.context_compressor.last_prompt_tokens = 1
    agent._structured_pruning_config = cfg


def _chars_for_tokens(tokens: int) -> int:
    """Return a character count that estimates to at least *tokens* tokens."""
    return tokens * 4


def _make_small_messages(count: int) -> list:
    """Return many short messages to exercise the count-based preflight gate."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"short message {i:03d}"}
        for i in range(count)
    ]


def _make_terminal_interaction(terminal_chars: int, call_id: str = "call-terminal-1") -> list:
    """Return an assistant tool_call + terminal tool_result pair."""
    return [
        {
            "role": "assistant",
            "content": "I'll run a terminal command",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "name": "terminal", "content": "x" * terminal_chars},
    ]


def _make_history_with_terminal(
    protected_content: str,
    terminal_chars: int,
    filler_count: int = 10,
) -> list:
    """Return a history with protected messages and an old terminal output."""
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "previous turn"},
    ]
    history.extend(_make_terminal_interaction(terminal_chars))
    history.append({"role": "assistant", "content": protected_content})
    history.extend(_make_small_messages(filler_count))
    return history


def _pruned_view(messages: list, tools: list | None = None) -> dict:
    """Build a structured-pruning result dict for a test hook."""
    return {
        "effective_messages": messages,
        "effective_system_prompt": "You are helpful.",
        "effective_tools": tools,
        "accounting": {
            "pruned_count": 1,
            "saved_tokens": 1000,
            "redacted": [{"id": "call-terminal-1", "class": "terminal_tool_output"}],
        },
    }


def _make_policy_hook(
    cfg: dict[str, bool | int | str],
    *,
    preflight_pruned,
    pre_api_pruned,
):
    """Return a test hook that simulates TokenJuice soft/hard pressure routing.

    The hook inspects the unpruned contribution pressure and the available
    disposable ``terminal_tool_output`` tokens, then returns the supplied
    effective view only when safe candidates can meet the configured target.
    Otherwise it fails open (returns ``None``) so Hermes compression fires.
    """
    threshold_tokens = cfg["tokenjuice_prompt_pruning_threshold_tokens"]
    trigger_tokens = int(threshold_tokens * cfg["tokenjuice_prompt_pruning_trigger_ratio"] / 100)
    soft_target = int(threshold_tokens * cfg["tokenjuice_prompt_pruning_soft_target_ratio"] / 100)
    hard_target = int(threshold_tokens * cfg["tokenjuice_prompt_pruning_hard_target_ratio"] / 100)

    def _hook(contributions, **kwargs):
        total = sum(c.get("token_estimate", 0) for c in contributions if isinstance(c, dict))
        if total < trigger_tokens:
            return None

        disposable_tokens = sum(
            c.get("token_estimate", 0)
            for c in contributions
            if isinstance(c, dict)
            and c.get("class") == "terminal_tool_output"
            and c.get("prune_policy") != "never"
        )

        pruned_total = total - disposable_tokens
        if total >= threshold_tokens:
            if pruned_total > hard_target:
                return None
        elif pruned_total > soft_target:
            return None

        phase = kwargs.get("phase", "preflight")
        view = preflight_pruned if phase == "preflight" else pre_api_pruned
        if callable(view):
            view = view(contributions)
        return view

    return _hook


class TestStructuredPruningDelaysCompaction:
    """Structured pruning lowers the exact pressure consumed by both gates."""

    def test_below_trigger_no_pruning_no_compression(self, agent):
        """Below trigger, the hook returns None and compression never fires."""
        cfg = _STRUCTURED_PRUNING_TEST_CONFIG
        _setup_structured_pruning_agent(agent, cfg)

        # Enough messages to pass the cheap count gate, but well below trigger tokens.
        conversation_history = _make_small_messages(30)
        canonical_snapshot = copy.deepcopy(conversation_history)
        hook_phases = []

        def _capture_hook(contributions, **kwargs):
            hook_phases.append(kwargs.get("phase"))
            return None

        agent._structured_context_prune = _capture_hook
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        preflight_pressures = []

        def _spy_preflight_estimate(*args, **kwargs):
            value = estimate_request_tokens_rough(*args, **kwargs)
            preflight_pressures.append(value)
            return value

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.turn_context.estimate_request_tokens_rough", side_effect=_spy_preflight_estimate),
        ):
            result = agent.run_conversation("hello", conversation_history=conversation_history)

        assert result["completed"] is True
        assert "preflight" in hook_phases
        mock_compress.assert_not_called()
        # Canonical history is unchanged.
        assert conversation_history == canonical_snapshot
        # Provider payload kept the full canonical view (system prompt + messages).
        provider_kwargs = agent.client.chat.completions.create.call_args.kwargs
        provider_messages = provider_kwargs.get("messages", [])
        assert len(provider_messages) == len(conversation_history) + 2
        # Preflight pressure stayed below the trigger.
        assert all(
            p < cfg["tokenjuice_prompt_pruning_threshold_tokens"] * 0.8
            for p in preflight_pressures
        )

    def test_soft_pressure_preflight_avoids_compression(self, agent):
        """Soft pressure at the preflight gate is pruned before compression fires."""
        cfg = _STRUCTURED_PRUNING_TEST_CONFIG
        _setup_structured_pruning_agent(agent, cfg)

        trigger_tokens = int(cfg["tokenjuice_prompt_pruning_threshold_tokens"] * 0.8)
        # Protected messages ~1000 tokens + terminal ~6500 tokens -> ~7500 tokens (soft).
        protected = "x" * _chars_for_tokens(1000)
        terminal_chars = _chars_for_tokens(6500)
        conversation_history = _make_history_with_terminal(protected, terminal_chars, filler_count=30)
        canonical_snapshot = copy.deepcopy(conversation_history)

        pruned_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "run command"},
        ]
        agent._structured_context_prune = _make_policy_hook(
            cfg,
            preflight_pruned=_pruned_view(pruned_messages),
            pre_api_pruned=_pruned_view(pruned_messages),
        )

        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        preflight_pressures = []

        def _spy_preflight_estimate(*args, **kwargs):
            value = estimate_request_tokens_rough(*args, **kwargs)
            preflight_pressures.append(value)
            return value

        pre_api_msg_pressures = []

        def _spy_pre_api_messages(messages):
            value = estimate_messages_tokens_rough(messages)
            pre_api_msg_pressures.append(value)
            return value

        should_compress_inputs = []

        def _spy_should_compress(tokens=None):
            should_compress_inputs.append(tokens)
            return tokens >= agent.context_compressor.threshold_tokens

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.turn_context.estimate_request_tokens_rough", side_effect=_spy_preflight_estimate),
            patch("agent.conversation_loop.estimate_messages_tokens_rough", side_effect=_spy_pre_api_messages),
            patch.object(agent.context_compressor, "should_compress", side_effect=_spy_should_compress),
        ):
            result = agent.run_conversation("run command", conversation_history=conversation_history)

        assert result["completed"] is True
        mock_compress.assert_not_called()
        # Both gates consumed pruned pressure below the trigger.
        assert all(p < trigger_tokens for p in preflight_pressures)
        assert all(p < trigger_tokens for p in pre_api_msg_pressures)
        assert all(t < trigger_tokens for t in should_compress_inputs if t is not None)
        provider_messages = agent.client.chat.completions.create.call_args.kwargs.get("messages", [])
        assert provider_messages == pruned_messages
        # Canonical history is unchanged.
        assert conversation_history == canonical_snapshot
        assert any(
            msg.get("role") == "tool" and "x" * terminal_chars in msg.get("content", "")
            for msg in result["messages"]
        )

    def test_soft_pressure_pre_api_avoids_compression(self, agent):
        """Soft pressure at the pre-API gate is pruned before compression fires."""
        cfg = {
            **_STRUCTURED_PRUNING_TEST_CONFIG,
            "tokenjuice_prompt_pruning_protect_recent_tool_interactions": 0,
        }
        _setup_structured_pruning_agent(agent, cfg)

        trigger_tokens = int(cfg["tokenjuice_prompt_pruning_threshold_tokens"] * 0.8)
        # Preflight is below trigger; a tool result later pushes pre-API into soft pressure.
        conversation_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        canonical_snapshot = copy.deepcopy(conversation_history)
        user_message = "run terminal"

        tc = SimpleNamespace(
            id="call-terminal-2",
            type="function",
            function=SimpleNamespace(name="terminal", arguments="{}"),
        )
        tool_resp = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        final_resp = _mock_response(content="Done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [tool_resp, final_resp]

        agent.valid_tool_names = {"terminal", "web_search"}

        tool_result_chars = _chars_for_tokens(7000)

        def _inject_terminal_result(assistant_message, messages, effective_task_id, api_call_count=0):
            for _tc in assistant_message.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": _tc.id,
                    "name": "terminal",
                    "content": "x" * tool_result_chars,
                })

        pruned_api_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-terminal-2",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-terminal-2", "name": "terminal", "content": "[pruned]"},
        ]
        agent._structured_context_prune = _make_policy_hook(
            cfg,
            preflight_pruned=None,
            pre_api_pruned=_pruned_view(pruned_api_messages),
        )

        pre_api_msg_pressures = []

        def _spy_pre_api_messages(messages):
            value = estimate_messages_tokens_rough(messages)
            pre_api_msg_pressures.append(value)
            return value

        should_compress_inputs = []

        def _spy_should_compress(tokens=None):
            should_compress_inputs.append(tokens)
            return tokens >= agent.context_compressor.threshold_tokens

        with (
            patch.object(agent, "_execute_tool_calls", side_effect=_inject_terminal_result),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.conversation_loop.estimate_messages_tokens_rough", side_effect=_spy_pre_api_messages),
            patch.object(agent.context_compressor, "should_compress", side_effect=_spy_should_compress),
        ):
            result = agent.run_conversation(user_message, conversation_history=conversation_history)

        assert result["completed"] is True
        assert result["final_response"] == "Done"
        mock_compress.assert_not_called()
        # The second (post-tool) pre-API estimate consumed the pruned view.
        assert all(p < trigger_tokens for p in pre_api_msg_pressures)
        assert all(t < trigger_tokens for t in should_compress_inputs if t is not None)
        create_calls = agent.client.chat.completions.create.call_args_list
        assert len(create_calls) == 2
        assert create_calls[1].kwargs.get("messages") == pruned_api_messages
        # Canonical history still carries the original tool result.
        assert conversation_history == canonical_snapshot
        assert any(
            msg.get("role") == "tool" and "x" * tool_result_chars in msg.get("content", "")
            for msg in result["messages"]
        )

    def test_hard_pressure_avoids_compression_when_candidates_sufficient(self, agent):
        """Hard pressure with enough disposable candidates: prune, do not compress."""
        cfg = _STRUCTURED_PRUNING_TEST_CONFIG
        _setup_structured_pruning_agent(agent, cfg)

        trigger_tokens = int(cfg["tokenjuice_prompt_pruning_threshold_tokens"] * 0.8)
        # Protected ~1000 tokens + terminal ~8000 tokens -> ~9000 tokens (hard).
        protected = "x" * _chars_for_tokens(1000)
        terminal_chars = _chars_for_tokens(8000)
        conversation_history = _make_history_with_terminal(protected, terminal_chars)
        canonical_snapshot = copy.deepcopy(conversation_history)

        pruned_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "run command"},
        ]
        agent._structured_context_prune = _make_policy_hook(
            cfg,
            preflight_pruned=_pruned_view(pruned_messages),
            pre_api_pruned=_pruned_view(pruned_messages),
        )

        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        preflight_pressures = []

        def _spy_preflight_estimate(*args, **kwargs):
            value = estimate_request_tokens_rough(*args, **kwargs)
            preflight_pressures.append(value)
            return value

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.turn_context.estimate_request_tokens_rough", side_effect=_spy_preflight_estimate),
        ):
            result = agent.run_conversation("run command", conversation_history=conversation_history)

        assert result["completed"] is True
        mock_compress.assert_not_called()
        assert all(p < trigger_tokens for p in preflight_pressures)
        assert conversation_history == canonical_snapshot

    def test_hard_pressure_falls_back_to_compression_when_candidates_insufficient(self, agent):
        """Hard pressure where safe candidates cannot meet target falls through to compression."""
        cfg = _STRUCTURED_PRUNING_TEST_CONFIG
        _setup_structured_pruning_agent(agent, cfg)

        # Protected ~7000 tokens + terminal ~3000 tokens -> ~10000 tokens (hard).
        # Clearing terminal leaves ~7000 tokens, still above hard target 5525.
        protected = "x" * _chars_for_tokens(7000)
        terminal_chars = _chars_for_tokens(3000)
        conversation_history = _make_history_with_terminal(protected, terminal_chars)

        agent._structured_context_prune = _make_policy_hook(
            cfg,
            preflight_pruned=None,
            pre_api_pruned=None,
        )

        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation("run command", conversation_history=conversation_history)

        assert result["completed"] is True
        mock_compress.assert_called()
        assert result["final_response"] == "OK"

    def test_llm_request_fallback_alone_does_not_stop_compression(self, agent):
        """With the structured seam disabled, request-time pruning alone cannot prevent compression.

        The existing ``llm_request`` / ``prune_request_messages`` fallback runs
        late, after both compression gates have already made their decisions from
        the unpruned view. This test proves compression still fires when only
        that fallback is available.
        """
        cfg = dict(_STRUCTURED_PRUNING_TEST_CONFIG)
        cfg["tokenjuice_prompt_pruning_enabled"] = False
        _setup_structured_pruning_agent(agent, cfg)

        # Same shape as the hard-insufficient case; unpruned pressure is above threshold.
        protected = "x" * _chars_for_tokens(7000)
        terminal_chars = _chars_for_tokens(3000)
        conversation_history = _make_history_with_terminal(protected, terminal_chars)

        # No agent-bound hook; config disabled so apply_structured_context_prune is no-op.
        agent._structured_context_prune = MagicMock()

        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation("run command", conversation_history=conversation_history)

        assert result["completed"] is True
        agent._structured_context_prune.assert_not_called()
        mock_compress.assert_called()
