import asyncio  # noqa: F401  # noqa: ANYIO_OK
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    PlatformConfig,
    SendResult,
)
from gateway.receipts import DeliveryOutcome, DeliveryReceipt, DeliveryStatus
from gateway.run import GatewayRunner, _strip_response_attachments_for_direct_send
from gateway.session import SessionSource


class _ReceiptAdapter(BasePlatformAdapter):
    def __init__(self, result: SendResult) -> None:
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.MATRIX)
        self._result = result

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return self._result

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        user_id="@rabbit:example.org",
    )


@pytest.mark.parametrize("rabbit_first", [False, True])
@pytest.mark.asyncio
async def test_receipt_event_never_merges_with_matrix_debounce(
    rabbit_first: bool,
) -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))
    matrix_event = MessageEvent(text="matrix", source=_source())
    rabbit_event = MessageEvent(
        text="rabbit",
        source=_source(),
        metadata={"gateway_session_strict": True, "rabbit_ingress": True},
        allow_gateway_control=False,
        receipt=DeliveryReceipt(),
    )
    existing, incoming = (
        (rabbit_event, matrix_event) if rabbit_first else (matrix_event, rabbit_event)
    )

    assert not adapter._can_merge_text_debounce_events(existing, incoming)

    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    assert not adapter._is_queue_text_debounce_candidate(rabbit_event)
    await adapter._queue_text_debounce("matrix:bound", existing)
    await adapter._queue_text_debounce("matrix:bound", incoming)

    assert adapter._pending_messages["matrix:bound"] is existing
    buffered = adapter._text_debounce_store()["matrix:bound"]
    assert buffered.event is incoming
    assert buffered.event.text in {"matrix", "rabbit"}
    if buffered.task is not None:
        buffered.task.cancel()
        await asyncio.gather(buffered.task, return_exceptions=True)


def test_queued_receipt_text_matches_visible_delivery_text() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))

    visible = _strip_response_attachments_for_direct_send(
        "visible answer\nMEDIA:/tmp/internal.mp3\n[[audio_as_voice]]",
        adapter,
    )

    assert visible == "visible answer"


@pytest.mark.asyncio
async def test_receipt_keeps_first_terminal_outcome() -> None:
    receipt = DeliveryReceipt()

    receipt.complete(DeliveryOutcome(DeliveryStatus.DELIVERED, response="first"))
    receipt.complete(DeliveryOutcome(DeliveryStatus.DELIVERY_FAILED, error="late"))

    assert await receipt.wait() == DeliveryOutcome(
        DeliveryStatus.DELIVERED,
        response="first",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_response", "send_result", "expected_status"),
    [
        ("answer", SendResult(success=True, message_id="m1"), DeliveryStatus.DELIVERED),
        (None, SendResult(success=True), DeliveryStatus.NO_RESPONSE),
        (
            "answer",
            SendResult(success=False, error="matrix down"),
            DeliveryStatus.DELIVERY_FAILED,
        ),
    ],
)
async def test_base_processing_resolves_exact_delivery_outcome(
    handler_response: str | None,
    send_result: SendResult,
    expected_status: DeliveryStatus,
) -> None:
    adapter = _ReceiptAdapter(send_result)

    async def handler(event: MessageEvent) -> str | None:
        return handler_response

    adapter.set_message_handler(handler)
    receipt = DeliveryReceipt()
    event = MessageEvent(text="question", source=_source(), receipt=receipt)

    await adapter._process_message_background(event, "matrix:@rabbit:example.org")

    outcome = await receipt.wait()
    assert outcome.status is expected_status
    assert outcome.response == handler_response
    if expected_status is DeliveryStatus.DELIVERY_FAILED:
        assert outcome.error == "matrix down"


@pytest.mark.asyncio
async def test_base_processing_returns_exact_streamed_response() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))

    async def handler(event: MessageEvent) -> None:
        setattr(event, "_streamed_final_response", "exact streamed answer")

    adapter.set_message_handler(handler)
    receipt = DeliveryReceipt()
    event = MessageEvent(text="question", source=_source(), receipt=receipt)

    await adapter._process_message_background(event, "matrix:@rabbit:example.org")

    assert await receipt.wait() == DeliveryOutcome(
        DeliveryStatus.DELIVERED,
        response="exact streamed answer",
    )


@pytest.mark.asyncio
async def test_deferred_receipt_waits_for_queued_turn_delivery() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True, message_id="m-queued"))
    receipt = DeliveryReceipt()
    root = MessageEvent(text="root question", source=_source())
    queued = MessageEvent(text="queued question", source=_source(), receipt=receipt)
    calls = 0

    async def handler(message: MessageEvent) -> str | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            queued.receipt_deferred = True
            adapter._pending_messages["matrix:@rabbit:example.org"] = queued
            return "root answer"
        return "queued answer"

    adapter.set_message_handler(handler)

    await adapter._process_message_background(root, "matrix:@rabbit:example.org")

    assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
        DeliveryStatus.DELIVERED,
        response="queued answer",
    )
    assert calls == 2


@pytest.mark.asyncio
async def test_startup_restore_queue_owns_receipt_until_replay() -> None:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._startup_restore_queue = []
    runner._adapter_for_source = lambda source: None
    receipt = DeliveryReceipt()
    event = MessageEvent(text="early", source=_source(), receipt=receipt)

    runner._queue_startup_restore_event(event)

    assert event.receipt_deferred is True
    assert await runner._drain_startup_restore_queue() == 0
    assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
        DeliveryStatus.UNAVAILABLE
    )


@pytest.mark.asyncio
async def test_startup_restore_replay_failure_completes_receipt() -> None:
    class _FailingAdapter:
        async def handle_message(self, event: MessageEvent) -> None:
            raise RuntimeError("internal detail")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._startup_restore_queue = []
    runner._adapter_for_source = lambda source: _FailingAdapter()
    receipt = DeliveryReceipt()
    event = MessageEvent(text="early", source=_source(), receipt=receipt)
    runner._queue_startup_restore_event(event)

    assert await runner._drain_startup_restore_queue() == 0
    assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
        DeliveryStatus.DELIVERY_FAILED,
        error="RuntimeError",
    )


@pytest.mark.asyncio
async def test_receipt_events_bypass_debounce_and_preserve_fifo() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.MATRIX: adapter}
    runner._queued_events = {}
    events = [
        MessageEvent(text=f"turn-{index}", source=_source(), receipt=DeliveryReceipt())
        for index in range(3)
    ]

    accepted = [
        runner._queue_or_replace_pending_event("matrix:bound", event)
        for event in events
    ]

    assert accepted == [True, True, True]
    assert adapter._pending_messages["matrix:bound"] is events[0]
    assert runner._queued_events["matrix:bound"] == events[1:]
    assert all(event.receipt_deferred for event in events)


@pytest.mark.asyncio
async def test_conversation_reset_cancels_all_deferred_queue_receipts() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.MATRIX: adapter}
    runner._queued_events = {}
    receipts = [DeliveryReceipt() for _ in range(3)]
    events = [
        MessageEvent(text=f"turn-{index}", source=_source(), receipt=receipt)
        for index, receipt in enumerate(receipts)
    ]
    for event in events:
        assert runner._queue_or_replace_pending_event("matrix:bound", event)

    runner._clear_conversation_scope("matrix:bound", reason="test_reset")

    assert "matrix:bound" not in adapter._pending_messages
    assert "matrix:bound" not in runner._queued_events
    for receipt in receipts:
        assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
            DeliveryStatus.CANCELLED
        )


@pytest.mark.asyncio
async def test_queue_capacity_rejects_receipt_without_waiting() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.MATRIX: adapter}
    runner._queued_events = {}
    runner._BUSY_QUEUE_MAX_PENDING = 0
    receipt = DeliveryReceipt()
    event = MessageEvent(text="overflow", source=_source(), receipt=receipt)

    accepted = runner._queue_or_replace_pending_event("matrix:bound", event)

    assert accepted is False
    assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
        DeliveryStatus.REJECTED
    )


@pytest.mark.asyncio
async def test_busy_handler_rejects_receipt_when_fifo_is_full() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.MATRIX: adapter}
    runner._queued_events = {}
    runner._BUSY_QUEUE_MAX_PENDING = 0
    runner._draining = False
    runner._is_user_authorized = lambda source: True
    runner._effective_busy_input_mode = lambda source: "queue"
    runner._adapter_for_source = lambda source: adapter
    receipt = DeliveryReceipt()
    event = MessageEvent(text="overflow", source=_source(), receipt=receipt)

    handled = await runner._handle_active_session_busy_message(
        event,
        "matrix:bound",
    )

    assert handled is True
    assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
        DeliveryStatus.REJECTED
    )


@pytest.mark.asyncio
async def test_busy_handler_rejects_unauthorized_receipt_immediately() -> None:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._is_user_authorized = lambda source: False
    receipt = DeliveryReceipt()
    event = MessageEvent(text="forbidden", source=_source(), receipt=receipt)

    handled = await runner._handle_active_session_busy_message(
        event,
        "matrix:bound",
    )

    assert handled is True
    assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
        DeliveryStatus.REJECTED
    )


@pytest.mark.asyncio
async def test_busy_handler_rejects_receipt_pinned_to_replaced_active_session() -> None:
    class _SessionStore:
        async def lookup_by_session_key(self, session_key: str) -> SimpleNamespace:
            return SimpleNamespace(session_key=session_key, session_id="new-session")

    adapter = _ReceiptAdapter(SendResult(success=True))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.MATRIX: adapter}
    runner._async_session_store = _SessionStore()
    runner._queued_events = {}
    runner._draining = False
    runner._is_user_authorized = lambda source: True
    runner._effective_busy_input_mode = lambda source: "queue"
    runner._adapter_for_source = lambda source: adapter
    runner._session_state("matrix:bound").turn.agent = SimpleNamespace(
        session_id="old-session"
    )
    receipt = DeliveryReceipt()
    event = MessageEvent(
        text="stale active turn",
        source=_source(),
        metadata={
            "gateway_session_key": "matrix:bound",
            "gateway_session_id": "new-session",
            "gateway_session_strict": True,
        },
        receipt=receipt,
    )

    handled = await runner._handle_active_session_busy_message(
        event,
        "matrix:bound",
    )

    assert handled is True
    assert "matrix:bound" not in adapter._pending_messages
    assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
        DeliveryStatus.REJECTED
    )


@pytest.mark.asyncio
async def test_restart_drain_does_not_ack_rejected_receipt_as_queued() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))
    adapter._send_with_retry = AsyncMock()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.MATRIX: adapter}
    runner._queued_events = {}
    runner._BUSY_QUEUE_MAX_PENDING = 0
    runner._draining = True
    runner._is_user_authorized = lambda source: True
    runner._effective_busy_input_mode = lambda source: "queue"
    runner._adapter_for_source = lambda source: adapter
    runner._queue_during_drain_enabled = lambda mode: True
    receipt = DeliveryReceipt()
    event = MessageEvent(text="overflow", source=_source(), receipt=receipt)

    handled = await runner._handle_active_session_busy_message(
        event,
        "matrix:bound",
    )

    assert handled is True
    adapter._send_with_retry.assert_not_awaited()
    assert await asyncio.wait_for(receipt.wait(), timeout=0.1) == DeliveryOutcome(
        DeliveryStatus.REJECTED
    )


@pytest.mark.asyncio
async def test_final_delivery_receipt_does_not_rebind_root_event() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True, message_id="m2"))
    root_receipt = DeliveryReceipt()
    queued_receipt = DeliveryReceipt()
    event = MessageEvent(text="first", source=_source(), receipt=root_receipt)

    async def handler(message: MessageEvent) -> str:
        root_receipt.complete(
            DeliveryOutcome(DeliveryStatus.DELIVERED, response="first answer")
        )
        message.final_delivery_receipt = queued_receipt
        return "queued answer"

    adapter.set_message_handler(handler)

    await adapter._process_message_background(event, "matrix:@rabbit:example.org")

    assert event.receipt is root_receipt
    assert await root_receipt.wait() == DeliveryOutcome(
        DeliveryStatus.DELIVERED,
        response="first answer",
    )
    assert await asyncio.wait_for(
        queued_receipt.wait(), timeout=0.1
    ) == DeliveryOutcome(
        DeliveryStatus.DELIVERED,
        response="queued answer",
    )


@pytest.mark.asyncio
async def test_final_delivery_receipt_gets_cancellation() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True))
    root_receipt = DeliveryReceipt()
    final_receipt = DeliveryReceipt()
    event = MessageEvent(text="first", source=_source(), receipt=root_receipt)
    event.final_delivery_receipt = final_receipt

    async def handler(message: MessageEvent) -> None:
        raise asyncio.CancelledError

    adapter.set_message_handler(handler)

    with pytest.raises(asyncio.CancelledError):
        await adapter._process_message_background(event, "matrix:@rabbit:example.org")

    assert await final_receipt.wait() == DeliveryOutcome(DeliveryStatus.CANCELLED)


@pytest.mark.asyncio
async def test_receipt_contains_bounded_visible_text() -> None:
    adapter = _ReceiptAdapter(SendResult(success=True, message_id="m3"))

    async def handler(event: MessageEvent) -> str:
        return f"{'x' * 5000}\nMEDIA:/tmp/internal.mp3"

    adapter.set_message_handler(handler)
    receipt = DeliveryReceipt()
    event = MessageEvent(text="question", source=_source(), receipt=receipt)

    await adapter._process_message_background(event, "matrix:@rabbit:example.org")

    outcome = await receipt.wait()
    assert outcome.status is DeliveryStatus.DELIVERED
    assert outcome.response == "x" * 4096
