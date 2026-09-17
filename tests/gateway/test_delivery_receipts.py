import asyncio  # noqa: F401  # noqa: ANYIO_OK

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    PlatformConfig,
    SendResult,
)
from gateway.receipts import DeliveryOutcome, DeliveryReceipt, DeliveryStatus
from gateway.run import _strip_response_attachments_for_direct_send
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
