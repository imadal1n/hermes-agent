import asyncio  # noqa: F401  # noqa: ANYIO_OK
import json
from datetime import datetime

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.rabbit_ingress import RabbitIngressConfig, RabbitIngressServer
from gateway.receipts import DeliveryOutcome, DeliveryReceipt, DeliveryStatus
from gateway.session import SessionEntry, SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        user_id="@rabbit:example.org",
    )


class _SessionStore:
    def __init__(self, entry: SessionEntry | None) -> None:
        self.entry = entry

    def lookup_by_session_key(self, session_key: str) -> SessionEntry | None:
        return (
            self.entry if self.entry and self.entry.session_key == session_key else None
        )


class _IngressAdapter:
    def __init__(self) -> None:
        self.event: MessageEvent | None = None

    async def handle_message(self, event: MessageEvent) -> None:
        self.event = event
        assert event.receipt is not None
        event.receipt.complete(
            DeliveryOutcome(DeliveryStatus.DELIVERED, response=f"reply:{event.text}")
        )


class _Runner:
    def __init__(self, entry: SessionEntry | None) -> None:
        self._running = True
        self.session_store: _SessionStore = _SessionStore(entry)
        self.adapter: _IngressAdapter = _IngressAdapter()
        self.adapters: dict[Platform, _IngressAdapter] = {Platform.MATRIX: self.adapter}


async def _request(
    socket_path: str, payload: dict[str, str | int]
) -> dict[str, str | int]:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    writer.write(json.dumps(payload).encode() + b"\n")
    await writer.drain()
    response = json.loads(await reader.readline())
    writer.close()
    await writer.wait_closed()
    return response


@pytest.mark.asyncio
async def test_unix_ingress_dispatches_through_bound_matrix_event(tmp_path) -> None:
    source = _source()
    entry = SessionEntry(
        session_key="matrix:bound",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
    )
    runner = _Runner(entry)
    socket_path = tmp_path / "rabbit.sock"
    config = RabbitIngressConfig(
        enabled=True,
        socket_path=socket_path,
        session_key=entry.session_key,
        matrix_room_id=source.chat_id,
        matrix_user_id=source.user_id or "",
        request_timeout_seconds=1.0,
        max_frame_bytes=4096,
    )
    server = RabbitIngressServer(runner, config)
    await server.start()

    response = await _request(
        str(socket_path),
        {"protocol_version": 1, "request_id": "r1", "text": "hello"},
    )
    await server.stop()

    assert response == {
        "protocol_version": 1,
        "request_id": "r1",
        "status": "delivered",
        "response": "reply:hello",
    }
    assert runner.adapter.event is not None
    assert runner.adapter.event.source == source
    assert runner.adapter.event.metadata["gateway_session_key"] == entry.session_key
    assert not socket_path.exists()


@pytest.mark.asyncio
async def test_unix_ingress_rejects_session_origin_mismatch(tmp_path) -> None:
    source = _source()
    entry = SessionEntry(
        session_key="matrix:bound",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
    )
    runner = _Runner(entry)
    socket_path = tmp_path / "rabbit.sock"
    server = RabbitIngressServer(
        runner,
        RabbitIngressConfig(
            enabled=True,
            socket_path=socket_path,
            session_key=entry.session_key,
            matrix_room_id="!other:example.org",
            matrix_user_id=source.user_id or "",
            request_timeout_seconds=1.0,
            max_frame_bytes=4096,
        ),
    )
    await server.start()

    response = await _request(
        str(socket_path),
        {"protocol_version": 1, "request_id": "r2", "text": "hello"},
    )
    await server.stop()

    assert response["status"] == "rejected"
    assert runner.adapter.event is None


@pytest.mark.asyncio
async def test_unix_ingress_sanitizes_invalid_request_error(tmp_path) -> None:
    source = _source()
    entry = SessionEntry(
        session_key="matrix:bound",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
    )
    socket_path = tmp_path / "rabbit.sock"
    server = RabbitIngressServer(
        _Runner(entry),
        RabbitIngressConfig(
            enabled=True,
            socket_path=socket_path,
            session_key=entry.session_key,
            matrix_room_id=source.chat_id,
            matrix_user_id=source.user_id or "",
            request_timeout_seconds=1.0,
            max_frame_bytes=4096,
        ),
    )
    await server.start()

    response = await _request(
        str(socket_path),
        {"protocol_version": 2, "request_id": "bad", "text": "hello"},
    )
    await server.stop()

    assert response == {
        "protocol_version": 1,
        "request_id": "",
        "status": "invalid_request",
        "error": "invalid request",
    }


@pytest.mark.asyncio
async def test_unix_ingress_refuses_non_socket_replacement(tmp_path) -> None:
    socket_path = tmp_path / "rabbit.sock"
    socket_path.write_text("keep", encoding="utf-8")
    runner = _Runner(None)
    server = RabbitIngressServer(
        runner,
        RabbitIngressConfig(
            enabled=True,
            socket_path=socket_path,
            session_key="matrix:bound",
            matrix_room_id="!room:example.org",
            matrix_user_id="@rabbit:example.org",
            request_timeout_seconds=1.0,
            max_frame_bytes=4096,
        ),
    )

    with pytest.raises(RuntimeError, match="not a Unix socket"):
        await server.start()

    assert socket_path.read_text(encoding="utf-8") == "keep"
