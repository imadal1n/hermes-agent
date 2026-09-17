import asyncio  # noqa: F401  # noqa: ANYIO_OK
import json
import os
from datetime import datetime

import pytest
from pydantic import ValidationError

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.rabbit_ingress import (
    RabbitIngressConfig,
    RabbitIngressServer,
    RabbitRequest,
)
from gateway.receipts import DeliveryOutcome, DeliveryStatus
from gateway.session import SessionEntry, SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        user_id="@rabbit:example.org",
    )


def _entry() -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key="matrix:bound",
        session_id="session-1",
        created_at=now,
        updated_at=now,
        origin=_source(),
    )


class _SessionStore:
    def __init__(self, entry: SessionEntry | None) -> None:
        self.entry = entry

    def lookup_by_session_key(self, session_key: str) -> SessionEntry | None:
        if self.entry is not None and self.entry.session_key == session_key:
            return self.entry
        return None


class _WaitingAdapter:
    def __init__(self) -> None:
        self.event: MessageEvent | None = None
        self.started = asyncio.Event()

    async def handle_message(self, event: MessageEvent) -> None:
        self.event = event
        self.started.set()


class _Runner:
    def __init__(
        self, entry: SessionEntry | None, *, with_adapter: bool = True
    ) -> None:
        self._running = True
        self.session_store = _SessionStore(entry)
        self.adapter = _WaitingAdapter()
        self.adapters = {Platform.MATRIX: self.adapter} if with_adapter else {}


def _config(tmp_path, **overrides) -> RabbitIngressConfig:
    values = {
        "enabled": True,
        "socket_path": tmp_path / "rabbit.sock",
        "session_key": "matrix:bound",
        "matrix_room_id": "!room:example.org",
        "matrix_user_id": "@rabbit:example.org",
        "request_timeout_seconds": 1.0,
        "max_frame_bytes": 4096,
    }
    values.update(overrides)
    return RabbitIngressConfig.model_validate(values)


def _request() -> RabbitRequest:
    return RabbitRequest(protocol_version=1, request_id="r1", text="hello")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "with_adapter"),
    [(None, True), (_entry(), False)],
)
async def test_start_degrades_when_matrix_binding_is_unavailable(
    tmp_path,
    entry: SessionEntry | None,
    with_adapter: bool,
) -> None:
    config = _config(tmp_path)
    server = RabbitIngressServer(_Runner(entry, with_adapter=with_adapter), config)

    assert await server.start() is False
    assert not config.socket_path.exists()


def test_enabled_invalid_environment_is_fatal(monkeypatch) -> None:
    prefix = "HERMES_RABBIT_INGRESS_"
    for key in tuple(os.environ):
        if key.startswith(prefix):
            monkeypatch.delenv(key)
    monkeypatch.setenv(f"{prefix}ENABLED", "true")

    with pytest.raises(ValidationError):
        RabbitIngressConfig.from_environment()


@pytest.mark.asyncio
async def test_dispatch_times_out_pending_delivery(tmp_path) -> None:
    runner = _Runner(_entry())
    server = RabbitIngressServer(
        runner,
        _config(tmp_path, request_timeout_seconds=0.01),
    )

    assert await server._dispatch(_request()) == DeliveryOutcome(DeliveryStatus.TIMEOUT)
    assert runner.adapter.event is not None
    receipt = runner.adapter.event.receipt
    assert receipt is not None
    assert await receipt.wait() == DeliveryOutcome(DeliveryStatus.TIMEOUT)


@pytest.mark.asyncio
async def test_dispatch_cancellation_resolves_receipt(tmp_path) -> None:
    runner = _Runner(_entry())
    server = RabbitIngressServer(runner, _config(tmp_path))
    task = asyncio.create_task(server._dispatch(_request()))
    await runner.adapter.started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.adapter.event is not None
    receipt = runner.adapter.event.receipt
    assert receipt is not None
    assert await receipt.wait() == DeliveryOutcome(DeliveryStatus.CANCELLED)


@pytest.mark.asyncio
async def test_stop_resolves_pending_delivery_as_shutdown(tmp_path) -> None:
    runner = _Runner(_entry())
    server = RabbitIngressServer(runner, _config(tmp_path))
    task = asyncio.create_task(server._dispatch(_request()))
    await runner.adapter.started.wait()

    await server.stop()

    assert await task == DeliveryOutcome(DeliveryStatus.SHUTDOWN)


@pytest.mark.asyncio
async def test_unix_ingress_rejects_unauthorized_peer(tmp_path) -> None:
    config = _config(tmp_path, allowed_peer_uid=os.getuid() + 1)
    server = RabbitIngressServer(_Runner(_entry()), config)
    assert await server.start() is True
    reader, writer = await asyncio.open_unix_connection(config.socket_path)
    writer.write(
        json.dumps({
            "protocol_version": 1,
            "request_id": "peer",
            "text": "hello",
        }).encode()
        + b"\n"
    )
    await writer.drain()

    response = json.loads(await reader.readline())
    writer.close()
    await writer.wait_closed()
    await server.stop()

    assert response == {
        "protocol_version": 1,
        "request_id": "",
        "status": "rejected",
        "error": "rejected",
    }
