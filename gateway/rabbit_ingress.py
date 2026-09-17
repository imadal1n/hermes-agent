import asyncio  # noqa: F401  # noqa: ANYIO_OK
import os
import socket
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.receipts import DeliveryOutcome, DeliveryReceipt, DeliveryStatus
from gateway.session import SessionEntry


_ENV_PREFIX: Final = "HERMES_RABBIT_INGRESS_"


@dataclass(frozen=True, slots=True)
class SocketPathValueError(ValueError):
    path: Path

    def __str__(self) -> str:
        return f"socket path must be absolute: {self.path}"


@dataclass(frozen=True, slots=True)
class UnsafeSocketPathError(RuntimeError):
    path: Path

    def __str__(self) -> str:
        return f"configured path is not a Unix socket: {self.path}"


class RabbitIngressConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    socket_path: Path
    session_key: str = Field(min_length=1)
    matrix_room_id: str = Field(min_length=1)
    matrix_user_id: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0)
    max_frame_bytes: int = Field(gt=0)
    allowed_peer_uid: int | None = Field(default=None, ge=0)

    @field_validator("socket_path")
    @classmethod
    def require_absolute_socket_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise SocketPathValueError(value)
        return value

    @classmethod
    def from_environment(cls) -> "RabbitIngressConfig | None":
        enabled = os.environ.get(f"{_ENV_PREFIX}ENABLED", "").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return None
        values = {
            "enabled": True,
            "socket_path": os.environ.get(f"{_ENV_PREFIX}SOCKET_PATH"),
            "session_key": os.environ.get(f"{_ENV_PREFIX}SESSION_KEY"),
            "matrix_room_id": os.environ.get(f"{_ENV_PREFIX}MATRIX_ROOM_ID"),
            "matrix_user_id": os.environ.get(f"{_ENV_PREFIX}MATRIX_USER_ID"),
            "request_timeout_seconds": os.environ.get(
                f"{_ENV_PREFIX}REQUEST_TIMEOUT_SECONDS"
            ),
            "max_frame_bytes": os.environ.get(f"{_ENV_PREFIX}MAX_FRAME_BYTES"),
            "allowed_peer_uid": os.environ.get(f"{_ENV_PREFIX}ALLOWED_PEER_UID"),
        }
        if values["allowed_peer_uid"] in {None, ""}:
            values["allowed_peer_uid"] = None
        return cls.model_validate(values)


class RabbitRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: Literal[1]
    request_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)


class RabbitResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: Literal[1] = 1
    request_id: str
    status: DeliveryStatus
    response: str | None = None
    error: str | None = None


class RabbitSessionStore(Protocol):
    def lookup_by_session_key(self, session_key: str) -> SessionEntry | None: ...


class RabbitMatrixAdapter(Protocol):
    async def handle_message(self, event: MessageEvent) -> None: ...


class RabbitIngressRunner(Protocol):
    @property
    def _running(self) -> bool: ...

    @property
    def session_store(self) -> RabbitSessionStore: ...

    @property
    def adapters(self) -> Mapping[Platform, RabbitMatrixAdapter]: ...


class RabbitIngressServer:
    def __init__(
        self, runner: RabbitIngressRunner, config: RabbitIngressConfig
    ) -> None:
        self._runner = runner
        self._config = config
        self._server: asyncio.AbstractServer | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._pending: set[DeliveryReceipt] = set()

    async def start(self) -> bool:
        path = self._config.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = None
        try:
            existing = path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISSOCK(existing.st_mode):
                raise UnsafeSocketPathError(path)
        if self._runner.adapters.get(Platform.MATRIX) is None:
            return False
        if (
            self._runner.session_store.lookup_by_session_key(self._config.session_key)
            is None
        ):
            return False
        if existing is not None:
            path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=path,
            limit=self._config.max_frame_bytes + 1,
        )
        created = path.lstat()
        self._socket_identity = (created.st_dev, created.st_ino)
        return True

    async def stop(self) -> None:
        for receipt in tuple(self._pending):
            receipt.complete(DeliveryOutcome(DeliveryStatus.SHUTDOWN))
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        path = self._config.socket_path
        try:
            current = path.lstat()
        except FileNotFoundError:
            return
        identity = (current.st_dev, current.st_ino)
        if stat.S_ISSOCK(current.st_mode) and identity == self._socket_identity:
            path.unlink()

    def _peer_allowed(self, writer: asyncio.StreamWriter) -> bool:
        allowed_uid = self._config.allowed_peer_uid
        if allowed_uid is None:
            return True
        transport_socket = writer.get_extra_info("socket")
        if transport_socket is None or not hasattr(socket, "SO_PEERCRED"):
            return False
        credentials = transport_socket.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        _pid, uid, _gid = struct.unpack("3i", credentials)
        return uid == allowed_uid

    async def _dispatch(self, request: RabbitRequest) -> DeliveryOutcome:
        if not self._runner._running:
            return DeliveryOutcome(
                DeliveryStatus.UNAVAILABLE, error="gateway unavailable"
            )
        adapter = self._runner.adapters.get(Platform.MATRIX)
        if adapter is None:
            return DeliveryOutcome(
                DeliveryStatus.UNAVAILABLE, error="Matrix unavailable"
            )
        entry = self._runner.session_store.lookup_by_session_key(
            self._config.session_key
        )
        if entry is None:
            return DeliveryOutcome(
                DeliveryStatus.REJECTED, error="session binding mismatch"
            )
        origin = entry.origin
        if (
            origin is None
            or origin.platform is not Platform.MATRIX
            or origin.chat_id != self._config.matrix_room_id
            or origin.user_id != self._config.matrix_user_id
        ):
            return DeliveryOutcome(
                DeliveryStatus.REJECTED, error="session binding mismatch"
            )
        receipt = DeliveryReceipt()
        self._pending.add(receipt)
        event = MessageEvent(
            text=request.text,
            source=origin,
            user_id=origin.user_id,
            user_name=origin.user_name,
            message_id=request.request_id,
            metadata={
                "gateway_session_key": entry.session_key,
                "gateway_session_id": entry.session_id,
                "gateway_session_strict": True,
                "rabbit_ingress": True,
            },
            receipt=receipt,
        )
        try:
            await adapter.handle_message(event)
            try:
                async with asyncio.timeout(self._config.request_timeout_seconds):
                    return await receipt.wait()
            except TimeoutError:
                outcome = DeliveryOutcome(DeliveryStatus.TIMEOUT)
                receipt.complete(outcome)
                return outcome
        except asyncio.CancelledError:
            receipt.complete(DeliveryOutcome(DeliveryStatus.CANCELLED))
            raise
        finally:
            self._pending.discard(receipt)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_id = ""
        try:
            if not self._peer_allowed(writer):
                outcome = DeliveryOutcome(
                    DeliveryStatus.REJECTED, error="peer rejected"
                )
            else:
                try:
                    frame = await reader.readuntil(b"\n")
                    if len(frame) - 1 > self._config.max_frame_bytes:
                        raise asyncio.LimitOverrunError("frame too large", len(frame))
                    request = RabbitRequest.model_validate_json(frame)
                    request_id = request.request_id
                    outcome = await self._dispatch(request)
                except (
                    asyncio.IncompleteReadError,
                    asyncio.LimitOverrunError,
                    UnicodeDecodeError,
                    ValidationError,
                ):
                    outcome = DeliveryOutcome(
                        DeliveryStatus.INVALID_REQUEST,
                        error="invalid request",
                    )
            payload = RabbitResponse(
                request_id=request_id,
                status=outcome.status,
                response=outcome.response,
                error=outcome.error,
            )
            writer.write(payload.model_dump_json(exclude_none=True).encode() + b"\n")
            await writer.drain()
        except (BrokenPipeError, ConnectionError):
            return
        finally:
            writer.close()
            await writer.wait_closed()
