import asyncio  # noqa: F401  # noqa: ANYIO_OK
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


MAX_DELIVERY_RESPONSE_CHARS: Final = 4096


class DeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    NO_RESPONSE = "no_response"
    REJECTED = "rejected"
    DELIVERY_FAILED = "delivery_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SHUTDOWN = "shutdown"
    INVALID_REQUEST = "invalid_request"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    status: DeliveryStatus
    response: str | None = None
    error: str | None = None


class DeliveryReceipt:
    def __init__(self) -> None:
        self._future: asyncio.Future[DeliveryOutcome] = (
            asyncio.get_running_loop().create_future()
        )

    def complete(self, outcome: DeliveryOutcome) -> bool:
        if self._future.done():
            return False
        self._future.set_result(outcome)
        return True

    async def wait(self) -> DeliveryOutcome:
        return await asyncio.shield(self._future)
