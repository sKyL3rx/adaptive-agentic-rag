from __future__ import annotations

import asyncio
from types import TracebackType

from observability import (
    RAG_INFLIGHT_REQUESTS,
    RAG_PENDING_REQUESTS,
)


class RequestExecutionLifecycle:
    """
    pending:
        waiting on either same-session serialization
        or the global request semaphore.

    inflight:
        actively executing an orchestration request
        with a semaphore permit.
    """

    def __init__(
        self,
        *,
        session_lock: asyncio.Lock,
        request_semaphore: asyncio.Semaphore,
    ) -> None:
        self._session_lock = session_lock
        self._request_semaphore = (
            request_semaphore
        )

        self._pending_counted = False
        self._session_acquired = False
        self._semaphore_acquired = False
        self._inflight_counted = False
    

    async def __aenter__(self) -> "RequestExecutionLifecycle":
                RAG_PENDING_REQUESTS.inc()
                self._pending_counted = True
                
                try:
                    await self._session_lock.acquire()
                    self._session_acquired = True
                    
                    await self._request_semaphore.acquire()
                    self._semaphore_acquired = True
                except BaseException:
                    self._release_resources()
                    raise
                
                finally:
                    if self._pending_counted:
                        RAG_PENDING_REQUESTS.dec()
                        self._pending_counted = False

                try:
                    RAG_INFLIGHT_REQUESTS.inc()
                    self._inflight_counted = True
                except BaseException:
                    self._release_resources()
                    raise

                return self

    async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                if self._inflight_counted:
                    RAG_INFLIGHT_REQUESTS.dec()
                    self._inflight_counted = False
                self._release_resources()
        
    def _release_resources(self) -> None:
                if self._semaphore_acquired:
                    self._request_semaphore.release()
                    self._semaphore_acquired = False

                if self._session_acquired:
                    self._session_lock.release()
                    self._session_acquired = False
            