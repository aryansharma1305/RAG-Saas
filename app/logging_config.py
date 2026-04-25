import logging
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger("rag_saas")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s",
    )
    request_id_filter = RequestIdFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(request_id_filter)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid4().hex
        started = time.perf_counter()
        request.state.request_id = request_id
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-Id"] = request_id
        logger.info(
            "%s %s status=%s elapsed_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            extra={"request_id": request_id},
        )
        return response
