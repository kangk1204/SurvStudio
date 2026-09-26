from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from survival_toolkit.errors import JobCancelledError

# ``warnings.catch_warnings`` saves and restores the process-wide filter list. Two threads
# entering and leaving it out of order restore each other's filters, which can leave an
# "ignore" filter installed for the rest of the process. The web server runs analyses in a
# thread pool, so every temporary filter change in SurvStudio goes through this lock.
_WARNING_FILTER_LOCK = threading.RLock()

# Set by the web layer for the duration of one analysis job; long loops poll it through
# raise_if_cancelled() so a job whose client disconnected stops instead of running on.
_CANCEL_EVENT: ContextVar[threading.Event | None] = ContextVar("survstudio_cancel_event", default=None)


@contextmanager
def suppressed_warnings(*categories: type[Warning]) -> Iterator[None]:
    """Silence the given warning categories (all warnings when none are given).

    Keep the wrapped block short: other threads cannot change warning filters while it
    runs, and warnings they emit meanwhile are filtered the same way.
    """

    with _WARNING_FILTER_LOCK:
        with warnings.catch_warnings():
            if categories:
                for category in categories:
                    warnings.simplefilter("ignore", category=category)
            else:
                warnings.simplefilter("ignore")
            yield


@contextmanager
def cancellation_scope(event: threading.Event) -> Iterator[None]:
    """Make ``event`` the cancellation signal for code running in this context."""

    token = _CANCEL_EVENT.set(event)
    try:
        yield
    finally:
        _CANCEL_EVENT.reset(token)


def cancellation_requested() -> bool:
    event = _CANCEL_EVENT.get()
    return event is not None and event.is_set()


def raise_if_cancelled() -> None:
    """Stop the current analysis if its request was cancelled (a cheap no-op otherwise)."""

    if cancellation_requested():
        raise JobCancelledError("The analysis was stopped because its request was cancelled.")
