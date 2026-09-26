from __future__ import annotations

from functools import wraps
from types import TracebackType
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

_PACKAGE_NAME = __name__.rsplit(".", 1)[0]


class SurvStudioError(Exception):
    """Base class for typed SurvStudio exceptions."""


class UserInputError(SurvStudioError, ValueError):
    """Validation or configuration error that is safe to show to the user."""


class NotFoundError(UserInputError):
    """User-facing missing-resource error."""


class DatasetNotFoundError(NotFoundError):
    """Requested dataset id does not exist in the active store."""


class ColumnNotFoundError(NotFoundError):
    """Requested dataframe column does not exist."""


class DependencyError(SurvStudioError, ImportError):
    """Optional dependency required for the requested workflow is unavailable."""


class JobCancelledError(SurvStudioError):
    """The request that started a running analysis went away, so the analysis stopped early."""


class InternalAnalysisError(SurvStudioError, ValueError):
    """Unexpected internal failure whose raw message must not be shown to users.

    It still subclasses ValueError so existing ``except ValueError`` fallbacks keep working;
    the original exception is preserved as ``__cause__`` for logging.
    """

    default_message = (
        "The analysis failed because of an unexpected internal error. "
        "Review the selected columns and settings; if the problem persists, report it with the server log."
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)


def _innermost_traceback_module(traceback: TracebackType | None) -> str:
    module_name = ""
    while traceback is not None:
        module_name = str(traceback.tb_frame.f_globals.get("__name__", ""))
        traceback = traceback.tb_next
    return module_name


def _raised_by_survstudio(exc: BaseException) -> bool:
    """True when the exception was raised by SurvStudio code rather than inside a library."""

    module_name = _innermost_traceback_module(exc.__traceback__)
    return module_name == _PACKAGE_NAME or module_name.startswith(f"{_PACKAGE_NAME}.")


def _is_numerical_library_error(exc: BaseException) -> bool:
    """Numerical failures (singular matrices, non-convergence) that the web layer classifies itself."""

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        return False
    if isinstance(exc, np.linalg.LinAlgError):
        return True
    return type(exc).__name__ == "ConvergenceError"


def is_programming_error(exc: BaseException) -> bool:
    """True for exceptions that signal a bug in SurvStudio rather than a data or numerical failure.

    Per-model and per-fold fallbacks record data-driven failures (singular designs, too few
    events, non-convergence) and carry on; they must re-raise these instead of reporting a
    coding error as "model failed on fold k". A ``TypeError`` raised inside a third-party
    library usually reflects unusable input (for example mixed-type columns) and is not
    treated as a bug here.
    """

    if isinstance(exc, (AttributeError, KeyError, NameError, AssertionError)):
        return True
    return isinstance(exc, TypeError) and _raised_by_survstudio(exc)


def must_propagate(exc: BaseException) -> bool:
    """True when a per-model or per-fold fallback must re-raise instead of recording a failure.

    Covers programming errors and cancellation of the whole analysis.
    """

    return isinstance(exc, JobCancelledError) or is_programming_error(exc)


def user_input_boundary(func: Callable[P, T]) -> Callable[P, T]:
    """Convert public-service validation failures into a typed user-facing error.

    Deliberate ``ValueError``s raised by SurvStudio code keep their message as a
    :class:`UserInputError`. ``TypeError``s and errors raised inside third-party libraries
    (pandas/numpy/scikit-learn internals such as "The truth value of a Series is ambiguous")
    become an :class:`InternalAnalysisError` with a generic message instead of leaking
    implementation details. Numerical library errors pass through unchanged.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except (SurvStudioError, ImportError, RuntimeError):
            raise
        except (TypeError, ValueError) as exc:
            if _is_numerical_library_error(exc):
                raise
            if isinstance(exc, ValueError) and _raised_by_survstudio(exc):
                message = str(exc).strip() or "The request could not be processed with the selected dataset and settings."
                raise UserInputError(message) from exc
            raise InternalAnalysisError() from exc

    return wrapper
