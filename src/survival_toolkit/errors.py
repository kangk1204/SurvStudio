from __future__ import annotations

from functools import wraps
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


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


def user_input_boundary(func: Callable[P, T]) -> Callable[P, T]:
    """Convert public-service validation failures into a typed user-facing error."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except (SurvStudioError, ImportError, RuntimeError):
            raise
        except (TypeError, ValueError) as exc:
            message = str(exc).strip() or "The request could not be processed with the selected dataset and settings."
            raise UserInputError(message) from exc

    return wrapper
