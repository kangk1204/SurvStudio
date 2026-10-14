"""Product event-role checks, separate from the locked numerical kernel."""
from __future__ import annotations

import pandas as pd
import re

from survival_toolkit.errors import UserInputError

_CENSOR_TOKENS = {"censor", "censored", "cens", "censoring"}
_EXPLICIT_EVENT_TOKENS = {"event", "death", "dead", "died"}
_CENSOR_NAME_AFFIXES = {
    "", "os", "pfs", "rfs", "dfs", "dss", "dmfs", "ttp", "efs", "tts", "tte", "css", "mfs",
    "pfi", "dfi", "ttf", "ttr", "ttnt", "ffs", "lrfs", "drfs", "status", "indicator", "ind", "flag",
}
_COMPACT_CENSOR_NAMES = {
    prefix + role + suffix
    for prefix in _CENSOR_NAME_AFFIXES
    for role in _CENSOR_TOKENS
    for suffix in _CENSOR_NAME_AFFIXES
}


def event_coding_warning(frame: pd.DataFrame, event_column: str | None) -> str | None:
    """Recognize a censor-labelled generic code without guessing its true meaning.

    Status/vital words alone do not turn a censor indicator into an event.
    Literal Dead/Alive values retain the kernel's normal label mapping.
    """
    if not event_column:
        return None
    from survival_toolkit.analysis import _column_name_tokens, _has_only_generic_binary_codes

    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", str(event_column))
    tokens = set(_column_name_tokens(name))
    censor_name = bool(tokens & _CENSOR_TOKENS) or "".join(_column_name_tokens(name)) in _COMPACT_CENSOR_NAMES
    if not censor_name or tokens & _EXPLICIT_EVENT_TOKENS:
        return None
    if event_column not in frame.columns or not _has_only_generic_binary_codes(frame[event_column]):
        return None
    return (
        f'"{event_column}" looks like a censoring indicator (usually 1, Yes, or True = censored), '
        "but SurvStudio needs an event indicator (1 = event). Check the source definition, "
        f"then add a recoded column such as event = 1 - {event_column} (Yes -> 0, No -> 1) "
        "and select that instead. Source values have not been changed."
    )


def reject_ambiguous_censor_event_column(frame: pd.DataFrame, event_column: str | None) -> None:
    message = event_coding_warning(frame, event_column)
    if message:
        raise UserInputError(message)
