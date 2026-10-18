"""An explicit source convention resolves ambiguous text without a magnitude guess."""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from survival_toolkit.analysis import load_dataframe, load_dataframe_from_path, profile_dataframe
from survival_toolkit.app import app


@pytest.mark.parametrize("number_format,expected", [
    ("decimal_dot", [2361.0, 580.0]),
    ("decimal_comma", [2.361, 580.0]),
])
def test_declared_convention_changes_only_its_explicit_interpretation(number_format, expected, tmp_path):
    content = b'time,event,treatment\n"2,361",1,None\n580,0,Drug\n'
    with pytest.raises(ValueError, match="read either as"):
        load_dataframe(content, "days.csv")
    frame = load_dataframe(content, "days.csv", number_format=number_format)
    assert frame.time.tolist() == pytest.approx(expected)
    assert frame.treatment.tolist() == ["None", "Drug"]
    assert profile_dataframe(frame, "test", "days.csv")["number_format"] == number_format
    path = tmp_path / "days.csv"
    path.write_bytes(content)
    assert load_dataframe_from_path(path, number_format=number_format).time.tolist() == pytest.approx(expected)


def test_declared_comma_reloads_dot_groups_from_original_text():
    frame = load_dataframe(b'time;event\n2.361;1\n580;0\n', "days.csv", number_format="decimal_comma")
    assert frame.time.tolist() == [2361, 580]


@pytest.mark.parametrize("number_format,value", [("decimal_dot", "0,113"), ("decimal_comma", "0.113")])
def test_contradictory_convention_refuses_instead_of_silently_rescaling(number_format, value):
    with pytest.raises(ValueError, match="does not match the selected Number format"):
        load_dataframe(f'time,event\n"{value}",1\n580,0\n'.encode(), "days.csv", number_format=number_format)


def test_declared_format_retains_numeric_text_for_review():
    frame = load_dataframe(b'time,event,age\n"2,361",1,unknown\n580,0,60\n', "days.csv", number_format="decimal_dot")
    assert frame.age.tolist() == ["unknown", "60"]


@pytest.mark.parametrize("number_format", ["auto", "decimal_dot", "decimal_comma"])
def test_number_format_never_reclassifies_literal_none_as_missing(number_format):
    content = b'time,event,category\n1,1.0,"1,234"\n2,None,None\n3,1.0,"2,234"\n4,None,None\n'
    frame = load_dataframe(content, "none.csv", number_format=number_format)
    assert frame.event.tolist() == ["1.0", "None", "1.0", "None"]
    assert frame.category.tolist() == ["1,234", "None", "2,234", "None"]
    assert frame.event.isna().sum() == frame.category.isna().sum() == 0


def test_number_format_rejects_unknown_values_and_non_text_files():
    with pytest.raises(ValueError, match="Number format must be"):
        load_dataframe(b"time,event\n1,1", "days.csv", number_format="guess")
    buf = io.BytesIO()
    pd.DataFrame({"time": [1, 2], "event": [1, 0]}).to_parquet(buf)
    with pytest.raises(ValueError, match="Automatic number format for Excel and Parquet"):
        load_dataframe(buf.getvalue(), "days.parquet", number_format="decimal_dot")


def test_upload_api_retains_explicit_choice_in_profile():
    with TestClient(app, base_url="http://localhost") as client:
        source = {"file": ("days.csv", b'time,event\n"2,361",1\n580,0\n', "text/csv")}
        assert client.post("/api/upload", files=source).status_code == 400
        response = client.post("/api/upload?number_format=decimal_dot", files=source)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["number_format"] == "decimal_dot"
        assert payload["preview"][0]["time"] == 2361
        assert client.post("/api/upload?number_format=unknown", files=source).status_code == 422
