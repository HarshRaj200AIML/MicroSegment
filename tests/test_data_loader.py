import pandas as pd

from src.data_loader import infer_column_roles, load_dataset


def test_csv_loader_and_column_roles(tmp_path):
    source = tmp_path / "sample.csv"
    pd.DataFrame({"district_code": [101, 102, 103], "count": [5, 7, 9], "urban": ["yes", "no", "yes"]}).to_csv(source, index=False)
    info = load_dataset(source)
    assert info.frame.shape == (3, 3)
    roles = infer_column_roles(info.frame)
    assert "district_code" in roles["identifier"]
    assert "count" in roles["numeric"]
    assert "urban" in roles["binary"]


def test_near_unique_text_values_are_identifiers_but_numeric_measures_are_not():
    frame = pd.DataFrame({
        "area_name": [f"Area {index}" for index in range(19)] + ["Area 0"],
        "observed_count": list(range(20)),
    })
    roles = infer_column_roles(frame)
    assert "area_name" in roles["identifier"]
    assert "observed_count" in roles["numeric"]


def test_unsupported_extension_is_clear(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("not a dataset", encoding="utf-8")
    try:
        load_dataset(source)
    except ValueError as error:
        assert "Supported formats" in str(error)
    else:
        raise AssertionError("Unsupported input should raise ValueError")


def test_html_table_saved_as_xls_is_loaded(tmp_path):
    source = tmp_path / "legacy_export.xls"
    source.write_text(
        "<html><body><table><tr><td>State</td><td>Count</td></tr>"
        "<tr><td>Example</td><td>1,250</td></tr></table></body></html>",
        encoding="utf-8",
    )
    frame = load_dataset(source).frame
    assert frame.to_dict(orient="records") == [{"State": "Example", "Count": 1250}]