import pandas as pd
import pytest

from saf import wrds_pull as wp


def schema(**types):
    return pd.DataFrame({"name": list(types), "nullable": True, "type": list(types.values()), "comment": None})


SPEC = wp.TableSpec("t", "lib", "tbl", ("id", "d", "x"), where=("flag = 'A'",),
                    date_col="d", start="2020-03-01", to_end=True, yearly=True, keys=("id", "d"))
SCHEMA = schema(id="INTEGER", d="DATE", x="NUMERIC(10, 4)", extra="VARCHAR")


def test_select_list_casts_numeric():
    assert wp.select_list(SPEC, SCHEMA) == ["id", "d", "x::float8 AS x"]


def test_select_list_all_columns_when_none():
    spec = wp.TableSpec("t", "lib", "tbl", None)
    assert wp.select_list(spec, SCHEMA) == ["id", "d", "x::float8 AS x", "extra"]


def test_select_list_reports_missing_columns():
    spec = wp.TableSpec("t", "lib", "tbl", ("id", "nope", "gone"))
    with pytest.raises(ValueError, match=r"\['nope', 'gone'\]"):
        wp.select_list(spec, SCHEMA)


def test_build_query():
    q = wp.build_query(SPEC, SCHEMA, pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31"))
    assert q == ("SELECT id, d, x::float8 AS x FROM lib.tbl WHERE (flag = 'A') AND (d >= '2020-03-01') "
                 "AND (d <= '2020-12-31') ORDER BY id, d")


def test_yearly_chunks_clip_to_start_and_end():
    c = wp.chunks(SPEC, "2022-06-30")
    assert [s for s, _, _ in c] == ["t_2020", "t_2021", "t_2022"]
    assert c[0][1] == pd.Timestamp("2020-03-01") and c[-1][2] == pd.Timestamp("2022-06-30")


def test_single_chunk_without_end_cap():
    spec = wp.TableSpec("f", "comp", "funda", ("gvkey",), date_col="datadate", start="1995-01-01")
    assert wp.chunks(spec, "2026-06-30") == [("f", pd.Timestamp("1995-01-01"), None)]


def test_specs_are_consistent():
    assert len(wp.BY_NAME) == len(wp.TABLES)
    for t in wp.TABLES:
        assert not t.yearly or (t.start and t.to_end), t.name
        assert not t.keys or set(t.keys) <= set(t.columns or t.keys), t.name


class FakeSession:
    def __init__(self, data):
        self.data, self.queries = data, []

    def describe(self, lib, table):
        return SCHEMA

    def sql(self, query, params=None, date_cols=None, retries=1):
        self.queries.append(query)
        year = query.split("d >= '")[1][:4]
        return self.data[year]


def test_pull_writes_files_manifest_and_resumes(tmp_path):
    data = {y: pd.DataFrame({"id": [1], "d": [f"{y}-06-01"], "x": [1.5]}) for y in ("2020", "2021")}
    s = FakeSession(data)
    man = wp.pull(s, tmp_path, "2021-12-31", tables=(SPEC,), log=lambda *_: None)
    assert sorted(man["file"]) == ["lib/t_2020.parquet", "lib/t_2021.parquet"]   # relative to data_dir
    assert (tmp_path / "meta" / "manifest.csv").exists()
    assert man["rows"].tolist() == [1, 1]
    df = wp.read_table(tmp_path, "t", folder="lib")
    assert len(df) == 2 and pd.api.types.is_datetime64_any_dtype(df["d"])

    (tmp_path / "lib" / "t_2021.parquet").unlink()
    s2 = FakeSession(data)
    man = wp.pull(s2, tmp_path, "2021-12-31", tables=(SPEC,), log=lambda *_: None)
    assert len(s2.queries) == 1 and "2021" in s2.queries[0]       # only the missing file
    assert len(man) == 2                                            # manifest row replaced, not duplicated


def test_date_sentinel_becomes_nat(tmp_path):
    spec = wp.TableSpec("m", "lib", "tbl", ("id", "d"))
    s = FakeSession({})
    s.sql = lambda q, **k: pd.DataFrame({"id": [1, 2], "d": ["2020-01-01", "9999-12-31"]})
    wp.pull(s, tmp_path, None, tables=(spec,), log=lambda *_: None)
    d = wp.read_table(tmp_path, "m", folder="lib")["d"]
    assert d.iloc[0] == pd.Timestamp("2020-01-01") and pd.isna(d.iloc[1])


def test_tables_go_to_source_folders():
    assert {t.folder for t in wp.TABLES} == {"crsp", "compustat", "ff"}
    assert wp.BY_NAME["crsp_dsf_v2"].folder == "crsp"
    assert wp.BY_NAME["comp_fundq"].folder == "compustat"
    assert wp.BY_NAME["ff_factors_daily"].folder == "ff"
