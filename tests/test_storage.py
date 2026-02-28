"""Tests for iferaopt.storage – Parquet, Zarr, DuckDB & Polars helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from iferaopt.storage import (
    list_zarr_dates,
    load_zarr_tensors,
    query_duckdb,
    read_parquet,
    save_zarr_tensors,
    scan_polars,
    write_parquet,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_table() -> pa.Table:
    """A small PyArrow table with symbol + date partition columns."""
    return pa.table(
        {
            "symbol": ["SPX", "SPX", "SPX", "NDX", "NDX"],
            "date": ["20250101", "20250101", "20250102", "20250101", "20250102"],
            "price": [4800.0, 4801.0, 4810.0, 17200.0, 17250.0],
            "volume": [100, 200, 150, 300, 250],
        }
    )


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


class TestParquetRoundTrip:
    """write_parquet → read_parquet with Hive partitioning by symbol+date."""

    def test_write_and_read_all(self, tmp_path: Path, sample_table: pa.Table) -> None:
        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        result = read_parquet(base)
        assert result.num_rows == sample_table.num_rows

    def test_filter_by_symbol(self, tmp_path: Path, sample_table: pa.Table) -> None:
        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        spx = read_parquet(base, symbols="SPX")
        assert spx.num_rows == 3  # 3 SPX rows in the fixture
        assert set(spx.column("symbol").to_pylist()) == {"SPX"}

    def test_filter_by_date(self, tmp_path: Path, sample_table: pa.Table) -> None:
        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        d = date(2025, 1, 2)
        rows = read_parquet(base, dates=d)
        assert rows.num_rows == 2  # one SPX + one NDX on 20250102

    def test_filter_by_symbol_and_date(self, tmp_path: Path, sample_table: pa.Table) -> None:
        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        rows = read_parquet(base, symbols="NDX", dates=date(2025, 1, 1))
        assert rows.num_rows == 1
        assert rows.column("price").to_pylist() == [17200.0]

    def test_column_selection(self, tmp_path: Path, sample_table: pa.Table) -> None:
        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        rows = read_parquet(base, columns=["price", "symbol"])
        assert set(rows.column_names) == {"price", "symbol"}

    def test_multiple_symbols_list(self, tmp_path: Path, sample_table: pa.Table) -> None:
        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        rows = read_parquet(base, symbols=["SPX", "NDX"])
        assert rows.num_rows == sample_table.num_rows

    def test_hive_directory_structure(self, tmp_path: Path, sample_table: pa.Table) -> None:
        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        # Hive partitioning should create symbol=…/date=…/ directories
        symbol_dirs = sorted(p.name for p in base.iterdir() if p.is_dir())
        assert "symbol=NDX" in symbol_dirs
        assert "symbol=SPX" in symbol_dirs


# ---------------------------------------------------------------------------
# Zarr round-trip
# ---------------------------------------------------------------------------


class TestZarrRoundTrip:
    """save_zarr_tensors → load_zarr_tensors."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        arrays = {
            "bars": np.random.rand(390, 5).astype(np.float32),
            "chain": np.random.rand(50, 8).astype(np.float32),
        }
        save_zarr_tensors(arrays, tmp_path, symbol="SPX", date_str="20250103")
        loaded = load_zarr_tensors(tmp_path, symbol="SPX", date_str="20250103")

        assert set(loaded.keys()) == {"bars", "chain"}
        np.testing.assert_array_almost_equal(loaded["bars"], arrays["bars"])
        np.testing.assert_array_almost_equal(loaded["chain"], arrays["chain"])

    def test_selective_keys(self, tmp_path: Path) -> None:
        arrays = {"a": np.array([1, 2, 3]), "b": np.array([4, 5])}
        save_zarr_tensors(arrays, tmp_path, symbol="SPX", date_str="20250103")
        loaded = load_zarr_tensors(tmp_path, symbol="SPX", date_str="20250103", keys=["a"])
        assert list(loaded.keys()) == ["a"]
        np.testing.assert_array_equal(loaded["a"], arrays["a"])

    def test_multi_symbol_isolation(self, tmp_path: Path) -> None:
        arr_spx = {"v": np.array([1.0, 2.0])}
        arr_ndx = {"v": np.array([10.0, 20.0])}
        save_zarr_tensors(arr_spx, tmp_path, symbol="SPX", date_str="20250103")
        save_zarr_tensors(arr_ndx, tmp_path, symbol="NDX", date_str="20250103")

        spx = load_zarr_tensors(tmp_path, symbol="SPX", date_str="20250103")
        ndx = load_zarr_tensors(tmp_path, symbol="NDX", date_str="20250103")
        np.testing.assert_array_equal(spx["v"], [1.0, 2.0])
        np.testing.assert_array_equal(ndx["v"], [10.0, 20.0])

    def test_store_path(self, tmp_path: Path) -> None:
        arrays = {"x": np.array([0])}
        path = save_zarr_tensors(arrays, tmp_path, symbol="SPX", date_str="20250103")
        assert path == tmp_path / "SPX" / "20250103.zarr"
        assert path.is_dir()


# ---------------------------------------------------------------------------
# list_zarr_dates
# ---------------------------------------------------------------------------


class TestListZarrDates:
    def test_lists_dates(self, tmp_path: Path) -> None:
        for d in ["20250101", "20250103", "20250102"]:
            save_zarr_tensors({"x": np.array([0])}, tmp_path, "SPX", d)
        assert list_zarr_dates(tmp_path, "SPX") == [
            "20250101",
            "20250102",
            "20250103",
        ]

    def test_empty_for_missing_symbol(self, tmp_path: Path) -> None:
        assert list_zarr_dates(tmp_path, "MISSING") == []


# ---------------------------------------------------------------------------
# DuckDB query
# ---------------------------------------------------------------------------


class TestDuckDB:
    def test_query_parquet(self, tmp_path: Path, sample_table: pa.Table) -> None:
        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        df = query_duckdb(
            "SELECT symbol, SUM(volume) AS total_vol "
            "FROM read_parquet('{base_dir}/**/*.parquet', hive_partitioning=true) "
            "GROUP BY symbol ORDER BY symbol",
            base_dir=base,
        )
        assert list(df["symbol"]) == ["NDX", "SPX"]
        assert list(df["total_vol"]) == [550, 450]


# ---------------------------------------------------------------------------
# Polars scan
# ---------------------------------------------------------------------------


class TestPolarsScan:
    def test_lazy_scan(self, tmp_path: Path, sample_table: pa.Table) -> None:
        import polars as pl

        base = tmp_path / "pq"
        write_parquet(sample_table, base)

        lf = scan_polars(base)
        result = lf.filter(pl.col("symbol") == "SPX").collect()
        assert result.shape[0] == 3
