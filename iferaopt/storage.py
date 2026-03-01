"""Local data storage utilities for Parquet and Zarr formats.

Provides helpers for:
- Writing and reading Hive-partitioned Parquet datasets (partitioned by symbol
  and date) with zstd compression.
- Saving and loading Zarr tensor stores for GPU-ready NumPy arrays, organised
  by symbol and date.
- Lightweight DuckDB and Polars query utilities for ad-hoc exploration of
  partitioned Parquet data.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def write_parquet(
    table: pa.Table,
    base_dir: str | Path,
    *,
    partition_cols: Sequence[str] = ("symbol", "date"),
    compression: str = "zstd",
    existing_data_behavior: str = "overwrite_or_ignore",
    max_rows_per_file: int = 5_000_000,
) -> None:
    """Write a PyArrow Table as a Hive-partitioned Parquet dataset.

    Parameters
    ----------
    table:
        Data to write.  Must contain the columns listed in *partition_cols*.
    base_dir:
        Root directory for the dataset (e.g. ``data/raw/quotes``).
    partition_cols:
        Column names used for Hive-style directory partitioning.
        Defaults to ``("symbol", "date")``.
    compression:
        Parquet compression codec (default ``"zstd"``).
    existing_data_behavior:
        How to handle existing data in the target directory.
    max_rows_per_file:
        Maximum number of rows per output file.
    """
    base_dir = Path(base_dir)
    partitioning = ds.partitioning(
        pa.schema([(col, table.schema.field(col).type) for col in partition_cols]),
        flavor="hive",
    )
    ds.write_dataset(
        table,
        base_dir=str(base_dir),
        partitioning=partitioning,
        format="parquet",
        file_options=ds.ParquetFileFormat().make_write_options(compression=compression),
        existing_data_behavior=existing_data_behavior,
        max_rows_per_file=max_rows_per_file,
    )


def read_parquet(
    base_dir: str | Path,
    *,
    symbols: str | Sequence[str] | None = None,
    dates: date | Sequence[date] | None = None,
    columns: Sequence[str] | None = None,
) -> pa.Table:
    """Read a Hive-partitioned Parquet dataset with optional filters.

    Parameters
    ----------
    base_dir:
        Root directory of the partitioned dataset.
    symbols:
        If provided, only load rows for these symbol(s).
    dates:
        If provided, only load rows for these date(s).
        Dates are matched against the ``date`` partition column as strings
        formatted ``YYYYMMDD``.
    columns:
        Subset of columns to read (``None`` = all).

    Returns
    -------
    pa.Table
    """
    base_dir = Path(base_dir)
    dataset = ds.dataset(str(base_dir), format="parquet", partitioning="hive")

    filters: list[ds.Expression] = []
    if symbols is not None:
        if isinstance(symbols, str):
            symbols = [symbols]
        filters.append(ds.field("symbol").isin(list(symbols)))
    if dates is not None:
        if isinstance(dates, date):
            dates = [dates]
        date_strings = [d.strftime("%Y%m%d") for d in dates]
        filters.append(ds.field("date").isin(date_strings))

    combined = filters[0] if filters else None
    for f in filters[1:]:
        combined = combined & f  # type: ignore[union-attr]

    return dataset.to_table(filter=combined, columns=columns)


# ---------------------------------------------------------------------------
# Zarr helpers
# ---------------------------------------------------------------------------


def save_zarr_tensors(
    arrays: dict[str, np.ndarray],
    base_dir: str | Path,
    symbol: str,
    date_str: str,
    *,
    compressor: str = "zstd",
    clevel: int = 5,
) -> Path:
    """Persist a dict of named NumPy arrays into a Zarr directory store.

    The store is located at ``<base_dir>/<symbol>/<date_str>.zarr``.

    Parameters
    ----------
    arrays:
        Mapping of array names to NumPy arrays (e.g.
        ``{"bars": np.ndarray, "chain": np.ndarray}``).
    base_dir:
        Root tensor directory (e.g. ``data/processed/tensors``).
    symbol:
        Instrument symbol (e.g. ``"SPX"``).
    date_str:
        Trading date as ``YYYYMMDD`` string.
    compressor:
        Compression algorithm name passed to ``numcodecs.Blosc``.
    clevel:
        Compression level (1–9).

    Returns
    -------
    Path to the created ``.zarr`` directory.
    """
    import zarr
    from zarr.codecs import BloscCodec

    base_dir = Path(base_dir)
    store_path = base_dir / symbol / f"{date_str}.zarr"
    store_path.mkdir(parents=True, exist_ok=True)

    comp = BloscCodec(cname=compressor, clevel=clevel)
    root = zarr.open_group(str(store_path), mode="w")

    for name, arr in arrays.items():
        root.create_array(name=name, data=arr, compressors=comp, overwrite=True)

    return store_path


def load_zarr_tensors(
    base_dir: str | Path,
    symbol: str,
    date_str: str,
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load arrays from a Zarr directory store.

    Parameters
    ----------
    base_dir:
        Root tensor directory.
    symbol:
        Instrument symbol.
    date_str:
        Trading date as ``YYYYMMDD`` string.
    keys:
        If supplied, load only these array names.  ``None`` loads all.

    Returns
    -------
    Dict mapping array names to NumPy arrays.
    """
    import zarr

    store_path = Path(base_dir) / symbol / f"{date_str}.zarr"
    root = zarr.open_group(str(store_path), mode="r")

    if keys is None:
        keys = list(root.keys())

    return {k: np.asarray(root[k]) for k in keys}


def list_zarr_dates(
    base_dir: str | Path,
    symbol: str,
) -> list[str]:
    """Return sorted list of date strings that have Zarr stores for *symbol*.

    Parameters
    ----------
    base_dir:
        Root tensor directory.
    symbol:
        Instrument symbol.

    Returns
    -------
    Sorted list of ``YYYYMMDD`` date strings.
    """
    symbol_dir = Path(base_dir) / symbol
    if not symbol_dir.is_dir():
        return []
    return sorted(p.stem for p in symbol_dir.iterdir() if p.suffix == ".zarr")


# ---------------------------------------------------------------------------
# DuckDB / Polars query utilities
# ---------------------------------------------------------------------------


def query_duckdb(
    sql: str,
    base_dir: str | Path | None = None,
) -> "pandas.DataFrame":
    """Run an arbitrary SQL query via DuckDB and return a pandas DataFrame.

    Parameters
    ----------
    sql:
        SQL query string.  May contain the literal ``{base_dir}`` which will
        be replaced by *base_dir* when that argument is not ``None``.
    base_dir:
        Optional path to substitute into the query.  When ``None`` the query
        is executed as-is (no substitution takes place).

    Example::

        query_duckdb(
            "SELECT * FROM '{base_dir}/**/*.parquet' WHERE symbol = 'SPX'",
            base_dir="data/processed/features",
        )
    """
    import duckdb

    if base_dir is not None:
        sql = sql.replace("{base_dir}", str(base_dir))

    return duckdb.execute(sql).df()


def scan_polars(
    base_dir: str | Path,
    *,
    hive_partitioning: bool = True,
) -> "polars.LazyFrame":
    """Return a Polars lazy scan over Hive-partitioned Parquet files.

    Parameters
    ----------
    base_dir:
        Root directory of the partitioned Parquet dataset.
    hive_partitioning:
        Whether to use Hive-style partition discovery.

    Returns
    -------
    polars.LazyFrame
    """
    import polars as pl

    return pl.scan_parquet(
        str(Path(base_dir) / "**/*.parquet"),
        hive_partitioning=hive_partitioning,
    )
