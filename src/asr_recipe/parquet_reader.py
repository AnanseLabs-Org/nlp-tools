from __future__ import annotations

from typing import Protocol

from asr_recipe.progress import NullProgressReporter, ProgressReporter


class TableLike(Protocol):
    def to_pylist(self) -> list[dict[str, object]]:
        """Return rows as Python dictionaries."""


class ParquetReader(Protocol):
    def read_rows(self, path: str, columns: list[str], limit: int | None = None) -> TableLike:
        """Read a subset of parquet columns from a parquet source."""


class PyArrowParquetReader:
    def __init__(self, progress: ProgressReporter | None = None) -> None:
        self.progress = progress or NullProgressReporter()

    def read_rows(self, path: str, columns: list[str], limit: int | None = None) -> TableLike:
        import fsspec
        import pyarrow as pa
        import pyarrow.parquet as pq

        if "*" in path:
            self.progress.emit(f"Scanning parquet shards: {path}")
            tables = []
            remaining = limit
            for handle in fsspec.open_files(path, "rb"):
                self.progress.emit(f"Reading shard: {handle.path}")
                with handle as stream:
                    table = pq.read_table(stream, columns=columns)
                if remaining is not None:
                    table = table.slice(0, remaining)
                    remaining -= table.num_rows
                tables.append(table)
                if remaining is not None and remaining <= 0:
                    break
            if not tables:
                raise FileNotFoundError(path)
            return pa.concat_tables(tables)

        self.progress.emit(f"Reading parquet: {path}")
        with fsspec.open(path, "rb") as handle:
            table = pq.read_table(handle, columns=columns)
        if limit is not None:
            table = table.slice(0, limit)
        return table
