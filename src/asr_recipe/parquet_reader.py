from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from asr_recipe.progress import NullProgressReporter, ProgressReporter


class TableLike(Protocol):
    def to_pylist(self) -> list[dict[str, object]]:
        """Return rows as Python dictionaries."""


class ParquetReader(Protocol):
    def read_rows(self, path: str, columns: list[str], limit: int | None = None) -> TableLike:
        """Read a subset of parquet columns from a parquet source."""

    def iter_batches(
        self,
        path: str,
        columns: list[str],
        batch_size: int = 1000,
    ) -> Iterator[tuple[str, list[dict[str, object]]]]:
        """Yield source path plus row batches."""


class PyArrowParquetReader:
    def __init__(self, progress: ProgressReporter | None = None) -> None:
        self.progress = progress or NullProgressReporter()

    def read_rows(self, path: str, columns: list[str], limit: int | None = None) -> TableLike:
        import pyarrow as pa

        if "*" in path:
            self.progress.emit(f"Scanning parquet shards: {path}")
            tables: list[pa.Table] = []
            remaining = limit
            for _, rows in self.iter_batches(path, columns=columns, batch_size=limit or 1000):
                table = pa.Table.from_pylist(rows)
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
        batches = list(self.iter_batches(path, columns=columns, batch_size=limit or 1000))
        if not batches:
            raise FileNotFoundError(path)
        table = pa.Table.from_pylist([row for _, rows in batches for row in rows])
        if limit is not None:
            table = table.slice(0, limit)
        return table

    def iter_batches(
        self,
        path: str,
        columns: list[str],
        batch_size: int = 1000,
    ) -> Iterator[tuple[str, list[dict[str, object]]]]:
        import fsspec
        import pyarrow.parquet as pq

        paths = fsspec.open_files(path, "rb") if "*" in path else [fsspec.open(path, "rb")]
        for handle in paths:
            source_path = getattr(handle, "path", path)
            self.progress.emit(f"Reading shard: {source_path}")
            with handle as stream:
                parquet_file = pq.ParquetFile(stream)
                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                    yield source_path, batch.to_pylist()
