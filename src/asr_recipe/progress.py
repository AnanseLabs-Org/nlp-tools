from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ProgressReporter(Protocol):
    def emit(self, message: str) -> None:
        """Emit a progress update."""


@dataclass(frozen=True)
class NullProgressReporter:
    def emit(self, message: str) -> None:
        return None
