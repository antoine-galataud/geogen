"""Stage every requested export before replacing any final output."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from geogen.reconstruction.types import ReconstructionError


class OutputTransaction:
    """Rollback replacements on failure; individual replacements are atomic."""

    def __init__(self, paths: list[Path]):
        resolved = [path.resolve() for path in paths]
        if len(resolved) != len(set(resolved)):
            raise ReconstructionError("output_conflict", "Requested output paths must be distinct")
        if any(path.exists() and not path.is_file() for path in resolved):
            raise ReconstructionError("output_conflict", "An output path is not a regular file")
        self.paths = resolved
        self.staged: dict[Path, Path] = {}
        self.backups: dict[Path, Path] = {}

    def __enter__(self):
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(
                    prefix=".geogen-", suffix=path.suffix, dir=path.parent
                )
                os.close(descriptor)
                self.staged[path] = Path(temporary)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def stage(self, path: Path) -> Path:
        return self.staged[path.resolve()]

    def commit(self):
        if any(not p.is_file() or p.stat().st_size == 0 for p in self.staged.values()):
            raise ReconstructionError("export_incomplete", "An export produced no complete file")
        published = []
        try:
            for path in self.paths:
                if path.exists():
                    descriptor, backup = tempfile.mkstemp(prefix=".geogen-backup-", dir=path.parent)
                    os.close(descriptor)
                    os.replace(path, backup)
                    self.backups[path] = Path(backup)
                os.replace(self.staged[path], path)
                published.append(path)
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            for path, backup in self.backups.items():
                os.replace(backup, path)
            self.backups.clear()
            raise

    def __exit__(self, *args):
        for path in (*self.staged.values(), *self.backups.values()):
            path.unlink(missing_ok=True)
