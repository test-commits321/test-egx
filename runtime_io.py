"""Atomic runtime files and a cross-process lock for each writer."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_text(path: Path, encoding='utf-8'):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = NamedTemporaryFile(mode='w', encoding=encoding, newline='', dir=path.parent,
                                prefix=f'.{path.name}.', suffix='.tmp', delete=False)
    temp = Path(handle.name)
    try:
        with handle:
            yield handle
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def dump_atomic(path: Path, obj):
    with atomic_text(path) as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


@contextmanager
def trial_lock(runtime: Path, filename='.trial.lock'):
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / filename).open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another process is updating {runtime}. Retry shortly.') from exc
        yield
