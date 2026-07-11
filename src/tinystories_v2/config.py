"""Declarative stage configs (TOML) and .env secret loading.

Every stage entrypoint reads one TOML file and writes artifacts to the
config's out_dir; stages share nothing in memory (PRD stage convention).
"""

import os
import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_env(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values.

    Values are secrets (HF token) — never log or print them.
    """
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
