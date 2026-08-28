"""Single .env loader used by every entry point.

Values already present in ``os.environ`` always win; the file only fills gaps.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(env_path: str | os.PathLike | None = None) -> Path | None:
    """Load KEY=VALUE lines from a .env file into os.environ (non-overriding).

    Resolution order: explicit ``env_path`` argument, the ``PRISM_ENV_FILE``
    env var, else the first ``.env`` found walking up from the current
    directory. Returns the path loaded, or None if no file was found.
    """
    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(Path(env_path))
    elif os.environ.get("PRISM_ENV_FILE"):
        candidates.append(Path(os.environ["PRISM_ENV_FILE"]))
    else:
        cur = Path.cwd()
        candidates.extend(parent / ".env" for parent in [cur, *cur.parents])

    for p in candidates:
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
            return p
    return None


def wandb_mode() -> str | None:
    """Mode to pass to ``wandb.init``.

    ``WANDB_MODE`` (online / offline / disabled) always wins. Otherwise, with
    no ``WANDB_API_KEY`` and no ``~/.netrc`` login, return ``"disabled"`` so a
    machine without W&B credentials trains instead of blocking on the
    interactive login prompt. Returns ``None`` (= W&B's own default) when
    credentials are present.
    """
    if os.environ.get("WANDB_MODE"):
        return None
    if os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").is_file():
        return None
    return "disabled"
