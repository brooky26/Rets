"""
Config loader: YAML base + environment variable overrides for secrets.

Why env vars at all: app_id/api_token/account_id/Supabase credentials must
not live in a committed YAML file once this runs on Railway. YAML stays
the source of truth for everything non-secret (symbols, thresholds,
windows, etc.); a small fixed set of security- and deployment-sensitive
fields can be overridden by environment variable at deploy time. Nothing
else is override-able by design — this is deliberately not a generic
"any config field can come from env" mechanism, to keep the override
surface small, obvious, and easy to audit.

Precedence: environment variable > YAML value > pydantic default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from configs.schema import PlatformConfig

# (env var name, dotted path into the raw config dict). Dotted path is
# resolved/created as needed — a missing intermediate dict is fine, a
# missing final key is fine, but a non-dict intermediate value is an error
# (should never happen against this schema's shape).
_ENV_OVERRIDES: list[tuple[str, str]] = [
    ("DERIV_APP_ID", "market_data.connection.app_id"),
    ("DERIV_API_TOKEN", "market_data.connection.api_token"),
    ("DERIV_ACCOUNT_ID", "market_data.connection.account_id"),
    ("DERIV_ACCOUNT_TYPE", "market_data.connection.ws_account_type"),
    ("STORAGE_BACKEND", "market_data.storage.backend"),
    ("SQLITE_PATH", "market_data.storage.sqlite_path"),
    ("SUPABASE_URL", "market_data.storage.supabase_url"),
    ("SUPABASE_KEY", "market_data.storage.supabase_key"),
]


def _set_dotted(d: dict, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cur = d
    for part in parts[:-1]:
        existing = cur.get(part)
        if existing is None:
            existing = {}
            cur[part] = existing
        elif not isinstance(existing, dict):
            raise TypeError(
                f"Cannot apply env override to '{dotted_path}': "
                f"'{part}' is a {type(existing).__name__}, not a mapping."
            )
        cur = existing
    cur[parts[-1]] = value


def apply_env_overrides(raw: dict) -> dict:
    """Mutates and returns `raw` with any matching environment variables
    applied on top of the YAML values. Only variables that are actually
    set in the environment are applied — an unset env var never clobbers
    a YAML value with None."""
    for env_name, dotted_path in _ENV_OVERRIDES:
        value = os.environ.get(env_name)
        if value is not None:
            _set_dotted(raw, dotted_path, value)
    return raw


def load_config(path: str | Path) -> PlatformConfig:
    """Load and validate the platform YAML config, then apply environment
    variable overrides for secrets/deployment fields (see _ENV_OVERRIDES).
    Raises pydantic.ValidationError with a clear field-level message if
    anything is missing or malformed after overrides are applied."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    raw = apply_env_overrides(raw)
    return PlatformConfig(**raw)
