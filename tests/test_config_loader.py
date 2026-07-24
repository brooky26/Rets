import os
from unittest.mock import patch

import pytest

from configs.loader import apply_env_overrides, load_config


@pytest.fixture
def base_raw():
    return {
        "market_data": {
            "connection": {"app_id": "yaml_app_id"},
            "storage": {"backend": "sqlite", "sqlite_path": "./yaml_path.db"},
        }
    }


def test_unset_env_vars_leave_yaml_values_untouched(base_raw):
    with patch.dict(os.environ, {}, clear=True):
        result = apply_env_overrides(base_raw)
    assert result["market_data"]["connection"]["app_id"] == "yaml_app_id"
    assert result["market_data"]["storage"]["sqlite_path"] == "./yaml_path.db"


def test_env_var_overrides_yaml_value(base_raw):
    with patch.dict(os.environ, {"DERIV_APP_ID": "env_app_id"}, clear=True):
        result = apply_env_overrides(base_raw)
    assert result["market_data"]["connection"]["app_id"] == "env_app_id"


def test_env_var_creates_missing_nested_keys():
    raw = {}
    with patch.dict(os.environ, {"DERIV_APP_ID": "env_app_id"}, clear=True):
        result = apply_env_overrides(raw)
    assert result["market_data"]["connection"]["app_id"] == "env_app_id"


def test_deriv_account_type_env_var_maps_to_real_schema_field(base_raw):
    """Regression test: this override previously targeted a nonexistent
    'account_type' key, which pydantic silently ignored (no model here sets
    extra='forbid') — so setting DERIV_ACCOUNT_TYPE had zero effect on the
    real field, ws_account_type. Verified two ways: the raw dict shape
    apply_env_overrides produces, AND a full load_config pass actually
    reads it back off the validated model (the level a silent-no-op bug
    would previously have slipped through undetected)."""
    with patch.dict(os.environ, {"DERIV_ACCOUNT_TYPE": "real"}, clear=True):
        result = apply_env_overrides(base_raw)
    assert result["market_data"]["connection"]["ws_account_type"] == "real"
    assert "account_type" not in result["market_data"]["connection"]


def test_deriv_account_type_env_var_takes_effect_through_load_config(tmp_path):
    config_yaml = tmp_path / "test_config.yaml"
    config_yaml.write_text(
        "market_data:\n"
        "  connection:\n"
        "    app_id: 'yaml_app_id'\n"
    )
    with patch.dict(os.environ, {"DERIV_ACCOUNT_TYPE": "real"}, clear=True):
        config = load_config(config_yaml)
    assert config.market_data.connection.ws_account_type == "real"


def test_multiple_overrides_apply_independently(base_raw):
    env = {
        "DERIV_APP_ID": "env_app_id",
        "SUPABASE_URL": "https://x.supabase.co",
        "SUPABASE_KEY": "sk_test",
        "STORAGE_BACKEND": "supabase",
    }
    with patch.dict(os.environ, env, clear=True):
        result = apply_env_overrides(base_raw)
    assert result["market_data"]["connection"]["app_id"] == "env_app_id"
    assert result["market_data"]["storage"]["backend"] == "supabase"
    assert result["market_data"]["storage"]["supabase_url"] == "https://x.supabase.co"
    assert result["market_data"]["storage"]["supabase_key"] == "sk_test"


def test_load_config_applies_env_override_end_to_end(tmp_path):
    yaml_content = """
market_data:
  connection:
    app_id: "yaml_app_id"
    api_token: "yaml_token"
    account_id: "yaml_account"
  storage:
    backend: "sqlite"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content)

    with patch.dict(os.environ, {"DERIV_APP_ID": "env_app_id"}, clear=True):
        config = load_config(config_path)

    assert config.market_data.connection.app_id == "env_app_id"
    assert config.market_data.connection.api_token == "yaml_token"  # unset env, YAML kept


def test_load_config_env_overrides_enable_supabase_backend(tmp_path):
    yaml_content = """
market_data:
  connection:
    app_id: "yaml_app_id"
    api_token: "yaml_token"
    account_id: "yaml_account"
  storage:
    backend: "sqlite"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml_content)

    env = {
        "STORAGE_BACKEND": "supabase",
        "SUPABASE_URL": "https://x.supabase.co",
        "SUPABASE_KEY": "sk_test",
    }
    with patch.dict(os.environ, env, clear=True):
        config = load_config(config_path)

    assert config.market_data.storage.backend == "supabase"
    assert config.market_data.storage.supabase_url == "https://x.supabase.co"
    assert config.market_data.storage.supabase_key == "sk_test"
