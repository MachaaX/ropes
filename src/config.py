import json
import os
from copy import deepcopy
from typing import Any, Dict, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default.json")

# configerror inherits from Exception class
class ConfigError(Exception):
    def __init__(self, message: str):
        super().__init__(message) # sets the Exception class's message attribute

    @classmethod
    def file_not_found(cls, path: str) -> "ConfigError":
        return cls(f"Config file not found: {path}")

    @classmethod
    def invalid_json(cls, path: str, exc: Exception) -> "ConfigError":
        return cls(f"Invalid JSON in: {path}\n{exc}")


def merge_default_and_custom_config(base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two config dicts, with `custom_config` taking precedence."""
    out = deepcopy(base_config)
    for k, v in (custom_config or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_default_and_custom_config(out[k], v)
        else:
            out[k] = v
    return out


def read_json(path: str) -> Dict[str, Any]:
    """Read and parse a JSON file, raising ConfigError on failure."""
    if not os.path.exists(path):
        raise ConfigError.file_not_found(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError.invalid_json(path, e) from e

def resolve_config_path(arg: str | None) -> str | None:
    """Resolve a config file path. Returns absolute path or None.
    If arg is just a filename, try <project>/config/<arg>.
    Otherwise treat arg as a path (absolute or relative).

    python app.py -> ./config/default.json
    python app.py my_config.json -> ./config/my_config.json
    python app.py -m /Users/me/temp.json → uses the absolute path directly
    """
    if arg is None:
        return None
    arg = os.path.expanduser(arg) # expand ~ to home dir
    
    # if path-like (contains / or \ or is absolute), use as-is
    if os.path.isabs(arg) or os.path.sep in arg or (os.path.altsep and os.path.altsep in arg):
        return os.path.abspath(arg)
      
    # if bare filename; look under /config
    guess = os.path.join(PROJECT_ROOT, "config", arg)
    return os.path.abspath(guess)


def load_config(override_path: str | None = None) -> Tuple[Dict[str, Any], str]:
    """Load default.json and deep-merge an optional override file.
    Returns: (config_dict, path_used_for_override_or_default)
    """
    base = read_json(DEFAULT_CONFIG_PATH)
    used = DEFAULT_CONFIG_PATH
    if override_path:
        override_path = resolve_config_path(override_path)
        override = read_json(override_path)
        base = merge_default_and_custom_config(base, override)
        # print(base)
        used = override_path
    return base, used


def ensure_config_dict_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Basic schema validation + fill map defaults for safety
    required_top = ["app", "data", "map", "fields", "voltage_bands"]
    for k in required_top:
        if k not in cfg:
            raise ConfigError(f"Missing top-level config key: {k}")

    # Required field names in CSV
    for fkey in ["id", "from", "to", "voltage", "type", "status", "geometry"]:
        if fkey not in cfg["fields"]:
            raise ConfigError(f"Missing fields.{fkey} in config")

    # Map defaults
    cfg.setdefault("map", {}).setdefault("line", {}).setdefault("width", 2)
    cfg.setdefault("map", {}).setdefault("initial_view", {}).setdefault("fallback_zoom", 5)

    return cfg