"""Configuration module for nanobot."""

from nanobot.config.loader import get_config_path, load_config
from nanobot.config.paths import (
    get_data_dir,
    get_logs_dir,
    get_media_dir,
    get_runtime_subdir,
    get_workspace_path,
)
from nanobot.config.schema import Config

__all__ = [
    "Config",
    "load_config",
    "get_config_path",
    "get_data_dir",
    "get_runtime_subdir",
    "get_media_dir",
    "get_logs_dir",
    "get_workspace_path",
]
