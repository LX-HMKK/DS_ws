import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"

PATH_KEYS = {
    ("calibration", "result_file"),
    ("calibration", "image_output_dir"),
    ("model", "digit_model_path"),
}


def _resolve_project_path(value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def load_app_config(config_path=None):
    selected_path = config_path or os.environ.get("DS_CONFIG_PATH") or DEFAULT_CONFIG_PATH
    selected_path = Path(selected_path)

    with selected_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for section, key in PATH_KEYS:
        if section in config and key in config[section]:
            config[section][key] = _resolve_project_path(config[section][key])

    return config
