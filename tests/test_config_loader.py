import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ConfigLoaderTest(unittest.TestCase):
    def write_config(self, directory: Path) -> Path:
        config_path = directory / "app.yaml"
        config_path.write_text(
            """
camera:
  exposure_auto: false
  exposure_time: 8000.0
  frame_timeout_ms: 1000
calibration:
  result_file: configs/camera_calibration.yaml
  image_output_dir: calibration_images
  chessboard_size: [11, 8]
  square_size: 30.0
  min_images: 15
  crop: [124, 1924, 324, 2124]
measurement:
  rect_width_mm: 170
  rect_height_mm: 267
  camera_offset_mm: -90.0
model:
  digit_model_path: DS_NUM.bin
  conf_thres: 0.30
  iou_thres: 0.6
serial:
  choice:
    port: /dev/ttyS1
    baudrate: 115200
  power:
    port: /dev/ttyS3
    baudrate: 115200
""".strip(),
            encoding="utf-8",
        )
        return config_path

    def test_load_app_config_resolves_project_relative_paths(self):
        from tools.config_loader import load_app_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = self.write_config(Path(tmp))
            config = load_app_config(config_path)
            repo_root = Path(__file__).resolve().parents[1]

        self.assertEqual(
            config["calibration"]["result_file"],
            str(repo_root / "configs" / "camera_calibration.yaml"),
        )
        self.assertEqual(
            config["calibration"]["image_output_dir"],
            str(repo_root / "calibration_images"),
        )
        self.assertEqual(config["model"]["digit_model_path"], str(repo_root / "DS_NUM.bin"))

    def test_load_app_config_uses_environment_override(self):
        from tools.config_loader import load_app_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = self.write_config(Path(tmp))
            with patch.dict(os.environ, {"DS_CONFIG_PATH": str(config_path)}):
                config = load_app_config()

        self.assertEqual(config["camera"]["exposure_time"], 8000.0)
        self.assertEqual(config["serial"]["choice"]["port"], "/dev/ttyS1")


if __name__ == "__main__":
    unittest.main()
