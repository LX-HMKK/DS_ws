import tempfile
import unittest
from pathlib import Path


class HikrobotPathTest(unittest.TestCase):
    def test_selects_linux_arm64_bundled_library(self):
        from tools.hikrobot_paths import select_bundled_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib = root / "drivers" / "hikrobot" / "lib" / "arm64" / "libMvCameraControl.so"
            lib.parent.mkdir(parents=True)
            lib.write_bytes(b"")

            selected = select_bundled_library(root, system_name="Linux", machine="aarch64")

        self.assertEqual(selected, lib)

    def test_selects_linux_amd64_bundled_library(self):
        from tools.hikrobot_paths import select_bundled_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib = root / "drivers" / "hikrobot" / "lib" / "amd64" / "libMvCameraControl.so"
            lib.parent.mkdir(parents=True)
            lib.write_bytes(b"")

            selected = select_bundled_library(root, system_name="Linux", machine="x86_64")

        self.assertEqual(selected, lib)

    def test_windows_does_not_select_linux_so(self):
        from tools.hikrobot_paths import select_bundled_library

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib = root / "drivers" / "hikrobot" / "lib" / "amd64" / "libMvCameraControl.so"
            lib.parent.mkdir(parents=True)
            lib.write_bytes(b"")

            selected = select_bundled_library(root, system_name="Windows", machine="AMD64")

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
