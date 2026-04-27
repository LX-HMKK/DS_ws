import subprocess
import sys
import unittest


class PathSetupTest(unittest.TestCase):
    def test_scripts_bootstrap_supports_new_project_layout_imports(self):
        code = (
            "from scripts._bootstrap import configure_paths;"
            "configure_paths();"
            "import importlib.util;"
            "import tools.config_loader;"
            "assert importlib.util.find_spec('drivers.send_data') is not None;"
            "assert importlib.util.find_spec('modules.detect') is not None;"
            "assert importlib.util.find_spec('drivers.hikrobot.HIK_CAM') is not None;"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
