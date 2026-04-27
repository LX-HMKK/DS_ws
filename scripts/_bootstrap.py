import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_PATHS = (
    PROJECT_ROOT,
    PROJECT_ROOT / "modules",
    PROJECT_ROOT / "tools",
)


def configure_paths():
    for path in reversed(IMPORT_PATHS):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
