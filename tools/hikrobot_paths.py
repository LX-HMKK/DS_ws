import platform
from pathlib import Path


LINUX_ARCH_DIRS = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


def select_bundled_library(repo_root, system_name=None, machine=None):
    system_name = system_name or platform.system()
    machine = (machine or platform.machine()).lower()

    if system_name != "Linux":
        return None

    arch_dir = LINUX_ARCH_DIRS.get(machine)
    if arch_dir is None:
        return None

    candidate = Path(repo_root) / "drivers" / "hikrobot" / "lib" / arch_dir / "libMvCameraControl.so"
    if candidate.is_file():
        return candidate
    return None


def mvs_sdk_roots():
    return [
        r"C:\Program Files (x86)\MVS",
        r"C:\Program Files\MVS",
        r"C:\Program Files (x86)\Common Files\MVS",
        r"C:\Program Files\Common Files\MVS",
    ]


def windows_runtime_dirs(sdk_root):
    root = Path(sdk_root)
    return [
        root / "Runtime" / "Win64_x64",
        root / "Runtime" / "Win32_i86",
        root / "bin",
    ]


def windows_python_import_dirs(sdk_root):
    root = Path(sdk_root)
    return [
        root / "Development" / "Samples" / "Python" / "MvImport",
        root / "Development" / "Samples" / "Python" / "MvImport64",
        root / "MvImport",
    ]
