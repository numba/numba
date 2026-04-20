import subprocess
import sys
from pathlib import Path
from typing import Final

_HERE = Path(__file__).resolve().parent

MYPY_CONFIG: Final = _HERE / "stubtest" / "mypy.ini"
ALLOWLIST: Final = _HERE / "stubtest" / "allowlist.txt"


def main() -> int:
    result = subprocess.run([
        sys.executable,
        "-m",
        "mypy.stubtest",
        "--ignore-disjoint-bases",
        "--mypy-config-file",
        MYPY_CONFIG,
        "--allowlist",
        ALLOWLIST,
        *sys.argv[1:],
        "numba",
    ])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
