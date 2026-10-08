"""Validate that a built wheel contains Numba type information files."""
import glob
import sys
import zipfile
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _expand_wheels(args):
    wheels = []
    for arg in args:
        if glob.has_magic(arg):
            matches = [Path(p) for p in glob.glob(arg)]
        else:
            matches = [Path(arg)]
        if not matches:
            raise AssertionError("no wheel files match '{}'".format(arg))
        wheels.extend(matches)

    seen = set()
    unique = []
    for wheel in wheels:
        wheel = wheel.resolve()
        if wheel in seen:
            continue
        seen.add(wheel)
        if not wheel.exists():
            raise AssertionError("wheel file does not exist: {}".format(wheel))
        if wheel.suffix != ".whl":
            raise AssertionError("not a wheel file: {}".format(wheel))
        unique.append(wheel)
    return unique


def _expected_type_files():
    root = _repo_root()
    numba_dir = root / "numba"
    expected = [numba_dir / "py.typed"]
    expected.extend(sorted(numba_dir.rglob("*.pyi")))

    missing = [p for p in expected if not p.exists()]
    if missing:
        msg = "expected source files are missing:\n{}".format(
            "\n".join("  {}".format(p) for p in missing)
        )
        raise AssertionError(msg)

    pyi_count = sum(1 for p in expected if p.suffix == ".pyi")
    if pyi_count == 0:
        raise AssertionError("no .pyi files found in source tree")

    return [p.relative_to(root).as_posix() for p in expected]


def _check_wheel(wheel, expected):
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())

    missing = [name for name in expected if name not in names]
    if missing:
        msg = "{} is missing type information files:\n{}".format(
            wheel,
            "\n".join("  {}".format(name) for name in missing),
        )
        raise AssertionError(msg)

    print("{} contains {} type information files".format(wheel, len(expected)))


def main(args):
    if not args:
        raise AssertionError("expected at least one wheel path")

    wheels = _expand_wheels(args)
    expected = _expected_type_files()
    for wheel in wheels:
        _check_wheel(wheel, expected)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
