"""Post-build sanity tests for a built osx-arm64 wheel.

Run after ``patch_wheel_dylib.py`` from the wheel-builder workflow.

Currently checks:
- No absolute ``LC_RPATH`` entries on any ``.so`` (build-env rpaths
  like ``/Users/runner/miniconda3/envs/test/lib`` must not leak into
  shipped wheels). See PR #10532 / issue #10535.
"""
import sys
import tempfile
import zipfile
import subprocess as subp
from pathlib import Path


def test_no_abs_lc_rpath(extracted_root):
    for so in Path(extracted_root).rglob('*.so'):
        lc_lines = subp.check_output(
            ['otool', '-l', str(so)]).decode().splitlines()
        for i, line in enumerate(lc_lines):
            if 'cmd LC_RPATH' in line:
                rp = lc_lines[i + 2].split()[1]
                assert not rp.startswith('/'), \
                    'absolute LC_RPATH {} in {}'.format(rp, so.name)


def main(whl):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(tmp)
        test_no_abs_lc_rpath(tmp)


if __name__ == '__main__':
    main(*sys.argv[1:])
