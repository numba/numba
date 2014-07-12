"""
Uses conda to run and test all supported python + numpy versions.
"""

from __future__ import print_function
import itertools
import subprocess
import os
import sys

if '-q' in sys.argv[1:]:
    NPY = '18',
else:
    NPY = '16', '17', '18'
PY = '26', '27', '33'
RECIPE_DIR = "./buildscripts/condarecipe.local"


def main():
    failfast = '-v' in sys.argv[1:]

    args = "conda build %s --no-binstar-upload" % RECIPE_DIR

    failures = []
    for py, npy in itertools.product(PY, NPY):
        if py == '33' and npy == '16':
            # Skip python3 + numpy16
            continue

        os.environ['CONDA_PY'] = py
        os.environ['CONDA_NPY'] = npy

        try:
            subprocess.check_call(args.split())
        except subprocess.CalledProcessError as e:
            failures.append((py, npy, e))
            if failfast:
                break

    print("=" * 80)
    if failures:
        for py, npy, err in failures:
            print("Test failed for python %s numpy %s" % (py, npy))
            print(err)
    else:
        print("All Passed")


if __name__ == '__main__':
    main()

