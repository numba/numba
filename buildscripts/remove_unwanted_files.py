"""
Workaround for a conda-build bug where failing to compile some Python files
results in a build failure.

See https://github.com/conda/conda-build/issues/1001
"""

import os
import sys


py2_only_files = []

py3_only_files = [
    'numba/tests/annotation_usecases.py',
    ]


def remove_files(basedir):
    """
    Remove unwanted files from the current source tree
    """
    if sys.version_info >= (3,):
        removelist = py2_only_files
        msg = "Python 2-only file"
    else:
        removelist = py3_only_files
        msg = "Python 3-only file"
    for relpath in removelist:
        path = os.path.join(basedir, relpath)
        print("Removing %s %r" % (msg, relpath))
        os.remove(path)


if __name__ == "__main__":
    remove_files('.')
