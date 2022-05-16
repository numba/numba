#!/usr/bin/env python
"""
Run code coverage sampling over the test suite, and produce
an HTML report in "htmlcov".
"""

import os
import shutil
import sys

try:
    import coverage
except ImportError:
    raise ImportError("Please install the coverage module "
                      "(https://pypi.python.org/pypi/coverage/)")


if __name__ == "__main__":
    # We disallow single-process mode, since some hacks are needed for
    # multiprocess'ed coverage to work.
    for arg in sys.argv:
        if arg.startswith('-m') or arg.startswith('--multiprocess'):
            print("Coverage can only be run single-threaded; multiprocess "
                  "requested. Aborting.")
            sys.exit(1)

    # We must start coverage before importing the package under test,
    # otherwise some lines will be missed.
    config_file = os.path.join(
        os.path.dirname(__file__),
        'coverage.conf')
    os.environ['COVERAGE_PROCESS_START'] = config_file
    cov = coverage.coverage(config_file=config_file)
    cov.start()

    from numba.testing import _runtests

    html_dir = 'htmlcov'
    try:
        _runtests.main(*sys.argv)
    except SystemExit:
        pass
    finally:
        cov.stop()
        cov.save()
        cov.combine()
    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
    cov.html_report(directory=html_dir)
