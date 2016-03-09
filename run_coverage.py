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
    # We must start coverage before importing the package under test,
    # otherwise some lines will be missed.
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        '.coveragerc')
    os.environ['COVERAGE_PROCESS_START'] = config_file
    cov = coverage.coverage(config_file=config_file)
    cov.start()

    from numba import runtests

    html_dir = 'htmlcov'
    try:
        # NOTE: we force single-process mode, since some hacks are needed
        # for multiprocess'ed coverage to work.
        runtests.main(*sys.argv[1:])
    except SystemExit:
        pass
    finally:
        cov.stop()
        cov.save()
        cov.combine()
    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
    cov.html_report(directory=html_dir)
