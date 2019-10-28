#!/usr/bin/env python

import pytest
import os

# ensure full tracebacks are available and no help messages appear in test mode
os.environ['NUMBA_DEVELOPER_MODE'] = '1'

def _work(slice_common=None, nprocs=None):
    # There's a big note at the bottom of:
    # https://docs.pytest.org/en/latest/usage.html
    # about calling `pytest.main()` multiple times from the same process as
    # imported modules are cached, seems like this would be safe as the test
    # runner is being run as a batch job

    common_args = ['-rs', "--durations=0", "--strict", 'numba/tests']

    if nprocs is not None:
        common_args = ['-n', nprocs] + common_args

    # run the "gitdiff" tests first
    stat = pytest.main(common_args + ['--runtype=gitdiff'],)
    if stat:
        msg = "Failing early. Testing subset target %s" % "gitdiff"
        raise RuntimeError(msg, stat)

    # then the common ones
    common_extra = ['--runtype=common']
    if slice_common is not None:
        common_extra.append('--slice=%s' % slice_common)
    pytest.main(common_args + common_extra,)

    # then the specific ones
    pytest.main(common_args + ['--runtype=specific'],)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice-common', help='the slice for the common tests')
    parser.add_argument('-m', help='the number of processes to use')
    args = parser.parse_args()
    _work(args.slice_common, args.m)
