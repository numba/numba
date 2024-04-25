import os
import sys
import pickle
import subprocess as subp
from textwrap import indent
from tempfile import NamedTemporaryFile

from numba.testing import run_tests
from numba.testing.main import _MinimalResult, ParallelTestResult, _FakeStringIO


ENV_SHARED_TEMPFILE = "_NUMBA_TESTING_SHARED_TEMPFILE"


def subproc_test_runner(testpath, env, *, show_timing, verbose, buffer):
    cmd = [
        sys.executable,
        "-m",
        "numba.testing.subproc_runner",  # this file
        "-m1",
        str(testpath),
    ]
    if show_timing:
        cmd.append("--show-timing")
    if verbose:
        cmd.append("-v")
    if buffer:
        cmd.append("-b")

    subenv = {**os.environ, **env}
    try:
        subenv["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_RCFILE"]
    except KeyError:
        pass  # ignored

    child_result = run_popen(cmd, secret=testpath.encode(), env=subenv)
    return child_result


def run_popen(cmd: list[str], *, secret: bytes, env: dict = {}):
    with NamedTemporaryFile(delete=True) as tmpfile:
        myenv = {**env, ENV_SHARED_TEMPFILE: tmpfile.name}
        try:
            subp.check_output(
                cmd, env=myenv, stderr=subp.STDOUT, encoding="utf-8"
            )
        except subp.CalledProcessError as e:
            print(f"STDOUT\n{indent(e.stdout, '  ')}\nEND", file=sys.__stderr__)
            raise
        # else:
        #     print(f"STDOUT\n{indent(out, '  ')}\nEND")

        child_result = pickle.load(tmpfile)
        return child_result


def main():
    os.environ["SUBPROC_TEST"] = "1"
    os.environ["_NUMBA_TEST_NO_PRINT_ERROR"] = "1"
    outfilename = os.environ[ENV_SHARED_TEMPFILE]
    with open(outfilename, mode='wb') as outfile:
        result = run_tests(argv=sys.argv)
        assert isinstance(result, ParallelTestResult)

        if result.showAll:
            result.stream = _FakeStringIO(
                indent(
                    result.stream.getvalue(),
                    f'{os.getpid()}] ',
                    lambda x: True,
                )
            )
        pickle.dump(_MinimalResult(result), file=outfile)


if __name__ == "__main__":
    main()
