"""runtest_multiproc.py

This is used to run tests in parallel processes using the multiprocessing module
and be better tolerant to tests that can timeout or segfaulting. This scripts
gather all the tests and schedule each TestCase class to run in a fresh process.
Logs from each process are stored under "testlogs" and categorized into "failed"
or "passed". This scripts is useful for bootstrapping Numba on a new Python
version or other high impact changes.

Usage:
    runtest_multiproc.py --run [<args>...]
    runtest_multiproc.py --count

Options:
    args: arguments passed to runtests --list
    --run: Run tests in parallel processes.
    --count: Print failed tests stats by processing log files.

Example:

    ```
    runtest_multiproc.py --run -- numba.tests --random=0.3
    ```

    To run 30% of tests chosen randomly from `numba.tests`.
    Use `--` to separate listing arguments from command arguments.

Requires:

    `numba` package to be import-able either by installing `numba` or running
    this script from the the source root.
"""
import re
import sys
from pathlib import Path
from pprint import pprint
import subprocess as subp
from docopt import docopt
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


PATH_LOG = Path("testlogs")
PATH_FAILED = PATH_LOG / "failed"
PATH_PASSED = PATH_LOG / "passed"


def run_tests(test_args):
    PATH_FAILED.mkdir(parents=True, exist_ok=True)
    PATH_PASSED.mkdir(parents=True, exist_ok=True)

    listing = subp.check_output(
        "python -m numba.runtests --list".split() + [*test_args],
        encoding="utf8",
    )
    tests = []
    for line in listing.splitlines():
        if line.startswith("numba."):
            tests.append(line)
        else:
            print(line)

    groups = defaultdict(list)
    for t in tests:
        comps = t.split(".")
        k = ".".join(comps[:4])
        groups[k].append(t)

    def runner(testgroup):
        cmdargs = " ".join(groups[testgroup])
        # Using -m=1 so tests are cancelled if they run too long due to timeout.
        cmd = f"python -m numba.runtests -m=1 -vb {cmdargs}"
        print("RUNNING", testgroup)
        try:
            output = subp.check_output(
                cmd.split(), stderr=subp.STDOUT, stdin=subp.DEVNULL
            )
        except subp.CalledProcessError as e:
            output = e.output
            path = PATH_FAILED
        else:
            path = PATH_PASSED

        with open(path / f"{testgroup}.log", "wb") as fout:
            fout.write(output)

        return testgroup

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as pool:
        try:
            futures = [pool.submit(runner, testgroup) for testgroup in groups]
            for future in as_completed(futures):
                print("DONE", future.result())
        except KeyboardInterrupt:
            pool.shutdown(wait=True, cancel_futures=True)
    print("END")


REGEX_ERROR = re.compile(r"(errors|failures|unexpectedSuccess)=(\d+)")


def count_tests():
    failed = {}
    timedout = {}

    problems = []

    for file in Path("testlogs/failed").glob("*.log"):
        with open(file, "r") as fin:
            body = fin.read().strip()
        print(file)
        lines = body.splitlines()
        cur_timedout = []
        for ln in reversed(lines):
            if ln.startswith("- "):
                cur_timedout.append(ln.strip("- '\""))
            else:
                break
        if cur_timedout:
            timedout[file] = cur_timedout
        else:
            last_line = lines[-1]
            if last_line.startswith("Parallel: "):
                last_line = lines[-2]
            if not last_line.startswith("FAILED"):
                problems.append(f"ERROR: {file} is malformed")
            failed[file] = last_line

    print("FAILED".center(80, "-"))
    pprint(failed)
    print("TIMEDOUT".center(80, "-"))
    pprint(timedout)

    stats = {"timedout": sum(len(v) for v in timedout.values())}
    for _fp, msg in failed.items():
        _gather_failed_stats(msg, stats)

    pprint(stats)

    for p in problems:
        print(p)


def _gather_failed_stats(line, stats):
    for fail_type, count in REGEX_ERROR.findall(line):
        orig = stats.setdefault(fail_type, 0)
        stats[fail_type] = orig + int(count)


if __name__ == "__main__":
    arguments = docopt(__doc__, version="1.0")
    action_run = arguments["--run"]
    test_args = arguments["<args>"]
    action_count = arguments["--count"]

    if action_count:
        count_tests()
    elif action_run:
        # Strip '--'
        if test_args and test_args[0] == "--":
            test_args = test_args[1:]
        run_tests(test_args)
    else:
        print("Please specify either --run or --count")
        sys.exit(1)
