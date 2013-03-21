# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
import subprocess

from os.path import join, dirname

from numba import PY3

# doctest compatible for jit or autojit numba functions
from numba.testing.test_support import testmod

EXCLUDE_TEST_PACKAGES = []

def exclude_package_dirs(dirs):
    for exclude_pkg in EXCLUDE_TEST_PACKAGES:
        if exclude_pkg in dirs:
            dirs.remove(exclude_pkg)


def qualified_test_name(root):
    qname = root.replace("/", ".").replace("\\", ".").replace(os.sep, ".") + "."
    offset = qname.rindex('numba.')
    return qname[offset:]

def whitelist_match(whitelist, modname):
    if whitelist:
        return any(item in modname for item in whitelist)
    return True

def map_returncode_to_message(retcode):
    if retcode < 0:
        retcode = -retcode
        return signal_to_name.get(retcode, "Signal %d" % retcode)

    return ""

def find_testdirs():
    import numba

    numba_pkg = os.path.dirname(os.path.abspath(numba.__file__))
    for root, dirs, files in os.walk(numba_pkg):
        for dir in dirs:
            if dir in ('tests',):
                yield os.path.join(root, dir)
                dirs.remove(dir)
                break

try:
    import signal
except ImportError:
    signal_to_name = {}
else:
    signal_to_name = dict((signal_code, signal_name)
                           for signal_name, signal_code in vars(signal).items()
                               if signal_name.startswith("SIG"))

def test(whitelist=None, blacklist=None, print_failures_only=False):
    # FIXME
    # temporarily disable pycc test on win32
    if sys.platform.startswith('win32'):
        blacklist = ['test_pycc_tresult']

    runner = TestRunner(whitelist, blacklist, print_failures_only)
    testdirs = find_testdirs()

    for testdir in testdirs:
        runner.run(testdir)

    sys.stdout.write("ran test files: failed: (%d/%d)\n" % (runner.failed,
                                                            runner.ran))


class TestRunner(object):

    def __init__(self, whitelist, blacklist, print_failures_only):
        self.ran = 0
        self.failed = 0
        self.whitelist = whitelist
        self.blacklist = blacklist
        self.print_failures_only = print_failures_only

    def run(self, testdir):
        for root, dirs, files in os.walk(testdir):
            qname = qualified_test_name(root)
            exclude_package_dirs(dirs)

            for fn in files:
                if fn.startswith('test_') and fn.endswith('.py'):
                    modname, ext = os.path.splitext(fn)
                    modname = qname + modname

                    if not whitelist_match(self.whitelist, modname):
                        continue
                    if self.blacklist and whitelist_match(self.blacklist,
                                                          modname):
                        continue

                    self.ran += 1
                    if not self.print_failures_only:
                        sys.stdout.write("running %-61s" % (modname,))

                    process = subprocess.Popen([sys.executable, '-m', modname],
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
                    out, err = process.communicate()

                    if process.returncode == 0:
                        if not self.print_failures_only:
                            sys.stdout.write("SUCCESS\n")
                    else:
                        if self.print_failures_only:
                            sys.stdout.write("running %-61s" % (modname,))

                        sys.stdout.write("FAILED: %s\n" % map_returncode_to_message(
                                                        process.returncode))
                        if PY3:
                            out = str(out, encoding='UTF-8')
                            err = str(err, encoding='UTF-8')
                        sys.stdout.write(out)
                        sys.stdout.write(err)
                        sys.stdout.write("-" * 80)
                        sys.stdout.write('\n')
                        self.failed += 1


def nose_run(module=None):
    import nose.config
    import __main__

    #os.environ["NOSE_EXCLUDE"] = "(test_all|test_all_noskip|.*compile_with_pycc.*|bytecode)"
    #os.environ["NOSE_VERBOSE"] = "4"

    result = nose.main()
    return len(result.errors), len(result.failures)
