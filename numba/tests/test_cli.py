# Tests for the CLI

import os
import subprocess
import sys
import threading

import unittest
from numba.tests.support import TestCase


def run_cmd(cmdline, env=os.environ, timeout=60):
    popen = subprocess.Popen(cmdline,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             env=env)

    timeout_timer = threading.Timer(timeout, popen.kill)
    try:
        timeout_timer.start()
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError(
                "process failed with code %s: stderr follows\n%s\n" %
                (popen.returncode, err.decode()))
        return out.decode(), err.decode()
    finally:
        timeout_timer.cancel()
    return None, None


class TestCLi(TestCase):

    def test_as_module_exit_code(self):
        cmdline = [sys.executable, "-m", "numba"]
        with self.assertRaises(AssertionError) as raises:
            run_cmd(cmdline)

        self.assertIn("process failed with code 1", str(raises.exception))

    def test_as_module(self):
        cmdline = [sys.executable, "-m", "numba", "-s"]
        o, _ = run_cmd(cmdline)
        self.assertIn("System info", o)


if __name__ == '__main__':
    unittest.main()
