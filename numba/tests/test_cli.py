# Tests for the CLI

import os
import subprocess
import sys
import threading
import json
from tempfile import TemporaryDirectory

import unittest
from numba.tests.support import TestCase
import numba.misc.numba_sysinfo as nsi


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

    def test_json_sysinfo(self):
        with TemporaryDirectory() as d:
            path = os.path.join(d, "test_json_sysinfo.json")
            cmdline = [sys.executable, "-m", "numba", "--sys-json", path]
            run_cmd(cmdline)
            with self.subTest(msg=f"{path} exists"):
                self.assertTrue(os.path.exists(path))
            with self.subTest(msg="json load"):
                with open(path, 'r') as f:
                    info = json.load(f)
            safe_contents = {
                int: (
                    nsi._cpu_count,
                ),
                float: (
                    nsi._runtime,
                ),
                str: (
                    nsi._start,
                    nsi._start_utc,
                    nsi._machine,
                    nsi._cpu_name,
                    nsi._platform_name,
                    nsi._os_name,
                    nsi._os_version,
                    nsi._python_comp,
                    nsi._python_impl,
                    nsi._python_version,
                    nsi._llvm_version,
                ),
                bool: (
                    nsi._cu_dev_init,
                    nsi._svml_state,
                    nsi._svml_loaded,
                    nsi._svml_operational,
                    nsi._llvm_svml_patched,
                    nsi._tbb_thread,
                    nsi._openmp_thread,
                    nsi._wkq_thread,
                ),
                list: (
                    nsi._errors,
                    nsi._warnings,
                ),
                dict: (
                    nsi._numba_env_vars,
                ),
            }
            for t, keys in safe_contents.items():
                for k in keys:
                    with self.subTest(k=k):
                        self.assertIsInstance(info[k], t)


if __name__ == '__main__':
    unittest.main()
