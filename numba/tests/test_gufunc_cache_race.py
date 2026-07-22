"""
Test for race condition in GUFunc caching (GitHub issue #10128).

The race condition occurs when two processes compile the same @guvectorize
function simultaneously. With a global itertools.count(), different processes
can assign different ABI version tags to the same function depending on
compilation order (i.e. which other functions were compiled first). This can
cause a mismatch between the main .nbc cache file and the guf-*.nbc wrapper
cache file, leading to segfaults.

The fix (using defaultdict(int) keyed by func_qualname) ensures each
function gets a deterministic ABI version regardless of compilation order.

This test reproduces the race condition deterministically using two concurrent
processes with file-based barriers injected at
``IndexDataCacheFile._save_data`` (the same breakpoint location used in the
original bug report).

Timeline that triggers the bug (without the fix):

1. Process A starts, compiles fast_gufunc (uid=1) and caches it.
2. Process A reaches _save_data for slow_gufunc's function cache (uid=2
   because it already compiled fast_gufunc, giving it abi:v2).
3. Process B starts, finds fast_gufunc cached, compiles slow_gufunc from
   scratch (uid=1 since it's the first function B compiles -> abi:v1).
4. Process A saves slow_gufunc.nbc (abi:v2) and guf-slow_gufunc.nbc (abi:v2).
5. Process B saves slow_gufunc.nbc (abi:v1), overwriting A's version.
6. Process B finds guf-slow_gufunc.nbc already cached (from step 4) and
   loads it -> the wrapper references abi:v2 names but the function data has
   abi:v1 -> ``get_pointer_to_function`` returns null -> SIGSEGV.

With the fix, both processes assign uid=1 to slow_gufunc (independent per-
function counter), so the ABI tags always match regardless of write order.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from numba.tests.support import TestCase


# Module with two gufuncs. fast_gufunc compiles quickly, slow_gufunc is the
# one that triggers the race condition.
GUFUNC_MODULE = textwrap.dedent("""\
    import numpy as np
    from numba import guvectorize, types

    @guvectorize([(types.float64[:], types.float64[:])],
                 '(n)->(n)', cache=True)
    def fast_gufunc(x, result):
        for i in range(x.shape[0]):
            result[i] = x[i] * 2.0

    @guvectorize([(types.float64[:], types.float64[:])],
                 '(n)->(n)', cache=True)
    def slow_gufunc(x, result):
        for i in range(x.shape[0]):
            result[i] = x[i] + 1.0
""")


# Worker script that monkeypatches IndexDataCacheFile._save_data with
# file-based barrier synchronization, then imports the gufunc module to
# trigger compilation and caching.
WORKER_SCRIPT = textwrap.dedent("""\
    import os
    import sys
    import time

    worker_id = sys.argv[1]
    barrier_dir = sys.argv[2]
    module_dir = sys.argv[3]

    # Monkeypatch _save_data to inject barriers
    import numba.core.caching as _caching

    _original_save_data = _caching.IndexDataCacheFile._save_data
    _save_count = [0]

    def _barrier_save_data(self, name, data):
        _save_count[0] += 1
        count = _save_count[0]

        # Signal: "I'm about to save, here's the filename"
        sig_path = os.path.join(barrier_dir, f'{worker_id}_ready_{count}')
        with open(sig_path, 'w') as f:
            f.write(name)

        # Wait for go signal from orchestrator
        go_path = os.path.join(barrier_dir, f'{worker_id}_go_{count}')
        deadline = time.monotonic() + 120
        while not os.path.exists(go_path):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Worker {worker_id} timed out at save #{count} "
                    f"for {name}"
                )
            time.sleep(0.01)

        return _original_save_data(self, name, data)

    _caching.IndexDataCacheFile._save_data = _barrier_save_data

    # Import the gufunc module to trigger compilation
    sys.path.insert(0, module_dir)
    import numpy as np
    from gufunc_module import fast_gufunc, slow_gufunc

    # Actually call the gufuncs to trigger compilation
    x = np.array([1.0, 2.0, 3.0])
    fast_gufunc(x)
    slow_gufunc(x)

    # Signal done
    done_path = os.path.join(barrier_dir, f'{worker_id}_done')
    with open(done_path, 'w') as f:
        f.write(f'saves={_save_count[0]}')
""")


# Verification script: loads from cache and calls the gufunc.
# Without the fix, this segfaults due to ABI mismatch.
VERIFY_SCRIPT = textwrap.dedent("""\
    import sys
    import numpy as np

    sys.path.insert(0, sys.argv[1])
    from gufunc_module import slow_gufunc

    x = np.array([1.0, 2.0, 3.0])
    result = slow_gufunc(x)
    expected = x + 1.0
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("OK")
""")


class TestGUFuncCacheRace(TestCase):
    """Reproduce the gufunc caching race condition from GitHub issue #10128.

    Uses two concurrent processes with barrier synchronization at
    ``_save_data`` to create the exact interleaving that corrupts the cache.
    """

    def _wait_for(self, path, timeout=120, proc=None):
        """Wait for a barrier signal file to appear.

        If *proc* is given, also monitor the subprocess: if it exits before
        the file appears, fail immediately with its stderr instead of waiting
        for the full timeout (this catches segfaults quickly).
        """
        deadline = time.monotonic() + timeout
        while not os.path.exists(path):
            if time.monotonic() > deadline:
                self.fail(f"Timeout waiting for {path}")
            if proc is not None and proc.poll() is not None:
                stderr = proc.stderr.read().decode()
                self.fail(
                    f"Process exited with code {proc.returncode} while "
                    f"waiting for {os.path.basename(path)}:\n{stderr}"
                )
            time.sleep(0.01)

    def _signal(self, path):
        """Create a go signal file."""
        with open(path, 'w') as f:
            f.write('go')

    def test_concurrent_gufunc_caching_no_segfault(self):
        """Two processes compile the same gufuncs; verify no ABI mismatch."""
        tmpdir = tempfile.mkdtemp(prefix='numba_cache_race_test_')
        barrier_dir = os.path.join(tmpdir, 'barriers')
        module_dir = os.path.join(tmpdir, 'module')
        os.makedirs(barrier_dir)
        os.makedirs(module_dir)

        proc_a = None
        proc_b = None

        try:
            # Write test files
            with open(os.path.join(module_dir, 'gufunc_module.py'), 'w') as f:
                f.write(GUFUNC_MODULE)
            worker_path = os.path.join(tmpdir, 'worker.py')
            with open(worker_path, 'w') as f:
                f.write(WORKER_SCRIPT)
            verify_path = os.path.join(tmpdir, 'verify.py')
            with open(verify_path, 'w') as f:
                f.write(VERIFY_SCRIPT)

            python = sys.executable

            # -- Start Worker A --
            # A compiles both gufuncs from scratch.
            # Expected _save_data calls:
            #   A_ready_1: fast_gufunc function .nbc
            #   A_ready_2: fast_gufunc wrapper  guf-.nbc
            #   A_ready_3: slow_gufunc function .nbc
            #   A_ready_4: slow_gufunc wrapper  guf-.nbc
            proc_a = subprocess.Popen(
                [python, worker_path, 'A', barrier_dir, module_dir],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

            # Let A save fast_gufunc completely (function + wrapper)
            self._wait_for(os.path.join(barrier_dir, 'A_ready_1'),
                           proc=proc_a)
            self._signal(os.path.join(barrier_dir, 'A_go_1'))

            self._wait_for(os.path.join(barrier_dir, 'A_ready_2'),
                           proc=proc_a)
            self._signal(os.path.join(barrier_dir, 'A_go_2'))

            # A is now ready to save slow_gufunc function -- hold it here
            self._wait_for(os.path.join(barrier_dir, 'A_ready_3'),
                           proc=proc_a)

            # -- Start Worker B --
            # B will find fast_gufunc cached (A just wrote it) but
            # slow_gufunc NOT cached (A hasn't saved it yet).
            # B compiles slow_gufunc from scratch.
            # With old code: B assigns uid=1 (first function B compiles)
            #   while A assigned uid=2 -> ABI mismatch.
            # With fix: both assign uid=1 for slow_gufunc -> match.
            proc_b = subprocess.Popen(
                [python, worker_path, 'B', barrier_dir, module_dir],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

            # Wait for B to finish compiling and be ready to save
            # slow_gufunc function data.
            self._wait_for(os.path.join(barrier_dir, 'B_ready_1'),
                           proc=proc_b)

            # Now let A save slow_gufunc function AND wrapper.
            # A writes both .nbc files before B gets to write anything.
            self._signal(os.path.join(barrier_dir, 'A_go_3'))
            self._wait_for(os.path.join(barrier_dir, 'A_ready_4'),
                           proc=proc_a)
            self._signal(os.path.join(barrier_dir, 'A_go_4'))

            # Wait for A to finish completely
            self._wait_for(os.path.join(barrier_dir, 'A_done'),
                           proc=proc_a)

            # Now let B save slow_gufunc function data.
            # This OVERWRITES A's slow_gufunc.nbc with B's version.
            # B will then check the guf-wrapper cache, find A's version
            # (written in A_go_4), and load it without saving.
            self._signal(os.path.join(barrier_dir, 'B_go_1'))

            # B may or may not have a second _save_data call depending on
            # whether it finds the wrapper cached. In the race condition
            # scenario, B finds A's wrapper and skips saving.
            # Monitor proc_b so we detect a crash (segfault) immediately
            # rather than waiting for the full timeout.
            self._wait_for(os.path.join(barrier_dir, 'B_done'), timeout=120,
                           proc=proc_b)

            # Check worker exit codes
            a_exit = proc_a.wait(timeout=30)
            b_exit = proc_b.wait(timeout=30)

            a_stderr = proc_a.stderr.read().decode()
            b_stderr = proc_b.stderr.read().decode()

            self.assertEqual(
                a_exit, 0,
                f"Worker A failed (exit={a_exit}):\n{a_stderr}",
            )
            self.assertEqual(
                b_exit, 0,
                f"Worker B failed (exit={b_exit}):\n{b_stderr}",
            )

            # -- Verify --
            # Load from the cache that was written by both workers.
            # Without the fix, this segfaults because slow_gufunc.nbc has
            # abi:v1 names but guf-slow_gufunc.nbc references abi:v2 names.
            result = subprocess.run(
                [python, verify_path, module_dir],
                capture_output=True, text=True, timeout=60,
            )

            self.assertEqual(
                result.returncode, 0,
                f"Verification failed (segfault from ABI mismatch?):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}",
            )
            self.assertIn("OK", result.stdout)

        finally:
            # Clean up subprocesses
            for proc in (proc_a, proc_b):
                if proc is not None:
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
