"""Concurrency tests for the gufunc cache (GitHub issue #10128).

A ``@guvectorize`` function is cached as two independent artifacts: the kernel
(``.nbc``) and the generated NumPy wrapper (``guf-*.nbc``).  The wrapper's
object code calls the kernel by its mangled symbol name, which carries a
per-compilation ABI tag (``v<uid>``).  That tag depends on what the compiling
process compiled first, so two processes can label the same function
differently.

Because the wrapper's cache key records the kernel symbol it was built
against, a wrapper is only ever retrieved for a kernel that defines that
symbol; a mismatch is a cache miss and a recompile.  Without that, the two
artifacts could be paired across processes, leaving the wrapper calling a
symbol nothing defines -- a null function pointer handed to NumPy, and a
SIGSEGV on the first call.

These tests drive the interleaving that produces such a pair, using file
barriers at ``IndexDataCacheFile._save_data`` (the breakpoint from the
original bug report).  They are deliberately end-to-end: the failure they
guard against is a crash in a fresh process that only reads the cache.

See also ``test_gufunc_wrapper_cache_key``, which covers the same invariant
without concurrency.
"""

import os
import subprocess
import sys
import time
import unittest

import numba
from numba.tests.support import TestCase, SerialMixin, temp_directory


# The workers are run as script files, so sys.path[0] is the script's own
# directory rather than the working directory: without this they would import
# whichever numba happens to be installed instead of the one under test.
_NUMBA_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(
    numba.__file__)))


def _subprocess_env():
    env = os.environ.copy()
    env['PYTHONPATH'] = _NUMBA_PARENT + os.pathsep + env.get('PYTHONPATH', '')
    return env


# Two independent gufuncs.  ``first`` is compiled before ``second`` so that a
# process compiling both assigns them consecutive ABI tags, while a process
# that finds ``first`` already cached assigns ``second`` the lower tag.
TWO_GUFUNC_MODULE = """\
import numpy as np
from numba import guvectorize, types

@guvectorize([(types.float64[:], types.float64[:])], '(n)->(n)', cache=True)
def first(x, result):
    for i in range(x.shape[0]):
        result[i] = x[i] * 2.0

@guvectorize([(types.float64[:], types.float64[:])], '(n)->(n)', cache=True)
def second(x, result):
    for i in range(x.shape[0]):
        result[i] = x[i] + 1.0
"""


# One gufunc with two signatures.  The ABI tag is assigned during compilation,
# so a cache *hit* on one signature does not advance the counter: a process
# that finds float32 cached labels float64 differently from a process that
# compiled both.  Both signatures share one kernel cache file and one wrapper
# cache file.
MULTISIG_GUFUNC_MODULE = """\
import numpy as np
from numba import guvectorize, types

@guvectorize([(types.float32[:], types.float32[:]),
              (types.float64[:], types.float64[:])],
             '(n)->(n)', cache=True)
def gu(x, result):
    for i in range(x.shape[0]):
        result[i] = x[i] + 1.0
"""


# Imports the use case with a file barrier around every cache write, so the
# orchestrator can interleave two processes' writes precisely.  The calls to
# make are passed in, keeping one worker for both scenarios.
WORKER_SCRIPT = '''\
import os
import sys
import time

worker_id, barrier_dir, module_dir, calls = sys.argv[1:5]

import numba.core.caching as _caching

_original_save_data = _caching.IndexDataCacheFile._save_data
_save_count = [0]


def _barrier_save_data(self, name, data):
    _save_count[0] += 1
    count = _save_count[0]

    # Announce the pending write, then block until released.
    ready = os.path.join(barrier_dir, '%s_ready_%d' % (worker_id, count))
    with open(ready, 'w') as f:
        f.write(name)

    go = os.path.join(barrier_dir, '%s_go_%d' % (worker_id, count))
    deadline = time.monotonic() + 120
    while not os.path.exists(go):
        if time.monotonic() > deadline:
            raise TimeoutError('worker %s stuck at save #%d for %s'
                               % (worker_id, count, name))
        time.sleep(0.01)

    return _original_save_data(self, name, data)


_caching.IndexDataCacheFile._save_data = _barrier_save_data

sys.path.insert(0, module_dir)
import numpy as np                                       # noqa: E402
import gufunc_module as m                                # noqa: E402

exec(calls)

with open(os.path.join(barrier_dir, worker_id + '_done'), 'w') as f:
    f.write(str(_save_count[0]))
'''


# Reads the cache the workers left behind, in a process that compiles nothing.
# This is where a mismatched kernel/wrapper pair crashes.
VERIFY_SCRIPT = '''\
import sys
import numpy as np

sys.path.insert(0, sys.argv[1])
import gufunc_module as m

exec(sys.argv[2])
print("OK")
'''


class _BarrierRaceTest(SerialMixin, TestCase):
    """Machinery for driving two cache writers through a fixed interleaving."""

    # Spawning several compiling subprocesses each is too heavy to also run
    # them against each other under the parallel test runner.
    _numba_parallel_test_ = False

    def setUp(self):
        self.tmpdir = temp_directory('test_gufunc_cache_race')
        self.barrier_dir = os.path.join(self.tmpdir, 'barriers')
        self.module_dir = os.path.join(self.tmpdir, 'module')
        self.log_dir = os.path.join(self.tmpdir, 'logs')
        for d in (self.barrier_dir, self.module_dir, self.log_dir):
            os.makedirs(d)
        self.worker_path = os.path.join(self.tmpdir, 'worker.py')
        with open(self.worker_path, 'w') as f:
            f.write(WORKER_SCRIPT)
        self.verify_path = os.path.join(self.tmpdir, 'verify.py')
        with open(self.verify_path, 'w') as f:
            f.write(VERIFY_SCRIPT)
        self._procs = []

    def tearDown(self):
        for proc, out, err in self._procs:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # Close before the temp tree is removed: on Windows an open
            # handle inside it makes the cleanup fail.
            out.close()
            err.close()

    def write_module(self, source):
        with open(os.path.join(self.module_dir, 'gufunc_module.py'), 'w') as f:
            f.write(source)

    def spawn(self, worker_id, calls):
        """Start a barrier worker.

        Output goes to files rather than pipes: a worker is held at a barrier
        for as long as the orchestration needs, and an undrained pipe would
        deadlock as soon as it emitted more than the buffer size.
        """
        out_path = os.path.join(self.log_dir, worker_id + '.out')
        err_path = os.path.join(self.log_dir, worker_id + '.err')
        out = open(out_path, 'wb')
        err = open(err_path, 'wb')
        proc = subprocess.Popen(
            [sys.executable, self.worker_path, worker_id, self.barrier_dir,
             self.module_dir, calls],
            stdout=out, stderr=err, env=_subprocess_env(),
        )
        self._procs.append((proc, out, err))
        return proc

    @staticmethod
    def _read(path):
        try:
            with open(path, 'rb') as f:
                return f.read().decode(errors='replace')
        except OSError:
            return ''

    def output_of(self, worker_id):
        return (self._read(os.path.join(self.log_dir, worker_id + '.out')),
                self._read(os.path.join(self.log_dir, worker_id + '.err')))

    def signal_path(self, worker_id, signal):
        return os.path.join(self.barrier_dir, '%s_%s' % (worker_id, signal))

    def fail_if_dead(self, worker_id, proc, doing):
        """Fail immediately if *proc* has exited, rather than waiting out a
        timeout. Signals written just before exit still count."""
        if proc.poll() is None:
            return
        if os.path.exists(self.signal_path(worker_id, 'done')):
            return
        _, err = self.output_of(worker_id)
        self.fail('worker %s exited with %s while %s\n%s'
                  % (worker_id, proc.returncode, doing, err))

    def wait_for(self, worker_id, signal, proc=None, timeout=120):
        """Block until a worker's barrier signal file appears."""
        path = self.signal_path(worker_id, signal)
        deadline = time.monotonic() + timeout
        while not os.path.exists(path):
            if proc is not None:
                self.fail_if_dead(worker_id, proc,
                                  'waiting for ' + signal)
            if time.monotonic() > deadline:
                self.fail('timed out waiting for %s_%s' % (worker_id, signal))
            time.sleep(0.01)

    def drain(self, worker_id, proc, next_index, timeout=120):
        """Release the worker's remaining writes until it signals done.

        How many writes are left is not fixed: whether the worker still has a
        wrapper of its own to save depends on what the other worker already
        wrote.  So this waits on the only reliable terminator -- the ``done``
        signal -- releasing each further write as it is announced, rather
        than guessing a count.  A worker that dies is reported at once
        instead of being mistaken for one that simply had nothing left to
        write.
        """
        done = self.signal_path(worker_id, 'done')
        deadline = time.monotonic() + timeout
        while not os.path.exists(done):
            self.fail_if_dead(worker_id, proc, 'draining its remaining writes')
            ready = self.signal_path(worker_id, 'ready_%d' % next_index)
            if os.path.exists(ready):
                self.release(worker_id, 'go_%d' % next_index)
                next_index += 1
            if time.monotonic() > deadline:
                self.fail('timed out draining worker %s' % worker_id)
            time.sleep(0.01)

    def release(self, worker_id, signal):
        with open(self.signal_path(worker_id, signal), 'w') as f:
            f.write('go')

    def assert_worker_ok(self, proc, worker_id):
        code = proc.wait(timeout=60)
        out, err = self.output_of(worker_id)
        self.assertEqual(code, 0, 'worker %s failed (exit=%s)\nstdout: %s\n'
                                  'stderr: %s' % (worker_id, code, out, err))

    def assert_cache_usable(self, calls):
        """A fresh reader of the cache must not crash and must be correct."""
        proc = subprocess.run(
            [sys.executable, self.verify_path, self.module_dir, calls],
            capture_output=True, text=True, timeout=300,
            env=_subprocess_env(),
        )
        self.assertEqual(
            proc.returncode, 0,
            'reading the cache failed with exit=%s (a negative code is a '
            'crash, i.e. the ABI mismatch this test guards against)\n'
            'stdout: %s\nstderr: %s'
            % (proc.returncode, proc.stdout, proc.stderr),
        )
        self.assertIn('OK', proc.stdout)


class TestGUFuncCacheRace(_BarrierRaceTest):
    """Two gufuncs, two processes, different compilation histories."""

    CALLS = ('m.first(np.ones(3));'
             ' m.second(np.ones(3))')

    VERIFY = ('x = np.arange(3, dtype=np.float64);'
              ' np.testing.assert_allclose(m.second(x), x + 1.0)')

    def test_concurrent_gufunc_caching_no_segfault(self):
        """Interleave so ``second``'s kernel and wrapper come from different
        processes.

        A compiles both gufuncs, so it labels ``second`` with the second tag.
        B starts once ``first`` is cached, so ``second`` is the only thing it
        compiles and it uses the first tag.  A is held until B has compiled,
        then allowed to write both of its ``second`` artifacts; B's kernel
        write then lands on top of A's.

        The wrapper B needs must not be satisfied by the wrapper A wrote.
        """
        self.write_module(TWO_GUFUNC_MODULE)

        # A: saves are (1) first kernel, (2) first wrapper,
        #               (3) second kernel, (4) second wrapper.
        proc_a = self.spawn('A', self.CALLS)
        for i in (1, 2):
            self.wait_for('A', 'ready_%d' % i, proc=proc_a)
            self.release('A', 'go_%d' % i)

        # Hold A with ``second`` compiled but unwritten.
        self.wait_for('A', 'ready_3', proc=proc_a)

        # B now finds ``first`` cached and ``second`` not, so it compiles
        # only ``second`` -- and gives it a different tag than A did.
        proc_b = self.spawn('B', self.CALLS)
        self.wait_for('B', 'ready_1', proc=proc_b)

        # Let A write both of its ``second`` artifacts, then finish.
        self.release('A', 'go_3')
        self.wait_for('A', 'ready_4', proc=proc_a)
        self.release('A', 'go_4')
        self.wait_for('A', 'done', proc=proc_a)

        # B overwrites the kernel. Whether it also writes a wrapper is the
        # crux: it must not silently adopt A's.
        self.release('B', 'go_1')
        self.drain('B', proc_b, next_index=2)

        self.assert_worker_ok(proc_a, 'A')
        self.assert_worker_ok(proc_b, 'B')
        self.assert_cache_usable(self.VERIFY)


class TestGUFuncMultiSigCacheRace(_BarrierRaceTest):
    """One gufunc, two signatures, one partially warm cache.

    The tag is assigned during compilation, so a cache hit on float32 does
    not advance it.  A process that compiles both signatures labels float64
    with the second tag; a process that finds float32 cached labels float64
    with the first.  Both signatures share a single kernel cache file and a
    single wrapper cache file, so the two processes contend over the same
    entries.
    """

    CALLS = ('m.gu(np.ones(3, dtype=np.float32));'
             ' m.gu(np.ones(3, dtype=np.float64))')

    VERIFY = ('x = np.arange(3, dtype=np.float64);'
              ' np.testing.assert_allclose(m.gu(x), x + 1.0);'
              ' y = np.arange(3, dtype=np.float32);'
              ' np.testing.assert_allclose(m.gu(y), y + 1.0)')

    def test_multi_signature_partial_cache_no_segfault(self):
        self.write_module(MULTISIG_GUFUNC_MODULE)

        # Both signatures are compiled before either wrapper is built, so A's
        # writes are: (1) float32 kernel, (2) float64 kernel,
        #             (3) float32 wrapper, (4) float64 wrapper.
        proc_a = self.spawn('A', self.CALLS)
        self.wait_for('A', 'ready_1', proc=proc_a)
        self.release('A', 'go_1')

        # Hold A with float32 written and float64 not yet written.
        self.wait_for('A', 'ready_2', proc=proc_a)

        # B therefore finds float32 cached and float64 missing: float64 is
        # the only signature it compiles, and it labels it with the tag A
        # gave float32.
        proc_b = self.spawn('B', self.CALLS)
        self.wait_for('B', 'ready_1', proc=proc_b)

        # Let A write its float64 kernel and both wrappers.
        for i in (2, 3, 4):
            self.release('A', 'go_%d' % i)
            if i < 4:
                self.wait_for('A', 'ready_%d' % (i + 1), proc=proc_a)
        self.wait_for('A', 'done', proc=proc_a)

        self.release('B', 'go_1')
        self.drain('B', proc_b, next_index=2)

        self.assert_worker_ok(proc_a, 'A')
        self.assert_worker_ok(proc_b, 'B')
        self.assert_cache_usable(self.VERIFY)


if __name__ == '__main__':
    unittest.main()
