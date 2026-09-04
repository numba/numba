"""Regression tests for the gufunc wrapper cache key (#10128).

A ``@guvectorize`` function is cached as two independent artifacts: the kernel
(``.nbc``) and the generated NumPy wrapper (``guf-*.nbc``).  The wrapper's
object code calls the kernel by its mangled symbol name, which carries a
per-compilation ABI tag (``v<uid>``).

Historically the wrapper's cache key did not mention that symbol, so a wrapper
built against one kernel symbol was a valid cache hit for a kernel defining a
different one.  Concurrent processes could write such a pair; loading it left
the wrapper calling an undefined symbol, which resolved to NULL and segfaulted
on the first call.
"""

import glob
import os
import pickle
import subprocess
import sys
import textwrap
import unittest

import numpy as np

import numba
from numba import guvectorize
from numba.np.ufunc.wrappers import GufWrapperCache
from numba.tests.support import TestCase, SerialMixin, temp_directory, \
    override_config


def _plain_kernel(x):
    # Module-level so that it has a readable source file, as the cache
    # locators require.
    return x + 1


class _StubCodegen(object):
    def magic_tuple(self):
        return ('stub-target',)


class TestGufWrapperCacheKey(TestCase):
    """The kernel symbol must participate in the wrapper's index key."""

    def _key_for(self, symbol):
        cache = GufWrapperCache(_plain_kernel, symbol)
        return cache._index_key('float64(float64)', _StubCodegen())

    def test_same_symbol_same_key(self):
        self.assertEqual(self._key_for('_ZN1fB2v1E'),
                         self._key_for('_ZN1fB2v1E'))

    def test_different_symbol_different_key(self):
        # The regression: these two differ *only* in the ABI tag. Before the
        # fix they produced equal keys, so either wrapper was a hit for
        # either kernel.
        self.assertNotEqual(self._key_for('_ZN1fB2v1E'),
                            self._key_for('_ZN1fB2v2E'))


class TestGufWrapperCacheOnDisk(SerialMixin, TestCase):
    """The symbol reaches the on-disk index, not just the in-memory key."""

    def setUp(self):
        self.cache_dir = temp_directory('test_guf_wrapper_cache_key')

    @staticmethod
    def _read_index(path):
        with open(path, 'rb') as f:
            pickle.load(f)          # version prefix
            _stamp, overloads = pickle.loads(f.read())
        return overloads

    def test_wrapper_index_keys_carry_kernel_symbol(self):
        with override_config('CACHE_DIR', self.cache_dir):
            @guvectorize(['void(float64[:], float64[:])'], '(n)->(n)',
                         cache=True, nopython=True)
            def double_it(x, out):
                for i in range(x.shape[0]):
                    out[i] = x[i] * 2

            arr = np.arange(4, dtype=np.float64)
            np.testing.assert_allclose(double_it(arr), arr * 2)

        indices = glob.glob(os.path.join(self.cache_dir, '**', 'guf-*.nbi'),
                            recursive=True)
        self.assertTrue(indices, 'no guf-*.nbi index was written')

        for path in indices:
            overloads = self._read_index(path)
            self.assertTrue(overloads, 'empty wrapper index in %s' % path)
            for key in overloads:
                symbol = key[-1]
                self.assertIsInstance(symbol, str)
                # An Itanium-mangled kernel name carrying an ABI tag.
                self.assertTrue(symbol.startswith('_Z'),
                                'wrapper key does not end in a mangled '
                                'kernel symbol: %r' % (key,))


class TestRecompiledKernel(SerialMixin, TestCase):
    """A kernel recompiled under a new ABI tag must not reuse the old wrapper.

    The tag comes from a process-global counter, so a process that compiled
    something else first labels the same kernel differently.  Two such
    processes racing on one cache directory can leave a kernel written under
    one tag beside a wrapper built against another; the wrapper then calls a
    symbol nothing defines.
    """

    # No closure: the ABI tag still varies with what the process compiled
    # first, because it comes from a process-global counter.
    PLAIN_SRC = """
        from numba import guvectorize

        @guvectorize(['void(float64[:], float64[:])'], '(n)->(n)',
                     cache=True, nopython=True)
        def gf(x, out):
            for i in range(x.shape[0]):
                out[i] = x[i] + 5.0
    """

    DRIVER = """
        import numpy as np
        import uidmod

        arr = np.arange(4, dtype=np.float64)
        np.testing.assert_allclose(uidmod.gf(arr), arr + 5.0)
        cres, = uidmod.gf.gufunc_builder.nb_func.overloads.values()
        print('MANGLED:' + cres.fndesc.mangled_name)
    """

    def setUp(self):
        self.tempdir = temp_directory('test_guf_unstable_identity')
        self.cache_dir = os.path.join(self.tempdir, 'cache')
        self.mod_dir = os.path.join(self.tempdir, 'mod')
        os.makedirs(self.cache_dir)
        os.makedirs(self.mod_dir)

    def _write(self, src):
        """Write the use case once.

        The module must not be rewritten between runs: numba stamps its cache
        index with the source file's mtime and size, so a rewrite invalidates
        every entry and no run could ever reuse another's artifacts.
        """
        with open(os.path.join(self.mod_dir, 'uidmod.py'), 'w') as f:
            f.write(textwrap.dedent(src))

    def _run(self, seed, history=''):
        """Import the use case in a fresh process; return its kernel symbol."""
        env = os.environ.copy()
        # Pin the numba under test: the subprocess must not pick up a
        # different installation.
        parent = os.path.dirname(os.path.dirname(os.path.abspath(
            numba.__file__)))
        env['PYTHONPATH'] = os.pathsep.join(
            [self.mod_dir, parent, env.get('PYTHONPATH', '')])
        env['PYTHONHASHSEED'] = str(seed)
        env['NUMBA_CACHE_DIR'] = self.cache_dir
        code = textwrap.dedent(history) + textwrap.dedent(self.DRIVER)
        proc = subprocess.run([sys.executable, '-c', code], env=env,
                              capture_output=True, timeout=300)
        self.assertEqual(proc.returncode, 0,
                         'subprocess failed (rc=%s):\n%s'
                         % (proc.returncode, proc.stderr.decode()))
        out = proc.stdout.decode()
        line, = [ln for ln in out.splitlines()
                 if ln.startswith('MANGLED:')]
        return line[len('MANGLED:'):]

    def _wrapper_symbols_on_disk(self):
        symbols = set()
        pattern = os.path.join(self.cache_dir, '**', 'guf-*.nbi')
        for path in glob.glob(pattern, recursive=True):
            for key in TestGufWrapperCacheOnDisk._read_index(path):
                symbols.add(key[-1])
        return symbols

    def test_recompiled_kernel_does_not_reuse_stale_wrapper(self):
        """A deterministic stand-in for the #10128 race.

        The ABI tag comes from a process-global counter, so a process that
        compiled something else first labels the same kernel differently.
        Concurrently, two such processes write a kernel under one tag and a
        wrapper built against another.  Here that interleaving is produced
        without threads: run once, drop just the *kernel* index so the next
        run must recompile, and run again with the counter shifted.

        The second run must not pick up the first run's wrapper, which calls
        a symbol the newly compiled kernel does not define.
        """
        self._write(self.PLAIN_SRC)
        first = self._run(seed=1)

        # Drop the kernel index only, leaving the wrapper cache populated.
        removed = 0
        for path in glob.glob(os.path.join(self.cache_dir, '**', '*.nbi'),
                              recursive=True):
            if not os.path.basename(path).startswith('guf-'):
                os.unlink(path)
                removed += 1
        self.assertTrue(removed, 'no kernel index was written')

        # Shift the ABI tag counter, then force a kernel recompile.
        second = self._run(
            seed=1,
            history='from numba import njit\nnjit(lambda v: v + 1)(1.0)\n',
        )

        self.assertNotEqual(first, second,
                            'the two runs agreed on an ABI tag, so this test '
                            'no longer exercises a mismatched pair')
        self.assertEqual(self._wrapper_symbols_on_disk(), {first, second})


if __name__ == '__main__':
    unittest.main()
