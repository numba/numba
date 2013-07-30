from __future__ import print_function, division, absolute_import
import tempfile
import textwrap

from numba import jit, autojit

# Thanks to @bfredl

def test_fetch_latest_source():
    """
    When reloading new versions of the same module into the same session (i.e.
    an interactive ipython session), numba sometimes gets the wrong version of
    the source from inspect.getsource()
    """
    f = tempfile.NamedTemporaryFile()
    fn = f.name
    # fn = "/tmp/numbatest.py"

    f.write(textwrap.dedent("""
    @jit('i8()')
    def test():
        return 0
    """))
    f.flush()
    exec open(fn)

    f.write(textwrap.dedent("""
    @jit('i8()')
    def test():
        return 1
    """))
    f.flush()
    exec open(fn)

    assert test() == test.py_func() # gives 0 == 1

def test_no_auto_reload():
    """
    In this case autojit 'sees' the new version of the source even if it
    hasn't been reloaded. This could be fixed by fetching the ast directly
    at declaration time rather that at first compilation (2nd commit)
    """
    f = tempfile.NamedTemporaryFile()
    fn = f.name

    f.write(textwrap.dedent("""
    @autojit
    def test2():
        return 0
    """))
    f.flush()
    exec open(fn)

    f.write(textwrap.dedent("""
    @autojit
    def test2():
        return 1
    """))
    f.flush()

    # note that we don't reexec the file

    assert test2() == test2.py_func() # gives 1 == 0

test_fetch_latest_source()
test_no_auto_reload()