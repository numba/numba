from __future__ import print_function, division, absolute_import
import os
import contextlib

from numba import jit, autojit
from numba.tests.issues import issue_305_helper1, issue_305_helper2

try:
    from imp import reload
except ImportError:
    pass

# Thanks to @bfredl

env = { '__file__': __file__, '__name__': __name__ ,
        'jit': jit, 'autojit': autojit }

root = os.path.dirname(os.path.abspath(__file__))
fn1 = os.path.join(root, 'issue_305_helper1.py')
fn2 = os.path.join(root, 'issue_305_helper2.py')

new_source1 = """
from numba import jit, autojit

@jit('i8()')
def test():
    return 1
"""

new_source2 = """
from numba import jit, autojit

@autojit
def test2():
    return 1
"""

@contextlib.contextmanager
def newsource(fn, mod, reexec=True):
    old_source = open(fn).read()
    try:
        open(fn, 'w').write(new_source1)
        if reexec: reload(mod)
        yield
    finally:
        open(fn, 'w').write(old_source)


def test_fetch_latest_source():
    """
    When reloading new versions of the same module into the same session (i.e.
    an interactive ipython session), numba sometimes gets the wrong version of
    the source from inspect.getsource()
    """
    with newsource(fn1, issue_305_helper1):
        assert issue_305_helper1.test() == issue_305_helper1.test.py_func()

def test_no_auto_reload():
    """
    In this case autojit 'sees' the new version of the source even if it
    hasn't been reloaded. This could be fixed by fetching the ast directly
    at declaration time rather that at first compilation (2nd commit)
    """
    with newsource(fn2, issue_305_helper2, reexec=False):
        print(issue_305_helper2.test2(), issue_305_helper2.test2.py_func())
        assert issue_305_helper2.test2() == issue_305_helper2.test2.py_func()

test_fetch_latest_source()
test_no_auto_reload()