import os
import sys
import pytest
import numpy as np
import platform

arch_set = set(['armv7l', 'aarch64', 'ppc64le', 'x86', 'x86_64'])
os_set = set(['linux', 'win32', 'osx'])


def bit_check(mark):
    """ Checks 32/64 bit marker
    """
    x = int(mark.args[0])
    assert x in (32, 64), "only 32 and 64 bits supported"
    return x == 64 and sys.maxsize > 2 ** 32


def os_check(mark):
    """ Checks OS
    """
    for os in mark.args:
        assert os in os_set, "os unknown: %s" % os
        if sys.platform.startswith(os):
            return True
    return False


def arch_check(mark):
    """ Checks architecture
    """
    for arch in mark.args:
        assert arch in arch_set, "arch unknown: %s" % os
        if arch == platform.machine():
            return True
    return False


def supports_parfors():
    """ Checks if parfors is supported
    """
    _windows_py27 = (sys.platform.startswith('win32') and
                     sys.version_info[:2] == (2, 7))
    _32bit = sys.maxsize <= 2 ** 32
    return not (_32bit or _windows_py27)

def has_lapack():
    try:
        import scipy.linalg.cython_lapack
        return True
    except ImportError:
        return False

# converts numerical or string version X.Y.Z to tuple(X, Y, Z)
def ver2tuple(z): return tuple(map(int, str(z).split('.')))


def gen_version_check(this_version):
    def version_check(mark):
        minimum = mark.kwargs.get('min', None)
        maximum = mark.kwargs.get('max', None)
        ok = True
        if minimum is not None:
            ok &= this_version >= ver2tuple(minimum)
        if maximum is not None:
            ok &= this_version < ver2tuple(maximum)
        return ok
    return version_check


python_version_check = gen_version_check(sys.version_info[:3])
numpy_version_check = gen_version_check(ver2tuple(np.__version__))

support_dict = {
                # type
                'serial': False,

                # feature
                'parfors': supports_parfors(),
                'needs_lapack': has_lapack(),

                # system
                'only_os': os_check,
                'only_arch': arch_check,
                'not_OS': lambda x: not os_check(x),
                'not_arch': lambda x: not arch_check(x),
                'bits': bit_check,
                'python_version': python_version_check,
                'numpy_version': numpy_version_check,
                }


def check_support(mark):
    supported = support_dict.get(mark.name)
    if supported is None:
        msg = "Could not find support checking function for mark: %s" % mark
        raise ValueError(msg)
    if isinstance(supported, bool):
        return supported
    else:
        return supported(mark)


def pytest_addoption(parser):
    parser.addoption("--runtype",  action="store", default="all",
                    help="run tests common to all configurations")

    parser.addoption("--slice",  action="store", default="None",
                    help="run the given 'slice' through the selected tests")


def pytest_collection_modifyitems(session, config, items):

    ty = config.getoption("runtype")
    keep = []
    if ty == "all":
        keep.extend(items)
    elif ty == "specific":
        # any marker, skip or whatever is considered specific, the only
        # marker considered "special" is the use of 'serial' as a marker by
        # itself.
        for item in items:
            if item.own_markers:
                if not (len(item.own_markers) == 1 and
                        item.own_markers[0].name == "serial"):
                    keep.append(item)
    elif ty == "serial":
        # anything with 'serial' in the markers is considered serial
        for item in items:
            if item.own_markers:
                for mark in item.own_markers:
                    if mark.name == "serial":
                        keep.append(item)
    elif ty == "common":
        # anything unmarked is 'common'
        for item in items:
            if not item.own_markers:
                keep.append(item)
    elif ty == "gitdiff":
        try:
            from git import Repo
        except ImportError:
            raise ValueError("gitpython needed for git functionality")
        repo = Repo('.')
        branch_commits = [x for x in repo.iter_commits('origin/master..HEAD')]
        run = []
        for c in branch_commits:
            for fname, stat in c.stats.files.items():
                # make any change to a test and it runs the whole module!
                if 'numba/tests' in fname:
                    run.append(os.path.join(os.path.dirname(__file__),
                                            fname))
        for item in items:
            if item.module.__file__ in run:
                keep.append(item)
    else:
        raise ValueError("Unknown type specified in `--runtype`: %s" % ty)

    # clobber existing items
    items.clear()
    items.extend(keep)

    # use slicing if given
    sls = config.getoption("slice")
    default_slice = None
    l = {}
    if sls != "None":  # need to parse
        exec("sl = slice(%s)" % sls, l)
        default_slice = l['sl']
        new = items[default_slice]
        items.clear()
        items.extend(new)

    print("\n", "-" * 80)
    print("Test target '%s' has: %s active tests." % (ty, len(items)))
