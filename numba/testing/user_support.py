# -*- coding: utf-8 -*-

"""
Doctest support exposed to numba users.
"""

from __future__ import print_function, division, absolute_import

import sys
import doctest

from numba.testing import doctest_support

doctest_options = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

def testmod(module=None, run=True, optionflags=0, verbosity=2):
    """
    Tests a doctest modules with numba functions. When run in nosetests, only
    populates module.__test__, when run as main, runs the doctests.

    module: the module to run the doctests in
    run: whether to run the doctests or just build a __test__ dict
    verbosity: verbosity level passed to unittest.TextTestRunner

        The defualt is 2

    optionflags: doctest options (e.g. doctest.ELLIPSIS)
    """
    if module is None:
        mod_globals = sys._getframe(1).f_globals
        modname = mod_globals['__name__']
        module = __import__(modname)

    doctest_support.testmod(
        module,
        run_doctests=run,
        optionflags=optionflags,
        verbosity=verbosity,
    )
