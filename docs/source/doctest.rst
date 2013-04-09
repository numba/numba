
.. _doctest:

***************************************
Using doctest with Numba
***************************************

Numba allows numba functions decorated with ``jit`` or ``autojit`` to
participate in doctest modules. Instead of using the ``doctest`` module,
one has to use ``numba.testmod`` instead::

    import numba

    @numba.autojit
    def func(value):
        """
        >>> func(10.0)
        10.0
        """
        return value

    numba.testmod()

.. function:: numba.testmod(module=None, run=True, optionflags=0, verbosity=2)

    :param module: python module to process for doctest functions
    :param run: whether to run the doctests or only build the ``module.__test__`` dict
    :param optionflags: doctest options (e.g. ``doctest.ELLIPSIS``)
    :param verbosity: verbosity level passed to unittest.TextTestRunner

