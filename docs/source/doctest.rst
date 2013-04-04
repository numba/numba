
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

