"""This module contains imports to activate experimental Numba features.

Analog to the __future__ module from Python, this module contains feature flags
in the form of imports which can be used to opt-in to experimental Numba
features that will become the default in a future version of Numba.

In order to use such a feature simply import it, for example::

    from numba.future import disable_reflected_list
    disable_reflected_list()  # call to make flake8/linter happy

As features mature and become the default the functions defined here will
become no-operations, i.e. they will simply cease to have an effect. To ensure
forward compatibility, the functions will never be removed.

"""


def disable_reflected_list():
    """Disable the Numba reflected list.

    This will disable support for the Numba reflected list in favour of the
    Numba typed list. This implies two things

    a) Any Python lists that are arguments to a Numba compiled function will
    automatically be converted to a immutable Numba typed lists. The advantage
    of this is that nested Python lists can now be used as arguments to Numba
    compiled functions. The disadvantage is that such lists can not longer be
    mutated. If you need to mutate the list within a Numba compiled function,
    please consider creating a Numba typed-list outside of the function and
    then passing that as an argument.

    For example::

        from numba.future import disable_reflected_list
        disable_reflected_list()

        z = [1, 2, 3]  # z is a Python list

        @njit
        def foo(lst):
            acc = 0
            for i in lst:  # lst is now an immutable typed list
                acc += i
            return acc

        foo(z)

    If you need to mutate the list within the compiled function, use a typed
    list to begin with::

        from numba.typed import List

        z = List()  # mutable typed list
        for i in (2, 3, 5, 7):
            z.append(i)

        @njit
        def foo(lst):
            lst.append(23)

        foo(z)  # this will append 23 to z

    b) Any built-ins to instantiate lists, such as ``list`` and ``[]`` will
    result in a Numba typed list when used within a Numba compiled function.

    For example::

        @njit
        def foo():
            a = []         # empty typed list
            b = list()     # empty typed list too
            c = [1, 2, 3]  # typed list initialised with three integers

    """
    pass
