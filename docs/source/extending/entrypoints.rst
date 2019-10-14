Registering Extensions with Entry Points
========================================

Often, third party packages will have a user-facing API as well as define
extensions to the Numba compiler.  In those situations, the new types and
overloads can registered with Numba when the package is imported by the user.
However, there are situations where a Numba extension would not normally be
imported directly by the user, but must still be registered with the Numba
compiler.  An example of this is the `numba-scipy
<https://github.com/numba/numba-scipy>`_ package, which adds support for some
SciPy functions to Numba.  The end user does not need to ``import
numba_scipy`` to enable compiler support for SciPy, the extension only needs
to be installed in the Python environment.

Numba discovers extensions using the `entry points
<https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins>`_
feature of ``setuptools``.  This allows a Python package to register an
initializer function that will be called before ``numba`` compiles for the
first time.  The delay ensures that the cost of importing extensions is
deferred until it is necessary.


Adding Support for the "Init" Entry Point
-----------------------------------------

A package can register an initialization function with Numba by adding the
``entry_points`` argument to the ``setup()`` function call in ``setup.py``:

.. code-block:: python

    setup(
        ...,
        entry_points={
            "numba_extensions": [
                "init = numba_scipy:_init_extension",
            ],
        },
        ...
    )

Numba currently only looks for the ``init`` entry point in the
``numba_extensions`` group.  The entry point should be a function (any name,
as long as it matches what is listed in ``setup.py``) that takes no arguments,
and the return value is ignored.  This function should register types,
overloads, or call other Numba extension APIs.  The order of initialization of
extensions is undefined.
