.. _jit-module:

============================================
Automatic module jitting with ``jit_module``
============================================

A common usage pattern is to have an entire module containing user-defined
functions that all need to be jitted. One option to accomplish this is to
manually apply the ``@jit`` decorator to each function definition. This approach
works and is great in many cases. However, for large modules with many functions,
manually ``jit``-wrapping each function definition can be tedious. For these
situations, Numba provides another option, the ``jit_module`` function, to
automatically replace functions declared in a module with their ``jit``-wrapped
equivalents.

It's important to note the conditions under which ``jit_module`` will *not*
impact a function:

1. Functions which have already been wrapped with a Numba decorator (e.g.
   ``jit``, ``vectorize``, ``cfunc``, etc.) are not impacted by ``jit_module``.

2. Functions which are declared outside the module from which ``jit_module``
   is called are not automatically ``jit``-wrapped.

3. Function declarations which occur logically after calling ``jit_module``
   are not impacted.

All other functions in a module will have the ``@jit`` decorator automatically
applied to them. See the following section for an example use case.

.. note:: This feature is for use by module authors. ``jit_module`` should not
    be called outside the context of a module containing functions to be jitted.


Example usage
=============

Let's assume we have a Python module we've created, ``mymodule.py`` (shown
below), which contains several functions. Some of these functions are defined
in ``mymodule.py`` while others are imported from other modules. We wish to have
all the functions which are defined in ``mymodule.py`` jitted using
``jit_module``.

.. _jit-module-usage:

.. code-block:: python

   # mymodule.py

   from numba import jit, jit_module

   def inc(x):
      return x + 1
   
   def add(x, y):
      return x + y
   
   import numpy as np
   # Use NumPy's mean function
   mean = np.mean
   
   @jit(nogil=True)
   def mul(a, b):
      return a * b
   
   jit_module(nopython=True, error_model="numpy")

   def div(a, b):
       return a / b

There are several things to note in the above example:

- Both the ``inc`` and ``add`` functions will be replaced with their
  ``jit``-wrapped equivalents with :ref:`compilation options <jit-options>`
  ``nopython=True`` and ``error_model="numpy"``.

- The ``mean`` function, because it's defined *outside* of ``mymodule.py`` in
  NumPy, will not be modified.

- ``mul`` will not be modified because it has been manually decorated with
  ``jit``.

- ``div`` will not be automatically ``jit``-wrapped because it is declared
  after ``jit_module`` is called.

When the above module is imported, we have:

.. code-block:: python

   >>> import mymodule
   >>> mymodule.inc
   CPUDispatcher(<function inc at 0x1032f86a8>)
   >>> mymodule.mean
   <function mean at 0x1096b8950>


API
===
.. warning:: This feature is experimental. The supported features may change
    with or without notice.

.. autofunction:: numba.jit_module

