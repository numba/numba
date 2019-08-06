.. _jit-module:

============================================
Automatic module jitting with ``jit_module``
============================================

Numba provides a :func:`numba.jit` decorator for compiling functions to machine code. A common usage pattern is to have an entire user-defined module containing functions that all need to be jitted. One option to accomplish this is to manually apply the ``@jit`` decorator to each function definition. This works and is great in many cases. However, for large modules with many functions, manually ``jit``-wrapping each function definition can be tedious. As such, Numba provides another option, the ``jit_module`` function, to automatically replace all functions defined in a module with their ``jit``-wrapped equivalents. Note that if a function has already been decorated with ``jit``, then ``jit_module`` will have no impact on the function. 

.. note:: This feature is for use by module authors. ``jit_module`` should not
    be called outside the context of a module containing functions to be jitted.


Example usage
=============

Let's assume we have a Python module we've created, ``mymodule.py`` (shown below), which contains several functions. Some of these functions are defined in other modules and some are defined in ``mymodule.py``. We wish to have all the functions which are defined in ``mymodule.py`` jitted using ``jit_module``.

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
   
   @jit(nogil=True)  # Overrides the module level options specified below
   def mul(a, b):
      return a * b
   
   jit_module(__name__, nopython=True, error_model="numpy")

There are several things to note here:

- Both the ``inc`` and ``add`` functions will be replaced with their ``jit``-wrapped equivalents with :ref:`compilation options <jit-options>` ``nopython=True`` and ``error_model="numpy"``.

- The ``mean`` function, because it defined *outside* of ``mymodule.py`` in NumPy, will not be modified.

- The ``mul`` function will not be modified because it has been manually decorated with ``jit``, which has priority over the module-level ``jit`` options specified in the ``jit_module`` call. 

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

