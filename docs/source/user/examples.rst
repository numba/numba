========
Examples
========


Mandelbrot
----------

.. literalinclude:: /../../examples/mandel.py


.. _example-movemean:

Moving average
--------------

.. literalinclude:: /../../examples/movemean.py


Multi-threading
---------------

The code below showcases the potential performance improvement when
using the :ref:`nogil <jit-nogil>` feature.  For example, on a 4-core machine,
I get the following results printed out::

   numpy (1 thread)       145 ms
   numba (1 thread)       128 ms
   numba (4 threads)       35 ms

.. note::
   Under Python 3, you can use the standard `concurrent.futures
   <https://docs.python.org/3/library/concurrent.futures.html>`_ module
   rather than spawn threads and dispatch tasks by hand.

.. literalinclude:: /../../examples/nogil.py
