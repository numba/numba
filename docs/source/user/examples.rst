========
Examples
========


Mandelbrot
----------

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_mandelbrot`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_mandelbrot.begin
   :end-before: magictoken.ex_mandelbrot.end
   :dedent: 12
   :linenos:

.. _example-movemean:

Moving average
--------------

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_moving_average`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_moving_average.begin
   :end-before: magictoken.ex_moving_average.end
   :dedent: 12
   :linenos:

Multi-threading
---------------

The code below showcases the potential performance improvement when
using the :ref:`nogil <jit-nogil>` feature.  For example, on a 4-core machine,
the following results were printed::

   numpy (1 thread)       145 ms
   numba (1 thread)       128 ms
   numba (4 threads)       35 ms

.. note::
   If preferred it's possible to use the standard `concurrent.futures
   <https://docs.python.org/3/library/concurrent.futures.html>`_ module
   rather than spawn threads and dispatch tasks by hand.

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_no_gil`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_no_gil.begin
   :end-before: magictoken.ex_no_gil.end
   :dedent: 12
   :linenos:
