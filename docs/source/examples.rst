Examples
==============
.. _examples:

A Simple Function
-----------------

Suppose we want to write an image-processing function in Python.  Here's how it might look.

.. code-block:: python 

   import numpy
   
   def filter2d(image, filt):
       M, N = image.shape
       Mf, Nf = filt.shape
       Mf2 = Mf // 2
       Nf2 = Nf // 2
       result = numpy.zeros_like(image)
       for i in range(Mf2, M - Mf2):
           for j in range(Nf2, N - Nf2):
               num = 0.0
               for ii in range(Mf):
                   for jj in range(Nf):
                       num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii, j-Nf2+jj])
               result[i, j] = num
       return result
   
   # This kind of quadruply-nested for-loop is going to be quite slow.
   # Using Numba we can compile this code to LLVM which then gets 
   # compiled to machine code: 
   
   from numba import double, jit
   
   fastfilter_2d = jit(double[:,:](double[:,:], double[:,:]))(filter2d)
   
   # Now fastfilter_2d runs at speeds as if you had first translated
   # it to C, compiled the code and wrapped it with Python
   image = numpy.random.random((100, 100))
   filt = numpy.random.random((10, 10))
   res = fastfilter_2d(image, filt)

Numba actually produces two functions.   The first function is the
low-level compiled version of filter2d.  The second function is the
Python wrapper to that low-level function so that the function can be
called from Python.   The first function can be called from other
numba functions to eliminate all python overhead in function calling. 

.. _classes:

Classes
-------
.. literalinclude:: /../../examples/numbaclasses.py

Closures
--------
.. literalinclude:: /../../examples/closure.py

.. _structexample:

Structs
-------
.. literalinclude:: /../../examples/structures.py


.. _pointerexample:

Pointers
--------
.. literalinclude:: /../../examples/pointers.py

Objects
-------
.. literalinclude:: /../../examples/objects.py

Mandelbrot
----------
.. literalinclude:: /../../examples/mandel.py


Filterbank Correlation
----------------------
.. literalinclude:: /../../examples/fbcorr.py

Multi threading
---------------
.. literalinclude:: /../../examples/multithread.py

Strings and libc
----------------

.. literalinclude:: /../../examples/strings.py