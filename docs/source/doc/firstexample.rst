First Example
==============

A Simple Function
-----------------

Suppose we want to write an image-processing function in Python.  Here's how it might look.

.. code-block:: python 

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

This kind of quadruply-nested for-loop is going to be quite slow.  Using Numba we can compile this code to LLVM which then gets compiled to machine code: 

.. code-block:: python

   from numba import double
   from numba.decorators import jit

   fastfilter_2d = jit(arg_types = [double[:,:], double[:,:]], ret_type=double[:,:])(filter2d)

   # Now fastfilter_2d runs at speeds as if you had translated it to C
   res = fastfilter_2d(image, filt)


