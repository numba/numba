Generalized Ufuncs
==================

The `GUFuncVectorize` module of NumbaPro creates a fast "generalized ufunc" from Numba-compiled code.
Traditional ufuncs perform element-wise operations, whereas generalized ufuncs operate on entire
sub-arrays. Unlike other NumbaPro Vectorize classes, the GUFuncVectorize constructor takes
an additional signature of the generalized ufunc.


Imports
-------

::

	from numbapro import float32, float64, int32
	from numbapro import guvectorize
	import numpy as np

Basic generalized ufuncs are available in Numba. In addition to the basic generalized
ufuncs, NumbaPro provides a CUDA-accelerated generalized ufunc.

.. autofunction:: numbapro.guvectorize

To see how generalized ufuncs work, we refer the reader to
`Numba gufuncs <http://numba.pydata.org/numba-doc/dev/arrays.html#generalized-ufunc-definition>`_.

Generalized CUDA ufuncs
-----------------------
Generalized ufuncs may also be executed on the GPU using CUDA, analogous to the CUDA ufunc functionality.
Jump to the `documentation for CUDA ufunc <CUDAufunc.html>`_ for continued discussion on generalized ufuncs.

