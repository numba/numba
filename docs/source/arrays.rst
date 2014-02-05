******
Arrays
******

Support for NumPy arrays is a key focus of Numba development and is currently
undergoing extensive refactorization and improvement. Most capabilities of NumPy
arrays are supported by Numba in object mode, and a few features are supported in
nopython mode too (with much more to come).

A few noteworthy limitations of arrays at this time:

* Arrays can be passed in to a function in nopython mode, but not returned.
  Arrays can only be returned in object mode.
* New arrays can only be created in object mode.
* Currently there are no bounds checking for array indexing and slicing.
* NumPy array ufunc support in nopython node is incomplete at this time. Most
  if not all ufuncs should work in object mode though.
* Array slicing only works in object mode.

