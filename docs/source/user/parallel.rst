.. _parallel:

=========================================
Automatic parallelization option to @jit.
=========================================

.. seealso:: :ref:`parallel_jit_option`

See the above link for how to enable this experimental Numba feature that
attempts to automatically parallelize and perform other optimizations on
a function.  

Some operations in a function, e.g., Numpy element-wise addition on an array, 
are known to have parallel semantics.  Some functions contain many such 
operations and while each operation could be parallelized individually, such
an approach often has lackluster performance due to poor cache behavior.  
Instead, with this experimental feature, Numba identifies such operations, 
fuses them together into one pass over the data, eliminates unnecessary
intermediate arrays, and parallelizes their execution using the 
:func:`~numba.guvectorize` mechanism.

Say something about speedups here?

Supported Operations
====================

In this section, we give a list of all the operations that have parallel
semantics and for which we attempt to parallelize.

Say something about multi-dimensional array support.

Say something about non-Numpy operations supported.

Say something about which Numpy operations aren't supported.

Say something about reductions.

Something about new features coming soon?

Example
=======

In this section, we give an example of how this feature applies to the
Black-Scholes option pricing::

	import numba
	import numpy as np
	import math

	@numba.vectorize(nopython=True)
	def cndf2(inp):
		out = 0.5 + 0.5 * math.erf((math.sqrt(2.0)/2.0) * inp)
		return out

	@numba.jit(nopython=True,parallel=True)
	def blackscholes(sptprice, strike, rate, volatility, timev):
		logterm = np.log(sptprice / strike)
		powterm = 0.5 * volatility * volatility
		den = volatility * np.sqrt(timev)
		d1 = (((rate + powterm) * timev) + logterm) / den
		d2 = d1 - den
		NofXd1 = cndf2(d1)
		NofXd2 = cndf2(d2)
		futureValue = strike * np.exp(- rate * timev)
		c1 = futureValue * NofXd2
		call = sptprice * NofXd1 - c1
		put  = call - futureValue + sptprice
		return put

There are several things to note here:

* Say something about the required declaration for cndf2.

* Say how all the operations are Numpy operations that fuse down to a single loop.

