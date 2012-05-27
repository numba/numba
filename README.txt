Extensible type objects for Python
==================================

Often Python extensions needs to communicate things on the ABI level
about PyObjects. In essence, one would like more slots in PyTypeObject
for a custom purpose (dictionary lookups would be too slow).
This tiny library implements a metaclass that can be used for this.

SEP 200 contains more background and API documentation:

https://github.com/numfocus/sep/blob/master/sep200.rst

Contents:

 * ``include/extensibletype.h`` should be included by any library
   that wants to implement SEP 200

 * ``demo/`` contains (crufty) proof-of-concept and benchmarking code
