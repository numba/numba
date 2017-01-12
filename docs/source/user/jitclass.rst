.. _jitclass:

=======================================
Compiling python classes with @jitclass
=======================================

.. note::

  This is a early version of jitclass support. Not all compiling features are
  exposed or implemented, yet.


Numba supports code generation for classes via the :func:`numba.jitclass`
decorator.  A class can be marked for optimization using this decorator along
with a specification of the types of each field.  We call the resulting class
object a jitclass.  All methods of a jitclass is compiled into nopython
functions.  The data of a jitclass instance is allocated on the heap as a
C-compatible structure so that any compiled functions can have direct access
to the underlying data, bypassing the interpreter.


Basic usage
===========

Here's an example of a jitclass::

  import numpy as np
  from numba import jitclass          # import the decorator
  from numba import int32, float32    # import the types

  spec = [
      ('value', int32),               # a simple scalar field
      ('array', float32[:]),          # an array field
  ]

  @jitclass(spec)
  class Bag(object):
      def __init__(self, value):
          self.value = value
          self.array = np.zeros(value, dtype=np.float32)

      @property
      def size(self):
          return self.array.size

      def increment(self, val):
          for i in range(self.size):
              self.array[i] = val
          return self.array


(see full example at `examples/jitclass.py` from the source tree)

In the above example, a ``spec`` is provided as a list of 2-tuples.  The tuples
contain the name of the field and the numba type of the field.  Alternatively,
user can use a dictionary (an ``OrderedDict`` preferrably for stable field
ordering), which maps field names to types.

The definition of the class requires at least a ``__init__`` method for
initializing each defined fields.  Uninitialized fields contains garbage data.
Methods and properties (getters and setters only) can be defined.  They will be
automatically compiled.


Support operations
==================

The following operations of jitclasses work in both the interpreter and numba
compiled functions:

* calling the jitclass class object to construct a new instance
  (e.g. ``mybag = Bag(123)``);
* read/write access to attributes and properties (e.g. ``mybag.value``);
* calling methods (e.g. ``mybag.increment(3)``);

Using jitclasses in numba compiled function is more efficient.
Short methods can be inlined (at the discretion of LLVM inliner).
Attributes access are simply reading from a C structure.
Using jitclasses from the intpreter has the same overhead of calling any
numba compiled function from the interpreter.  Arguments and return values
must be unboxed or boxed between python objects and native representation.
Values encapsulated by a jitclass does not get boxed into python object when
the jitclass instance is handed to the interpreter.  It is during attribute
access to the field values that they are boxed.


Limitations
===========

* A jitclass class object is treated as a function (the constructor) inside
  a numba compiled function.
* ``isinstance()`` only works in the interpreter.
* Manipulating jitclass instances in the interpreter is not optimized, yet.
* Support for jitclasses are available on CPU only.
  (Note: Support for GPU devices is planned for a future release.)


The decorator: ``@jitclass``
============================

.. autofunction:: numba.jitclass
