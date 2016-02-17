
Extending Numba
===============

.. module:: numba.extending

This chapter describes how to extend Numba to make it recognize and support
additional operations, functions or types.  Numba provides two categories
of APIs to this end:

* The high-level APIs provide abstracted entry points which are sufficient
  for simple uses.  They require little knowledge of Numba's internal
  compilation chain.

* The low-level APIs reflect Numba's internal compilation chain and allow
  flexible interaction with its various layers, but require more effort
  and experience with Numba internals.

It may be helpful for readers of this chapter to also read some of the
documents in the :doc:`developer manual </developer/index>`, especially
the :doc:`architecture document </developer/architecture>`.


.. toctree::

   high-level.rst
   low-level.rst
   interval-example.rst

