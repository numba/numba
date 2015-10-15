
============================
Compiling code ahead of time
============================

.. _pycc:

While Numba's main use case is :term:`Just-in-Time compilation`, it also
provides a facility for :term:`Ahead-of-Time compilation` (AOT).


Overview
========

Benefits
--------

#. AOT compilation produces a compiled extension module which does not depend
   on Numba: you can distribute the module on machines which do not have
   Numba installed (but Numpy is required).

#. There is no compilation overhead at runtime (but see the
   :ref:`cache <jit-cache>` option), nor any overhead of importing
   Numba.

.. seealso::
   Compiled extension modules are discussed in the
   `Python packaging user guide <https://packaging.python.org/en/latest/extensions/>`_.


Limitations
-----------

#. AOT compilation only allows for regular functions, not :term:`ufuncs <ufunc>`.

#. You have to specify function signatures explicitly.

#. Each exported function can have only one signature (but you can export
   several different signatures under different names).

#. AOT compilation produces generic code for your CPU's architectural family
   (for example "x86-64"), while JIT compilation produces code tailored
   to your particular CPU model.


Usage
=====

Standalone example
------------------

::

   from numba.pycc import CC

   cc = CC('my_module')
   # Uncomment the following line to print out the compilation steps
   #cc.debug = True

   @cc.export('multf', 'f8(f8, f8)')
   @cc.export('multi', 'i4(i4, i4)')
   def mult(a, b):
       return a * b

   @cc.export('square', 'f8(f8)')
   def square(a):
       return a ** 2

   if __name__ == "__main__":
       cc.compile()


If you run this Python script, it will generate an extension module named
``my_module``.  Depending on your platform, the actual filename may be
``my_module.so``, ``my_module.pyd``, ``my_module.cpython-34m.so``, etc.

The generated module has three functions: ``multf``, ``multi`` and ``square``.
``multi`` operates on 32-bit integers (``i4``), while ``multf`` and ``square``
operate on double-precision floats (``f8``)::

   >>> import my_module
   >>> dir(my_module)
   ['__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'multf', 'multi', 'square']
   >>> my_module.__file__
   '/home/antoine/workdir/my_module.cpython-34m.so'
   >>> my_module.multi(3, 4)
   12
   >>> my_module.square(1.414)
   1.9993959999999997


Distutils integration
---------------------

You can also integrate the compilation step for your extension modules
in your ``setup.py`` script, using distutils or setuptools::

   from distutils.core import setup

   from source_module import cc

   setup(...,
         ext_modules=[cc.distutils_extension()])


The ``source_module`` above is the module defining the ``cc`` object.
Extensions compiled like this will be automatically included in the
build files for your Python project, so you can distribute them inside
binary packages such as wheels or Conda packages.


Signature syntax
----------------

The syntax for exported signatures is the same as in the ``@jit``
decorator.  You can read more about it in the :func:`numba.jit` reference.

