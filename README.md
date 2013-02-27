Numba
=====

Numba is an Open Source NumPy-aware optimizing compiler for Python 
sponsored by Continuum Analytics, Inc.  It uses the
remarkable LLVM compiler infrastructure to compile Python syntax to
machine code.

It is aware of NumPy arrays as typed memory regions and so can speed-up
code using NumPy arrays.  Other, less well-typed code will be translated 
to Python C-API calls effectively removing the "interpreter" but not removing
the dynamic indirection.  

Numba is also not a tracing jit.  It *compiles* your code before it gets
run either using run-time type information or type information you provide
in the decorator.

Numba is a mechanism for producing machine code from Python syntax and typed
data structures such as those that exist in NumPy. 

Dependencies
============

Dependencies:

  * LLVM 3.1 or 3.2
  * llvmpy (from llvmpy/llvmpy fork)
  * numpy (version 1.6 or higher)
  * Meta (from numba/Meta fork (optional))
  * Cython (build dependency only)
  * nose (for unit tests)
  * argparse (for pycc)

Installing
=================

The easiest way to install numba and get updates is by using the Anaconda
Distribution: http://continuum.io/anacondace.html

Custom Python Environments
==========================

If you're not using anaconda, you will need LLVM with RTTI enabled:

* Compile LLVM 3.2

```bash
    $ wget http://llvm.org/releases/3.2/llvm-3.2.src.tar.gz 
    $ tar zxvf llvm-3.2.src.tar.gz
    $ ./configure --enable-optimized
    $ # Be sure your compiler architecture is same as version of Python you will use
    $ #  e.g. -arch i386 or -arch x86_64.  It might be best to be explicit about this.
    $ make install
```

* Installing Numba

```bash
    $ git clone https://github.com/numba/numba.git
    $ cd numba
    $ pip install -r requirements.txt
    $ python setup.py install
```

or simply

```bash
    $ pip install numba
```

Documentation
=============

http://numba.pydata.org/numba-doc/dev/index.html

Mailing Lists
=============

* Follow Numba
Join the numba mailinglist numba-users@continuum.io
https://groups.google.com/a/continuum.io/d/forum/numba-users

Some old archives are at http://librelist.com/browser/numba/

* See if our sponsor can help you (which can help this project)
http://www.continuum.io

Website
=======

http://numba.pydata.org
