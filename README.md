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

  * LLVM 3.2 or 3.3
  * llvmpy (from llvmpy/llvmpy fork)
  * numpy (version 1.6 or higher)
  * argparse (for pycc in python2.6)

Installing
=================

The easiest way to install numba and get updates is by using the Anaconda
Distribution: https://store.continuum.io/cshop/anaconda/

```bash
    $ conda install numba
```

If you wanted to compile Numba from source,
it is recommended to use conda environment to maintain multiple isolated
development environments.  To create a new environment for Numba development:

```bash
    $ conda create -p ~/dev/mynumba python numpy llvmpy
```

To select the installed version, append "=VERSION" to the package name,
where, "VERSION" is the version number.  For example:

```bash
    $ conda create -p ~/dev/mynumba python=2.7 numpy=1.6 llvmpy
```

to use Python 2.7 and Numpy 1.6.


Custom Python Environments
==========================

If you're not using anaconda, you will need LLVM with RTTI enabled:

* Compile LLVM 3.2

See https://github.com/llvmpy/llvmpy for the most up-to-date instructions.

```bash
    $ wget http://llvm.org/releases/3.2/llvm-3.2.src.tar.gz
    $ tar zxvf llvm-3.2.src.tar.gz
    $ ./configure --enable-optimized --prefix=LLVM_BUILD_DIR
    $ # It is recommended to separate the custom build from the default system
    $ # package.
    $ # Be sure your compiler architecture is same as version of Python you will use
    $ #  e.g. -arch i386 or -arch x86_64.  It might be best to be explicit about this.
    $ REQUIRES_RTTI=1 make install
```

* Install llvmpy

```bash
    $ git clone https://github.com/llvmpy/llvmpy
    $ cd llvmpy
    $ LLVM_CONFIG_PATH=LLVM_BUILD_DIR/bin/llvm-config python setup.py install
```

* Installing Numba

```bash
    $ git clone https://github.com/numba/numba.git
    $ cd numba
    $ pip install -r requirements.txt
    $ python setup.py build_ext --inplace
    $ python setup.py install
```

or simply

```bash
    $ pip install numba
```

**NOTE:** Make sure you install *distribute* instead of setuptools. Using setuptools
          may mean that source files do not get cythonized and may result in an
          error during installation.

Documentation
=============

http://numba.pydata.org/numba-doc/dev/index.html

Mailing Lists
=============

Join the numba mailing list numba-users@continuum.io :

https://groups.google.com/a/continuum.io/d/forum/numba-users

Some old archives are at: http://librelist.com/browser/numba/

Website
=======

See if our sponsor can help you (which can help this project): http://www.continuum.io

http://numba.pydata.org

Continuous Integration
======================

https://travis-ci.org/numba/numba
