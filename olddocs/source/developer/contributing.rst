=====================
Contributing to Numba
=====================

We welcome people who want to make contributions to Numba, big or small!
Even simple documentation improvements are encouraged.  If you have
questions, don't hesitate to ask them (see below).


Communication
=============

Mailing-list
------------

We have a public mailing-list that you can e-mail at numba-users@continuum.io.
If you have any questions about contributing to Numba, it is ok to ask them
on this mailing-list.  You can subscribe and read the archives on
`Google Groups <https://groups.google.com/a/continuum.io/forum/#!forum/numba-users>`_,
and there is also a `Gmane mirror <http://news.gmane.org/gmane.comp.python.numba.user>`_
allowing NNTP access.

Bug tracker
-----------

We use the `Github issue tracker <https://github.com/numba/numba/issues>`_
to track both bug reports and feature requests.


Workflow
========

If you want to contribute, we recommend you fork our `Github repository
<https://github.com/numba/numba>`_, then create a branch representing
your work.  When your work is ready, you should submit it as a pull
request from the Github interface.

If you want, you can submit a pull request even when you haven't finished
working.  This can be useful to gather feedback, or to stress your changes
against the :ref:`continuous integration <travis_ci>` platorm.  In this
case, please prepend ``[WIP]`` to your pull request's title.

Build environment
-----------------

Numba has a number of dependencies (mostly numpy and llvmpy) with
non-trivial build instructions.  Unless you want to build those dependencies
yourself, we recommend you use
`Conda <http://conda.pydata.org/miniconda.html>`_ to create a dedicated
development environment and install precompiled versions of those
dependencies there::

   $ <path_to_miniconda>/conda create -n numbaenv python=3.4 llvmpy numpy

.. note::
   This installs an environment based on Python 3.4, but you can of course
   choose another version supported by Numba.

To activate the environment for the current shell session::

   $ source <path_to_miniconda>/activate numbaenv

.. note::
   Those instructions are for a standard Linux shell.  You may need to
   adapt them for other platforms.

Once the environment is activated, you have a dedicated Python with the
requested dependencies::

   $ python
   Python 3.4.1 |Continuum Analytics, Inc.| (default, May 19 2014, 13:02:41)
   [GCC 4.1.2 20080704 (Red Hat 4.1.2-54)] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import llvm
   >>> llvm.__version__
   '0.12.7'

Building Numba
--------------

For a quick development workaround, we recommend you build Numba inside
its source checkout::

   $ python setup.py build_ext --inplace

This assumes you have a working C compiler and runtime on your development
system.

Running tests
-------------

Numba is validated using a test suite comprised of various kind of tests
(unit tests, functional tests). The test suite is written using the
standard :py:mod:`unittest` framework.

The various test modules are inside the ``numba/tests`` directory. There
are two entry points to run the test suite:

* if you want to run the whole test suite (which will take a couple of
  minutes), call the ``runtests.py`` script; in particular, the ``-m`` flag
  will parallelize the test suite into several processes::

  $ python runtests.py -m

* if you want to run an individual test module, invoke it as a python
  module, for example::

  $ python -m numba.tests.test_closure

Both the global test runner ``runtests.py`` and individual modules allow you
to pass various options to influence test running and report.  Pass ``-h``
or ``--help`` to get a glimpse at those options.


Development rules
=================

Code reviews
------------

Any non-trivial change should go through a code review by one or several of
the core developers.  The recommended process is to submit a pull request
on github.

A code review should try to assess the following criteria:

* general design and correctness
* code structure and maintainability
* coding conventions
* docstrings, comments
* test coverage

Coding conventions
------------------

All Python code should follow :pep:`8`.  Our C code doesn't have a
well-defined coding style (would it be nice to follow :pep:`7`?).
Code and documentation should generally fit within 80 columns, for
maximum readability with all existing tools (such as code review UIs).

Stability
---------

The repository's ``master`` branch is expected to be stable at all times.
This translates into the fact that the test suite passes without errors
on all supported platforms (see below).  This also means that a pull request
also needs to pass the test suite before it is merged in.

.. _travis_ci:

Platform support
----------------

Numba is to be kept compatible with Python 2.6, 2.7, 3.3 and 3.4 under
at least Linux, OS X and Windows.  Also, Numpy versions 1.6 and upwards
are supported.

We don't expect invidual contributors to test those combinations
themselves! Instead, we have a continuous integration platform.  Part of
the platform is hosted at `Travis-CI <https://travis-ci.org/numba/numba>`_.
Each time you submit a pull request, a corresponding build will be started
at Travis-CI and check that Numba builds and tests without any errors.
You can expect this to take less than 20 minutes.

Some platforms (such as Windows) cannot be hosted by Travis-CI, and the
Numba team has therefore access to a separate platform provided by
`Continuum <http://continuum.io>`_, our sponsor. We hope parts of that
infrastructure can be made public in the future.
