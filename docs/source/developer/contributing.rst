
Contributing to Numba
=====================

We welcome people who want to make contributions to Numba, big or small!
Even simple documentation improvements are encouraged.  If you have
questions, don't hesitate to ask them (see below).


Communication
-------------

Mailing-list
''''''''''''

We have a public mailing-list that you can e-mail at numba-users@continuum.io.
If you have any questions about contributing to Numba, it is ok to ask them
on this mailing-list.  You can subscribe and read the archives on
`Google Groups <https://groups.google.com/a/continuum.io/forum/#!forum/numba-users>`_,
and there is also a `Gmane mirror <http://news.gmane.org/gmane.comp.python.numba.user>`_
allowing NNTP access.

.. _report-bugs:

Bug tracker
''''''''''''

We use the `Github issue tracker <https://github.com/numba/numba/issues>`_
to track both bug reports and feature requests.  If you report an issue,
please include specifics:

* what you are trying to do;
* which operating system you have and which version of Numba you are running;
* how Numba is misbehaving, e.g. the full error traceback, or the unexpected
  results you are getting;
* as far as possible, a code snippet that allows full reproduction of your
  problem.

Getting set up
--------------

If you want to contribute, we recommend you fork our `Github repository
<https://github.com/numba/numba>`_, then create a branch representing
your work.  When your work is ready, you should submit it as a pull
request from the Github interface.

If you want, you can submit a pull request even when you haven't finished
working.  This can be useful to gather feedback, or to stress your changes
against the :ref:`continuous integration <travis_ci>` platorm.  In this
case, please prepend ``[WIP]`` to your pull request's title.

.. _buildenv:

Build environment
'''''''''''''''''

Numba has a number of dependencies (mostly `Numpy <http://www.numpy.org/>`_
and `llvmlite <https://github.com/numba/llvmlite>`_) with non-trivial build
instructions.  Unless you want to build those dependencies yourself, we
recommend you use `Conda <http://conda.pydata.org/miniconda.html>`_ to
create a dedicated development environment and install precompiled versions
of those dependencies there.

First add the Binstar ``numba`` channel so as to get development builds of
the llvmlite library::

   $ conda config --add channels numba

Then create an environment with the right dependencies::

   $ <path_to_miniconda>/conda create -n numbaenv python=3.5 llvmlite numpy

.. note::
   This installs an environment based on Python 3.5, but you can of course
   choose another version supported by Numba.

To activate the environment for the current shell session::

   $ source <path_to_miniconda>/activate numbaenv

.. note::
   Those instructions are for a standard Linux shell.  You may need to
   adapt them for other platforms.

Once the environment is activated, you have a dedicated Python with the
required dependencies::

   $ python
   Python 3.4.2 |Continuum Analytics, Inc.| (default, Oct 21 2014, 17:16:37)
   [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import llvmlite
   >>> llvmlite.__version__
   '0.2.0-3-g9f60cd1'

Building Numba
''''''''''''''

For a convenient development workflow, we recommend you build Numba inside
its source checkout::

   $ python setup.py build_ext --inplace

This assumes you have a working C compiler and runtime on your development
system.  You will have to run this command again whenever you modify
C files inside the Numba source tree.

Running tests
'''''''''''''

Numba is validated using a test suite comprised of various kind of tests
(unit tests, functional tests). The test suite is written using the
standard :py:mod:`unittest` framework.

The tests can be executed via ``python -m numba.runtests``.  If you are
running Numba from a source checkout, you can type ``./runtests.py``
as a shortcut.  Various options are supported to influence test running
and reporting.  Pass ``-h`` or ``--help`` to get a glimpse at those options.
Examples:

* to list all available tests::

    $ python -m numba.runtests -l

* to list tests from a specific (sub-)suite::

    $ python -m numba.runtests -l numba.tests.test_usecases

* to run those tests::

    $ python -m numba.runtests -l numba.tests.test_usecases

* to run all tests in parallel, using multiple sub-processes::

    $ python -m numba.runtests -m
    
* For a detailed list of all options::

    $ python -m numba.runtests -h


Development rules
-----------------

Code reviews
''''''''''''

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
''''''''''''''''''

All Python code should follow :pep:`8`.  Our C code doesn't have a
well-defined coding style (would it be nice to follow :pep:`7`?).
Code and documentation should generally fit within 80 columns, for
maximum readability with all existing tools (such as code review UIs).

Stability
'''''''''

The repository's ``master`` branch is expected to be stable at all times.
This translates into the fact that the test suite passes without errors
on all supported platforms (see below).  This also means that a pull request
also needs to pass the test suite before it is merged in.

.. _travis_ci:

Platform support
''''''''''''''''

Numba is to be kept compatible with Python 2.7, 3.4 and 3.5 under
at least Linux, OS X and Windows.  Also, Numpy versions 1.7 and upwards
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


Documentation
-------------

The numba documentation is split over two repositories:

* This documentation is in the ``docs`` directory inside the
  `Numba repository <https://github.com/numba/numba>`_.

* The `Numba homepage <http://numba.pydata.org>`_ has its sources in a
  separate repository at https://github.com/numba/numba-webpage


Main documentation
''''''''''''''''''

This documentation is under the ``docs`` directory of the `Numba repository`_.
It is built with `Sphinx <http://sphinx-doc.org/>`_, which is available
using conda or pip.

To build the documentation, you need the basicstrap theme and
its dependencies::

   $ pip install sphinxjp.themes.basicstrap
   $ pip install sphinxjp.themecore

You can edit the source files under ``docs/source/``, after which you can
build and check the documentation::

   $ make html
   $ open _build/html/index.html

Core developers can upload this documentation to the Numba website
at http://numba.pydata.org by using the ``gh-pages.py`` script under ``docs``::

   $ python gh-pages.py version  # version can be 'dev' or '0.16' etc

then verify the repository under the ``gh-pages`` directory and use
``git push``.

Web site homepage
'''''''''''''''''

The Numba homepage on http://numba.pydata.org can be fetched from here:
https://github.com/numba/numba-webpage

After pushing documentation to a new version, core developers will want to
update the website.  Some notable files:

* ``index.rst``       # Update main page
* ``_templates/sidebar_versions.html``    # Update sidebar links
* ``doc.rst``         # Update after adding a new version for numba docs
* ``download.rst``    # Updata after uploading new numba version to pypi

After updating run::

   $ make html

and check out ``_build/html/index.html``.  To push updates to the Web site::

   $ python _scripts/gh-pages.py

then verify the repository under the ``gh-pages`` directory.  Make sure the
``CNAME`` file is present and contains a single line for ``numba.pydata.org``.
Finally, use ``git push`` to update the website.

