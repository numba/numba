
Contributing to Numba
=====================

We welcome people who want to make contributions to Numba, big or small!
Even simple documentation improvements are encouraged.  If you have
questions, don't hesitate to ask them (see below).


Communication
-------------

Real-time Chat
''''''''''''''

Numba uses Gitter for public real-time chat.  To help improve the
signal-to-noise ratio, we have two channels:

* `numba/numba <https://gitter.im/numba/numba>`_: General Numba discussion,
  questions, and debugging help.
* `numba/numba-dev <https://gitter.im/numba/numba-dev>`_: Discussion of PRs,
  planning, release coordination, etc.

Both channels are public, but we may ask that discussions on numba-dev move to
the numba channel.  This is simply to ensure that numba-dev is easy for core
developers to keep up with.

Note that the Github issue tracker is the best place to report bugs.  Bug
reports in chat are difficult to track and likely to be lost.

Forum
.....

Numba uses Discourse as a forum for longer running threads such as design
discussions and roadmap planning. There are various categories available and it
can be reached at: `numba.discourse.group <https://numba.discourse.group/>`_.

Weekly Meetings
'''''''''''''''

The core Numba developers have a weekly video conference to discuss roadmap,
feature planning, and outstanding issues.  These meetings are entirely public,
details are posted on
`numba.discourse.group Announcements <https://numba.discourse.group/c/announcements/>`_
and everyone is welcome to join the discussion. Minutes will be taken and will
be posted to the
`Numba wiki <https://github.com/numba/numba/wiki/Meeting-Minutes>`_.

.. _report-numba-bugs:

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
against the :ref:`continuous integration <continuous_integration_testing>`
platform.  In this case, please prepend ``[WIP]`` to your pull request's title.

.. _buildenv:

Build environment
'''''''''''''''''

Numba has a number of dependencies (mostly `NumPy <http://www.numpy.org/>`_ and
`llvmlite <https://github.com/numba/llvmlite>`_) with non-trivial build
instructions.  Unless you want to build those dependencies yourself, we
recommend you use `conda <http://conda.pydata.org/miniconda.html>`_ to create a
dedicated development environment and install precompiled versions of those
dependencies there. Read more about the Numba dependencies here:
`numba-source-install-check`.

When working with a source checkout of Numba you will also need a development
build of llvmlite. These are available from the ``numba/label/dev`` channel on
`anaconda.org <https://anaconda.org/numba/llvmlite>`_.

To create an environment with the required dependencies, noting the use of the 
double-colon syntax (``numba/label/dev::llvmlite``) to install the latest
development version of the llvmlite library::

   $ conda create -n numbaenv python=3.10 numba/label/dev::llvmlite numpy scipy jinja2 cffi

.. note::
   This installs an environment based on Python 3.10, but you can of course
   choose another version supported by Numba.  To test additional features,
   you may also need to install ``tbb`` and/or ``llvm-openmp``. Check the
   dependency list above for details.

To activate the environment for the current shell session::

   $ conda activate numbaenv

.. note::
   These instructions are for a standard Linux shell.  You may need to
   adapt them for other platforms.

Once the environment is activated, you have a dedicated Python with the
required dependencies::

    $ python
    Python 3.10.3 (main, Mar 28 2022, 04:26:28) [Clang 12.0.0 ] on darwin
    Type "help", "copyright", "credits" or "license" for more information.

    >>> import llvmlite
    >>> llvmlite.__version__
    0.39.0dev0+61.gf27ac6f


Building Numba
''''''''''''''

For a convenient development workflow, we recommend you build Numba inside
its source checkout::

   $ git clone git@github.com:numba/numba.git
   $ cd numba
   $ python setup.py build_ext --inplace

This assumes you have a working C compiler and runtime on your development
system.  You will have to run this command again whenever you modify
C files inside the Numba source tree.

The ``build_ext`` command in Numba's setup also accepts the following
arguments:

- ``--noopt``: This disables optimization when compiling Numba's CPython
  extensions, which makes debugging them much easier. Recommended in
  conjunction with the standard ``build_ext`` option ``--debug``.
- ``--werror``: Compiles Numba's CPython extensions with the ``-Werror`` flag.
- ``--wall``: Compiles Numba's CPython extensions with the ``-Wall`` flag.

Note that Numba's CI and the conda recipe for Linux build with the ``--werror``
and ``--wall`` flags, so any contributions that change the CPython extensions
should be tested with these flags too.

.. _running-tests:

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

    $ python -m numba.runtests numba.tests.test_usecases

* to run all tests in parallel, using multiple sub-processes::

    $ python -m numba.runtests -m

* For a detailed list of all options::

    $ python -m numba.runtests -h

The numba test suite can take a long time to complete.  When you want to avoid
the long wait,  it is useful to focus on the failing tests first with the
following test runner options:

* The ``--failed-first`` option is added to capture the list of failed tests
  and to re-execute them first::

    $ python -m numba.runtests --failed-first -m -v -b

* The ``--last-failed`` option is used with ``--failed-first`` to execute
  the previously failed tests only::

    $ python -m numba.runtests --last-failed -m -v -b

When debugging, it is useful to turn on logging.  Numba logs using the
standard ``logging`` module.  One can use the standard ways (i.e.
``logging.basicConfig``) to configure the logging behavior.  To enable logging
in the test runner, there is a ``--log`` flag for convenience::

    $ python -m numba.runtests --log

To enable :ref:`runtime type-checking <type_anno_check>`, set the environment
variable ``NUMBA_USE_TYPEGUARD=1`` and use `runtests.py` from the source root
instead. For example::

    $ NUMBA_USE_TYPEGUARD=1 python runtests.py


Running coverage
''''''''''''''''

Coverage reports can be produced using `coverage.py
<https://coverage.readthedocs.io/en/stable/index.html>`_. To record coverage
info for the test suite, run::

    coverage run -m numba.runtests <runtests args>

Next, combine coverage files (potentially for multiple runs) with::

    coverage combine

The combined output can be transformed into various report formats - see the
`coverage CLI usage reference
<https://coverage.readthedocs.io/en/stable/cmd.html#command-line-usage>`_.
For example, to produce an HTML report, run::

    coverage html

Following this command, the report can be viewed by opening ``htmlcov/index.html``.


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
* docstrings, comments and release notes (if necessary)
* test coverage


Policy on large scale changes to code formatting
''''''''''''''''''''''''''''''''''''''''''''''''

Please note that pull requests making large scale changes to format the code
base are in general not accepted. Such changes often increase the likelihood of
merge conflicts for other pull requests, which inevitably take time and
resources to resolve. They also require a lot of effort to check as Numba aims
to compile code that is valid even if it is not ideal. For example, in a test of
``operator.eq``::

    if x == None: # Valid code, even if the recommended form is `if x is None:`

This tests Numba's compilation of comparison with ``None``, and therefore
should not be changed, even though most style checkers will suggest it should.

This policy has been adopted by the core developers so as to try and make best
use of limited resources. Whilst it would be great to have an extremely tidy
code base, priority is given to fixes and features over code formatting changes.


Coding conventions
''''''''''''''''''

All Python code should follow :pep:`8`.  Our C code doesn't have a
well-defined coding style (would it be nice to follow :pep:`7`?).
Code and documentation should generally fit within 80 columns, for
maximum readability with all existing tools (such as code review UIs).

Numba uses `Flake8 <http://flake8.pycqa.org/en/latest/>`_ to ensure a consistent
Python code format throughout the project. ``flake8`` can be installed
with ``pip`` or ``conda`` and then run from the root of the Numba repository::

    flake8 numba

Optionally, you may wish to setup `pre-commit hooks <https://pre-commit.com/>`_
to automatically run ``flake8`` when you make a git commit. This can be
done by installing ``pre-commit``::

    pip install pre-commit

and then running::

    pre-commit install

from the root of the Numba repository. Now ``flake8`` will be run each time
you commit changes. You can skip this check with ``git commit --no-verify``.

Numba has started the process of using `type hints <https://www.python.org/dev/peps/pep-0484/>`_ in its code base. This
will be a gradual process of extending the number of files that use type hints, as well as going from voluntary to
mandatory type hints for new features. `Mypy <http://mypy-lang.org/>`_ is used for automated static checking.

At the moment, only certain files are checked by mypy. The list can be found in ``mypy.ini``. When making changes to
those files, it is necessary to add the required type hints such that mypy tests will pass. Only in exceptional
circumstances should ``type: ignore`` comments be used.

If you are contributing a new feature, we encourage you to use type hints, even if the file is not currently in the
checklist. If you want to contribute type hints to enable a new file to be in the checklist, please add the file to the
``files`` variable in ``mypy.ini``, and decide what level of compliance you are targeting. Level 3 is basic static
checks, while levels 2 and 1 represent stricter checking. The levels are described in details in ``mypy.ini``.

There is potential for confusion between the Numba module ``typing`` and Python built-in module ``typing`` used for type
hints, as well as between Numba types---such as ``Dict`` or ``Literal``---and ``typing`` types of the same name.
To mitigate the risk of confusion we use a naming convention by which objects of the built-in ``typing`` module are
imported with an ``pt`` prefix. For example, ``typing.Dict`` is imported as ``from typing import Dict as ptDict``.

Release Notes
'''''''''''''

Pull Requests that add significant user-facing modifications may need to be mentioned in the release notes.
To add a release note, a short ``.rst`` file needs creating containing a summary of the change and it needs to be placed in 
``docs/upcoming_changes``. The file ``docs/upcoming_changes/README.rst`` details the format
and file naming conventions.

Stability
'''''''''

The repository's ``main`` branch is expected to be stable at all times.
This translates into the fact that the test suite passes without errors
on all supported platforms (see below).  This also means that a pull request
also needs to pass the test suite before it is merged in.

.. _platform_support:

Platform support
''''''''''''''''

Every commit to the main branch is automatically tested on all of the
platforms Numba supports. This includes ARMv8, POWER8, and NVIDIA GPUs.
The build system however is internal to Anaconda, so we also use
`Azure <https://dev.azure.com/numba/numba/_build>`_ to provide public continuous
integration information for as many combinations as can be supported by the
service.  Azure CI automatically tests all pull requests on Windows, OS X and
Linux, as well as a sampling of different Python and NumPy versions. If you see
problems on platforms you are unfamiliar with, feel free to ask for help in your
pull request. The Numba core developers can help diagnose cross-platform
compatibility issues. Also see the :ref:`continuous integration
<continuous_integration_testing>` section on how public CI is implemented.

.. _continuous_integration_testing:

Continuous integration testing
''''''''''''''''''''''''''''''

The Numba test suite causes CI systems a lot of grief:

#. It's huge, 9000+ tests.
#. In part because of 1. and that compilers are pretty involved, the test suite
   takes a long time to run.
#. There's sections of the test suite that are deliberately designed to stress
   systems almost to the point of failure (tests which concurrently compile and
   execute with threads and fork processes etc).
#. The combination of things that Numba has to test well exceeds the capacity of
   any public CI system, (Python versions x NumPy versions x Operating systems
   x Architectures x feature libraries (e.g. SVML) x threading backends
   (e.g. OpenMP, TBB)) and then there's CUDA too and all its version
   variants.

As a result of the above, public CI is implemented as follows:

#. The combination of OS x Python x NumPy x Various Features in the testing
   matrix is designed to give a good indicative result for whether "this pull
   request is probably ok".
#. When public CI runs it:

   #. Looks for files that contain tests that have been altered by the proposed
      change and runs these on the whole testing matrix.
   #. Runs a subset of the test suite on each part of the testing matrix. i.e.
      slice the test suite up by the number of combinations in the testing
      matrix and each combination runs one chunk. This is done for speed,
      because public CI cannot cope with the load else.

If a Pull Request (PR) changes CUDA code or will affect the CUDA target, it
needs to be run on `gpuCI <https://gpuci.gpuopenanalytics.com/job/numba/>`_.
This can be triggered by one of the Numba maintainers commenting ``run gpuCI
tests`` on the PR discussion. This runs the CUDA testsuite with various CUDA
toolkit versions on Linux, to provide some initial confidence in the
correctness of the changes with respect to CUDA. Following approval, the PR
will also be run on Numba's build farm to test other configurations with CUDA
(including Windows, which is not tested by gpuCI).

If the PR is not CUDA-related but makes changes to something that the core
developers consider risky, then it will also be run on the Numba farm just to
make sure. The Numba project's private build and test farm will actually
exercise all the applicable tests on all the combinations noted above on real
hardware!


.. _type_anno_check:

Type annotation and runtime type checking
'''''''''''''''''''''''''''''''''''''''''

Numba is slowly gaining type annotations. To facilitate the review of pull
requests that are incrementally adding type annotations, the test suite uses
`typeguard`_ to perform runtime type checking. This helps verify the validity
of type annotations.

To enable runtime type checking in the test suite, users can use
`runtests.py`_ in the source root as the test runner and set environment
variable ``NUMBA_USE_TYPEGUARD=1``. For example::

    $ NUMBA_USE_TYPEGUARD=1 python runtests.py numba.tests

Things that help with pull requests
'''''''''''''''''''''''''''''''''''

Even with the mitigating design above public CI can get overloaded which causes
a backlog of builds. It's therefore really helpful when opening pull requests if
you can limit the frequency of pushing changes. Ideally, please squash commits
to reduce the number of patches and/or push as infrequently as possible. Also,
once a pull request review has started, please don't rebase/force push/squash
or do anything that rewrites history of the reviewed code as GitHub cannot track
this and it makes it very hard for reviewers to see what has changed.

The core developers thank everyone for their cooperation with the above!

Why is my pull request/issue seemingly being ignored?
'''''''''''''''''''''''''''''''''''''''''''''''''''''

Numba is an open source project and like many similar projects it has limited
resources. As a result, it is unfortunately necessary for the core developers to
associate a priority with issues/pull requests (PR). A great way to move your
issue/PR up the priority queue is to help out somewhere else in the project so
as to free up core developer time. Examples of ways to help:

* Perform an initial review on a PR. This often doesn't require compiler
  engineering knowledge and just involves checking that the proposed patch is of
  good quality, fixes the problem/implements the feature, is well tested and
  documented.
* Debug an issue, there are numerous issues which `"need triage" <https://github.com/numba/numba/issues?q=is%3Aissue+is%3Aopen+label%3Aneedtriage>`_
  which essentially involves debugging the reported problem. Even if you cannot
  get right to the bottom of a problem, leaving notes about what was discovered
  for someone else is also helpful.
* Answer questions/provide help for users on `discourse <https://numba.discourse.group/>`_
  and/or `gitter.im <https://gitter.im/numba/numba>`_.

The core developers thank everyone for their understanding with the above!

Documentation
-------------

The Numba documentation is split over two repositories:

* This documentation is in the ``docs`` directory inside the
  `Numba repository <https://github.com/numba/numba>`_.

* The `Numba homepage <https://numba.pydata.org>`_ has its sources in a
  separate repository at https://github.com/numba/numba.github.com.


Main documentation
''''''''''''''''''

This documentation is under the ``docs`` directory of the `Numba repository`_.
It is built with `Sphinx <http://sphinx-doc.org/>`_, `numpydoc
<https://numpydoc.readthedocs.io/>`_ and the
`sphinx-rtd-theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_.

To install all dependencies for building the documentation, use::

   $ conda install sphinx numpydoc sphinx_rtd_theme

You can edit the source files under ``docs/source/``, after which you can
build and check the documentation under ``docs/``::

   $ make html
   $ open _build/html/index.html

Web site homepage
'''''''''''''''''

The Numba homepage on https://numba.pydata.org can be fetched from here:
https://github.com/numba/numba.github.com


.. _typeguard: https://typeguard.readthedocs.io/en/latest/
.. _runtests.py: https://github.com/numba/numba/blob/main/runtests.py
