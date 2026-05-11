Coding Style
============


All Python code should follow :pep:`8`.  Our C code doesn't have a
well-defined coding style (would it be nice to follow :pep:`7`?).
Code and documentation should generally fit within 80 columns, for
maximum readability with all existing tools (such as code review UIs).

Numba uses `Flake8 <http://flake8.pycqa.org/en/latest/>`_ to ensure a consistent
Python code format throughout the project. ``flake8`` can be installed
with ``pip`` or ``conda`` and then run from the root of the Numba repository::

    flake8 numba

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


.. _type_anno_check:

Type annotation and runtime type checking
''''''''''''''''''''''''''''''''''''''''''

Numba is slowly gaining type annotations. To facilitate the review of pull
requests that are incrementally adding type annotations, the test suite uses
`typeguard`_ to perform runtime type checking. This helps verify the validity
of type annotations.

To enable runtime type checking in the test suite, users can use
`runtests.py`_ in the source root as the test runner and set environment
variable ``NUMBA_USE_TYPEGUARD=1``. For example::

    $ NUMBA_USE_TYPEGUARD=1 python runtests.py numba.tests


.. _typeguard: https://typeguard.readthedocs.io/en/latest/
.. _runtests.py: https://github.com/numba/numba/blob/main/runtests.py
