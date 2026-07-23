Numba Release Process
=====================

The goal of the Numba release process -- from a high level perspective -- is to
publish source and binary artifacts that correspond to a given version
number. This usually involves a sequence of individual tasks that must be
performed in the correct order and with diligence. Numba and llvmlite are
commonly released in lockstep since there is usually a one-to-one mapping
between a Numba version and a corresponding llvmlite version.

This section contains various notes and templates that can be used to create a
Numba release checklist on the Numba Github issue tracker. This is an aid for
the maintainers during the release process and helps to ensure that all tasks
are completed in the correct order and that no tasks are accidentally omitted.

If new or additional items do appear during release, please do remember to add
them to the checklist templates. Also note that the release process itself is
always a work in progress. This means that some of the information here may be
outdated. If you notice this please do remember to submit a pull-request to
update this document.

All release checklists are available as Gitub issue templates. To create a new
release checklist simply open a new issue and select the correct template.


Primary Release Candidate Checklist
-----------------------------------

This is for the first/primary release candidate for minor release i.e. the
first release of every series. It is special, because during this release, the
release branch will have to be created. Release candidate indexing begins at 1.

.. literalinclude:: ../../../.github/ISSUE_TEMPLATE/first_rc_checklist.md
    :language: md
    :lines: 9-

`Open a primary release checklist <https://github.com/numba/numba/issues/new?template=first_rc_checklist.md>`_.

Subsequent Release Candidates, Final Releases and Patch Releases
----------------------------------------------------------------

Releases subsequent to the first release in a series usually involves a series
of cherry-picks, the recipe is therefore slightly different.

.. literalinclude:: ../../../.github/ISSUE_TEMPLATE/sub_rc_checklist.md
    :language: md
    :lines: 9-

`Open a subsequent release checklist <https://github.com/numba/numba/issues/new?template=sub_rc_checklist.md>`_.

Generating Release Notes
------------------------

The Numba release notes consist of two parts and there are two tools which need
to be invoked.

* Release summary and overview of noteworthy items: use ``towncrier``
* Pull-request and author list: use ``maint/gitlog2changelog.py``


Using ``towncrier``
...................

Before each release, generate the changelog using Towncrier.
This will collect all news fragments, combine them, and output
the result to a single changelog file. During the changelog creation
user is asked if the existing news fragments are to be deleted.
In case of release notes generation, they are to be deleted so
that they aren't accidentally included in post-release changelog creation.

.. code-block:: bash

   towncrier build --version=0.xx.x


Using ``maint/gitlog2changelog.py``
...................................

The script ``maint/gitlog2changelog.py`` is used to generate the list of
pull-requests and authors. To prepare to use it:

* Install dependencies: ``conda install docopt pygithub gitpython``.
* Generate a fine-grained Personal Access Token on Github with read access to
  public repositories. This can be done in
  `Github Personal Access Tokens settings
  <https://github.com/settings/tokens?type=beta>`_.
* Establish the base commit for the changelog. There are two known approaches
  here:

  * Finding a base-commit using ``git``. This is the common commit between
    ``main`` and the last release branch, which can be determined by running
    ``git merge-base main <branch>``. For example, ``branch`` may be
    ``release0.58`` if generating the release notes for the 0.59 release.
  * Using the development commit for the release. For example for the
    `0.59.0rc1` release this would be `0.59.0dev0`.

The script can then be invoked in the root of the repository with:

.. code-block:: bash

   python maint/gitlog2changelog.py --token="<token>" --beginning="<base commit>" \
                                    --repo="numba/numba" --digits=4

This uses the token and commit established above. The ``--digits`` arguments
specifies the number of digits in pull request numbers - this is presently 4 but
will roll over to 5 soon after the time of writing - when that happens, it will
be necessary to run the script twice, once with ``--digits=4`` and once with
``--digits=5`` and combine the results.

The script should output the release notes in a form that can be pasted into the
top of the ``CHANGE_LOG`` file in the repo root. Truncated example output looks
like:

.. code-block:: text

   Pull-Requests:

   * PR `#8990 <https://github.com/numba/numba/pull/8990>`_: Removed extra block copying in InlineWorker (`kc611 <https://github.com/kc611>`_)
   * PR `#9048 <https://github.com/numba/numba/pull/9048>`_: Dynamically allocate parfor schedule. (`DrTodd13 <https://github.com/DrTodd13>`_)

   <... some output omitted ...>

   Authors:

   * `apmasell <https://github.com/apmasell>`_
   * `DrTodd13 <https://github.com/DrTodd13>`_

   <... some output omitted ...>

Note that, the list may contain duplicates and thus you need to manually check
and eliminate these duplictes! The duplicates are commonly the result of
pull-requests that have been commited to the `main` branch and have then been
cherry picked to a `release` branch. This can happen when issues are fixed for
release candidates or when resolved issues are backported for patch releases.

Note also, that you must manually add the pull-request for the changelog itself
once it has been opened. This must be done after opening the pull-request
itself, since the link and number for the pull-request will not exists
beforehand.
