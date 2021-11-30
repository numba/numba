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
