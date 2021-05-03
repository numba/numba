Numba Release Checklist Templates
==================================

The goal of the Numba release process -- from a birds eye perspective -- is to
publish source and binary artifacts that correspond to the given version
number. This usually involves a sequence of individual tasks that must be
performed in the correct order and with diligence. Numba and llvmlite are
commonly released in lockstep since there is usually a one-to-one mapping
between a Numba version and a corresponding llvmlite version.

This section contains various notes and templates that can be used to create a
Numba release checklist on the Numba Github issue tracker. This is the aid for
the maintainers during the release process and helps to ensure that all tasks
are completed in the correct order and that no tasks are accidentally omitted.

Historically, the release checklists were copied from one release to the next.
While this worked OKish for many years, often times tasks were lost during
repeated copy and paste. As a result this document was created to keep track of
all potential tasks. All templates within this document are written in Github
Markdown and are available as issue templates on Github. This means,
to create a new release checklist, simply open a new issue and select the
correct template.

If new or additional items do appear during release, please do remember to add
them to the checklist templates. Note also, that the release process itself is
always a work in progress. This means, some of the information here may be
outdated. If you detect this, please do remember to submit a pull-request to
update this document.


Primary Release Candidate Checklist
-----------------------------------

This is for the first/primary release candidate for minor release. I.e. the
first release of every series. It is special, because during this release, the
release branch will have to be created. Release candidate indexing begins at 0.

.. literalinclude:: first_rc_checklist.yml
    :language: md
    :lines: 9-


Second Releases Candidates, Final Releases and Patch Releases
-------------------------------------------------------------

A patch release usually involves a series of cherry-picks, so the recipe is
slightly different.

.. literalinclude:: sub_rc_checklist.yml
    :language: md
    :lines: 9-
