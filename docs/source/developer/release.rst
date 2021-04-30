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
Markdown and can be copied verbatim to a GitHub issue to create a release
checklist.

If new or additional items do appear during release, please do remember to add
them to the checklist templates. Note also, that the release process itself is
always a work in progress. This means, some of the information here may be
outdated. If you detect this, please do remember to submit a pull-request to
update this document.


Minor Version Release Candidates
--------------------------------

This is for first minor version release candidates. I.e. the first release of
every series. It is special, because during this release, the release branch
will have to be created. Release candidate indexing begins at 0.

.. code-block::

    ## Numba X.Y.Z

    * [ ] merge to master:
        - [ ] "remaining Pull-Requests from milestone"
    * [ ] merge change log changes
        - [ ] "PR with changelog entries
    * [ ] Create X.Y release branch
    * [ ] pin llvmlite to `>=0.A.0rc1,<0.A+1.0`
    * [ ] annotated tag X.Y.Zrc0 on release branch
    * [ ] build and upload conda packages on buildfarm (check "upload")
    * [ ] build wheels (`$PYTHON_VERSIONS`) on the buildfarm
    * [ ] upload wheels and sdist to PyPI (upload from `ci_artifacts`)
    * [ ] verify packages uploaded to Anaconda Cloud and move to `numba/label/main`
    * [ ] verify wheels for all platforms arrived on PyPi
    * [ ] verify ReadTheDocs build
    * [ ] clean up `ci_artifacts`
    * [ ] review, merge and check execution of release notebook
    * [ ] send RC announcement email / post announcement to discourse group
    * [ ] post link to Twitter
    * [ ] post link to python-announce-list@python.org

    ### Post Release:

    * [ ] tag X.Y+1.0dev0 to start new development cycle on `master`
    * [ ] update llvmlite dependency spec to match next version via PR to `master`
    * [ ] update release checklist template
    * [ ] close milestone (and then close this release issue)

Second Releases Candidates, Final Releases and Patch Releases
-------------------------------------------------------------

A patch release usually involves a series of cherry-picks, so the recipe is
slightly different.

.. code-block::

    ## numba X.Y.Z

    * [ ] cherry-pick items from the X.Y.Z milestone into a PR
    * [ ] merge change log modifications and cherry-picks to X.Y release branch
    * [ ] https://github.com/numba/numba/pull/XXXX
    * [ ] annotated tag X.Y.Z on release branch (no `v` prefix)
    * [ ] build and upload conda packages on buildfarm (check "upload")
    * [ ] build wheels (`$PYTHON_VERSIONS`) on the buildfarm
    * [ ] upload wheels and sdist to PyPI (upload from `ci_artifacts`)
    * [ ] verify packages uploaded to Anaconda Cloud and move to `numba/label/main`
    * [ ] verify wheels for all platforms arrived on PyPi
    * [ ] verify ReadTheDocs build
    * [ ] clean up `ci_artifacts`
    * [ ] send RC/FINAL announcement email / post announcement to discourse group
    * [ ] post link to Twitter
    * [ ] post link to python-announce-list@python.org

    ### Post release

    * [ ] cherry-pick change-log modifications to main branch (`master`)
    * [ ] update release checklist template
    * [ ] ping Anaconda Distro team to trigger a build for `defaults` (FINAL ONLY)
    * [ ] close milestone (and then close this release issue)
