---
name: First Release Candidate Checklist (maintainer only)
about: Checklist template for the first release of every series
title: Numba X.Y.Zrc1 Checklist (FIXME)
labels: task

---


## Numba X.Y.Z

* [ ] Merge to master.
    - [ ] "remaining Pull-Requests from milestone".
* [ ] Merge change log changes.
    - [ ] "PR with changelog entries".
* [ ] Create X.Y release branch.
* [ ] Pin llvmlite to `>=0.A.0rc1,<0.A+1.0`.
* [ ] Annotated tag X.Y.Zrc1 on release branch.
* [ ] Build and upload conda packages on buildfarm (check "upload").
* [ ] Build wheels (`$PYTHON_VERSIONS`) on the buildfarm.
* [ ] Upload wheels and sdist to PyPI (upload from `ci_artifacts`).
* [ ] Verify packages uploaded to Anaconda Cloud and move to `numba/label/main`.
* [ ] Verify wheels for all platforms arrived on PyPi.
* [ ] Initialize and verify ReadTheDocs build.
* [ ] Clean up `ci_artifacts`.
* [ ] Send RC announcement email / post announcement to discourse group.
* [ ] Post link to Twitter.

### Post Release:

* [ ] Tag X.Y+1.0dev0 to start new development cycle on `master`.
* [ ] Update llvmlite dependency spec to match next version via PR to `master`.
* [ ] Update release checklist template with any additional bullet points that
      may have arisen during the release.
* [ ] Close milestone (and then close this release issue).
