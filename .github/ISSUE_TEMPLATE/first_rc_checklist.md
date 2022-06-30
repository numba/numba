---
name: First Release Candidate Checklist (maintainer only)
about: Checklist template for the first release of every series
title: Numba X.Y.Zrc1 Checklist (FIXME)
labels: task

---


## Numba X.Y.Z

* [ ] Merge to main.
    - [ ] "remaining Pull-Requests from milestone".
* [ ] Check Numba's version support table documentation. Update via PR if
      needed.
* [ ] Review deprecation schedule and notices. Make PRs if need be.
* [ ] Merge change log changes.
    - [ ] "PR with changelog entries".
* [ ] Create X.Y release branch.
* [ ] Dependency version pinning on release branch
  * [ ] Pin llvmlite to `>=0.A.0rc1,<0.A+1.0`.
  * [ ] Pin NumPy if needed
  * [ ] Pin TBB if needed
* [ ] Annotated tag X.Y.Zrc1 on release branch (no `v` prefix).
* [ ] Build and upload conda packages on buildfarm (check "upload").
* [ ] Build wheels and sdist on the buildfarm (check "upload").
* [ ] Verify packages uploaded to Anaconda Cloud and move to `numba/label/main`.
* [ ] Upload wheels and sdist to PyPI (upload from `ci_artifacts`).
* [ ] Verify wheels for all platforms arrived on PyPi.
* [ ] Initialize and verify ReadTheDocs build.
* [ ] Send RC announcement email / post announcement to discourse group.
* [ ] Post link to Twitter.

### Post Release:

* [ ] Clean up `ci_artifacts` by moving files to sub-directories
* [ ] Tag X.Y+1.0dev0 to start new development cycle on `main`.
* [ ] Update llvmlite dependency via PR to `main`, PR includes version updates
      to:
  * [ ] `setup.py`
  * [ ] `numba/__init__.py`
  * [ ] `docs/environment.yml`
  * [ ] `buildscripts/incremental/setup_conda_environment.sh`
  * [ ] `buildscripts/incremental/setup_conda_environment.cmd`
  * [ ] `buildscripts/condarecipe.local/meta.yaml`
* [ ] Update release checklist template with any additional bullet points that
      may have arisen during the release.
* [ ] Close milestone (and then close this release issue).
