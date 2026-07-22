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
* [ ] Review deprecation schedule and notices. Make PRs if need be. (Note that
  deprecation notices for features that have been removed are kept in the
  documentation for two more releases.)
* [ ] Create changelog using instructions at: `docs/source/developer/release.rst`
* [ ] Merge change log changes.
    - [ ] "PR with changelog entries".
* [ ] Create X.Y release branch.
* [ ] Create PR against the release branch to make `numba/testing/main.py`
      to refer to `origin/releaseX.Y` instead of `origin/main`.
* [ ] Dependency version pinning on release branch:
  * [ ] Pin llvmlite to `0.A.*`.
  * [ ] Pin NumPy if needed (see previous release for details).
    * [ ] `buildscripts/condarecipe.local/meta.yaml`
    * [ ] `numba/__init__.py`
    * [ ] `setup.py`
    * [ ] `docs/environment.yml`
  * [ ] Pin TBB if needed.
* [ ] Run the HEAD of the release branch through GHA and confirm:
  * [ ] conda build and test has passed.
  * [ ] wheel build and test has passed.
* [ ] Annotated tag `X.Y.Zrc1` on release branch (no `v` prefix).
* [ ] Upload conda packages, wheels and sdist using GHA.
* [ ] Make sure that packages arrived on PyPI and on anaconda.org on labels `numba/label/dev` and `numba/label/main`.
* [ ] Initialize and verify ReadTheDocs build.
* [ ] Post announcement to discourse group and ping the release testers group
  using `@RC_Testers`.
* [ ] Post link to X and to Mastodon and...

### Post Release:

* [ ] Tag `X.Y+1.0dev0` to start new development cycle on `main`.
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
