---
name: Subsequent Release Candidate Checklist (maintainer only)
about: Checklist template for all subsequent releases (RC 2-N, FINAL and PATCH) of every series
title: Numba X.Y.Zrc1 Checklist (FIXME)
labels: task

---


## numba X.Y.Z

* [ ] Cherry-pick items from the X.Y.Z milestone into a PR.
* [ ] Merge change log modifications and cherry-picks to X.Y release branch.
  * [ ] https://github.com/numba/numba/pull/XXXX
* [ ] Review, merge and check execution of release notebook. (FINAL ONLY)
* [ ] Annotated tag X.Y.Z on release branch (no `v` prefix).
* [ ] Build and upload conda packages on buildfarm (check "upload").
* [ ] Build wheels (`$PYTHON_VERSIONS`) on the buildfarm.
* [ ] Upload wheels and sdist to PyPI (upload from `ci_artifacts`).
* [ ] Verify packages uploaded to Anaconda Cloud and move to
  `numba/label/main`.
* [ ] Verify wheels for all platforms arrived on PyPi.
* [ ] Verify ReadTheDocs build.
* [ ] Clean up `ci_artifacts`.
* [ ] Send RC/FINAL announcement email / post announcement to discourse group.
* [ ] Post link to Twitter.
* [ ] Post link to python-announce-list@python.org.

### Post release

* [ ] Update release checklist template.
* [ ] Ping Anaconda Distro team to trigger a build for `defaults` (FINAL ONLY).
* [ ] Create a release on Github at https://github.com/numba/numba/releases (FINAL ONLY).
* [ ] Close milestone (and then close this release issue).
