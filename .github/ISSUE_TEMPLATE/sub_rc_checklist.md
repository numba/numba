---
name: Subsequent Release Candidate Checklist (maintainer only)
about: Checklist template for all subsequent releases (RC 2-N, FINAL and PATCH) of every series
title: Numba X.Y.Zrc1 Checklist (FIXME)
labels: task

---


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

* [ ] cherry-pick change-log modifications to main branch (`master`) via PR
* [ ] update release checklist template
* [ ] ping Anaconda Distro team to trigger a build for `defaults` (FINAL ONLY)
* [ ] close milestone (and then close this release issue)
