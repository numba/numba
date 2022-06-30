---
name: Subsequent Release Candidate Checklist (maintainer only)
about: Checklist template for all subsequent releases (RC 2-N, FINAL and PATCH) of every series
title: Numba X.Y.Zrc1 Checklist (FIXME)
labels: task

---


## numba X.Y.Z

* [ ] Cherry-pick items from the X.Y.Z milestone into a PR.
* [ ] Update the "version support table" in the documentation with the final
  release date (FINAL ONLY).
  * [ ] Make, approve and merge PR against the X.Y release branch.
  * [ ] Create a cherry-pick PR of the above against `main`.
* [ ] Approve change log modifications and cherry-pick.
* [ ] Merge change log modifications and cherry-picks to X.Y release branch.
  * [ ] https://github.com/numba/numba/pull/XXXX
* [ ] Review, merge and check execution of release notebook. (FINAL ONLY)
* [ ] Annotated tag X.Y.Z on release branch (no `v` prefix).
* [ ] Build and upload conda packages on buildfarm (check `upload`).
* [ ] Build wheels and sdist on the buildfarm (check "upload").
* [ ] Verify packages uploaded to Anaconda Cloud and move to
  `numba/label/main`.
* [ ] Upload wheels and sdist to PyPI (upload from `ci_artifacts`).
* [ ] Verify wheels for all platforms arrived on PyPi.
* [ ] Verify ReadTheDocs build.
* [ ] Send RC/FINAL announcement email / post announcement to discourse group.
* [ ] Post link to Twitter.
* [ ] Post link to python-announce-list@python.org.

### Post release

* [ ] Snapshot Build Farm config
* [ ] Clean up `ci_artifacts` by moving files to subdirectories
* [ ] Update release checklist template with any additional bullet points that
      may have arisen during the release.
* [ ] Ping Anaconda Distro team to trigger a build for `defaults` (FINAL ONLY).
* [ ] Create a release on Github at https://github.com/numba/numba/releases (FINAL ONLY).
* [ ] Close milestone (and then close this release issue).
