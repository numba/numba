---
name: Subsequent Release Candidate Checklist (maintainer only)
about: Checklist template for all subsequent releases (RC 2-N, FINAL and PATCH) of every series
title: Numba X.Y.Zrc1 Checklist (FIXME)
labels: task

---


## numba X.Y.Z

* [ ] Cherry-pick items from the X.Y.Z milestone into a cherry-pick PR.
* [ ] Update the "version support table" in the documentation with the final
  release date (FINAL ONLY) and add to cherry-pick PR
* [ ] Update `CHANGE_LOG` on cherry-pick PR
* [ ] Check if any dependency pinnings need an update (e.g. NumPy)
* [ ] Approve cherry-pick PR
* [ ] Merge cherry-pick PR to X.Y release branch.
  * [ ] https://github.com/numba/numba/pull/XXXX
* [ ] Review, merge and check execution of release notebook. (FINAL ONLY)
* [ ] Run the HEAD of the release branch through the build farm and confirm:
  * [ ] Build farm CPU testing has passed.
  * [ ] Build farm CUDA testing has passed
  * [ ] Build farm wheel testing has passed
* [ ] Annotated tag X.Y.Z on release branch (no `v` prefix).
  `git tag -am "Version X.Y.Z" X.Y.Z`
* [ ] Build and upload conda packages on buildfarm (check `upload`).
* [ ] Build wheels and sdist on the buildfarm (check "upload").
* [ ] Verify packages uploaded to Anaconda Cloud and move to
  `numba/label/main`.
* [ ] Upload wheels and sdist to PyPI (upload from `ci_artifacts`).
* [ ] Verify wheels for all platforms arrived on PyPi.
* [ ] Verify ReadTheDocs build.
* [ ] Create a release on Github at https://github.com/numba/numba/releases (FINAL ONLY).
* [ ] Post link to X and to Mastodon and...
* [ ] Post announcement to discourse group and ping the release testers group
  using `@RC_Testers` (RC ONLY).
* [ ] Post link to python-announce-list@python.org.

### Post release

* [ ] Cherry-pick change-log and version support table modifications to `main`
* [ ] Snapshot Build Farm config
* [ ] Clean up `ci_artifacts` by moving files to subdirectories
* [ ] Update release checklist template with any additional bullet points that
      may have arisen during the release.
* [ ] Ping Anaconda Distro team to trigger a build for `defaults` (FINAL ONLY).
* [ ] Close milestone (and then close this release issue).
