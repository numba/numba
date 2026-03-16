---
name: Subsequent Release Candidate Checklist (maintainer only)
about: Checklist template for all subsequent releases (RC 2-N, FINAL and PATCH) of every series
title: Numba X.Y.Z Checklist (FIXME)
labels: task

---


## numba X.Y.Z

* [ ] Cherry-pick items from the X.Y.Z milestone into a cherry-pick PR.
* [ ] Update the "version support table" in the documentation with the final
  release date (FINAL ONLY) and add to cherry-pick PR.
* [ ] Update `CHANGE_LOG` on cherry-pick PR.
* [ ] Check if any dependency pinnings need an update (e.g. NumPy).
* [ ] Approve cherry-pick PR.
* [ ] Merge cherry-pick PR to X.Y release branch.
  * [ ] https://github.com/numba/numba/pull/XXXX
* [ ] Run the HEAD of the release branch through GHA and confirm:
  * [ ] conda build and test has passed.
  * [ ] wheel build and test has passed.
* [ ] Annotated tag X.Y.Z on release branch (no `v` prefix).
  `git tag -am "Version X.Y.Z" X.Y.Z`
* [ ] Build conda packages and wheels on GHA.
* [ ] Using the upload workflow, upload the conda packages and wheels and confirm they have arrived.
* [ ] Verify ReadTheDocs build.
* [ ] Create a release on Github at https://github.com/numba/numba/releases (FINAL ONLY).
* [ ] Post announcement to discourse group and ping the release testers group
  using `@RC_Testers` (RC ONLY).

### Post release

* [ ] Cherry-pick change-log and version support table modifications to `main`.
* [ ] Update release checklist template with any additional bullet points that
      may have arisen during the release.
* [ ] Ping Anaconda Distro team to trigger a build for `defaults` (FINAL ONLY).
* [ ] Close milestone (and then close this release issue).
