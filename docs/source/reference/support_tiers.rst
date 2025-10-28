.. _support_tiers:

Support Policy
==============

This section aims to answer questions like:

* Which OS/hardware/platform/package format(s) are supported?
* Why is ``X`` supported and not ``Y``?
* What is needed for ``Z`` to be supported?


Definitions:
------------

The following terms are defined (with examples):

* hardware target: the target hardware e.g. ``x86_64`` or ``aarch64``
* operating system: the operating system on which the code will run, note that
  this may also include emulation or pseudo-emulation. e.g. Windows subsystem
  for Linux or OSX Rosetta.
* packaging system: a package building system, that which takes source code and
  turns it into a package for distribution. e.g. ``conda build``, ``pip wheel``
  or ``rpmbuild``.
* package: an artefact produced by the package system suitable for use in the
  distribution system. e.g. instances of a wheel, an rpm or a source tarball.
* package type: the type of a package. e.g. wheel, conda, source.
* distribution system: the system for distributing packages to users e.g. the
  PyPI package index, Anaconda repositories, a Linux distribution.
* release: shipped binary artefacts present on a distribution system that have
  been build from a specific git tag for a specific operating system and
  hardware target. A release comprises a tuple (project, hardware target,
  operating system, package type, distribution system). e.g. (Numba project,
  ``aarch64``, Linux, wheel, PyPI) or (llvmlite project, ``x86_64``, Windows,
  conda, Anaconda repositories).
* source release: shipped source artefacts present on a distribution system that
  have been built from a specific git tag. A source release comprises a tuple
  (project, package type, distribution system) e.g. (Numba project, tarball,
  GitHub)
* CI/CD: continuous integration and delivery for the provision of releases.
  e.g. a release performed using Github Actions
* Numba maintainers/maintainers: The maintainers of the projects under the
  GitHub Numba organisation.


Support Tiers:
--------------

Support is split into tiers.


Tier 1: Releases that are maintained by the Numba maintainers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guarantees for Tier 1:

* The Numba maintainers will maintain public CI/CD and release the software.
* Patches that break CI/CD will not be accepted.
* Release candidates will be produced for major changes. There will be minimum
  period of two weeks of testing available for the first release candidate of
  any release.

Conditions of being in Tier 1:

* A conda based Python distribution for the operating system and hardware target
  is required. This is because the public CI/CD for the Numba stack is driven by
  conda and conda packages (even for wheel builds). The purpose of this
  condition is to reduce the overall CI/CD maintenance burden.
* The operating system and hardware target must be supported by GitHub Actions
  on the free-tier.
* The release package type must be conda or wheel.
* The distribution system must be PyPI or Anaconda repositories.


Tier 1 status is also contingent upon the continued provision of support and
suitable releases from core dependencies:

* Python
* NumPy
* LLVM

If a core dependency stops supporting, deprecates or fails to provide a suitable
release then the platform automatically transitions to Tier 2.

Currently supported Tier 1 releases for the Numba and llvmlite projects:

* Conda packages on the Anaconda dot org distribution system released to the
  ``numba`` channel (using conda nomenclature, with clarifications in
  brackets):

  * ``osx-arm64`` (OSX on Apple silicon)
  * ``linux-64`` (Linux on ``x86_64``)
  * ``linux-aarch64`` (Linux on ``aarch64``)
  * ``win-64`` (Windows on ``x86_64``)

* Wheel packages on the PyPI distribution system  (using conda nomenclature,
  with clarifications in brackets):

  * ``osx-arm64`` (OSX on Apple silicon)
  * ``linux-64`` (Linux on ``x86_64``)
  * ``linux-aarch64`` (Linux on ``aarch64``)
  * ``win-64`` (Windows on ``x86_64``)

Along with the above releases, two further Tier 1 releases exist for the Numba
and llvmlite projects so as to help with Python ecosystem support, Numba
maintainers either maintain or help to maintain these releases.

* Conda packages on the Anaconda dot org distribution system released as part of
  the Anaconda distribution for Python (using conda nomenclature, with
  clarifications in brackets):

  * ``osx-arm64`` (OSX on Apple silicon)
  * ``linux-64`` (Linux on ``x86_64``)
  * ``linux-aarch64`` (Linux on ``aarch64``)
  * ``win-64`` (Windows on ``x86_64``)

* Conda packages on the Anaconda dot org distribution system released as part of
  the Conda-Forge distribution for Python (using conda nomenclature, with
  clarifications in brackets):

  * ``osx-arm64`` (OSX on Apple silicon)
  * ``linux-64`` (Linux on ``x86_64``)
  * ``linux-aarch64`` (Linux on ``aarch64``)
  * ``win-64`` (Windows on ``x86_64``)

Tier 2a: Releases that are built by large software distributors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tier 1 releases should avoid breaking releases built by large software
distributors but this is a non-blocking condition for Tier 1 releases. The Numba
maintainers will offer support and advice to help ensure that any breakages are
fixed. Patches associated with maintenance of Tier 2a releases will be accepted
so long as, in the view of the maintainers, they do not add a significant
maintenance burden. CI/CD and distribution systems are provided externally and
are not maintained by the Numba maintainers. The following are examples of Tier
2a releases:

* Linux distributions:

  * RHEL/Fedora/Rocky
  * Debian/Ubuntu

* BSD distributions:

  * FreeBSD
  * NetBSD
  * OpenBSD
  * DragonFly BSD


Tier 2b: General community support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A "best effort" level of support will be supplied to assist with any of the
categories listed in the "definitions" above. Patches to the project source code
and build system will be accepted so long as, in the view of the maintainers,
they do not add a significant maintenance burden or result in large amounts of
code that cannot be tested. CI/CD and distribution, if any, are provided by
systems external to the Numba project and are not maintained by the Numba
maintainers.

Examples include:

* Conda and wheel packages not listed in Tier 1
* Hardware targets:

  * ``s390x``
  * ``ppc64le``
  * ``RISC-V``

* Operating system/hardware target:

  * Windows on ARM.


How to move Tiers?
------------------

A release can be moved from Tier 1 to Tier 2 if any of the conditions of being
in Tier 1 are broken. Equally, the reverse applied, should a Tier 2 target meet
the conditions to be in Tier 1 then it can be moved. A move between Tiers is
only accepted via a proposal mode in GitHub issue on the project and subsequent
discussion and approval by the maintainers (for example following discussion in
a maintainer or public meeting). Ultimately, the Numba maintainers have the
final say over what is in Tier 1 as they are long term committed to the project
and carry the on going maintenance burden.


Additional Accelerator and Special Hardware support
---------------------------------------------------

Accelerator support is evaluated independently from release Tiers. GPU or
accelerator support requires:

* Vendor tool chain availability on the host hardware target and operating
  system.
* Maintainer expertise or strong community contribution.
* Availability of appropriate testing infrastructure.
