Numba Mission Statement
=======================

Introduction
------------

This document is the mission statement for the Numba project. It exists to
provide a clear description of the purposes and goals of the project.  As such,
this document provides background on Numba's users and use-cases, and outlines
the project's overall goals.

This is a living document:

=========================== =============
The first revision date is: May 2022
The last updated date is:   May 2022
The next review date is:    November 2022
=========================== =============

Background
----------

The Numba project provides tools to improve the performance of Python software.
It comprises numerous facilities including just-in-time (JIT) compilation,
extension points for library authors, and a compiler toolkit on which new
computational acceleration technologies can be explored and built.

The range of use-cases and applications that can be targeted by Numba includes,
but is not limited to:

* Scientific Computing
* Computationally intensive tasks
* Numerically oriented applications
* Data science utilities and programs

The user base of Numba includes anyone needing to perform intensive
computational work, including users from a wide range of disciplines, examples
include:

* The most common use case, a user wanting to JIT compile some numerical
  functions.
* Users providing JIT accelerated libraries for domain specific use cases e.g.
  scientific researchers.
* Users providing JIT accelerated libraries for use as part of the numerical
  Python ecosystem.
* Those writing more advanced JIT accelerated libraries containing their own
  domain specific data types etc.
* Compiler engineers who explore new compiler use-cases and/or need a custom
  compiler.
* Hardware vendors looking to extend Numba to provide Python support for their
  custom silicon or new hardware.

Project Goals
-------------

The primary aims of the Numba project are:

* To make it easier for Python users to write high performance code.
* To have a core package with a well defined and pragmatically selected feature
  scope that meets the needs of the user base without being overly complex.
* To provide a compiler toolkit for Python that is extensible and can be
  customized to meet the needs of the user base. This comes with the expectation
  that users potentially need to invest time and effort to extend and/or
  customize the software themselves.
* To support both the Python core language/standard libraries and NumPy.
* To consistently produce high quality software:

  * Feature stability across versions.
  * Well established and tested public APIs.
  * Clearly documented deprecation cycles.
  * Internally stable code base.
  * Externally tested release candidates.
  * Regular releases with a predictable and published release cycle.
  * Maintain suitable infrastructure for both testing and releasing. With as
    much in public as feasible.

* To make it as easy as possible for people to contribute.
* To have a maintained public roadmap which will also include areas under active
  development.
* To have a governance document in place and it working in practice.
* To ensure that Numba receives timely updates for its core dependencies: LLVM,
  NumPy and Python.
