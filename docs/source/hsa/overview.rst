========
Overview
========

Numba supports HSA APU programming by directly compiling a restricted subset
of Python code into HSA kernels and device functions following the HSA
execution model.  Kernels written in Numba appear to have direct access
to NumPy arrays.

Terminology
===========

Several important terms in the topic of HSA programming are listed here:

- *kernels*: a GPU function launched by the host and executed on the device
- *device function*: a GPU function executed on the device which can only be
  called from the device (i.e. from a kernel or another device function)


Requirements
============

This is a preview of the HSA feature.  We only support Kavari on 64-bit
Ubuntu at this time.  Please consult offical documentation at
`this documentation <https://github.com/HSAFoundation/HSA-Docs-AMD/wiki/HSA-Platforms-&-Installation>`_.
about system requirement.


Installation
============

Follow `this document <https://github.com/HSAFoundation/HSA-Docs-AMD/wiki/HSA-Platforms-&-Installation#installation-overview>`_
for installation instructions to enable HSA support for the system.
Be sure to use the ``.deb`` packages to simplify the process.
Aftwards, the following libraries must be added to your ``LD_LIBRARY_PATH``:

* libhsakmt.so.1
* libhsa-runtime64.so
* libhsa-runtime-ext64.so


``libhsa-runtime64.so`` and ``libhsa-runtime-ext64.so`` are in ``/opt/hsa/lib``
``libhsakmt.so.1`` has no default location and is available from
https://github.com/HSAFoundation/HSA-Drivers-Linux-AMD

The current implementation uses the stable LLVM compiler from AMD.
To install, use ``.deb`` file from https://github.com/HSAFoundation/HSAIL-HLC-Stable
This will install the binaries to ``/opt/amd/bin``, which is expected by Numba.



