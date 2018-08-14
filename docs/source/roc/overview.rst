========
Overview
========

Numba supports `AMD ROC GPU <https://rocm.github.io/>`_ programming by directly
compiling a restricted subset of Python code into HSA kernels and device
functions following the HSA execution model. Kernels written in Numba appear to
have direct access to NumPy arrays.

Terminology
===========

Several important terms in the topic of HSA programming are listed here:

- *kernels*: a GPU function launched by the host and executed on the device
- *device function*: a GPU function executed on the device which can only be
  called from the device (i.e. from a kernel or another device function)


Requirements
============

`This document <https://github.com/RadeonOpenCompute/ROCm#are-you-ready-to-rock>`__
describes the requirements for using ROC. Essentially an AMD dGPU is needed
(Fiji, Polaris and Vega families) and a CPU which supports PCIe Gen3 and PCIe
Atomics (AMD Ryzen and EPYC, and Intel CPUs >= Haswell), full details are in the
linked document. Further a linux operating system is needed, those supported and
tested are also listed in the linked document.

Installation
============

Follow `this document <https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories>`__
for installation instructions to enable ROC support for the system.
Be sure to use the binary packages for the system's linux distribution to
simplify the process. At this point the install should be tested by running::

    $ /opt/rocm/bin/rocminfo

the output of which should list at least two HSA Agents, at least one of which
should be a CPU and at least one of which should be a dGPU.

Assuming the installation is working correctly, the ROC support for Numba is
provided by the ``roctools`` package which can be installed via ``conda``, along
with Numba, from the Numba channel as follows (creating an env called
``numba_roc``)::

    $ conda create -n numba_roc -c numba numba roctools

Activating the env, and then running the Numba diagnostic tool should confirm
that Numba is running with ROC support enabled, e.g.::

    $ source activate numba_roc
    $ numba -s

The output of ``numba -s`` should contain a section similar to::

    __ROC Information__
    ROC available                       : True
    Available Toolchains                : librocmlite library, ROC command line tools

    Found 2 HSA Agents:
    Agent id  : 0
        vendor: CPU
        name: Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz
        type: CPU

    Agent id  : 1
        vendor: AMD
        name: gfx803
        type: GPU

    Found 1 discrete GPU(s)             : gfx803

confirming that ROC is available, listing the available toolchains and
displaying the HSA Agents and dGPU count.
