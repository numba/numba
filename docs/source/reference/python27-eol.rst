===========================
Python 2.7 End of Life Plan
===========================

As per `PEP 373 <http://legacy.python.org/dev/peps/pep-0373/>`_, Python 2.7
`will cease to be supported in 2020 <https://pythonclock.org/>`_.  Like many
projects, the Numba team has to consider how to time its own end of Python 2.7
support.  Given how deeply Numba must interface with the Python interpreter,
supporting both Python 2 and 3 creates quite a development and testing burden.
In addition, Numba (specifically via llvmlite) has to deal with some tricky
compiler issues on Windows, where LLVM requires Visual Studio 2015 or later,
but Python 2.7 extensions must be built with Visual Studio 2008.  Needless to
say, the goal with this plan is to support our Python 2.7 user base (~13% of 
conda package downloads and 25-30% of PyPI downloads as of July 2019), but
also clearly signal that *now is the time to switch to Python 3 if you have
not already*.

Python 2.7 users of Numba should also be aware of `NumPy's timeline for ending Python 2.7 support
<https://github.com/numpy/numpy/blob/067cb067cb17a20422e51da908920a4fbb3ab851/doc/neps/nep-0014-dropping-python2.7-proposal.rst>`_.
Due to Numba's tight coupling with NumPy, the NumPy timeline has 
informed the Numba timeline below.

Timeline
========

This timeline was revised in August 2019 to reflect the reality that we wanted
to get more improvements into Numba before ending Python 2.7 support.  Note
that the Numba versions mentioned below are more concrete than the dates.

The end of Python 2.7 support in Numba will be staged:

* **Numba 0.47**: (~Dec 2019) This is will be the last version of Numba to
  support both Python 2 and 3.
* **Numba 0.48 development cycle**: (Jan-Mar 2020) Python 2 support will be 
  removed from ``master`` branch. If any critical bug fixes are discovered in
  the 0.47.0 release, they will be backported to create 0.47.x patch releases.
* **Numba 0.48**: (~Mar 2020) This will be the first release of Numba that is
  Python 3 only.  We will set the metadata on the package to require Python
  3.5 or later.
* After Feb 2020, the Numba core developers will stop doing patch releases to
  the 0.47.x series for Python 2.

If there are concerns about the above timeline, please
`raise an issue <https://github.com/numba/numba/issues>`_ in our issue
tracker.