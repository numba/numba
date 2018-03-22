===========================
Python 2.7 End of Life Plan
===========================

As per `PEP 373 <http://legacy.python.org/dev/peps/pep-0373/>`_, Python 2.7 will cease to be supported in 2020, though `no exact date has been officially selected yet <https://pythonclock.org/>`_.  Like many projects, the Numba team has to consider how to time its own end of Python 2.7 support.  Given how deeply Numba must interface with the Python interpreter, supporting both Python 2 and 3 creates quite a development and testing burden.  In addition, Numba (specifically via llvmlite) has to deal with some specifically tricky compiler issues on Windows, where LLVM requires Visual Studio 2015 or later, but Python 2.7 extensions must be built with Visual Studio 2008.  Needless to say, the goal with this plan is to support our Python 2.7 user base (~30% of downloads as of February 2018), but also clearly signal that *now is the time to switch to Python 3 if you have not already*.

Python 2.7 users of Numba should also be aware of `NumPy's timeline for ending Python 2.7 support <https://github.com/numpy/numpy/blob/master/doc/neps/nep-0014-dropping-python2.7-proposal.rst>`_.  Due to Numba's tight coupling with NumPy, the NumPy timeline has strongly informed the Numba timeline below.

Timeline
========

The end of Python 2.7 support in Numba will be staged:

* **December 2018**: Tag and release Numba 1.x.0.  Create a Python 2.7 branch based on this release.  
* Critical fixes until **January 1, 2020** will be backported to the Python 2.7 branch and released as Numba 1.x.y.
* No new features will be added to the Python 2.7 branch, but we will continue to do automated testing of it with new NumPy releases.
* **January 1, 2019**: we will slim down the Numba master branch by removing all the Python 2.7 compatibility code and release Numba 1.(x+1).0, which will be functionally identical to Numba 1.x.0.
* **January 1, 2020**: The Numba developers will stop supporting the Python 2.7 branch.

If there are concerns about the above timeline, please `raise an issue <https://github.com/numba/numba/issues>`_ in our issue tracker.