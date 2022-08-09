from __future__ import absolute_import

# NOTE: The following imports are adapted to use as a vendored subpackage.
# from https://github.com/cloudpipe/cloudpickle/blob/d3279a0689b769d5315fc6ff00cd0f5897844526/cloudpickle/init.py
from .cloudpickle import *  # noqa
from .cloudpickle_fast import CloudPickler, dumps, dump  # noqa

# Conform to the convention used by python serialization libraries, which
# expose their Pickler subclass at top-level under the  "Pickler" name.
Pickler = CloudPickler

__version__ = '1.6.0'
