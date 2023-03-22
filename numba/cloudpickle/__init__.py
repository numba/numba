from __future__ import absolute_import

# NOTE: The following imports are adapted to use as a vendored subpackage.
# from https://github.com/cloudpipe/cloudpickle/commit/f31859b1dd83fa691f4f7f797166b262c9acb8e7
from .cloudpickle import *  # noqa
from .cloudpickle_fast import CloudPickler, dumps, dump  # noqa

# Conform to the convention used by python serialization libraries, which
# expose their Pickler subclass at top-level under the  "Pickler" name.
Pickler = CloudPickler

__version__ = '2.2.0'
