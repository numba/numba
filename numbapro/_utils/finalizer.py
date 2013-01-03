'''
Modified C-level finalizer by Benjamin Peterson <benjamin@python.org>
Available at http://code.activestate.com/recipes/577242-calling-c-level-finalizers-without-__del__/
'''
import sys
import traceback
import weakref

class OwnerRef(weakref.ref):
    """A simple weakref.ref subclass, so attributes can be added."""
    pass

def _run_finalizer(ref):
    """Internal weakref callback to run finalizers"""
    del _finalize_refs[ref.owner]
    for item, finalizer in ref.items:
        try:
            finalizer(item)
        except Exception:
            print>>sys.stderr, "Exception running {}:".format(finalizer)
            traceback.print_exc()

_finalize_refs = {}

def track(owner, item, finalizer):
    """Register an object for finalization.

        ``owner`` is the the object which is responsible for ``item``.
        ``finalizer`` will be called with ``item`` as its only argument when
        ``owner`` is destroyed by the garbage collector.
        """
    if id(owner) in _finalize_refs:
        ref = _finalize_refs[id(owner)]
    else:
        ref = OwnerRef(owner, _run_finalizer)
        ref.owner = id(owner)
        ref.items = []
    ref.items.append((item, finalizer))
    _finalize_refs[id(owner)] = ref

class OwnerMixin(object):
    def _finalizer_track(self, item):
        if not hasattr(self, '_finalize'):
            raise AttributeError("%s must define a _finalize method" % self)
        track(self, item, type(self)._finalize)
