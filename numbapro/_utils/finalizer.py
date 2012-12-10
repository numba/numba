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
    del _finalize_refs[id(ref)]
    finalizer = ref.finalizer
    item = ref.item
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
    ref = OwnerRef(owner, _run_finalizer)
    ref.owner = weakref.ref(owner)
    ref.item = item
    ref.finalizer = finalizer
    _finalize_refs[id(ref)] = ref

class OwnerMixin(object):
    def _finalizer_track(self, item):
        if not hasattr(self, '_finalize'):
            raise AttributeError("%s must define a _finalize method" % self)
        track(self, item, type(self)._finalize)
