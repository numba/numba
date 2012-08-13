
# Create a new callable object
#  that creates a fast version of Python code using LLVM
# It maintains the generic function call for situations
#  where it cannot figure out a fast version, and specializes
#  based on the types that are passed in. 
#  It maintains a dictionary keyed by python code + 
#   argument-types with a tuple of either 
#       (bitcode-file-name, function_name)
#  or (llvm mod object and llvm func object)

class CallSite(object):
    # Must support
    # func = CallSite(func)
    # func = CallSite()(func)
    # func = Callsite(*args, **kwds)(func)
    #  args[0] cannot be callable
    def __init__(self, *args, **kwds):
        # True if this instance is now a function 
        self._isfunc = False
        self._args = args
        if len(args) > 1 and callable(args[0]):
            self._tocall = args[0]
            self._isfunc = True
            self._args = args[1:]

        def __call__(self, *args, **kwds):
            if self._isfunc:
                return self._tocall(*args, **kwds)
            else:
                if len(args) < 1 or not callable(args[0]):
                        raise ValueError, "decorated object must be callable"
                self._tocall = args[0]
                self._isfunc = True
                return self

# A simple fast-vectorize example was removed because it only supports one
#  use-case --- slower NumPy vectorize is included here instead.
#  The required code is still in _ext.c which is not compiled by default
#   and here is the decorator:
#def vectorize(func):
#    global __tr_map__
#    try:
#        if func not in __tr_map__:
#            t = Translate(func)
#            t.translate()
#            __tr_map__[func] = t
#        else:
#            t = __tr_map__[func]
#        return t.make_ufunc()
#    except Exception as msg:
#        print "Warning: Could not create fast version...", msg
#        import traceback
#        traceback.print_exc()
#        import numpy
#        return numpy.vectorize(func)

from numpy import vectorize

from translate import Translate

# The __tr_map__ global maps from Python functions to a Translate
# object.  This added reference prevents the translator and its
# generated LLVM code from being garbage collected when we leave the
# scope of a decorator.

# See: https://github.com/ContinuumIO/numba/issues/5

__tr_map__ = {}

def jit(*args, **kws):
    def _jit(func):
        global __tr_map__
        llvm = kws.pop('llvm', True)
        if func in __tr_map__:
            print("Warning: Previously compiled version of %r may be "
                  "garbage collected!" % (func,))
        t = Translate(func, *args, **kws)
        t.translate()
        __tr_map__[func] = t
        return t.get_ctypes_func(llvm)
    return _jit

