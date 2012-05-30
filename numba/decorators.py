
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

# A simple fast-vectorize example

from translate import Translate

# The __tr_map__ global maps from Python functions to a Translate
# object.  This added reference prevents the translator and its
# generated LLVM code from being garbage collected when we leave the
# scope of a decorator.

# See: https://github.com/ContinuumIO/numba/issues/5

__tr_map__ = {}

def vectorize(func):
    global __tr_map__
    try:
        if func not in __tr_map__:
            t = Translate(func)
            t.translate()
            __tr_map__[func] = t
        else:
            t = __tr_map__[func]
        return t.make_ufunc()
    except:
	print "Warning: Could not create fast version..."
	import numpy
	return numpy.vectorize(func)

# XXX Proposed name; compile() would mask builtin of same name.
def numba_compile(*args, **kws):
    def _numba_compile(func):
        global __tr_map__
        llvm = kws.pop('llvm', True)
        if func not in __tr_map__:
            t = Translate(func, *args, **kws)
            t.translate()
            __tr_map__[func] = t
        else:
            t = __tr_map__[func]
        return t.get_ctypes_func(llvm)
    return _numba_compile

from kerneltranslate import Translate as KernelTranslate

def numba_kompile(*args, **kws):
    def _numba_kompile(func):
        global __tr_map__
        llvm = kws.pop('llvm', True)
        if func not in __tr_map__:
            t = KernelTranslate(func)
            t.translate()
            __tr_map__[func] = t
        else:
            t = __tr_map__[func]
        return t.get_ctypes_func(llvm)
    return _numba_kompile
