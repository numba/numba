
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

def vectorize(func):
    try:
	t = Translate(func)
	t.translate()
	print t.mod
	return t.make_ufunc()
    except:
	print "Warning: Could not create fast version..."
	import numpy
	return numpy.vectorize(func)
