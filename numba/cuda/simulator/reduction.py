from numba.six.moves import reduce as pyreduce

def Reduce(func):
    def reduce_wrapper(seq, init=None):
        # Numba's CUDA reduce allows an empty sequence to be reduced with no
        # initializer but functools.reduce does not.
        if len(seq) == 0 and init == None:
            init = 0
        if init is not None:
            return pyreduce(func, seq, init)
        else:
            return pyreduce(func, seq)
    return reduce_wrapper

reduce = Reduce
