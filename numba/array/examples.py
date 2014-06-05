import numba.array as numbarray
import numpy as np
from timeit import repeat
from numba import vectorize


def simple_example():

    a1 = numbarray.Array(data=np.arange(10000000, dtype='f8'))
    a2 = numbarray.Array(data=np.arange(10000000, dtype='f8'))

    result = a1 + a2 + a1*2 + a2*3

    print result.__repr__()

    # force eval
    result.eval(debug=True)

    print result.__repr__()


def reduce_example():

    a1 = numbarray.Array(data=np.arange(10))
    a2 = numbarray.Array(data=np.arange(10))

    result = a1 + a2

    # force eval
    total = numbarray.reduce_(lambda x,y: x+y, result, 0)

    print total


def deferred_data_example():

    a1 = numbarray.Array(name='a1')
    a2 = numbarray.Array(name='a2')

    result = a1 + a2

    print result.__repr__()

    # force eval with concrete data
    print result.eval(a1=np.arange(10, dtype='i8'), a2=np.arange(10, dtype='i8'))
    
    # force eval with different concrete data
    print result.eval(a1=np.arange(20, dtype='f8'), a2=np.arange(20, dtype='f8'))
    

def python_mode_example():

    a1 = numbarray.Array(data=np.arange(10))
    a2 = numbarray.Array(data=np.arange(10))

    result = a1 + a2

    print result.eval(python=True)


def slice_example():

    a1 = numbarray.Array(data=np.arange(10))
    a2 = numbarray.Array(data=np.arange(10))

    result = a1 + a2
    print result.__repr__()

    result = result[0:5] + result[5:]
    print result.eval(debug=False)


def assignment_example():

    a1 = numbarray.Array(data=np.arange(20))
    a2 = numbarray.Array(data=np.arange(10))

    a1 = a1 + a1
    a1[0:10] = a2 * a2

    print a1


if __name__ == '__main__':
    simple_example()
    #reduce_example()
    #deferred_data_example()
    #python_mode_example()
    #slice_example()
    #assignment_example()
