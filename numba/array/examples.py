from __future__ import print_function, absolute_import
import numba.array as numbarray
import numpy as np
from timeit import repeat
from numba import vectorize


def simple_example():

    a1 = numbarray.Array(data=np.arange(100, dtype='f8'))
    a2 = numbarray.Array(data=np.arange(100, dtype='f8'))

    result = a1 + a2 + a1*2 + a2*3

    print(result.__repr__())

    # force eval
    result.eval(debug=True)

    print(result.__repr__())


def reduce_example():

    a1 = numbarray.Array(data=np.arange(10))
    a2 = numbarray.Array(data=np.arange(10))

    result = a1 + a2

    # force eval
    total = numbarray.reduce_(lambda x,y: x+y, result, 0)

    print(total)


def deferred_data_example():
    
    # build array expression with two variable arrays whose data
    # will be specified later
    a1 = numbarray.Array(name='a1')
    a2 = numbarray.Array(name='a2')
    result = a1 + a2

    print(result.__repr__())

    # force eval with concrete data
    print(result.eval(a1=np.arange(10, dtype='f8'), a2=np.arange(10, dtype='f8')))

    print(result.__repr__())
    
    # force eval with different concrete data
    print(result.eval(a1=np.ones(20), a2=np.ones(20)))

    # build array expression with one variable array and one concrete array
    a1 = numbarray.Array(name='a1')
    a2 = numbarray.arange(10)
    result = a1 + a2

    print(result.__repr__())

    # force eval with concrete data
    print(result.eval(a1=np.arange(10, dtype='f8')))
    
    # force eval with different concrete data
    print(result.eval(a1=np.ones(10, dtype='f8')))
    

def python_mode_example():

    a1 = numbarray.Array(data=np.arange(10))
    a2 = numbarray.Array(data=np.arange(10))

    result = a1 + a2

    print(result.eval(use_python=True))


def slice_example():

    a1 = numbarray.Array(data=np.arange(10))
    a2 = numbarray.Array(data=np.arange(10))

    result = a1 + a2
    print(result.__repr__())

    result = result[0:5] + result[5:]
    print(result.eval(debug=True))


def assignment_example():

    a1 = numbarray.Array(data=np.arange(20))
    a2 = numbarray.Array(data=np.arange(10))

    a1 = a1 + a1
    a1[0:10] = a2 * a2

    print(a1)


def where_example():

    a1 = numbarray.Array(data=np.array([1,2,3]))
    a2 = numbarray.Array(data=np.array([4,5,6]))
    cond = numbarray.Array(data=np.array([True,False,True]))

    result = numbarray.where(cond, a1, a2)
    print(result.eval(debug=True))


if __name__ == '__main__':
    simple_example()
    reduce_example()
    deferred_data_example()
    python_mode_example()
    slice_example()
    assignment_example()
    #where_example()
