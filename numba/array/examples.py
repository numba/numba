from array import Array, abs_, add, reduce_
import numpy as np


def simple_example():

    a1 = Array(data=np.arange(10))
    a2 = Array(data=np.arange(10))
    a3 = Array(data=np.arange(10))

    result = a1 + a2 + a3

    print result.__repr__()

    # force eval
    print result

    print result.__repr__()


def reduce_example():

    a1 = Array(data=np.arange(10))
    a2 = Array(data=np.arange(10))

    result = a1 + a2

    # force eval
    total = add.reduce(result)

    print total


'''def deferred_data_example():

    a1 = Array(name='a1')
    a2 = Array(name='a2')

    result = a1 + a2

    print result.__repr__()

    # force eval with concrete data
    print result(a1=np.arange(10, dtype='i8'), a2=np.arange(10, dtype='i8'))
    
    # attach to concrete data
    result2 = result(a1=np.arange(10, dtype='f8'), a2=np.arange(10, dtype='f8'))

    # force eval
    print result2 + a1'''


def python_mode_example():

    a1 = Array(data=np.arange(10))
    a2 = Array(data=np.arange(10))

    result = a1 + a2

    print result.eval(python=True)


if __name__ == '__main__':
    simple_example()
    reduce_example()
    #deferred_data_example
    #python_mode_example()

