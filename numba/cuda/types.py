from numba.core import types


class Dim3(types.Type):
    def __init__(self):
        super().__init__(name='Dim3')


dim3 = Dim3()
