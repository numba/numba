class Type(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class Array(Type):
    LAYOUTS = frozenset(['C', 'F', 'CS', 'FS', 'A'])

    def __init__(self, dtype, ndim, layout):
        if layout not in self.LAYOUT:
            raise ValueError("Invalid layout '%s'" % layout)
        name = "array(%s, %sd, %s)" % (dtype, ndim, layout)
        super(Array, self).__init__(self, name)
        self.dtype = dtype
        self.ndim = ndim
        self.layout = layout


class Tuple(Type):
    def __init__(self, *elements):
        name = "(%s)" % ', '.join(str(e) for e in elements)
        super(Tuple, self).__init__(name)
        self.elements = elements


pyobject = Type('pyobject')

int32 = Type('int32')
int64 = Type('int64')
intp = int64

float32 = Type('float32')
float64 = Type('float64')

complex64 = Type('complex64')
complex128 = Type('complex128')

range_type = Type('range')