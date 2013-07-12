__all__ = ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
           'uint64', 'float32', 'float64', 'complex64', 'complex128',
           'arraytype')

class Integer(object):
    def __init__(self, signed, bitwidth):
        self.signed = signed
        self.bitwidth = bitwidth

    def __eq__(self, other):
        if isinstance(other, Integer):
            return (self.signed == other.signed and
                    self.bitwidth == other.bitwidth)

    def __repr__(self):
        if self.signed:
            return 'int%d' % self.bitwidth
        else:
            return 'uint%d' % self.bitwidth


class Signed(Integer):
    def __init__(self, bitwidth):
        super(Signed, self).__init__(True, bitwidth)

    def coerce(self, other):
        if isinstance(other, Signed):
            return (other.bitwidth - self.bitwidth) // 8
        elif isinstance(other, Unsigned):
            return (other.bitwidth - self.bitwidth) // 8 * 2
        elif isinstance(other, Float):
            return (other.bitwidth - self.bitwidth) // 8 + 50
        elif isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8 + 100


class Unsigned(Integer):
    def __init__(self, bitwidth):
        super(Unsigned, self).__init__(False, bitwidth)

    def coerce(self, other):
        if isinstance(other, Unsigned):
            return (other.bitwidth - self.bitwidth) // 8
        elif isinstance(other, Signed):
            return (other.bitwidth - self.bitwidth) // 8 * 2
        elif isinstance(other, Float):
            return (other.bitwidth - self.bitwidth) // 8 + 50
        elif isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8 + 100


class Float(object):
    def __init__(self, bitwidth):
        self.bitwidth = bitwidth

    def coerce(self, other):
        if isinstance(other, Float):
            return (other.bitwidth - self.bitwidth) // 8
        elif isinstance(other, Integer):
            factor = 0 if other.signed else 1
            return (other.bitwidth - self.bitwidth) // 8 - 50 - 1
        elif isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8 + 50

    def __eq__(self, other):
        if isinstance(other, Float):
            return self.bitwidth == other.bitwidth

    def __repr__(self):
        return 'float%d' % self.bitwidth


class Complex(object):
    def __init__(self, bitwidth):
        self.bitwidth = bitwidth

    def coerce(self, other):
        if isinstance(other, Complex):
            return (other.bitwidth - self.bitwidth) // 8

    def __eq__(self, other):
        if isinstance(other, Complex):
            return self.bitwidth == other.bitwidth

    def __repr__(self):
        return 'complex%d' % self.bitwidth


class Array(object):
    def __init__(self, element, shape, layout):
        assert isinstance(shape, tuple)
        assert layout in 'CFA'
        self.shape = shape
        self.element = element
        self.layout = layout

    def coerce(self, other):
        return (self == other)

    def __eq__(self, other):
        return (isinstance(other, Array) and
                self.element == other.element and
                self.shape == other.shape and
                self.layout == other.layout)

    def __repr__(self):
        return 'complex%d' % self.bitwidth


int8  = Signed(8)
int16 = Signed(16)
int32 = Signed(32)
int64 = Signed(64)

uint8  = Unsigned(8)
uint16 = Unsigned(16)
uint32 = Unsigned(32)
uint64 = Unsigned(64)

float32 = Float(32)
float64 = Float(64)

complex64 = Complex(64)
complex128 = Complex(128)

arraytype = Array


