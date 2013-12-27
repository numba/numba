import itertools


class Type(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class Dummy(Type):
    """
    For type that does not really have a representation.
    """


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

boolean = bool_ = Type('bool')

int32 = Type('int32')
int64 = Type('int64')
intp = int32 if tuple.__itemsize__ == 4 else int64

uint32 = Type('uint32')
uint64 = Type('uint64')

float32 = Type('float32')
float64 = Type('float64')

complex64 = Type('complex64')
complex128 = Type('complex128')

range_type = Dummy('range')
range_state32_type = Type('range_state32')
range_state64_type = Type('range_state64')
range_iter32_type = Type('range_iter32')
range_iter64_type = Type('range_iter64')

signed_domain = frozenset([int32, int64])
unsigned_domain = frozenset([uint32, uint64])
integer_domain = signed_domain | unsigned_domain
real_domain = frozenset([float32, float64])

domains = unsigned_domain, signed_domain, real_domain


class TypeLattice(object):
    def __init__(self):
        self.conns = {}
        self.types = set()

    def connect(self, fromty, toty, weight=1.):
        key = fromty, toty
        self.conns[key] = weight
        self.types.add(fromty)
        self.types.add(toty)

    def build(self):
        alltypes = tuple(self.types)
        pending = []

        # initialize
        for k in itertools.product(alltypes, alltypes):
            a, b = k
            rk = b, a
            if a == b:
                self.conns[k] = 0
            elif k not in self.conns:
                if rk in self.conns:
                    self.conns[k] = -self.conns[rk]
                else:
                    pending.append(k)

        # span first expansion
        while pending:
            before = len(pending)
            tried = []
            for k in pending:
                a, b = k
                rk = b, a
                for t in alltypes:
                    k1 = a, t
                    k2 = t, b
                    if k1 in self.conns and k2 in self.conns:
                        w = self.conns[k1] + self.conns[k2]
                        self.conns[k] = w
                        self.conns[rk] = -w
                        break
                else:
                    tried.append(k)
            pending = tried
            after = len(pending)
            assert after < before, "Not making progress"

        return self.conns


def _build_type_lattice():
    lattice = TypeLattice()
    # Write out all promotion rules
    # int
    lattice.connect(int32, int64)
    # uint
    lattice.connect(uint32, uint64)
    # uint -> int
    lattice.connect(uint32, int32, weight=1.5)
    lattice.connect(uint64, int64, weight=1.5)
    # real
    lattice.connect(float32, float64)
    # complex
    lattice.connect(complex64, complex128)
    # int -> real
    lattice.connect(int32, float32, weight=1.75)
    lattice.connect(int32, float64)
    lattice.connect(int64, float64, weight=1.75)
    # real -> complex
    lattice.connect(float32, complex64)
    lattice.connect(float64, complex128)

    return lattice.build()


type_lattice = _build_type_lattice()


