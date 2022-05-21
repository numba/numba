from numba import types
from numba.cuda.stubs import _vector_type_stubs


class SimulatedVectorType:
    attributes = ['x', 'y', 'z', 'w']

    def __init__(self, *args):
        args_flattened = []
        for arg in args:
            if isinstance(arg, SimulatedVectorType):
                args_flattened += arg.as_list()
            else:
                args_flattened.append(arg)
        self._attrs = self.attributes[:len(args_flattened)]
        assert len(self._attrs) == len(args_flattened), \
            f"{len(self._attrs)} != {len(args_flattened)}"

        for arg, attr in zip(args_flattened, self._attrs):
            setattr(self, attr, arg)

    def as_list(self):
        return [getattr(self, attr) for attr in self._attrs]


def make_simulated_vector_type(num_elements, name):
    obj = type(name, (SimulatedVectorType,), {
        "num_elements": num_elements,
        "base_type": types.float32,
        "name": name
    })
    obj.user_facing_object = obj
    return obj


def _initialize():
    _simulated_vector_types = {}
    for stub in _vector_type_stubs:
        num_elements = int(stub.__name__[-1])
        _simulated_vector_types[stub.__name__] = (
            make_simulated_vector_type(num_elements, stub.__name__)
        )
        for alias in stub.aliases:
            _simulated_vector_types[alias] = (
                make_simulated_vector_type(num_elements, alias)
            )
    return _simulated_vector_types


_simulated_vector_types = _initialize()
