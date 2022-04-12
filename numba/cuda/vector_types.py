# CUDA built-in Vector Types
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types

from typing import List, Tuple

from numba import types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.extending import make_attribute_wrapper, models, register_model
from numba.core.imputils import Registry as ImplRegistry
from numba.core.typing.templates import ConcreteTemplate
from numba.core.typing.templates import Registry as TypingRegistry
from numba.core.typing.templates import signature
from numba.cuda import stubs
from numba.cuda.errors import CudaLoweringError

typing_registry = TypingRegistry()
impl_registry = ImplRegistry()

register = typing_registry.register
register_attr = typing_registry.register_attr
register_global = typing_registry.register_global
lower = impl_registry.lower


def once(func):
    def wrapper(*args, **kwargs):
        if not hasattr(func, "_has_run"):
            func._has_run = True
            return func(*args, **kwargs)
    return wrapper


# TODO: double check the type mapping is correct for all platforms.
# TODO: see numba.core.types.__init__.py, create aliases such as int32x4=int4?
_vector_type_to_base_types = {
    "char": types.char,
    "short": types.int16,
    "int": types.int32,
    "long": types.int32,
    "longlong": types.int64,
    "uchar": types.uchar,
    "ushort": types.uint16,
    "uint": types.uint32,
    "ulong": types.uint32,
    "ulonglong": types.uint64,
    "float": types.float32,
    "double": types.float64
}


class VectorType(types.Type):
    def __init__(self, name, base_type, attr_names, user_facing_object):
        self._base_type = base_type
        self._attr_names = attr_names
        self._user_facing_object = user_facing_object
        super().__init__(name=name)

    @property
    def base_type(self):
        return self._base_type

    @property
    def attr_names(self):
        return self._attr_names

    @property
    def num_elements(self):
        return len(self._attr_names)

    @property
    def user_facing_object(self):
        return self._user_facing_object


def make_vector_type(
    name: str, base_type: types.Type, attr_names: List[str], user_facing_object
) -> types.Type:
    """Create a vector type.

    Parameters
    ----------
    name: str
        The name of the type.
    base_type: numba.types.Type
        The primitive type for each element in the vector.
    attr_names: list of str
        Name for each attribute.
    """

    class _VectorType(VectorType):
        """Internal instantiation of VectorType."""

        pass

    class VectorTypeModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [(attr_name, base_type) for attr_name in attr_names]
            super().__init__(dmm, fe_type, members)

    vector_type = _VectorType(name, base_type, attr_names, user_facing_object)
    register_model(_VectorType)(VectorTypeModel)
    for attr_name in attr_names:
        make_attribute_wrapper(_VectorType, attr_name, attr_name)

    return vector_type


def enable_vector_type_ctor(
    vector_type: VectorType, overloads: List[Tuple[types.Type]]
):
    """Create typing and lowering for vector type constructor.

    Parameters
    ----------
    vector_type: VectorType
        The type whose constructor to type and lower.
    overloads: List of argument types tuples
        A list containing different overloads of the factory function. Each
        base type in the tuple should either be primitive type or VectorType.
    """
    ctor = vector_type.user_facing_object

    @register
    class CtorTemplate(ConcreteTemplate):
        key = ctor
        cases = [signature(vector_type, *arglist) for arglist in overloads]

    register_global(ctor, types.Function(CtorTemplate))

    # Lowering

    def make_lower_factory(fml_arg_list):
        """Meta function to create a lowering for the factory function. Flattens
        the arguments by converting vector_type into load instructions for each
        of its attributes. Such as float2 -> float2.x, float2.y.
        """

        def lower_factory(context, builder, sig, actual_args):
            # A list of elements to assign from
            source_list = []
            # Convert the list of argument types to a list of load IRs.
            for argidx, fml_arg in enumerate(fml_arg_list):
                if isinstance(fml_arg, VectorType):
                    pxy = cgutils.create_struct_proxy(fml_arg)(
                        context, builder, actual_args[argidx]
                    )
                    source_list += [
                        getattr(pxy, attr) for attr in fml_arg.attr_names
                    ]
                else:
                    # assumed primitive type
                    source_list.append(actual_args[argidx])

            if len(source_list) != vector_type.num_elements:
                raise CudaLoweringError(
                    f"Unmatched number of source elements ({len(source_list)}) "
                    "and target elements ({vector_type.num_elements})."
                )

            out = cgutils.create_struct_proxy(vector_type)(context, builder)

            for attr_name, source in zip(vector_type.attr_names, source_list):
                setattr(out, attr_name, source)
            return out._getvalue()

        return lower_factory

    for arglist in overloads:
        lower_factory = make_lower_factory(arglist)
        lower(ctor, *arglist)(lower_factory)


def lower_vector_type_binops(
    binop, vector_type: VectorType, overloads: List[Tuple[types.Type]]
):
    """Lower binops for ``vector_type``

    Parameters
    ----------
    binop: operation
        The binop to lower
    vector_type: VectorType
        The type to lower op for.
    overloads: List of argument types tuples
        A list containing different overloads of the binop. Should be one of:
            - vector_type x vector_type
            - primitive_type x vector_type
            - vector_type x primitive_type.
        If one of the oprand is primitive_type, the operation is broadcasted.
    """
    # Should we assume the above are the only possible cases?
    class Vector_op_template(ConcreteTemplate):
        key = binop
        cases = [signature(vector_type, *arglist) for arglist in overloads]

    def make_lower_op(fml_arg_list):
        def op_impl(context, builder, sig, actual_args):
            def _make_load_IR(typ, actual_arg):
                if isinstance(typ, VectorType):
                    pxy = cgutils.create_struct_proxy(typ)(
                        context, builder, actual_arg
                    )
                    oprands = [getattr(pxy, attr) for attr in typ.attr_names]
                else:
                    # Assumed primitive type, broadcast
                    oprands = [
                        actual_arg for _ in range(vector_type.num_elements)
                    ]
                return oprands

            def element_wise_op(lhs, rhs, res, attr):
                setattr(
                    res,
                    attr,
                    context.compile_internal(
                        builder,
                        lambda x, y: binop(x, y),
                        signature(types.float32, types.float32, types.float32),
                        (lhs, rhs),
                    ),
                )

            lhs_typ, rhs_typ = fml_arg_list
            # Construct a list of load IRs
            lhs = _make_load_IR(lhs_typ, actual_args[0])
            rhs = _make_load_IR(rhs_typ, actual_args[1])

            if not len(lhs) == len(rhs) == vector_type.num_elements:
                raise TypingError(
                    f"Unmatched number of lhs elements ({len(lhs)}), rhs "
                    f"elements ({len(rhs)}) and target elements "
                    f"({vector_type.num_elements})."
                )

            out = cgutils.create_struct_proxy(vector_type)(context, builder)
            for attr, l, r in zip(vector_type.attr_names, lhs, rhs):
                element_wise_op(l, r, out, attr)

            return out._getvalue()

        return op_impl

    register_global(binop, types.Function(Vector_op_template))
    for arglist in overloads:
        impl = make_lower_op(arglist)
        lower(binop, *arglist)(impl)


def vector_type_ctor_overloads(vty: VectorType) -> List[Tuple[types.Type]]:
    return [(vty.base_type,) * vty.num_elements]


vector_types : List[VectorType] = []


@once
def initialize_once():
    for stub in stubs._vector_type_stubs:
        type_name = stub.__name__
        base_type = _vector_type_to_base_types[type_name[:-1]]
        num_elements = int(type_name[-1])
        attributes = stubs._vector_type_attribute_names[:num_elements]
        vector_type = make_vector_type(type_name, base_type, attributes, stub)
        vector_types.append(vector_type)

    for vty in vector_types:
        enable_vector_type_ctor(vty, vector_type_ctor_overloads(vty))
