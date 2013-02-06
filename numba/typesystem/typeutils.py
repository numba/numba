from numba.typesystem import *

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

def is_obj(type):
    return type.is_object or type.is_array

native_type_dict = {}
for native_type in minitypes.native_integral:
    native_type_dict[(native_type.itemsize, native_type.signed)] = native_type

def promote_to_native(int_type):
    return native_type_dict[int_type.itemsize, int_type.signed]

def promote_closest(context, int_type, candidates):
    """
    promote_closest(Py_ssize_t, [int_, long_, longlong]) -> longlong
    """
    for candidate in candidates:
        promoted = context.promote_types(int_type, candidate)
        if promoted.itemsize == candidate.itemsize and promoted.signed == candidate.signed:
            return candidate

    return candidates[-1]

def get_type(ast_node):
    """
    :param ast_node: a Numba or Python AST expression node
    :return: the type of the expression node
    """
    return ast_node.variable.type

def index_type(type):
    "Result of indexing a value of the given type with an integer index"
    if type.is_array:
        result = type.copy()
        result.ndim -= 1
        if result.ndim == 0:
            result = result.dtype
    else:
        result = type.base_type

    return result

def require(property, *ast_nodes):
    "Assert that the types of the given nodes meets a certainrequirement"
    for ast_node in ast_nodes:
        if not getattr(get_type(ast_node), property):
            typename = property[3:] # remove 'is_' prefix
            raise error.NumbaError(ast_node, "Expected an %s" % (typename,))
