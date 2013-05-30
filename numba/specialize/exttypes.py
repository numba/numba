import ast

import numba
from numba import *
from numba import error
from numba import typesystem
from numba import visitors
from numba import nodes
from numba import function_util
from numba.exttypes import virtual
from numba.traits import traits, Delegate

class ExtensionTypeLowerer(visitors.NumbaTransformer):
    """
    Lower extension type attribute accesses and method calls.
    """

    def get_handler(self, ext_type):
        if ext_type.is_extension and not ext_type.is_autojit_exttype:
            return StaticExtensionHandler()
        else:
            assert ext_type.is_autojit_exttype, ext_type
            return DynamicExtensionHandler()

    # ______________________________________________________________________
    # Attributes

    def visit_ExtTypeAttribute(self, node):
        """
        Resolve an extension attribute.
        """
        handler = self.get_handler(node.ext_type)
        self.visitchildren(node)
        return handler.handle_attribute_lookup(self.env, node)

    # ______________________________________________________________________
    # Methods

    def visit_NativeFunctionCallNode(self, node):
        if node.signature.is_bound_method:
            assert isinstance(node.function, nodes.ExtensionMethod)
            self.visitlist(node.args)
            node = self.visit_ExtensionMethod(node.function, node)
        else:
            self.visitchildren(node)

        return node

    def visit_ExtensionMethod(self, node, call_node=None):
        """
        Resolve an extension method. We currently only support immediate
        calls of extension methods.
        """
        if call_node is None:
            raise error.NumbaError(node, "Referenced extension method '%s' "
                                         "must be called" % node.attr)

        handler = self.get_handler(node.ext_type)
        return handler.handle_method_call(self.env, node, call_node)

#------------------------------------------------------------------------
# Handle Static VTable Attributes and Methods
#------------------------------------------------------------------------

class StaticExtensionHandler(object):
    """
    Handle attribute lookup and method calls for static extensions
    with C++/Cython-like virtual method tables and static object layouts.
    """

    def handle_attribute_lookup(self, env, node):
        """
        Resolve an extension attribute for a static object layout.

            ((attributes_struct *)
                 (((char *) obj) + attributes_offset))->attribute

        :node: ExtTypeAttribute AST node
        """
        ext_type = node.value.type
        offset = ext_type.attr_offset
        type = ext_type.attribute_table.to_struct()

        if isinstance(node.ctx, ast.Load):
            value_type = type.ref()         # Load result
        else:
            value_type = type.pointer()     # Use pointer for storage

        struct_pointer = nodes.value_at_offset(node.value, offset,
                                               value_type)
        result = nodes.StructAttribute(struct_pointer, node.attr,
                                       node.ctx, type.ref())

        return result

    def handle_method_call(self, env, node, call_node):
        """
        Resolve an extension method of a static (C++/Cython-like) vtable:

            typedef {
                double (*method1)(double);
                ...
            } vtab_struct;

            vtab_struct *vtab = *(vtab_struct **) (((char *) obj) + vtab_offset)
            void *method = vtab[index]
        """
        # Make the object we call the method on clone-able
        node.value = nodes.CloneableNode(node.value)

        ext_type = node.value.type
        offset = ext_type.vtab_offset

        vtable_struct = ext_type.vtab_type.to_struct()
        vtable_struct_type = vtable_struct.ref()

        vtab_struct_pointer_pointer = nodes.value_at_offset(
            node.value, offset,vtable_struct_type.pointer())
        vtab_struct_pointer = nodes.DereferenceNode(vtab_struct_pointer_pointer)

        vmethod = nodes.StructAttribute(vtab_struct_pointer, node.attr,
                                        ast.Load(), vtable_struct_type)

        # Insert first argument 'self' in args list
        args = call_node.args
        args.insert(0, nodes.CloneNode(node.value))
        result = nodes.NativeFunctionCallNode(node.type, vmethod, args)

        return result


#------------------------------------------------------------------------
# Handle Dynamic VTable Attributes and Methods
#------------------------------------------------------------------------

@traits
class DynamicExtensionHandler(object):
    """
    Handle attribute lookup and method calls for autojit extensions
    with dynamic perfect-hash-based virtual method tables and dynamic
    object layouts.
    """

    static_handler = StaticExtensionHandler()

    # TODO: Implement hash-based attribute lookup
    handle_attribute_lookup = Delegate('static_handler')

    def handle_method_call(self, env, node, call_node):
        """
        Resolve an extension method of a dynamic hash-based vtable:

            PyCustomSlots_Table ***vtab_slot = (((char *) obj) + vtab_offset)
            lookup_virtual_method(*vtab_slot)

        We may cache (*vtab_slot), but we may not cache (**vtab_slot), since
        compilations may regenerate the table.

        However, we could *preload* (**vtab_slot), where function calls
        invalidate the preload, if we were so inclined.
        """
        # Make the object we call the method on clone-able
        node.value = nodes.CloneableNode(node.value)

        ext_type = node.ext_type
        func_signature = node.type #typesystem.extmethod_to_function(node.type)
        offset = ext_type.vtab_offset

        # __________________________________________________________________
        # Retrieve vtab

        vtab_ppp = nodes.value_at_offset(node.value, offset,
                                         void.pointer().pointer())
        vtab_struct_pp = nodes.DereferenceNode(vtab_ppp)

        # __________________________________________________________________
        # Calculate pre-hash

        prehash = virtual.hash_signature(func_signature, func_signature.name)
        prehash_node = nodes.ConstNode(prehash, uint64)

        # __________________________________________________________________
        # Retrieve method pointer

        # A method is always present when it was given a static signature,
        # e.g. @double(double)
        always_present = node.attr in ext_type.vtab_type.methodnames
        args = [vtab_struct_pp, prehash_node]

        # lookup_impl = NumbaVirtualLookup()
        lookup_impl = DebugVirtualLookup()
        ptr = lookup_impl.lookup(env, always_present, node, args)
        vmethod = ptr.coerce(func_signature.pointer())
        vmethod = vmethod.cloneable

        # __________________________________________________________________
        # Call method pointer

        # Insert first argument 'self' in args list
        args = call_node.args
        args.insert(0, nodes.CloneNode(node.value))
        method_call = nodes.NativeFunctionCallNode(func_signature, vmethod, args)

        # __________________________________________________________________
        # Generate fallback

        # TODO: Subclassing!
        # if not always_present:
        #     # TODO: Enable this path and generate a phi for the result
        #     # Generate object call
        #     obj_args = [nodes.CoercionNode(arg, object_) for arg in args]
        #     obj_args.append(nodes.NULL)
        #     object_call = function_util.external_call(
        #         env.context, env.crnt.llvm_module,
        #         'PyObject_CallMethodObjArgs', obj_args)
        #
        #     # if vmethod != NULL: vmethod(obj, ...)
        #     # else: obj.method(...)
        #     method_call = nodes.if_else(
        #         ast.NotEq(),
        #         vmethod.clone, nodes.NULL,
        #         lhs=method_call, rhs=object_call)

        return method_call

#------------------------------------------------------------------------
# Method lookup
#------------------------------------------------------------------------

def call_jit(jit_func, args):
    return nodes.NativeCallNode(jit_func.signature, args, jit_func.lfunc)

class NumbaVirtualLookup(object):
    """
    Use a numba function from numba.utility.virtuallookup to look up virtual
    methods in a hash table.
    """

    def lookup(self, env, always_present, node, args):
        """
        :param node: ExtensionMethodNode
        :param args: [vtable_node, prehash_node]
        :return: The virtual method as a Node
        """
        from numba.utility import virtuallookup

        if always_present and False:
            lookup = virtuallookup.lookup_method
        else:
            lookup = virtuallookup.lookup_and_assert_method
            args.append(nodes.const(node.attr, c_string_type))

        vmethod = call_jit(lookup, args)
        return vmethod

class DebugVirtualLookup(object):
    """
    Use a C utility function from numba/utility/utilities/virtuallookup.c
    to look up virtual methods in a hash table.

    Use for debugging.
    """

    def lookup(self, env, always_present, node, args):
        args.append(nodes.const(node.attr, c_string_type))
        vmethod = function_util.utility_call(
            env.context, env.crnt.llvm_module,
            "lookup_method", args)
        return vmethod
