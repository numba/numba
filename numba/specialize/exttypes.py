import ast

import numba
from numba import *
from numba import error
from numba import typesystem
from numba import visitors
from numba import nodes
from numba.typesystem import is_obj, promote_closest, promote_to_native

class ExtensionTypeLowerer(visitors.NumbaTransformer):
    """
    Lower extension type attribute accesses and method calls.
    """

    def get_handler(self, ext_type):
        if ext_type.is_jit_extension:
            return StaticExtensionHandler()
        else:
            return DynamicExtensionHandler()

    # ______________________________________________________________________
    # Attributes

    def visit_ExtTypeAttribute(self, node):
        """
        Resolve an extension attribute.
        """
        handler = self.get_handler(node.ext_type)
        self.visitchildren(node)
        return handler.handle_attribute_lookup(node)

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
        return handler.handle_method_call(node, call_node)


#------------------------------------------------------------------------
# Handle Static VTable Attributes and Methods
#------------------------------------------------------------------------

class StaticExtensionHandler(object):
    """
    Handle attribute lookup and method calls for static extensions
    with C++/Cython-like virtual method tables and static object layouts.
    """

    def handle_attribute_lookup(self, node):
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

    def handle_method_call(self, node, call_node):
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
        struct_type = vtable_struct.ref()

        struct_pointer_pointer = nodes.value_at_offset(node.value, offset,
                                                       struct_type.pointer())
        struct_pointer = nodes.DereferenceNode(struct_pointer_pointer)

        vmethod = nodes.StructAttribute(struct_pointer, node.attr,
                                        ast.Load(), struct_type)

        # Insert first argument 'self' in args list
        args = call_node.args
        args.insert(0, nodes.CloneNode(node.value))
        result = nodes.NativeFunctionCallNode(node.type, vmethod, args)

        return result


#------------------------------------------------------------------------
# Handle Dynamic VTable Attributes and Methods
#------------------------------------------------------------------------

class DynamicExtensionHandler(object):
    """
    Handle attribute lookup and method calls for autojit extensions
    with dynamic perfect-hash-based virtual method tables and dynamic
    object layouts.
    """
