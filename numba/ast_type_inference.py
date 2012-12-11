import ast
import math
import cmath
import copy
import opcode
import types
import __builtin__ as builtins

import numba
from numba import *
from numba import error, transforms, closure, control_flow
from .minivect import minierror, minitypes
from . import translate, utils, _numba_types as numba_types
from .symtab import Variable
from . import visitors, nodes, error
from numba import stdio_util
from numba._numba_types import is_obj, promote_closest
from numba.utils import dump

import llvm.core
import numpy
import numpy as np

import logging

debug = False
#debug = True

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
if debug:
    logger.setLevel(logging.DEBUG)

def _parse_args(call_node, arg_names):
    result = dict.fromkeys(arg_names)

    # parse positional arguments
    i = 0
    for i, (arg_name, arg) in enumerate(zip(arg_names, call_node.args)):
        result[arg_name] = arg

    arg_names = arg_names[i:]
    if arg_names:
        # parse keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg in result:
                result[keyword.arg] = keyword.value

    return result

def no_keywords(node):
    if node.keywords or node.starargs or node.kwargs:
        raise error.NumbaError(
            node, "Function call does not support keyword or star arguments")


class BuiltinResolverMixin(transforms.BuiltinResolverMixinBase):
    """
    Resolve builtin calls for type inference. Only applies high-level
    transformations such as type coercions. A subsequent pass in
    LateSpecializer performs low-level transformations.
    """

    def _resolve_range(self, func, node, argtype):
        node.variable = Variable(numba_types.RangeType())
        args = self.visitlist(node.args)
        node.args = nodes.CoercionNode.coerce(args, dst_type=Py_ssize_t)
        return node

    _resolve_xrange = _resolve_range

    def _resolve_len(self, func, node, argtype):
        # Simplify len(array) to ndarray.shape[0]
        self._expect_n_args(func, node, 1)
        if argtype.is_array:
            shape_attr = nodes.ArrayAttributeNode('shape', node.args[0])
            new_node = nodes.index(shape_attr, 0)
            return self.visit(new_node)

        return None

    dst_types = {
        int: numba.int_,
        float: numba.double,
        complex: numba.complex128
    }

    def _resolve_int(self, func, node, argtype):
        # Resolve int(x) and float(x) to an equivalent cast
        self._expect_n_args(func, node, (0, 1, 2))
        dst_type = self.dst_types[func]

        if len(node.args) == 0:
            return nodes.ConstNode(func(0), dst_type)
        elif len(node.args) == 1:
            return nodes.CoercionNode(node.args[0], dst_type=dst_type)
        else:
            # XXX Moved the unary version to the late specializer,
            # what about the 2-ary version?
            arg1, arg2 = node.args
            if arg1.variable.type.is_c_string:
                assert dst_type.is_int
                return nodes.CoercionNode(
                    nodes.ObjectTempNode(
                        self.function_cache.call(
                            'PyInt_FromString', arg1,
                            nodes.NULL, arg2)),
                    dst_type=dst_type)
            return None

    _resolve_float = _resolve_int

    def _resolve_complex(self, func, node, argtype):
        if len(node.args) == 2:
            args = nodes.CoercionNode.coerce(node.args, double)
            result = nodes.ComplexNode(real=args[0], imag=args[1])
        else:
            result = self._resolve_int(func, node, argtype)

        return result

    def _resolve_abs(self, func, node, argtype):
        self._expect_n_args(func, node, 1)
        if argtype.is_complex:
            dst_type = double
            result_type = object_
        elif argtype.is_int:
            dst_type = argtype
            result_type = promote_closest(self.context, argtype, [long_, longlong])
        else:
            dst_type = argtype
            result_type = argtype

        node.variable = Variable(result_type)
        return nodes.CoercionNode(node, dst_type)

    def _resolve_pow(self, func, node, argtype):
        self._expect_n_args(func, node, (2, 3))
        return nodes.CoercionNode(node, argtype)

    def _resolve_round(self, func, node, argtype):
        self._expect_n_args(func, node, (1, 2))
        if argtype.is_float:
            dst_type = argtype
        elif argtype.is_int and (len(node.args) == 1 or
                                    self._is_math_function(node.args, round)):
            dst_type = argtype
        else:
            dst_type = object_

        node.variable = Variable(dst_type)
        return nodes.CoercionNode(node, double)

    def _resolve_globals(self, func, node, argtype):
        self._expect_n_args(func, node, 0)
        return nodes.ObjectInjectNode(self.func.func_globals)

    def _resolve_locals(self, func, node, argtype):
        self._expect_n_args(func, node, 0)
        raise error.NumbaError("locals() is not supported in numba functions")


class NumpyMixin(object):
    """
    Infer types for NumPy functionality. This includes:

        1) Figuring out dtypes

            e.g. np.double     -> double
                 np.dtype('d') -> double

        2) Function calls such as np.empty/np.empty_like/np.arange/etc
        3) Resolve the resulting type of indexing and slicing:

            - normalize ellipses
            - recognize newaxes
            - track how contiguity is affected (C or Fortran)
    """

    def _is_constant_index(self, node):
        return (isinstance(node, ast.Index) and
                isinstance(node.value, nodes.ConstNode))

    def _is_newaxis(self, node):
        v = node.variable
        return (self._is_constant_index(node) and
                node.value.pyval is None) or v.type.is_newaxis or v.type.is_none

    def _is_ellipsis(self, node):
        return (self._is_constant_index(node) and
                node.value.pyval is Ellipsis)

    def _unellipsify(self, node, slices, subscript_node):
        """
        Given an array node `node`, process all AST slices and create the
        final type:

            - process newaxes (None or numpy.newaxis)
            - replace Ellipsis with a bunch of ast.Slice objects
            - process integer indices
            - append any missing slices in trailing dimensions
        """
        type = node.variable.type

        if not type.is_array:
            assert type.is_object
            return minitypes.object_, node

        if (len(slices) == 1 and self._is_constant_index(slices[0]) and
                slices[0].value.pyval is Ellipsis):
            # A[...]
            return type, node

        result = []
        seen_ellipsis = False

        # Filter out newaxes
        newaxes = [newaxis for newaxis in slices if self._is_newaxis(newaxis)]
        n_indices = len(slices) - len(newaxes)

        full_slice = ast.Slice(lower=None, upper=None, step=None)
        full_slice.variable = Variable(numba_types.SliceType())
        ast.copy_location(full_slice, slices[0])

        # process ellipses and count integer indices
        indices_seen = 0
        for slice_node in slices[::-1]:
            slice_type = slice_node.variable.type
            if slice_type.is_ellipsis:
                if seen_ellipsis:
                    result.append(full_slice)
                else:
                    nslices = type.ndim - n_indices + 1
                    result.extend([full_slice] * nslices)
                    seen_ellipsis = True
            elif (slice_type.is_slice or slice_type.is_int or
                  self._is_newaxis(slice_node)):
                indices_seen += slice_type.is_int
                result.append(slice_node)
            else:
                # TODO: Coerce all object operands to integer indices?
                # TODO: (This will break indexing with the Ellipsis object or
                # TODO:  with slice objects that we couldn't infer)
                return minitypes.object_, nodes.CoercionNode(node,
                                                             minitypes.object_)

        # append any missing slices (e.g. a2d[:]
        result_length = len(result) - len(newaxes)
        if result_length < type.ndim:
            nslices = type.ndim - result_length
            result.extend([full_slice] * nslices)

        result.reverse()
        subscript_node.slice = ast.ExtSlice(result)
        ast.copy_location(subscript_node.slice, slices[0])

        # create the final array type and set it in value.variable
        result_dtype = node.variable.type.dtype
        result_ndim = node.variable.type.ndim + len(newaxes) - indices_seen
        if result_ndim > 0:
            result_type = result_dtype[(slice(None),) * result_ndim]
        elif result_ndim == 0:
            result_type = result_dtype
        else:
            result_type = minitypes.object_

        return result_type, node

    def _resolve_attribute_dtype(self, dtype):
        "Resolve the type for numpy dtype attributes"
        if dtype.is_numpy_attribute:
            numpy_attr = getattr(dtype.module, dtype.attr, None)
            if isinstance(numpy_attr, numpy.dtype):
                return numba_types.NumpyDtypeType(dtype=numpy_attr)
            elif issubclass(numpy_attr, numpy.generic):
                return numba_types.NumpyDtypeType(dtype=numpy.dtype(numpy_attr))

    def _get_dtype(self, node, args, dtype=None):
        "Get the dtype keyword argument from a call to a numpy attribute."
        for keyword in node.keywords:
            if keyword.arg == 'dtype':
                dtype = keyword.value.variable.type
                return self._resolve_attribute_dtype(dtype)
        else:
            # second argument is a dtype
            if args and args[0].variable.type.is_numpy_dtype:
                return args[0].variable.type

        return dtype

    def _resolve_empty_like(self, node):
        "Parse the result type for np.empty_like calls"
        args = node.args
        if args:
            arg = node.args[0]
            args = args[1:]
        else:
            for keyword in node.keywords:
                if keyword.arg == 'a':
                    arg = keyword.value
                    break
            else:
                return None

        type = arg.variable.type
        if type.is_array:
            dtype = self._get_dtype(node, args)
            if dtype is None:
                return type
            else:
                if not dtype.is_numpy_dtype:
                    return None
                return minitypes.ArrayType(dtype.resolve(), type.ndim)

    def _resolve_arange(self, node):
        "Resolve a call to np.arange()"
        dtype = self._get_dtype(node, node.args, numba_types.NumpyDtypeType(
            dtype=numpy.dtype(numpy.int64)))
        if dtype is not None:
            # return a 1D array type of the given dtype
            return dtype.resolve()[:]

    def _resolve_empty(self, node):
        args = _parse_args(node, ['shape', 'dtype', 'order'])
        if not args['shape'] or not args['dtype']:
            return None

        dtype = self._resolve_attribute_dtype(args['dtype'].variable.type)
        if dtype is None:
            return None

        shape_type = args['shape'].variable.type
        if shape_type.is_int:
            ndim = 1
        elif shape_type.is_tuple:
            ndim = shape_type.size
        else:
            return None

        return minitypes.ArrayType(dtype.resolve(), ndim)

    def _resolve_numpy_call(self, func_type, node):
        """
        Resolve a call of some numpy attribute or sub-attribute.
        """
        numpy_func = getattr(func_type.module, func_type.attr)
        if numpy_func in (numpy.zeros_like, numpy.ones_like,
                          numpy.empty_like):
            result_type = self._resolve_empty_like(node)
        elif numpy_func is numpy.arange:
            result_type = self._resolve_arange(node)
        elif numpy_func in (numpy.empty, numpy.zeros, numpy.ones):
            result_type = self._resolve_empty(node)
        else:
            result_type = None

        return result_type


class TypeInferer(visitors.NumbaTransformer, BuiltinResolverMixin,
                  NumpyMixin, closure.ClosureMixin, transforms.MathMixin):
    """
    Type inference. Initialize with a minivect context, a Python ast,
    and a function type with a given or absent, return type.

    Infers and checks types and inserts type coercion nodes.

    See transform.py for an overview of AST transformations.
    """

    def __init__(self, context, func, ast, closure_scope=None, **kwds):
        super(TypeInferer, self).__init__(context, func, ast, **kwds)

        self.given_return_type = self.func_signature.return_type
        self.return_type = None

        ast.symtab = self.symtab
        self.closure_scope = closure_scope
        ast.closure_scope = closure_scope
        ast.closures = []

        self.function_level = kwds.get('function_level', 0)

        self.init_locals()
        ast.have_return = False

    def infer_types(self):
        """
        Infer types for the function.
        """
        self.return_variable = Variable(self.given_return_type)
        self.ast = self.visit(self.ast)

        self.return_type = self.return_variable.type or void
        ret_type = self.func_signature.return_type

        if ret_type and ret_type != self.return_type:
            self.assert_assignable(ret_type, self.return_type)
            self.return_type = self.promote_types(ret_type, self.return_type)

        restype, argtypes = self.return_type, self.func_signature.args
        self.func_signature = minitypes.FunctionType(return_type=restype,
                                                     args=argtypes)
        if restype.is_struct or restype.is_complex:
            # Change signatures returning complex numbers or structs to
            # signatures taking a pointer argument to a complex number
            # or struct
            self.func_signature.struct_by_reference = True

    def init_global(self, global_name):
        globals = self.func_globals
        # Determine the type of the global, i.e. a builtin, global
        # or (numpy) module
        if (global_name not in globals and
                getattr(builtins, global_name, None)):
            type = numba_types.BuiltinType(name=global_name)
        else:
            # FIXME: analyse the bytecode of the entire module, to determine
            # overriding of builtins
            if isinstance(globals.get(global_name), types.ModuleType):
                type = numba_types.ModuleType(globals.get(global_name))
            else:
                type = numba_types.GlobalType(name=global_name)

        variable = Variable(type, name=global_name)
        self.symtab[global_name] = variable
        return variable

    def init_locals(self):
        arg_types = self.func_signature.args
        if (isinstance(self.ast, ast.FunctionDef) and
                len(arg_types) != len(self.ast.args.args)):
            raise error.NumbaError(
                    "Incorrect number of types specified in @jit()")

        self.symtab['None'] = Variable(numba_types.none, name='None',
                                       is_constant=True, constant_value=None)
        self.symtab['True'] = Variable(bool_, name='True', is_constant=True,
                                       constant_value=True)
        self.symtab['False'] = Variable(bool_, name='False', is_constant=True,
                                       constant_value=False)


        arg_types = list(arg_types)
        for local_name, local_type in self.locals.iteritems():
            if local_name not in self.symtab:
                self.symtab[local_name] = Variable(local_type, is_local=True,
                                                   name=local_name)
            variable = self.symtab[local_name]
            variable.type = local_type
            variable.promotable_type = False

            if local_name in self.argnames:
                idx = self.argnames.index(local_name)
                arg_types[idx] = local_type

        # Initialize argument types
        for var_name, arg_type in zip(self.local_names, arg_types):
            self.symtab[var_name].type = arg_type

        # Propagate argument types to first rename of the variable in the block
        for var in self.symtab.values():
            if var.parent_var and not var.parent_var.parent_var:
                var.type = var.parent_var.type
                if not var.type:
                    var.type = numba_types.UninitializedType(None)

        self.func_signature.args = tuple(arg_types)

        self.have_cfg = hasattr(self.ast, 'flow')
        if self.have_cfg:
            self.deferred_types = []
            self.resolve_variable_types()

        if debug and self.have_cfg:
            for block in self.ast.flow.blocks:
                for var in block.symtab.values():
                    if var.type and var.cf_references:
                        assert not var.type.is_unresolved
                        print "Variable after analysis: %s" % var

    def is_object(self, type):
        return type.is_object or type.is_array

    def promote_types(self, t1, t2):
        return self.context.promote_types(t1, t2)

    def promote_types_numeric(self, t1, t2):
        "Type promotion but demote objects to numeric types"
        if (t1.is_numeric or t2.is_numeric) and (self.is_object(t1) or
                                                 self.is_object(t2)):
            if t1.is_numeric:
                return t1
            else:
                return t2
        else:
            return self.promote_types(t1, t2)

    def promote(self, v1, v2):
        return self.promote_types(v1.type, v2.type)

    def assert_assignable(self, dst_type, src_type):
        self.promote_types(dst_type, src_type)

    def type_from_pyval(self, pyval):
        return self.context.typemapper.from_python(pyval)

    def handle_NameAssignment(self, assignment_node):
        if assignment_node is None:
            # ast.Name parameter to the function
            return

        if isinstance(assignment_node, ast.For):
            # Analyse target variable assignment
            return self.visit_For(assignment_node, visit_body=False)
            #print "handled", assignment_node.target.variable
        #elif (isinstance(assignment_node, ast.Assign) and
        #          isinstance(assignment_node.value, nodes.FuncDefExprNode)):
            # Don't analyse inner functions at this point
        #    pass
        else:
            return self.visit(assignment_node)
            #print "handled", assignment_node.targets[0].variable

    def handle_phi(self, node):
        # Merge point for different definitions
        incoming = [v for v in node.incoming
                          if not v.type or not v.type.is_uninitialized]
        assert incoming

        for v in incoming:
            if v.type is None:
                # We have not analyzed this definition yet, delay the type
                # resolution
                v.type = v.deferred_type
                self.deferred_types.append(v.type)

        incoming_types = [v.type for v in incoming]
        if len(incoming_types) > 1:
            promoted_type = numba_types.PromotionType(
                node.variable, self.context,incoming_types, assignment=True)
            promoted_type.simplify()
            node.variable.type = promoted_type.resolve()
        else:
            node.variable.type = incoming_types[0]

        #print "handled", node.variable
        return node

    def kill_unused_phis(self, cfg):
        """
        Kill phis which are not referenced. We need to do this bottom-up,
        i.e. in reverse topological dominator-tree order, since in SSA
        a definition always lexically precedes a reference.

        This is important, since it kills any unnecessary promotions (e.g.
        ones to object, which LLVM wouldn't be able to optimize out).
        """
        for block in cfg.blocks[::-1]:
            phi_nodes = []
            for i, phi in enumerate(block.phi_nodes):
                if phi.variable.cf_references:
                    # Used phi
                    phi_nodes.append(phi)
                else:
                    # Unused phi
                    logging.info("Killing phi: %s", phi)
                    block.symtab.pop(phi.variable.renamed_name)
                    for incoming_var in phi.incoming:
                        # A single definition can reach a block multiple times,
                        # remove= all references
                        refs = [ref for ref in incoming_var.cf_references
                                        if ref.variable is not phi.variable]
                        incoming_var.cf_references = refs

            block.phi_nodes = phi_nodes

    def analyse_assignments(self):
        """
        Analyze all variable assignments and phis.
        """
        cfg = self.ast.flow
        self.kill_unused_phis(cfg)

        self.function_level += 1
        for block in cfg.blocks:
            phis = []
            for phi in block.phi_nodes:
                phi = self.handle_phi(phi)
                if phi is not None:
                    phis.append(phi)
            block.phi_nodes = phis

            for stat in block.stats:
                if isinstance(stat, control_flow.NameAssignment):
                    #visitor.referenced = {}
                    #visitor.visit(stat.rhs)
                    #parents = [name_node.variable
                    #    for name_node in visitor.referenced.itervalues()]
                    assmnt = self.handle_NameAssignment(stat.assignment_node)
                    # TODO: inject back in AST...
                    stat.assignment_node = assmnt

        self.function_level -= 1

    def candidates(self, unvisited):
        "Types with in-degree zero"
        return [type for type in unvisited if len(type.parents) == 0]

    def add_resolved_parents(self, unvisited, start_points, strongly_connected):
        "Check for immediate resolved parents"
        for type in unvisited:
            for parent in type.parents:
                parent = strongly_connected.get(parent, parent)
                if self.is_resolved(parent) or len(parent.parents) == 0:
                    start_points.append(parent)

    def is_resolved(self, t):
        return not t.is_unresolved or (t.is_unresolved and not t.is_scc and not
                                       t.resolve().is_unresolved)

    def is_trivial_cycle(self, type):
        "Return whether the type directly refers to itself"
        return len(type.parents) == 1 and list(type.parents)[0] is type

    def _debug_type(self, start_point):
        if start_point.is_scc:
            print "scc", start_point, start_point.types
        else:
            print start_point

    def remove_resolved_type(self, start_point):
        "Remove a resolved type from the type graph"
        for child in start_point.children:
            if start_point in child.parents:
                child.parents.remove(start_point)

    def resolve_variable_types(self):
        """
        Resolve the types for all variable assignments. We run type inference
        on each assignment which builds a type graph in case of dependencies.
        The dependencies are resolved after type inference completes.
        """
        self.analyse_assignments()

        for deferred_type in self.deferred_types:
            deferred_type.update()

        #-------------------------------------------------------------------
        # Find all unresolved variables
        #-------------------------------------------------------------------
        unresolved = set()
        for block in self.ast.flow.blocks:
            for variable in block.symtab.itervalues():
                if variable.parent_var: # renamed variable
                    if variable.type.is_unresolved:
                        variable.type.resolve()
                        if variable.type.is_unresolved:
                            unresolved.add(variable.type)

        #-------------------------------------------------------------------
        # Find the strongly connected components (build a condensation graph)
        #-------------------------------------------------------------------
        unvisited = set(unresolved)
        strongly_connected = {}
        while unresolved:
            start_type = unresolved.pop()
            sccs = {}
            numba_types.kosaraju_strongly_connected(start_type, sccs)
            unresolved -= set(sccs)
            strongly_connected.update(sccs)

        #-------------------------------------------------------------------
        # Process type dependencies in topological order. Handle strongly
        # connected components specially.
        #-------------------------------------------------------------------

        if unvisited:
            sccs = dict((k, v) for k, v in strongly_connected.iteritems() if k is not v)

            # unvisited = set([strongly_connected[type] for type in unvisited])
            unvisited = set(strongly_connected.itervalues())
            original_unvisited = set(unvisited)
            visited = set()

            while unvisited:
                L = list(unvisited)
                start_points = self.candidates(unvisited)
                self.add_resolved_parents(unvisited, start_points, strongly_connected)
                if not start_points:
                    # Find and resolve any final reduced self-referential
                    # portions in the graph
                    for u in list(unvisited):
                        if self.is_trivial_cycle(u):
                            u.simplify()
                            if not u.resolve().is_unresolved:
                                unvisited.remove(u)

                    break

                while start_points:
                    start_point = start_points.pop()
                    assert (len(start_point.parents) == 0 or
                            self.is_trivial_cycle(start_point) or
                            self.is_resolved(start_point))
                    visited.add(start_point)
                    if start_point in unvisited:
                        unvisited.remove(start_point)

                    # self._debug_type(start_point)

                    if not self.is_resolved(start_point):
                        start_point.simplify()

                    if not (start_point.is_scc or start_point.is_deferred):
                        r = start_point
                        while r.is_unresolved:
                            resolved = r.resolve()
                            if resolved is r:
                                break
                            r = resolved
                        assert not r.is_unresolved

                    self.remove_resolved_type(start_point)
                    if start_point.is_scc:
                        for type in start_point.types:
                            self.remove_resolved_type(type)

                    children = (strongly_connected.get(c, c)
                                    for c in start_point.children
                                        if c not in visited)
                    start_points.extend(self.candidates(children))

        if unvisited:
            t = list(unvisited)[0] # for debugging
            self.error_unresolved_types(unvisited)

    def error_unresolved_types(self, unvisited):
        "Raise an exception for a circular dependence we can't resolve"
        def getvar(type):
            if type.is_scc:
                candidates = [type for type in type.scc if not type.is_scc]
                return type.scc[0].variable
            else:
                return type.variable

        def pos(type):
            assmnt = getvar(type).name_assignment
            if assmnt:
                assmnt_node = assmnt.assignment_node
                return error.format_pos(assmnt_node) or 'na'
            else:
                return 'na'

        type = sorted(unvisited, key=pos)[0]
        numba_types.error_circular(getvar(type))


    #------------------------------------------------------------------------
    # Visit methods
    #------------------------------------------------------------------------

    def visit(self, node):
        if node is Ellipsis:
            node = ast.Ellipsis()
        result = super(TypeInferer, self).visit(node)
        return result

    def visit_PhiNode(self, node):
        # Already handled
        return node

    #------------------------------------------------------------------------
    # Assignments
    #------------------------------------------------------------------------

    def _handle_unpacking(self, node):
        """
        Handle tuple unpacking. We only handle tuples, lists and numpy attributes
        such as shape on the RHS.
        """
        value_type = node.value.variable.type

        if len(node.targets) == 1:
            # tuple or list constant
            targets = node.targets[0].elts
        else:
            targets = node.targets

        valid_type = (value_type.is_carray or value_type.is_sized_pointer or
                      value_type.is_list or value_type.is_tuple)

        if not valid_type:
            self.error(node.value,
                       'Only NumPy attributes and list or tuple literals are '
                       'supported')
        elif value_type.size != len(targets):
            self.error(node.value,
                       "Too many/few arguments to unpack, got (%d, %d)" %
                                            (value_type.size, len(targets)))

        # Generate an assignment for each unpack
        result = []
        for i, target in enumerate(targets):
            if value_type.is_carray or value_type.is_sized_pointer:
                # C array
                value = nodes.index(node.value, i)
            else:
                # list or tuple literal
                value = node.value.elts[i]

            assmt = ast.Assign(targets=[target], value=value)
            result.append(self.visit(assmt))

        return result

    def visit_Assign(self, node):
        # Initialize inplace operator
        node.inplace_op = getattr(node, 'inplace_op', None)

        node.value = self.visit(node.value)
        if len(node.targets) != 1 or isinstance(node.targets[0], (ast.List,
                                                                  ast.Tuple)):
            return self._handle_unpacking(node)

        target = node.targets[0] = self.visit(node.targets[0])
        self.assign(target, node.value)

        lhs_var = target.variable
        rhs_var = node.value.variable
        if isinstance(target, ast.Name):
            node.value = nodes.CoercionNode(node.value, lhs_var.type)
        elif lhs_var.type != rhs_var.type:
            if lhs_var.type.is_array and rhs_var.type.is_array:
                # Let other code handle array coercions
                pass
            else:
                node.value = nodes.CoercionNode(node.value, lhs_var.type)

        return node

    def assign(self, lhs_node, rhs_node, rhs_var=None):
        lhs_var = lhs_node.variable
        if rhs_var is None:
            rhs_var = rhs_node.variable

        if lhs_var.type is None:
            lhs_var.type = rhs_var.type
        elif lhs_var.type != rhs_var.type:
            if lhs_var.name in self.locals:
                self.assert_assignable(lhs_var.type, rhs_var.type)
                if rhs_node:
                    rhs_node = nodes.CoercionNode(rhs_node, lhs_var.type)
                #if (lhs_var.type.is_numeric and rhs_var.type.is_numeric and
                #        lhs_var.promotable_type):
                #    lhs_var.type = self.promote_types(lhs_var.type, rhs_var.type)
            elif lhs_var.type.is_deferred:
                # Override type with new assignment of a deferred LHS and
                # update the type graph to link it together correctly
                assert lhs_var is lhs_var.type.variable
                deferred_type = lhs_var.type
                lhs_var.type = rhs_var.type
                deferred_type.update()
            elif isinstance(lhs_node, ast.Name):
                # Override type with new assignment
                lhs_var.type = rhs_var.type

        lhs_var.deleted = False
        return rhs_node

    #------------------------------------------------------------------------
    # Loops and Control Flow
    #------------------------------------------------------------------------

    def _get_iterator_type(self, node, iterator_type, target_type):
        "Get the type of an iterator Variable"
        if iterator_type.is_iterator:
            base_type = iterator_type.base_type
        elif iterator_type.is_array:
            if iterator_type.ndim > 1:
                slices = (slice(None),) * (iterator_type.ndim - 1)
                base_type = iterator_type.dtype[slices]
                base_type.is_c_contig = iterator_type.is_c_contig
                base_type.is_inner_contig = iterator_type.is_inner_contig
            else:
                base_type = iterator_type.dtype
        elif iterator_type.is_range:
            # base_type = target_type or Py_ssize_t
            base_type = Py_ssize_t
        else:
            raise error.NumbaError(
                node, "Cannot iterate over value of type %s" % (iterator_type,))

        return base_type

    def visit_For(self, node, visit_body=True):
        target = node.target
        #if not isinstance(target, ast.Name):
        #    self.error(node.target,
        #               "Only assignment to target names is supported.")

        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        base_type = self._get_iterator_type(node.iter, node.iter.variable.type,
                                            node.target.variable.type)
        self.assign(node.target, None, rhs_var=Variable(base_type))

        if visit_body:
            self.visitlist(node.body)
        if visit_body and node.orelse:
            self.visitlist(node.orelse)

        return node

    def visit_booltest(self, node):
        if isinstance(node.test, control_flow.ControlBlock):
            node.test.body[0] = nodes.CoercionNode(
                node.test.body[0], minitypes.bool_)
        else:
            node.test = nodes.CoercionNode(node.test, minitypes.bool_)

    def visit_While(self, node):
        self.generic_visit(node)
        self.visit_booltest(node)
        return node

    visit_If = visit_While

    def visit_IfExp(self, node):
        self.generic_visit(node)
        type_ = self.promote(node.body.variable, node.orelse.variable)
        node.variable = Variable(type_)
        node.test = nodes.CoercionNode(node.test, minitypes.bool_)
        node.orelse = nodes.CoercionNode(node.orelse, type_)
        node.body = nodes.CoercionNode(node.body, type_)
        return node

    #------------------------------------------------------------------------
    # Return
    #------------------------------------------------------------------------

    def visit_Return(self, node):
        if node.value is not None:
            # 'return value'
            self.ast.have_return = True
            value = self.visit(node.value)
            type = value.variable.type
            assert type is not None
        else:
            # 'return'
            value = None

        if value is None or type.is_none:
            # When returning None, set the return type to void.
            # That way, we don't have to deal with the PyObject reference.
            if self.return_variable.type is None:
                self.return_variable.type = minitypes.VoidType()
            value = None
        elif self.return_variable.type is None:
            self.return_variable.type = type
        elif self.return_variable.type != type:
            # TODO: in case of unpromotable types, return object?
            if self.given_return_type is None:
                self.return_variable.type = self.promote_types_numeric(
                                        self.return_variable.type, type)

            value = nodes.DeferredCoercionNode(value, self.return_variable)

        node.value = value
        return node

    #------------------------------------------------------------------------
    # 'with' statement
    #------------------------------------------------------------------------

    def visit_With(self, node):
        if (not isinstance(node.context_expr, ast.Name) or
                node.context_expr.id not in ('python', 'nopython')):
            raise error.NumbaError(
                node, "only 'with nopython' and 'with python' is supported "
                      "in with statements")

        if node.context_expr.id == 'nopython':
            node = self.visit(nodes.WithNoPythonNode(
                    body=node.body, lineno=node.lineno,
                    col_offset=node.col_offset))
        else:
            node = self.visit(nodes.WithPythonNode(
                    body=node.body, lineno=node.lineno,
                    col_offset=node.col_offset))

        if (node.body and isinstance(node.body[0], ast.Expr) and
                node.body[0].value == 'WITH_BLOCK'):
            node.body = node.body[1:]

        return node

    #------------------------------------------------------------------------
    # Variable Assignments and References
    #------------------------------------------------------------------------

    def visit_Name(self, node):
        from numba import functions
        node.name = node.id

        var = self.current_scope.lookup(node.id)
        is_none = var and node.id in ('None', 'True', 'False')
        in_closure_scope = self.closure_scope and node.id in self.closure_scope
        if var and (var.is_local or is_none):
            if isinstance(node.ctx, ast.Param) or is_none:
                variable = self.symtab[node.id]
            else:
                # Local variable
                local_variable = self.symtab[node.id]
                if (node.id in self.locals or local_variable.is_cellvar or
                        local_variable.is_freevar):
                    variable = local_variable
                else:
                    variable = node.variable
        elif in_closure_scope and not self.is_store(node.ctx):
            # Free variable
            # print node.id, node.ctx, self.ast.name
            closure_var = self.closure_scope[node.id]

            variable = Variable.from_variable(closure_var)
            variable.is_local = False
            variable.is_cellvar = False
            variable.is_freevar = True
            variable.promotable_type = False
            self.symtab[node.id] = variable
        else:
            # Global or builtin
            variable = self.init_global(node.id)
            if variable.type and not variable.type.is_deferred:
                if variable.type.is_global: # or variable.type.is_module:
                    # TODO: look up globals in dict at call time
                    obj = self.func_globals[node.name]
                    if not functions.is_numba_func(obj):
                        type = self.context.typemapper.from_python(obj)
                        return nodes.const(obj, type)
                elif variable.type.is_builtin:
                    # Rewrite builtin-ins later on, give other code the chance
                    # to handle them first
                    pass

        node.variable = variable
        if variable.type and variable.type.is_unresolved:
            variable.type = variable.type.resolve()
        return node

    #------------------------------------------------------------------------
    # Binary and Unary operations
    #------------------------------------------------------------------------

    def visit_BoolOp(self, node):
        "and/or expression"
        # NOTE: BoolOp.values can have as many items as possible.
        #       Only meta is doing 2 items.
        # if len(node.values) != 2:
        #     raise AssertionError
        assert node.values >= 2
        node.values = self.visitlist(node.values)
        node.values[:] = nodes.CoercionNode.coerce(node.values, minitypes.bool_)
        node.variable = Variable(minitypes.bool_)
        return node

    def visit_BinOp(self, node):
        # TODO: handle floor devision
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        if isinstance(node.op, ast.Pow):
            node = self.pow(node.left, node.right)
            return self.visit(node)

        v1, v2 = node.left.variable, node.right.variable
        promotion_type = self.promote(v1, v2)
        if not (v1.type.is_array and v2.type.is_array):
            # Don't coerce arrays to lesser or higher dimensionality
            # Broadcasting transforms should take care of this
            node.left, node.right = nodes.CoercionNode.coerce(
                                [node.left, node.right], promotion_type)

        node.variable = Variable(promotion_type)
        if isinstance(node.op, ast.FloorDiv):
            dst_type = self.promote(node.left.variable, node.right.variable)
            if dst_type.is_float or dst_type.is_int:
                node.op = ast.Div()
                node = nodes.CoercionNode(node, long_)
                node = nodes.CoercionNode(node, dst_type)

        return node

    def visit_UnaryOp(self, node):
        node.operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            node.operand = nodes.CoercionNode(node.operand, minitypes.bool_)
            node.variable = Variable(minitypes.bool_)
        else:
            node.variable = Variable(node.operand.variable.type)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        lhs = node.left
        comparators = node.comparators
        types = [lhs.variable.type] + [c.variable.type for c in comparators]
        if len(set(types))!=1:
            type = reduce(self.context.promote_types, types)
            node.left = nodes.CoercionNode(lhs, type)
            node.comparators = [nodes.CoercionNode(c, type)
                                for c in comparators]
        node.variable = Variable(minitypes.bool_)
        return node

    #------------------------------------------------------------------------
    # Indexing and Slicing
    #------------------------------------------------------------------------

    def _handle_struct_index(self, node, value_type):
        slice_type = node.slice.variable.type

        if not isinstance(node.slice, ast.Index) or not (
                slice_type.is_int or slice_type.is_c_string):
            raise error.NumbaError(node.slice,
                                   "Struct index must be a single string "
                                   "or integer")

        if not isinstance(node.slice.value, nodes.ConstNode):
            raise error.NumbaError(node.slice,
                                   "Struct index must be constant")

        field_idx = node.slice.value.pyval
        if slice_type.is_int:
            if field_idx > len(value_type.fields):
                raise error.NumbaError(node.slice,
                                       "Struct field index too large")

            field_name, field_type = value_type.fields[field_idx]
        else:
            field_name = field_idx

        return ast.Attribute(value=node.value, attr=field_name, ctx=node.ctx)

    def assert_index(self, type, node):
        if type.is_unresolved:
            type.make_assertion('is_int', node, "Expected an integer")
        elif not type.is_int:
            self.error(node, "Excepted an integer")

    def get_resolved_type(self, type):
        if type.is_unresolved:
            type.simplify()
            type = type.resolve()
            if type.is_promotion and len(type.types) == 2:
                type1, type2 = type.types
                if type1.is_deferred and type2.is_deferred:
                    return type
                elif type1.is_deferred:
                    return type2, type1
                elif type2.is_deferred:
                    return type1, type2
                else:
                    return None, type
        else:
            return type, None

    def create_deferred(self, node, deferred_cls):
        variable = Variable(None)
        deferred_type = deferred_cls(variable, self, node)
        variable.type = deferred_type
        node.variable = variable
        return deferred_type

    def visit_Subscript(self, node, visitchildren=True):
        if visitchildren:
            node.value = self.visit(node.value)
            node.slice = self.visit(node.slice)

        value = node.value
        value_type = node.value.variable.type
        deferred_type = self.create_deferred(node, numba_types.DeferredIndexType)
        if value_type and value_type.is_unresolved:
            deferred_type.dependences.append(node.value)
            deferred_type.update()
            return node

        slice_variable = node.slice.variable
        slice_type = slice_variable.type
        if value_type.is_array:
            # Handle array indexing
            if (slice_type.is_tuple and
                    isinstance(node.slice, ast.Index)):
                node.slice = node.slice.value

            slices = None
            if (isinstance(node.slice, ast.Index) or
                    slice_type.is_ellipsis or slice_type.is_slice):
                slices = [node.slice]
            elif isinstance(node.slice, ast.ExtSlice):
                slices = list(node.slice.dims)
            elif isinstance(node.slice, ast.Tuple):
                slices = list(node.slice.elts)

            if slices is None:
                if slice_type.is_tuple:
                    # result_type = value_type[slice_type.size:]
                    # TODO: use slice_variable.constant_value if available
                    result_type = minitypes.object_
                else:
                    result_type = minitypes.object_
            elif any(slice_node.variable.type.is_unresolved for slice_node in slices):
                for slice_node in slices:
                    if slice_type.variable.type.is_unresolved:
                        deferred_type.dependences.append(slice_node)

                deferred_type.update()
                result_type = deferred_type
            else:
                result_type, node.value = self._unellipsify(
                                    node.value, slices, node)

        elif value_type.is_carray:
            # Handle C array indexing
            if (not slice_variable.type.is_int and not
                    slice_variable.type.is_unresolved):
                self.error(node.slice, "Can only index with an int")
            if not isinstance(node.slice, ast.Index):
                self.error(node.slice, "Expected index")

            # node.slice = node.slice.value
            result_type = value_type.base_type

        elif value_type.is_struct:
            node = self._handle_struct_index(node, value_type)
            return self.visit(node)

        elif value_type.is_pointer:
            self.assert_index(slice_variable.type, node.slice)
            result_type = value_type.base_type

        elif value_type.is_object:
            result_type = object_

        elif value_type.is_c_string:
            # Handle string indexing
            if slice_type.is_int:
                result_type = char
            elif slice_type.is_slice:
                result_type = c_string_type
            elif slice_type.is_unresolved:
                deferred_type.dependences.append(node.slice)
                deferred_type.update()
                result_type = deferred_type
            else:
                # TODO: check for insanity
                node.value = nodes.CoercionNode(node.value, object_)
                node.slice = nodes.CoercionNode(node.slice, object_)
                result_type = object_

        else:
            op = ('sliced', 'indexed')[slice_variable.type.is_int]
            raise error.NumbaError(node, "object of type %s cannot be %s" %
                                                            (value_type, op))

        node.variable.type = result_type
        return node

    def visit_Index(self, node):
        "Normal index"
        node.value = self.visit(node.value)
        variable = node.value.variable
        type = variable.type
        if (type.is_object and variable.is_constant and
                variable.constant_value is None):
            type = numba_types.NewAxisType()

        node.variable = Variable(type)
        return node

    def visit_Ellipsis(self, node):
        return nodes.ConstNode(Ellipsis, numba_types.EllipsisType())

    def visit_Slice(self, node):
        self.generic_visit(node)
        type = numba_types.SliceType()

        is_constant = False
        const = None

        values = [node.lower, node.upper, node.step]
        constants = []
        for value in values:
            if value is None:
                constants.append(None)
            elif value.variable.is_constant:
                constants.append(value.variable.constant_value)
            else:
                break
        else:
            is_constant = True
            const = slice(*constants)

        node.variable = Variable(type, is_constant=is_constant,
                                 constant_value=const)
        return node

    def visit_ExtSlice(self, node):
        self.generic_visit(node)
        node.variable = Variable(minitypes.object_)
        return node

    #------------------------------------------------------------------------
    # Constants
    #------------------------------------------------------------------------

    def visit_Num(self, node):
        return nodes.ConstNode(node.n)

    def visit_Str(self, node):
        return nodes.ConstNode(node.s)

    def visit_long(self, value):

        return nodes.ConstNode(value, long_)

    def _get_constants(self, constants):
        items = []
        constant_value = None
        for i, item_node in enumerate(constants):
            # long constants like 5L are direct values, not Nums!
            if isinstance(item_node, long):
                constants[i] = nodes.ConstNode(item_node, long_)
                items.append(item_node)
            elif item_node.variable.is_constant:
                items.append(item_node.variable.constant_value)
            else:
                return None
        return items

    def _get_constant_list(self, node):
        if not isinstance(node.ctx, ast.Load):
            return None

        return self._get_constants(node.elts)

    def visit_Tuple(self, node):
        self.visitlist(node.elts)
        constant_value = self._get_constant_list(node)
        if constant_value is not None:
            constant_value = tuple(constant_value)
        type = numba_types.TupleType(size=len(node.elts))
        node.variable = Variable(type, is_constant=constant_value is not None,
                                 constant_value=constant_value)
        return node

    def visit_List(self, node):
        node.elts = self.visitlist(node.elts)
        constant_value = self._get_constant_list(node)
        type = numba_types.ListType(size=len(node.elts))
        node.variable = Variable(type, is_constant=constant_value is not None,
                                 constant_value=constant_value)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        constant_keys = self._get_constants(node.keys)
        constant_values = self._get_constants(node.values)
        type = numba_types.DictType(size=len(node.keys))
        if constant_keys and constant_values:
            variable = Variable(type, is_constant=True,
                                constant_value=dict(zip(constant_keys,
                                                        constant_values)))
        else:
            variable = Variable(type)

        node.variable = variable
        return node

    #------------------------------------------------------------------------
    # Function and Method Calls
    #------------------------------------------------------------------------

    def _resolve_function(self, func_type, func_name):
        func = None

        if func_type.is_builtin:
            func = getattr(builtins, func_name)
        elif func_type.is_global:
            func = self.func.__globals__[func_name]
        elif func_type.is_module_attribute:
            func = getattr(func_type.module, func_type.attr)

        return func

    def _create_deferred_call(self, arg_types, call_node):
        "Set the statement as uninferable for now"
        deferred_type = self.create_deferred(call_node,
                                             numba_types.DeferredCallType)
        for arg, arg_type in zip(call_node.args, arg_types):
            if arg_type.is_unresolved:
                deferred_type.dependences.append(arg)

        deferred_type.update()
        return call_node

    def _resolve_external_call(self, call_node, func_type, py_func, arg_types):
        """
        Resolve a call to a function. If we know about the function,
        generate a native call, otherwise go through PyObject_Call().
        """
        if any(arg_type.is_unresolved for arg_type in arg_types) and not func_type == object_:
            result = self.function_cache.get_function(py_func, arg_types)
            if result is not None:
                signature, llvm_func, py_func = result
            else:
                return self._create_deferred_call(arg_types, call_node)
        else:
            signature, llvm_func, py_func = \
                    self.function_cache.compile_function(py_func, arg_types)

        if llvm_func is not None:
            new_node = nodes.NativeCallNode(signature, call_node.args,
                                            llvm_func, py_func)
        elif not call_node.keywords and self._is_math_function(
                                        call_node.args, py_func):
            new_node = self._resolve_math_call(call_node, py_func)
        else:
            new_node = nodes.ObjectCallNode(signature, call_node.func,
                                            call_node.args, call_node.keywords,
                                            py_func)

        if not func_type.is_object:
            raise error.NumbaError(
                call_node, "Cannot call object of type %s" % (func_type,))

        if new_node.type.is_object:
            new_node = self._resolve_return_type(func_type, new_node, call_node)

        return new_node

    def _resolve_method_calls(self, func_type, new_node, node):
        "Resolve special method calls"
        if ((func_type.base_type.is_complex or
             func_type.base_type.is_float) and
            func_type.attr_name == 'conjugate'):
            assert isinstance(node.func, ast.Attribute)
            if node.args or node.keywords:
                raise error.NumbaError(
                        "conjugate method of complex number does not "
                        "take arguments")
            if func_type.base_type.is_float:
                return node.func.value

            new_node = nodes.ComplexConjugateNode(node.func.value)
            new_node.variable = Variable(func_type.base_type)

        return new_node

    def _infer_complex_math(self, func_type, new_node, node, result_type):
        "Infer types for cmath.somefunc()"
        # Check for cmath.{sqrt,sin,etc}
        args = [nodes.const(1.0, float_)]
        is_math = self._is_math_function(args, func_type.value)
        if len(node.args) == 1 and is_math:
            new_node = nodes.CoercionNode(new_node, complex128)
            result_type = complex128

        return new_node, result_type

    def _resolve_return_type(self, func_type, new_node, node):
        """
        We are performing a call through PyObject_Call, but we may be able
        to infer a more specific return value than 'object'.
        """
        result_type = None
        is_numpy = func_type.is_numpy_attribute
        if is_numpy:
            result_type = self._resolve_numpy_call(func_type, node)

        if ((func_type.is_module_attribute and func_type.module is cmath) or
            (result_type is None and is_numpy)):
            new_node, result_type = self._infer_complex_math(
                                func_type, new_node , node, result_type)

        if result_type is None:
            result_type = object_

        new_node.variable = Variable(result_type)
        return new_node

    def _parse_signature(self, node, func_type):
        types = []
        for arg in node.args:
            if not arg.variable.type.is_cast:
                self.error(arg, "Expected a numba type")
            else:
                types.append(arg.variable.type)

        signature = func_type.dst_type(*types)
        new_node = nodes.const(signature, numba_types.CastType(signature))
        return new_node

    def visit_Call(self, node, visitchildren=True):
        if node.starargs or node.kwargs:
            raise error.NumbaError("star or keyword arguments not implemented")

        if self.visitchildren:
            node.func = self.visit(node.func)
            self.visitlist(node.args)
            self.visitlist(node.keywords)

        func_variable = node.func.variable
        func_type = func_variable.type

        func = self._resolve_function(func_type, func_variable.name)

        # TODO: Resolve variable types based on how they are used as arguments
        # TODO: in calls with known signatures
        new_node = None
        if func_type.is_builtin:
            # Call to Python built-in function
            node.variable = Variable(object_)
            new_node = self._resolve_builtin_call(node, func)
            if new_node is None:
                new_node = node
        elif func_type.is_function:
            # Native function call
            no_keywords(node)
            new_node = nodes.NativeFunctionCallNode(
                            func_variable.type, node.func, node.args,
                            skip_self=True)
        elif func_type.is_method:
            # Call to special object method
            no_keywords(node)
            new_node = self._resolve_method_calls(func_type, new_node, node)

        elif func_type.is_closure:
            # Call to closure/inner function
            return nodes.ClosureCallNode(func_type, node)

        elif func_type.is_ctypes_function:
            # Call to ctypes function
            no_keywords(node)
            new_node = nodes.CTypesCallNode(
                    func_type.signature, node.args, func_type,
                    py_func=func_type.ctypes_func)

        elif func_type.is_cast:
            # Call of a numba type
            # 1) double(value) -> cast value to double
            # 2) double() or double(object_, double), ->
            #       this specifies a function signature
            no_keywords(node)
            if len(node.args) != 1 or node.args[0].variable.type.is_cast:
                new_node = self._parse_signature(node, func_type)
            else:
                new_node = nodes.CoercionNode(node.args[0], func_type.dst_type,
                                              name="cast")

        if new_node is None:
            # All other type of calls:
            # 1) call to compiled/autojitting numba function
            # 2) call to some math or numpy math function (np.sin, etc)
            # 3) call to special numpy functions (np.empty, etc)
            # 4) generic call using PyObject_Call
            arg_types = [a.variable.type for a in node.args]
            new_node = self._resolve_external_call(node, func_type, func, arg_types)

        return new_node

    #------------------------------------------------------------------------
    # Attributes
    #------------------------------------------------------------------------

    def _resolve_module_attribute(self, node, type):
        "Resolve attributes of the numpy module or a submodule"
        attribute = getattr(type.module, node.attr)

        result_type = None
        if attribute is numpy.newaxis:
            result_type = numba_types.NewAxisType()
        elif attribute is numba.NULL:
            return numba_types.null_type
        elif type.is_numpy_module or type.is_numpy_attribute:
            result_type = numba_types.NumpyAttributeType(module=type.module,
                                                         attr=node.attr)
        elif type.is_numba_module:
            result_type = self.context.typemapper.from_python(attribute)
            if result_type == object_:
                result_type = None

        if result_type is None:
            result_type = numba_types.ModuleAttributeType(module=type.module,
                                                          attr=node.attr)

        return result_type

    def _resolve_ndarray_attribute(self, array_node, array_attr):
        "Resolve attributes of numpy arrays"
        return

    def is_store(self, ctx):
        return isinstance(ctx, ast.Store)

    def _resolve_extension_attribute(self, node, type):
        if node.attr in type.methoddict:
            return nodes.ExtensionMethod(node.value, node.attr)
        if node.attr not in type.symtab:
            if type.is_resolved or not self.is_store(node.ctx):
                raise error.NumbaError(
                    node, "Cannot access attribute %s of type %s" % (
                                                node.attr, type.name))

            # Create entry in type's symbol table, resolve the actual type
            # in the parent Assign node
            type.symtab[node.attr] = Variable(None)

        return nodes.ExtTypeAttribute(node.value, node.attr, node.ctx, type)

    def _resolve_struct_attribute(self, node, type):
        if not node.attr in type.fielddict:
            raise error.NumbaError(
                    node, "Struct %s has no field %r" % (type, node.attr))

        if isinstance(node.ctx, ast.Store):
            if not isinstance(node.value, (ast.Name, ast.Subscript,
                                           nodes.StructVariable)):
                raise error.NumbaError(
                        node, "Can only assign to struct attributes of "
                              "variables or array indices")
            node.value.ctx = ast.Store()

        return nodes.StructAttribute(node.value, node.attr, node.ctx,
                                     node.value.variable.type)

    def _resolve_complex_attribute(self, node, type):
        # TODO: make conplex a struct type
        if node.attr in ('real', 'imag'):
            if self.is_store(node.ctx):
                raise TypeError("Cannot assign to the %s attribute of "
                                "complex numbers" % node.attr)
            result_type = type.base_type
        else:
            raise AttributeError("'%s' of complex type" % node.attr)

        return result_type

    def visit_Attribute(self, node):
        node.value = self.visit(node.value)
        type = node.value.variable.type
        if node.attr == 'conjugate' and (type.is_complex or type.is_float):
            result_type = numba_types.MethodType(type, 'conjugate')
        elif type.is_complex:
            result_type = self._resolve_complex_attribute(node, type)
        elif type.is_struct:
            return self._resolve_struct_attribute(node, type)
        elif type.is_module and hasattr(type.module, node.attr):
            result_type = self._resolve_module_attribute(node, type)
        elif type.is_array and node.attr in ('data', 'shape', 'strides', 'ndim'):
            # handle shape/strides/ndim etc
            return nodes.ArrayAttributeNode(node.attr, node.value)
        elif type.is_extension:
            return self._resolve_extension_attribute(node, type)
        else:
            # use PyObject_GetAttrString
            node.value = nodes.CoercionNode(node.value, object_)
            result_type = object_

        node.variable = Variable(result_type)
        node.type = result_type
        return node

    def visit_ClosureScopeLoadNode(self, node):
        return node

    def visit_FuncDefExprNode(self, node):
        return self.visit(node.func_def)

    #
    ### Unsupported nodes
    #

    def visit_Global(self, node):
        raise error.NumbaError(node, "Global keyword")


class TypeSettingVisitor(visitors.NumbaTransformer):
    """
    Set node.type for all AST nodes after type inference from node.variable.
    Allows for deferred coercions (may be removed in the future).
    """

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.flow:
            for block in node.flow.blocks:
                for phi in block.phi_nodes:
                    self.handle_phi(phi)

        rettype = self.func_signature.return_type
        if rettype.is_unresolved:
            rettype = rettype.resolve()
            assert not rettype.is_unresolved
            self.func_signature.return_type = rettype

        return node

    def resolve(self, variable):
        """
        Resolve any resolved types, and resolve any final disconnected
        type graphs that haven't been simplified. This can be the case if
        the type of a variable does not depend on the type of a sub-expression
        which may be unresolved, e.g.:

            y = 0
            for i in range(...):
                x = int(y + 4)  # y is unresolved here, so we have
                                # promote(deferred(y), int)
                y += 1
        """
        if variable.type.is_unresolved:
            variable.type = variable.type.resolve()
            if variable.type.is_unresolved:
                variable.type = numba_types.resolve_var(variable)
            assert not variable.type.is_unresolved

    def visit(self, node):
        if hasattr(node, 'variable'):
            self.resolve(node.variable)
            node.type = node.variable.type
        return super(TypeSettingVisitor, self).visit(node)

    def handle_phi(self, node):
        for incoming_var in node.incoming:
            self.resolve(incoming_var)
        self.resolve(node.variable)
        node.type = node.variable.type
        return node

    def visit_Name(self, node):
        return node

    def visit_ExtSlice(self, node):
        self.generic_visit(node)
        types = [n.type for n in node.dims]
        if all(type.is_int for type in types):
            node.type = reduce(self.context.promote_types, types)
        else:
            node.type = object_

        return node

    def visit_DeferredCoercionNode(self, node):
        "Resolve deferred coercions"
        self.generic_visit(node)
        return nodes.CoercionNode(node.node, node.variable.type)
