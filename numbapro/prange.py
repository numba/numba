import ast
import types
import ctypes

import llvm.core
from llvm.core import Type, inline_function
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder import builder

import numba.decorators
from numba import *
from numba import error, visitors, nodes, template
from numba.minivect import  minitypes
from numba import  _numba_types as numba_types
from numba.symtab import Variable

from numba.ast_type_inference import no_keywords
from numba.template import temp_name, temp_var

from numbapro.vectorize import parallel

import logging
logger = logging.getLogger(__name__)

class PrangeType(numba_types):
    is_prange = True

def outline_prange_body(context, outer_py_func, outer_symtab, subnode, **kwargs):
    global name_counter

    v = VariableFindingVisitor(context, outer_py_func, subnode)
    v.visit(subnode)

    fields = []
    reductions = []
    for var_name, assmnt_node in v.assigned.iteritems():
        fields.append(var_name, assmt.value.type)
        if assmnt_node.inplace_op:
            reductions.append((var_name, assmnt_node.inplace_op))

    privates_struct_type = numba.struct(fields)
    args = [ast.Name('__numba_privates', ast.Param())]
    func_def = ast.FunctionDef(name=temp_name(),
                               args=arguments(args=args, vararg=None,
                                              kwarg=None, defaults=[]))
    func_def.func_signature = void(privates_struct_type)
    return func_def, privates_struct_type, reductions

class VariableFindingVisitor(visitors.NumbaVisitor):
    def __init__(self, context, func, ast, **kwargs):
        super(OutliningVisitor, self).__init__(context, func, ast, **kwargs)
        self.referenced = {}
        self.assigned = {}

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.lhs, ast.Name):
            assmts = self.assigned.setdefault(node.lhs.id, [])
            assmts.append(node)

    def visit_Name(self, node):
        self.referenced[node.id] = node

class PrangeNode(nodes.Node):
    _fields = ['start', 'stop', 'step', 'target', 'func_def', 'invoke_closure']

    target = None
    func_def = None
    invoke_closure = None

    # Struct type to hold private and reduction variables
    struct_type = None

    # [(var_name, ast_operator)]
    reductions = None

    def __init__(self, start, stop, step, **kwargs):
        super(PrangeNode, self).__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.step = step
        self.type = PrangeType()

class InvokeAndJoinThreads(nodes.Node):
    _fields = ['contexts', 'func_def']

class PrangeResolver(visitors.NumbaTransformer):
    """
    Runs after type inference and before closures are resolved.
    """

    prange = 0

    def visit_CallNode(self, node):
        func_type = node.func.type
        if (func_type.is_module_attribute and
                func_type.value is numba.prange):
            ast_type_inference.no_keywords(node)
            start, stop, step = None, None, None
            if len(node.args) == 0:
                raise error.NumbaError(node, "Expected at least one argument")
            elif len(node.args) == 1:
                stop, = node.args
            elif len(node.args) == 2:
                start, stop = node.args
            else:
                start, stop, step = node.args

            node = PrangeNode(start, stop, step)

        self.generic_visit(node)
        return node

    def visit_For(self, node):
        if node.iter.type.is_prange:
            if self.prange:
                raise error.NumbaError(node, "Cannot nest prange")

            prange_node = node.iter

            self.prange += 1
            body = self.visitlist(node.body)
            self.prange -= 1

            result = outline_prange_body(self.context, self.func, self.symtab,
                                         ast.Suite(body=node.body))
            closure, privates_struct_type, reductions = result

            prange_node.func_def = closure
            prange_node.struct_type = privates_struct_type
            prange_node.reductions = reductions

            node = prange_node
            return self.visit(node)

        self.generic_visit(node)
        return node

    def visit_PrangeNode(self, node):
        templ = """
            {{func_def}}

            $pack_struct

            $nsteps = ({{stop}} - {{start}}) / {{num_threads}}
            for $temp_i in range({{num_threads}}):
                $temp_struct.__numba_closure_scope = {{closure_scope}}
                $temp_struct.__numba_start = {{start}} + $temp_i * $nsteps
                $temp_struct.__numba_stop = $temp_struct.__numba_start + $nsteps
                $temp_struct.__numba_step = {{step}}
                $contexts[$temp_i] = $temp_struct

            # Undo any truncation
            $contexts[$temp_i].__numba_stop = {{stop}}

            {{invoke_and_join_threads}}
            #for $temp_i in range({{num_threads}}):
            #    {{invoke_thread}}

            #for $temp_i in range({{num_threads}}):
            #    {{join_thread}}

            for $temp_i in range({{num_threads}} - 1):
                $reductions

            $unpack_struct
        """

        # Update struct type with closure scope, start, stop and step
        node.struct_type.add_field('__numba_closure_scope', void.pointer())
        node.struct_type.add_field('__numba_start', npy_intp)
        node.struct_type.add_field('__numba_stop', npy_intp)
        node.struct_type.add_field('__numba_step', npy_intp)

        num_threads = 4
        contexts_array_type = minitypes.CArrayType(node.struct_type,
                                                   num_threads)

        # Create variables for template substitution
        nsteps = temp_var('nsteps')
        temp_i = temp_var('temp_i')
        contexts = temp_var('contexts', contexts_array_type)

        temp_struct = temp_var('temp_struct', node.struct_type)
        pack_struct = temp_var('pack_struct')
        unpack_struct = temp_var('unpack_struct')
        reductions = temp_var('reductions')

        lastprivates_struct = "%s[{{num_threads}} - 1]" % contexts

        # Create code for reductions and (last)privates
        for i, (name, type) in enumerate(node.struct_type.types):
            if name in node.reductions_dict:
                op = node.reductions_dict[name]
                default = get_reduction_default(op)
                reductions.codes.append(
                        "%s %s= %s[%s]" % (lastprivates_struct, op,
                                           contexts, temp_i))
            else:
                default = name
                unpack_struct.codes.append(
                        "%s = %s.%s" % (name, lastprivates_struct, name))

            pack_struct.codes.append("%s.%s = %s" % (temp_struct, name,
                                                     temp_i, default))

        # Interpolate code and variables and run type inference
        node.func_def = node.func_def.cloneable
        invoke = InvokeAndJoinThreads(contexts=contexts.node,
                                      func_def=node.func_def.clone)

        closure_scope = nodes.ClosureScopeLoadNode()
        subs = dict(
            func_def=node.func_def,
            closure_scope=closure_scope,
            invoke_and_join_threads=invoke,
            num_threads=nodes.const(num_threads, int_))

        vars = [temp_i, context, pack_struct, unpack_struct, reductions, nsteps]
        result = template.template_type_infer(self.context, temp, subs, vars)
        return self.visit(result)

class PrangeCodegenMixin(object):

    def visit_InvokeAndJoinThreads(self, node):
        p

class RunThreadPool(CDefinition, parallel.ParallelUFuncPosixMixin):

    _name_ = "invoke_prange"
    _argtys_ = [
        ('contexts', C.pointer(C.void_p)),
        ('start', C.npy_intp),
        ('stop', C.npy_intp),
        ('step', C.npy_intp),
        ('num_threads', C.npy_intp),
    ]

    def body(self, contexts, start, stop, step, num_threads):
        nsteps = (stop - start) / step

        worker = builder.CFuncRef(self.lfunc)
        worker()

    def specialize(self, lfunc, num_threads):
        self.lfunc = lfunc
        self.num_threads = num_threads
        self.specialize_name()