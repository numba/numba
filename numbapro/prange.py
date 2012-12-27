"""
Support for

    for i in numba.prange(...):
        ...
"""

import ast
import copy
import types
import ctypes
import warnings
import multiprocessing

try:
    NUM_THREADS = multiprocessing.cpu_count()
except NotImplementedError:
    warnings.warn("Unable to determine cpu count, assuming 2")
    NUM_THREADS = 2

import llvm.core
from llvm.core import Type, inline_function
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder import builder

import numba.decorators
from numba import *
from numba import (error, visitors, nodes, template, ast_type_inference,
                   transforms)
from numba.minivect import  minitypes
from numba import typesystem
from numba.symtab import Variable

from numba.ast_type_inference import no_keywords

from numbapro.vectorize import parallel, minivectorize

import logging
logger = logging.getLogger(__name__)

def get_reduction_op(op):
    # TODO: recognize invalid operators
    op = type(op)

    reduction_op = minivectorize.opmap[op]
    reduction_ops = {'-': '+', '/': '*'}
    if reduction_op in reduction_ops:
        reduction_op = reduction_ops[reduction_op]

    return reduction_op

def get_reduction_default(op):
    defaults = {
        '+': 0, '-': 0, '*': 1, '/': 1,
    }
    return defaults[op]

#def outline_prange_body(context, outer_py_func, outer_symtab, subnode, **kwargs):
#    # Find referenced and assigned variables
#    v = VariableFindingVisitor()
#    v.visit(subnode)
#
#    # Determine privates and reductions. Shared variables will be handled by
#    # the closure support.
#    fields = []
#    reductions = []
#    for var_name, op in v.assigned.iteritems():
#        type = outer_symtab[var_name].type
#        fields.append((var_name, type))
#        if op is not None:
#            reductions.append((var_name, op))
#
#    # Build a wrapper function around the prange body, accepting a struct
#    # of private variables as argument
#    for i, (name, type) in enumerate(fields):
#        if type is None:
#            fields[i] = name, typesystem.DeferredType(name)
#
#    privates_struct_type = numba.struct(fields)
#    privates_struct = ast.Name('__numba_privates', ast.Param())
#    privates_struct.type = privates_struct_type.pointer()
#    privates_struct.variable = Variable(privates_struct.type)
#
#    args = [privates_struct]
#    func_def = ast.FunctionDef(name=template.temp_name("prange_body"),
#                               args=ast.arguments(args=args, vararg=None,
#                                                  kwarg=None, defaults=[]),
#                               body=[subnode],
#                               decorator_list=[])
#    func_def.func_signature = void(privates_struct.type)
#    func_def.need_closure_wrapper = False
#    return func_def, privates_struct_type, reductions


class VariableFindingVisitor(visitors.VariableFindingVisitor):
    "Find referenced and assigned ast.Name nodes"
    def __init__(self):
        super(VariableFindingVisitor, self).__init__()
        self.referenced = {}
        self.assigned = {}

    def register_assignment(self, node, target, op):
        if isinstance(target, ast.Name):
            if op is None:
                redop = op
            else:
                redop = get_reduction_op(op)

            if target.id in self.assigned:
                previous_op = self.assigned[target.id]
                if ((previous_op is None and op is not None) or
                        (previous_op is not None and op is None)):
                    raise error.NumbaError(
                            node, "Reduction variable %r may not be "
                                  "assigned to" % target.id)
                else:
                    if redop != previous_op:
                        raise error.NumbaError(
                            node, "Incompatible reduction operators: "
                                  "(%s and %s) for variable %r" % (
                                            op, previous_op, target.id))
            else:
                self.assigned[target.id] = redop


def outline_prange_body(prange_node, body):
    # Find referenced and assigned variables
    v = VariableFindingVisitor()
    v.visitlist(body)

    # Determine privates and reductions. Shared variables will be handled by
    # the closure support.
    assigned = v.assigned.items()
    fields = dict((name, op) for name, op in assigned if op is None)
    reductions = dict((name, op) for name, op in assigned if op is None)

    privates_struct_type = numba.struct([])
    privates_struct = ast.Name('__numba_privates', ast.Param())

    args = [privates_struct]
    func_def = ast.FunctionDef(name=template.temp_name("prange_body"),
                               args=ast.arguments(args=args, vararg=None,
                                                  kwarg=None, defaults=[]),
                               body=body,
                               decorator_list=[])

    func_def.func_signature = void(privates_struct.type)
    func_def.need_closure_wrapper = False

    prange_node.closure = closure
    prange_node.privates_struct_type = privates_struct_type
    prange_node.fields = fields
    prange_node.reductions = reductions


prange_template = """
{{func_def}}

$pack_struct

$nsteps = ({{stop}} - {{start}}) / ({{step}} * {{num_threads_}})
for $i in range({{num_threads}}):
    $temp_struct.__numba_closure_scope = {{closure_scope}}
    $temp_struct.__numba_start = {{start}} + $i * {{step}} * $nsteps
    $temp_struct.__numba_stop = $temp_struct.__numba_start + {{step}} * $nsteps
    $temp_struct.__numba_step = {{step}}
    $contexts[$i] = $temp_struct

# Undo any truncation, don't use $i here, range() doesn't
# have py semantics yet
$contexts[{{num_threads}} - 1].__numba_stop = {{stop}}

# print "invoking..."
{{invoke_and_join_threads}}

$lastprivates = $contexts[{{num_threads}} - 1]
"""

def rewrite_prange(context, prange_node, target):
    templ = template.TemplateContext(context, prange_template)

    func_def = prange_node.func_def
    struct_type = prange_node.privates_struct_type

    # Allocate context for each thread
    num_threads = NUM_THREADS
    contexts_array_type = minitypes.CArrayType(struct_type,
                                               num_threads)

    # Create variables for template substitution
    nsteps = templ.temp_var('nsteps')
    temp_i = templ.temp_var('i', int32)
    contexts = templ.temp_var('contexts', contexts_array_type)
    temp_struct = templ.temp_var('temp_struct', struct_type)
    prange_node.lastprivates = temp.temp_var("lastprivates")

    pack_struct, unpack_struct, reductions = templ.code_vars(
                         'pack_struct')

    target_name = target.id
    struct_type.add_field(target_name, Py_ssize_t)

    # Create code for reductions and (last)privates
    for i, (name, type) in enumerate(struct_type.fields):
        pack_struct.codes.append("%s.%s = %s" % (temp_struct, name,
                                                 default))

    # Update struct type with closure scope, index variable, start,
    # stop and step
    struct_type.add_field('__numba_closure_scope', void.pointer())
    struct_type.add_field('__numba_start', npy_intp)
    struct_type.add_field('__numba_stop', npy_intp)
    struct_type.add_field('__numba_step', npy_intp)

    # Interpolate code and variables and run type inference
    func_def.type = func_def.func_signature
    func_def.is_prange_body = True

    num_threads_node = nodes.const(num_threads, Py_ssize_t)
    invoke = InvokeAndJoinThreads(contexts=contexts.node,
                                  func_def_name=func_def.name,
                                  struct_type=struct_type,
                                  target_name=target_name,
                                  num_threads=num_threads_node)

    closure_scope = nodes.ClosureScopeLoadNode()
    subs = dict(
        func_def=func_def,
        closure_scope=closure_scope,
        invoke_and_join_threads=invoke,
        num_threads=num_threads_node,
        start=prange_node.start,
        stop=prange_node.stop,
        step=prange_node.step)

    tree = templ.template(subs)
    prange_node.substitutions = templ.substitutions

    return tree

post_prange_template = """
#for $i in range({{num_threads}}):
#    {{invoke_thread}}

#for $i in range({{num_threads}}):
#    {{join_thread}}

# print "performing reductions"
for $i in range({{num_threads}}):
    $reductions

# print "unpacking lastprivates"
$unpack_struct
"""

def perform_reductions(context, prange_node):
    templ = template.TemplateContext(context, post_prange_template)
    templ.add_variable(prange_node.lastprivates)

    unpack_struct, reductions = templ.code_vars('unpack_struct', 'reductions')
    reductions.sep = "; "

    # Create code for reductions and (last)privates
    for i, (name, type) in enumerate(struct_type.fields):
        if name != target_name and name in prange_node.reductions:
            reduction_op = prange_node.reductions[name]
            default = get_reduction_default(reduction_op)
            reductions.codes.append(
                    "%s %s= %s[%s].%s" % (name, reduction_op,
                                          contexts, temp_i, name))
            # reductions.codes.append('print "%s:", %s' % (name, name))
        else:
            default = name
            unpack_struct.codes.append(
                    "%s = $lastprivates.%s" % (name, name))

    substitutions = {"num_threads": prange_node.substitutions["num_threads"]}
    result = temp.template(substitutions)
    return result

#------------------------------------------------------------------------
# prange nodes and types
#------------------------------------------------------------------------

class PrangeType(typesystem.RangeType):
    is_prange = True

class PrangeNode(nodes.Node):
    """
    Prange node. This replaces the For loop iterator in the initial stage.
    After type inference and before closure type inference it replaces the
    entire loop.

    Attributes:

        closure:
        privates_struct_type:
        fields:
        reductions:
    """

    _fields = ['start', 'stop', 'step']

    def __init__(self, start, stop, step, **kwargs):
        super(PrangeNode, self).__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.step = step
        self.type = PrangeType()

class PrangeBodyNode(nodes.Node):
    _fields = ['func_def']

class InvokeAndJoinThreads(nodes.Node):
    _fields = ['contexts', 'num_threads']

#------------------------------------------------------------------------
# prange visitors
#------------------------------------------------------------------------

class PrangeOutliner(visitors.NumbaTransformer):
    """
    Rewrite 'for i in prange(...): ...' before the control flow pass.
    """

    def match_global(self, node, expected_value):
        if isinstance(node, ast.Name) and node.id not in self.local_names:
            value = self.func_globals.get(node.id, None)
            return value is expected_value
        return False

    def is_numba_prange(self, node):
        return (self.match_global(node, numba.prange) or
                (isinstance(node, ast.Attribute) and node.attr == "prange" and
                 self.match_global(node.value, numba)))

    def visit_Call(self, node):
        if self.is_numba_prange(node):
            ast_type_inference.no_keywords(node)
            start, stop, step = self.visitlist(transforms.unpack_range_args(node))
            node = PrangeNode(start, nodes.CoercionNode(stop, Py_ssize_t), step)

        self.visitchildren(node)
        return node

    def error_check_prange(self, node):
        if self.prange:
            raise error.NumbaError(node, "Cannot nest prange")
        if node.orelse:
            raise error.NumbaError(node.orelse,
                                   "Else clause to prange not yet supported")

    def visit_For(self, node, **kwargs):
        node.iter = self.visit(node.iter)

        if not isinstance(node.iter, PrangeNode):
            self.visitchildren(node)
            return node

        self.error_check_prange(node)
        node.target = self.visit(node.target)

        self.prange += 1
        node.body = self.visitlist(node.body)
        self.prange -= 1

        # Outline function body
        prange_node = node.iter
        outline_prange_body(prange_node, node.body)

        # Clear For loop body
        node.body = []
        tree = rewrite_prange(self.context, prange_node, node.target)
        tree = self.visit(tree)

        return ast.Suite(body=[node, tree])


class PrangePrivatesReplacerMixin(object):
    """
    Rewrite private variables to accesses on the privates during
    closure type inference (closure type inference of the outer function, and
    type inference of the inner function).
    """

    privates_struct = None

    in_prange_body = 0

    def __init__(self, *args, **kwargs):
        super(PrangePrivatesReplacerMixin, self).__init__(*args, **kwargs)
        if getattr(self.ast, 'is_prange_body', False):
            self.rewrite_privates(self.ast)

    def rewrite_privates(self, func_def):
        privates_var = self.symtab['__numba_privates']
        privates_var.type = privates_var.type.base_type
        privates_var.need_arg_copy = False
        self.privates_struct_type = privates_var.type

        self.in_prange_body += 1
        func_def.body = [nodes.WithNoPythonNode(body=func_def.body)]
        super(PrangePrivatesReplacerMixin, self).visit_FunctionDef(func_def)
        self.in_prange_body -= 1

        # Update symtab with inferred types
        for name, type in self.privates_struct_type.fields:
            var = self.closure_scope.get(name, None)
            if var and not type == var.type:
                if type.is_deferred:
                    if not type.resolve().is_deferred:
                        var.type = type.resolve()
                else:
                    var.type = type

    def visit_Name(self, node):
        if self.in_prange_body:
            privates_struct = ast.Name('__numba_privates', node.ctx)
            privates_struct.type = self.privates_struct_type
            privates_struct.variable = Variable(privates_struct.type,
                                                is_local=True)

            if node.id in self.privates_struct_type.fielddict:
                self.symtab.pop(node.id, None)

                result = nodes.StructAttribute(
                        value=privates_struct, attr=node.id,
                        struct_type=self.privates_struct_type, ctx=node.ctx)
                if result.type.is_deferred:
                    result.type = result.type.resolve()
                return result

        return super(PrangePrivatesReplacerMixin, self).visit_Name(node)


class PrangeCodegenMixin(object):
    """
    Code generator mixin that handles InvokeAndJoinThreads. Uses llvm_cbuilder
    """

    def visit_InvokeAndJoinThreads(self, node):
        closure_type = self.symtab[node.func_def_name].type
        lfunc = closure_type.closure.lfunc
        # print lfunc
        lfunc_wrapper, lfunc_run = get_threadpool_funcs(
                             self.context, self.ee,
                             node.struct_type, node.target_name,
                             lfunc, closure_type.signature, node.num_threads)
        self.keep_alive(lfunc_wrapper)
        self.keep_alive(lfunc_run)

        contexts = self.visit(node.contexts)
        num_threads = self.visit(node.num_threads.coerce(int_))
        self.builder.call(lfunc_run, [contexts, num_threads])
        return None

_count = 0
def get_threadpool_funcs(context, ee, context_struct_type, target_name,
                         lfunc, signature, num_threads):
    """
    Get functions to run the closure in separate threads.

        context:
            the Numba/Minivect context

        context_struct_type:
            the struct type holding all private and reduction variables
    """
    global _count
    _count += 1

    context_cbuilder_type = builder.CStruct.from_numba_struct(
                                    context, context_struct_type)
    context_p_ltype = context_struct_type.pointer().to_llvm(context)

    lobject = object_.to_llvm(context)

    class KernelWrapper(CDefinition):
        """
        Implements a prange kernel wrapper that is invoked in each thread.
        Implements:

            for i in range(start, stop, step):
                worker(closure_scope, privates)
        """

        _name_ = "prange_kernel_wrapper_%d" % _count
        _argtys_ = [
            ('context', C.void_p),
        ]

        def body(self, context_p):
            context_p = context_p.cast(context_p_ltype)
            context = context_p.as_struct(context_cbuilder_type)

            def gf(name):
                "Get a field named __numba_<name>"
                return getattr(context, '__numba_' + name)

            start = self.var(C.npy_intp, gf('start'))
            stop = self.var(C.npy_intp, gf('stop'))
            step = self.var(C.npy_intp, gf('step'))
            length = stop - start
            nsteps = self.var(C.npy_intp, length / step)
            zero = self.constant(C.npy_intp, 0)
            with self.ifelse(length % step != zero) as ifelse:
                with ifelse.then():
                    nsteps += self.constant(C.npy_intp, 1)

            with self.for_range(nsteps) as (loop, i):
                getattr(context, target_name).assign(start)

                worker = CFunc(self, lfunc)
                if signature.args[0].is_closure_scope:
                    worker(gf('closure_scope').cast(lobject), context_p)
                else:
                    worker(context_p)

                start += step

            self.ret()

    class RunThreadPool(CDefinition, parallel.ParallelMixin):
        """
        Function that spawns the thread pool.
        """

        _name_ = "invoke_prange_%d" % _count
        _argtys_ = [
            ('contexts', context_p_ltype),
            ('num_threads', C.int),
        ]

        def body(self, contexts, num_threads):
            callback = ee.get_pointer_to_function(wrapper_lfunc)
            callback = self.constant(Py_uintptr_t.to_llvm(context), callback)
            callback = callback.cast(C.void_p)
            self._dispatch_worker(callback, contexts, num_threads)
            self.ret()

    wrapper_lfunc = KernelWrapper()(lfunc.module)
    # print wrapper_lfunc
    run_threadpool_def = RunThreadPool()
    run_threadpool_lfunc = run_threadpool_def(lfunc.module)
    return wrapper_lfunc, run_threadpool_lfunc

def prange(start=0, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    return range(start, stop, step)

numba.prange = prange