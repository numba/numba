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
from numba import (error, visitors, nodes, templating, ast_type_inference,
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


def create_prange_closure(prange_node, body, target):
    # Find referenced and assigned variables
    v = VariableFindingVisitor()
    v.visitlist(body)

    # Determine privates and reductions. Shared variables will be handled by
    # the closure support.
    assigned = v.assigned.items()
    privates = set(name for name, op in assigned if op is None)
    reductions = dict((name, op) for name, op in assigned if op is not None)

    if isinstance(target, ast.Name) and target.id in reductions:
        # Remove target variable from reductions if present
        reductions.pop(target.id)
        privates.add(target.id)

    privates_struct_type = numba.struct([])
    privates_struct = ast.Name('__numba_privates', ast.Param())

    args = [privates_struct]
    func_def = ast.FunctionDef(name=templating.temp_name("prange_body"),
                               args=ast.arguments(args=args, vararg=None,
                                                  kwarg=None, defaults=[]),
                               body=copy.deepcopy(body),
                               decorator_list=[])

    func_def.func_signature = void(privates_struct_type)
    func_def.need_closure_wrapper = False

    prange_node.privates_struct_type = privates_struct_type
    prange_node.privates = privates
    prange_node.reductions = reductions
    prange_node.func_def = func_def


prange_template = """
{{func_def}}

$pack_struct

$nsteps = ({{stop}} - {{start}}) / ({{step}} * {{num_threads}})
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

def rewrite_prange(context, prange_node, target, locals_dict):
    templ = templating.TemplateContext(context, prange_template)

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
    lastprivates = templ.temp_var("lastprivates")

    pack_struct = templ.code_var('pack_struct')

    target_name = target.id
    struct_type.add_field(target_name, Py_ssize_t)

    # Create code for reductions and (last)privates
    for name, reduction_op in prange_node.reductions.iteritems():
        default = get_reduction_default(reduction_op)
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
    func_def.prange_node = prange_node

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
    templ.update_locals(locals_dict)

    prange_node.num_threads_node = subs["num_threads"]
    prange_node.template_vars = {
                                  'contexts': contexts,
                                  'i': temp_i,
                                  'lastprivates': lastprivates,
                                }

    return tree

def typeof(name, expr):
    return "__numba_typeof(%s, %s)" % (name, expr)

def assign(name, expr):
    return "%s = %s" % (name, typeof(name, expr))

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
    templ = templating.TemplateContext(context, post_prange_template)

    unpack_struct, reductions = templ.code_vars('unpack_struct', 'reductions')
    reductions.sep = "; "

    getvar = prange_node.template_vars.get
    templ.add_variable(getvar("i"))

    # Create code for reductions and (last)privates
    for name, reduction_op in prange_node.reductions.iteritems():
        # Generate: x += contexts[i].x
        expr = "%s %s %s[%s].%s" % (name, reduction_op,
                                    getvar("contexts"), getvar("i"),
                                    name)
        reductions.codes.append(assign(name, expr))

    for name in prange_node.privates:
        # Generate: x += contexts[num_threads - 1].x
        expr = "%s.%s" % (getvar("lastprivates"), name)
        unpack_struct.codes.append(assign(name, expr))

    substitutions = { "num_threads": prange_node.num_threads_node }
    result = templ.template(substitutions)
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
    """

    _fields = ['start', 'stop', 'step']

    func_def = None                 # outlined prange closure body
    privates_struct_type = None     # numba.struct(var_name=var_type)
    privates = None                 # set([var_name])
    reductions = None               # { var_name: reduction_op }

    num_threads_node = None         # num_threads CloneNode
    template_vars = None            # { "template_var_name": $template_var }

    def __init__(self, start, stop, step, **kwargs):
        super(PrangeNode, self).__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.step = step
        self.type = PrangeType()

class PrangeBodyNode(nodes.Node):
    _fields = ['func_def']

class InvokeAndJoinThreads(nodes.UserNode):

    _fields = ['contexts', 'num_threads']

    def infer_types(self, type_inferer):
        type_inferer.visitchildren(self)
        return self

    def codegen(self, codegen):
        closure_type = codegen.symtab[node.func_def_name].type
        lfunc = closure_type.closure.lfunc
        # print lfunc
        lfunc_wrapper, lfunc_run = get_threadpool_funcs(
                             codegen.context, codegen.ee,
                             node.struct_type, node.target_name,
                             lfunc, closure_type.signature, node.num_threads)
        codegen.keep_alive(lfunc_wrapper)
        codegen.keep_alive(lfunc_run)

        contexts = codegen.visit(node.contexts)
        num_threads = codegen.visit(node.num_threads.coerce(int_))
        codegen.builder.call(lfunc_run, [contexts, num_threads])
        return None

class TypeofNode(nodes.UserNode):

    _fields = ["name", "expr"]

    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

    def infer_types(self, type_inferer):
        if type_inferer.analyse:
            return type_inferer.visit(self.expr)

        self.name = type_inferer.visit(self.name)
        self.type = self.name.variable.type
        return self


def make_privates_struct_type(privates_struct_type, names):
    """
    Update the struct of privates and reductions once we know the
    field types.
    """
    fielddict = dict((name.id, name.variable.type) for name in names)
    fields = numba.struct(**fielddict).fields

    privates_struct_type.fields = fields
    privates_struct_type.fielddict = fielddict
    privates_struct_type.update_mutated()


class VariableTypeInferingNode(nodes.UserNode):

    _fields = ["names", "pre_prange_code"]

    def __init__(self, variable_names, privates_struct_type):
        super(VariableTypeInferingNode, self).__init__()
        self.privates_struct_type = privates_struct_type

        self.names = []
        for varname in variable_names:
            self.names.append(ast.Name(id=varname, ctx=ast.Load()))

    def infer_types(self, type_inferer):
        type_inferer.visitchildren(self)
        make_privates_struct_type(self.privates_struct_type, self.names)
        return None


#------------------------------------------------------------------------
# prange visitors
#------------------------------------------------------------------------

class PrangeExpander(visitors.NumbaTransformer):
    """
    Rewrite 'for i in prange(...): ...' before the control flow pass.
    """

    prange = 0

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
        if self.is_numba_prange(node.func):
            ast_type_inference.no_keywords(node)
            start, stop, step = self.visitlist(transforms.unpack_range_args(node))
            node = PrangeNode(start, stop, step)

        self.visitchildren(node)
        return node

    def error_check_prange(self, node):
        if self.prange:
            raise error.NumbaError(node, "Cannot nest prange")
        if node.orelse:
            raise error.NumbaError(node.orelse,
                                   "Else clause to prange not yet supported")

    def visit_For(self, node):
        node.iter = self.visit(node.iter)

        if not isinstance(node.iter, PrangeNode):
            self.visitchildren(node)
            return node

        self.error_check_prange(node)
        node.target = self.visit(node.target)

        self.prange += 1
        node.body = self.visitlist(node.body)
        self.prange -= 1

        # Create prange closure
        prange_node = node.iter
        create_prange_closure(prange_node, node.body, node.target)

        # setup glue code
        pre_loop = rewrite_prange(self.context, prange_node, node.target,
                                  self.locals)
        post_loop = perform_reductions(self.context, prange_node)

        # infer glue code at the right place
        pre_loop_dont_infer = nodes.dont_infer(pre_loop)
        pre_loop_infer_now = nodes.infer_now(pre_loop, pre_loop_dont_infer)

        # infer the type of the struct of privates right after the loop
        allprivates = set(prange_node.privates) | set(prange_node.reductions)
        type = prange_node.privates_struct_type
        infer_privates_struct = VariableTypeInferingNode(allprivates, type)

        # Signal that we now have additional local variables
        self.invalidate_locals()

        return ast.Suite(body=[
            pre_loop_dont_infer,
            node,
            infer_privates_struct,
            pre_loop_infer_now,
            post_loop])


class PrangeCleanup(visitors.NumbaTransformer):
    """
    Clean up outlined prange loops after type inference (removes them entirely).
    """

    def visit_For(self, node):
        if not node.iter.variable.type.is_prange:
            self.visitchildren(node)
            return node

        return None


class PrangePrivatesReplacer(visitors.NumbaTransformer):
    """
    Rewrite private variables to accesses on the privates before
    closure type inference (closure type inference of the outer function, and
    type inference (and control flow analysis) of the inner function).
    """

    in_prange_closure = 0

    def visit_FunctionDef(self, node):
        if getattr(node, 'is_prange_body', False):
            prange_node = node.prange_node
            self.privates_struct_type = prange_node.privates_struct_type

            node.body = [nodes.WithNoPythonNode(body=node.body)]

            self.in_prange_closure += 1
            self.visitchildren(node)
            self.in_prange_closure -= 1

            self.invalidate_locals(node)
            self.invalidate_locals()
        else:
            self.visitchildren(node)

        return node

    def visit_Name(self, node):
        if self.in_prange_closure:
            if node.id in self.privates_struct_type.fielddict:
                privates_struct = ast.Name('__numba_privates', node.ctx)
                result = ast.Attribute(value=privates_struct,
                                       attr=node.id,
                                       ctx=node.ctx)
                return result

        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "__numba_typeof":
            return TypeofNode(node.args[0], node.args[1])

        self.visitchildren(node)
        return node

#----------------------------------------------------------------------------
# LLVM cbuilder prange utilities
#----------------------------------------------------------------------------

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


#----------------------------------------------------------------------------
# The actual prange function
#----------------------------------------------------------------------------

def prange(start=0, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    return range(start, stop, step)

numba.prange = prange