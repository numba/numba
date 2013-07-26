# -*- coding: utf-8 -*-

"""
Support for

    for i in numba.prange(...):
        ...

The implementation isn't particularly good, and should be greatly simplified
at some point.
"""

from __future__ import print_function, division, absolute_import

import ast
import copy
import warnings
import multiprocessing

try:
    NUM_THREADS = multiprocessing.cpu_count()
except NotImplementedError:
    warnings.warn("Unable to determine cpu count, assuming 2")
    NUM_THREADS = 2

import llvm.core
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder import builder

import numba.decorators
from numba import *
from numba import error, visitors, nodes, templating
from numba.minivect import  minitypes
from numba import typesystem, pipeline
from numba.type_inference import infer
from numba.specialize.loops import unpack_range_args

from numba import threads

opmap = {
    # Unary
    ast.Invert: '~',
    ast.Not: None, # not supported
    ast.UAdd: '+',
    ast.USub: '-',

    # Binary
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Mod: '%',
    ast.Pow: '**',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.BitAnd: '&',
    ast.FloorDiv: '//',

    # Comparison
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Is: None,
    ast.IsNot: None,
    ast.In: None,
    ast.NotIn: None,
}


import logging
logger = logging.getLogger(__name__)

def get_reduction_op(op):
    # TODO: recognize invalid operators
    op = type(op)

    reduction_op = opmap[op]
    reduction_ops = {'-': '+', '/': '*'}
    if reduction_op in reduction_ops:
        reduction_op = reduction_ops[reduction_op]

    return reduction_op

def get_reduction_default(op):
    defaults = {
        '+': 0, '-': 0, '*': 1, '/': 1,
    }
    return defaults[op]

class VariableFindingVisitor(visitors.VariableFindingVisitor):
    "Find referenced and assigned ast.Name nodes"
    def __init__(self):
        super(VariableFindingVisitor, self).__init__()
        self.reductions = {}

    def register_assignment(self, node, target, op):
        if isinstance(target, ast.Name):
            self.assigned.add(target.id)

            if op is None:
                redop = op
            else:
                redop = get_reduction_op(op)

            if target.id in self.reductions:
                previous_op = self.reductions[target.id]
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
            elif op:
                self.reductions[target.id] = redop

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            self.register_assignment(node, node.targets[0],
                                     getattr(node, 'inplace_op', None))


def create_prange_closure(env, prange_node, body, target):
    # Find referenced and assigned variables
    v = VariableFindingVisitor()
    v.visitlist(body)

    # Determine privates and reductions. Shared variables will be handled by
    # the closure support.
    privates = set(v.assigned) - set(v.reductions)
    reductions = v.reductions

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

    # Update outlined prange body closure

    func_signature = void(privates_struct_type.ref())
    # func_signature.struct_by_reference = True
    need_closure_wrapper = False
    locals_dict = { '__numba_privates': privates_struct_type.ref() }

    func_env = env.translation.make_partial_env(
        func_def,
        func_signature=func_signature,
        need_closure_wrapper=need_closure_wrapper,
        locals=locals_dict,
    )

    # Update prange node
    prange_node.func_env = func_env
    prange_node.privates_struct_type = privates_struct_type
    prange_node.privates = privates
    prange_node.reductions = reductions
    prange_node.func_def = func_def


prange_template = """
{{func_def}}
%s # function name; avoid warning about unused variable

$pack_struct

$nsteps = ({{stop}} - {{start}}) / ({{step}} * {{num_threads}})
for $i in range({{num_threads}}):
    $temp_struct.__numba_closure_scope = {{closure_scope}}
    $temp_struct.__numba_start = {{start}} + $i * {{step}} * $nsteps
    $temp_struct.__numba_stop = $temp_struct.__numba_start + {{step}} * $nsteps
    $temp_struct.__numba_step = {{step}}
    $contexts[$i] = $temp_struct

    # print "temp struct", $temp_struct.__numba_start, \
    #       $temp_struct.__numba_stop, {{step}}, $nsteps

# Undo any truncation, don't use $i here, range() doesn't
# have py semantics yet
$contexts[{{num_threads}} - 1].__numba_stop = {{stop}}

# print "invoking..."
{{invoke_and_join_threads}}

$lastprivates = $contexts[{{num_threads}} - 1]
"""

def kill_attribute_assignments(env, prange_node, temporaries):
    """
    Remove attribute assignments from the list of statements that need to
    be resolved before type inference.
    """
    func_env = env.translation.crnt
    kill_set = func_env.kill_attribute_assignments
    kill_set.update(temporaries)
    kill_set.update(prange_node.privates)
    kill_set.update(prange_node.reductions)

def rewrite_prange(env, prange_node, target, locals_dict, closures_dict):
    func_def = prange_node.func_def
    struct_type = prange_node.privates_struct_type

    templ = templating.TemplateContext(env.context,
                                       prange_template % func_def.name)

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

    if isinstance(target, ast.Name):
        target_name = target.id
        struct_type.add_field(target_name, Py_ssize_t)
    else:
        raise error.NumbaError(
            prange_node, "Only name target for prange is currently supported")

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
    # TODO: UNDO monkeypatching
    func_def.type = prange_node.func_env.func_signature
    func_def.is_prange_body = True
    func_def.prange_node = prange_node

    num_threads_node = nodes.const(num_threads, Py_ssize_t)

    invoke = InvokeAndJoinThreads(env, contexts=contexts.node,
                                  func_def_name=func_def.name,
                                  struct_type=struct_type,
                                  target_name=target_name,
                                  num_threads=num_threads_node,
                                  closures=closures_dict)

    closure_scope = nodes.ClosureScopeLoadNode()
    subs = dict(
        func_def=func_def,
        func_def_name=func_def.name,
        closure_scope=closure_scope,
        invoke_and_join_threads=invoke,
        num_threads=num_threads_node,
        start=nodes.UntypedCoercion(prange_node.start, Py_ssize_t),
        stop=nodes.UntypedCoercion(prange_node.stop, Py_ssize_t),
        step=nodes.UntypedCoercion(prange_node.step, Py_ssize_t),
    )

    tree = templ.template(subs)

    temporaries = {}
    templ.update_locals(temporaries)
    locals_dict.update(temporaries)

    kill_attribute_assignments(env, prange_node, temporaries)

    # TODO: Make this an SSA variable
    locals_dict[target_name] = Py_ssize_t

    prange_node.target = target
    prange_node.num_threads_node = num_threads_node #.clone
    prange_node.template_vars = {
                                  'contexts': contexts,
                                  'i': temp_i,
                                  'lastprivates': lastprivates,
                                  'nsteps': nsteps,
                                }

    # print(templ.substituted_template)
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

    target_name = ""
    if isinstance(prange_node.target, ast.Name):
        target_name = prange_node.target.id

    for name in prange_node.privates:
        # Generate: x += contexts[num_threads - 1].x
        expr = "%s.%s" % (getvar("lastprivates"), name)
        assmnt = assign(name, expr)

        if name == target_name:
            assmnt = "if %s > 0: %s" % (getvar("nsteps"), assmnt)

        unpack_struct.codes.append(assmnt)

    substitutions = { "num_threads": prange_node.num_threads_node }
    result = templ.template(substitutions)

    # print(templ.substituted_template)
    return result

#------------------------------------------------------------------------
# prange nodes and types
#------------------------------------------------------------------------

class PrangeType(typesystem.NumbaType):
    is_prange = True
    is_range = True

class PrangeNode(nodes.ExprNode):
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
    target = None                   # Target iteration variable

    num_threads_node = None         # num_threads CloneNode
    template_vars = None            # { "template_var_name": $template_var }

    def __init__(self, start, stop, step, **kwargs):
        super(PrangeNode, self).__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.step = step
        self.type = PrangeType()


class InvokeAndJoinThreads(nodes.UserNode):
    """
    contexts
        contexts array node (array of privates structs)
    num_threads
        num threads node

    func_def_name
        name of outlined prange body function
    target_name
        name of iteration target variable (e.g. 'i')
    struct_type
        privates struct type
    closures
        { closure_name : closure_node }
    """

    _fields = ['contexts', 'num_threads']

    def __init__(self, env, **kwargs):
        super(InvokeAndJoinThreads, self).__init__(**kwargs)
        self.env = env

    def infer_types(self, type_inferer):
        type_inferer.visitchildren(self)
        return self

    def build_wrapper(self, codegen):
        closure_type = self.closures[self.func_def_name].type
        lfunc = closure_type.closure.lfunc
        lfunc_pointer = closure_type.closure.lfunc_pointer

        KernelWrapper, RunThreadPool = get_threadpool_funcs(
            codegen.context,
            self.struct_type,
            self.target_name,
            lfunc,
            lfunc_pointer,
            closure_type.signature,
            self.num_threads,
            codegen.llvm_module)

        kernel_wrapper = nodes.LLVMCBuilderNode(self.env, KernelWrapper, None)
        run_threadpool = nodes.LLVMCBuilderNode(self.env, RunThreadPool, None,
                                                dependencies=[kernel_wrapper])
        lfunc_run = codegen.visit(run_threadpool)
        return lfunc_run

    def codegen(self, codegen):
        contexts = codegen.visit(self.contexts)
        num_threads = codegen.visit(self.num_threads.coerce(int_))

        lfunc_run = self.build_wrapper(codegen)
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
    fielddict.update(privates_struct_type.fielddict)
    fields = numba.struct(**fielddict).fields

    privates_struct_type.fields[:] = fields
    # privates_struct_type.fielddict = fielddict
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

    def visit_FunctionDef(self, node):
        if self.func_level == 0:
            node = self.visit_func_children(node)

        return node

    def match_global(self, node, expected_value):
        if isinstance(node, ast.Name) and node.id not in self.local_names:
            value = self.func_globals.get(node.id, None)
            return value is expected_value
        return False

    def is_numba_prange(self, node):
        return (self.match_global(node, prange) or
                (isinstance(node, ast.Attribute) and node.attr == "prange" and
                 self.match_global(node.value, numba)))

    def visit_Call(self, node):
        if self.is_numba_prange(node.func):
            infer.no_keywords(node)
            start, stop, step = self.visitlist(unpack_range_args(node))
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
        create_prange_closure(self.env, prange_node, node.body, node.target)

        # setup glue code
        pre_loop = rewrite_prange(self.env, prange_node, node.target,
                                  self.locals, self.closures)
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

        nodes.delete_control_blocks(node, self.ast.flow)
        return None


class PrangePrivatesReplacer(visitors.NumbaTransformer):
    """
    Rewrite private variables to accesses on the privates before
    closure type inference (closure type inference of the outer function, and
    type inference (and control flow analysis) of the inner function).
    """

    in_prange_closure = 0

    def visit_FunctionDef(self, node):
        """
        Analyse immedidate prange functions (not ones in closures).

        Don't re-analyze prange functions when the prange function closures
        themselves are compiled.
        """
        if getattr(node, 'is_prange_body', False) and self.func_level == 0:
            prange_node = node.prange_node
            self.privates_struct_type = prange_node.privates_struct_type

            node.body = [nodes.WithNoPythonNode(body=node.body)]

            self.in_prange_closure += 1
            self.visit_func_children(node)
            self.in_prange_closure -= 1

            self.invalidate_locals(node)
            self.invalidate_locals()
        else:
            self.visit_func_children(node)

        return node

    def visit_Name(self, node):
        if self.in_prange_closure:
            if node.id in self.privates_struct_type.fielddict:
                privates_struct = ast.Name('__numba_privates', ast.Load()) #node.ctx)
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
def get_threadpool_funcs(context, context_struct_type, target_name,
                         lfunc, lfunc_pointer, signature,
                         num_threads, llvm_module):
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

        def __init__(self, dependencies, ldependencies, **kwargs):
            super(KernelWrapper, self).__init__(**kwargs)

        def dispatch(self, context_struct_p, context_getfield,
                     lfunc, lfunc_pointer):
            """
            Call the closure with the closure scope and context arguments.
            We don't directly call the lfunc since there are linkage issues.
            """
            if signature.args[0].is_closure_scope:
                llvm_object_type = object_.to_llvm(context)
                closure_scope = context_getfield('closure_scope')
                closure_scope = closure_scope.cast(llvm_object_type)
                args = [closure_scope, context_struct_p]
            else:
                args = [context_struct_p]

            # Get the LLVM arguments
            llargs = [arg.handle for arg in args]

            # Get the LLVM pointer to the function
            lfunc_pointer = llvm.core.Constant.int(Py_uintptr_t.to_llvm(context),
                                                   lfunc_pointer)
            lfunc_pointer = self.builder.inttoptr(lfunc_pointer, lfunc.type)

            self.builder.call(lfunc_pointer, llargs)

        def body(self, context_p):
            context_struct_p = context_p.cast(context_p_ltype)
            context_struct = context_struct_p.as_struct(context_cbuilder_type)

            def context_getfield(name):
                "Get a field named __numba_<name>"
                return getattr(context_struct, '__numba_' + name)

            start = self.var(C.npy_intp, context_getfield('start'))
            stop = self.var(C.npy_intp, context_getfield('stop'))
            step = self.var(C.npy_intp, context_getfield('step'))

            length = stop - start
            nsteps = self.var(C.npy_intp, length / step)

            zero = self.constant(C.npy_intp, 0)

            with self.ifelse(length % step != zero) as ifelse:
                with ifelse.then():
                    nsteps += self.constant(C.npy_intp, 1)

            # self.debug("start", start, "stop", stop, "step", step)
            with self.for_range(nsteps) as (loop, i):
                getattr(context_struct, target_name).assign(start)
                self.dispatch(context_struct_p, context_getfield,
                              lfunc, lfunc_pointer)
                start += step

            self.ret()

        def specialize(self, *args, **kwargs):
            self._name_ = "__numba_kernel_wrapper_%s" % lfunc.name

    class RunThreadPool(CDefinition, threads.ParallelMixin):
        """
        Function that spawns the thread pool.
        """

        _name_ = "invoke_prange_%d" % _count
        _argtys_ = [
            ('contexts', context_p_ltype),
            ('num_threads', C.int),
        ]

        def __init__(self, dependencies, ldependencies, **kwargs):
            self.kernel_wrapper, = dependencies
            super(RunThreadPool, self).__init__(**kwargs)

        def body(self, contexts, num_threads):
            callback = self.kernel_wrapper.pointer
            callback = self.constant(Py_uintptr_t.to_llvm(context), callback)
            callback = callback.cast(C.void_p)
            self._dispatch_worker(callback, contexts, num_threads)
            self.ret()

        def specialize(self, *args, **kwargs):
            self._name_ = "__numba_run_threadpool_%s" % lfunc.name

    return KernelWrapper, RunThreadPool

#----------------------------------------------------------------------------
# The actual prange function
#----------------------------------------------------------------------------

def prange(start=0, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    return range(start, stop, step)

# numba.prange = prange
