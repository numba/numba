"""
Support for

    for i in numba.prange(...):
        ...
"""

import ast
import copy
import types
import ctypes

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
from numba import  _numba_types as numba_types
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

def outline_prange_body(context, outer_py_func, outer_symtab, subnode, **kwargs):
    global name_counter

    # Find referenced and assigned variables
    v = VariableFindingVisitor(context, outer_py_func, subnode, outer_symtab)
    v.visit(subnode)

    # Determine privates and reductions. Shared variables will be handled by
    # the closure support.
    fields = []
    reductions = []
    for var_name, op in v.assigned.iteritems():
        type = outer_symtab[var_name].type
        fields.append((var_name, type))
        if op is not None:
            reductions.append((var_name, op))

    # Build a wrapper function around the prange body, accepting a struct
    # of private variables as argument
    for i, (name, type) in enumerate(fields):
        if type is None:
            fields[i] = name, numba_types.DeferredType(name)

    privates_struct_type = numba.struct(fields)
    privates_struct = ast.Name('__numba_privates', ast.Param())
    privates_struct.type = privates_struct_type.pointer()
    privates_struct.variable = Variable(privates_struct.type)

    args = [privates_struct]
    func_def = ast.FunctionDef(name=template.temp_name("prange_body"),
                               args=ast.arguments(args=args, vararg=None,
                                                  kwarg=None, defaults=[]),
                               body=[subnode],
                               decorator_list=[])
    func_def.func_signature = void(privates_struct.type)
    func_def.need_closure_wrapper = False
    return func_def, privates_struct_type, reductions

class VariableFindingVisitor(visitors.NumbaVisitor):
    "Find referenced and assigned ast.Name nodes"
    def __init__(self, context, func, ast, outer_symtab, **kwargs):
        super(VariableFindingVisitor, self).__init__(context, func, ast, **kwargs)
        self.referenced = {}
        self.assigned = {}
        self.outer_symtab = outer_symtab

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
                # This will not be triggered, since the body is already
                # analyzed
                if (self.outer_symtab[target.id].type is None and
                        redop is not None):
                    raise error.NumbaError(
                            node, "Reduction variable %r must be "
                                  "initialized before the loop" % target.id)

                self.assigned[target.id] = redop

    def visit_Assign(self, node):
        self.generic_visit(node)
        self.register_assignment(node, node.targets[0], node.inplace_op)

    def visit_For(self, node):
        self.generic_visit(node)
        self.register_assignment(node, node.target, None)

    def visit_Name(self, node):
        self.referenced[node.id] = node


class PrangeType(numba_types.RangeType):
    is_prange = True

class PrangeNode(nodes.Node):
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

class PrangeTypeInfererMixin(PrangePrivatesReplacerMixin):
    """
    Rewrite prange() loops.
    """

    prange = 0

    def visit_Call(self, node):
        node.func = self.visit(node.func)

        func_type = node.func.variable.type
        if not (func_type.is_module_attribute and
                func_type.value is numba.prange):
            return super(PrangeTypeInfererMixin, self).visit_Call(node)

        ast_type_inference.no_keywords(node)
        start, stop, step = self.visitlist(transforms.unpack_range_args(node))
        node = PrangeNode(start, nodes.CoercionNode(stop, Py_ssize_t), step)
        return node

    def visit_For(self, node):
        node.iter = self.visit(node.iter)
        if node.iter.variable.type.is_prange:
            if self.prange:
                raise error.NumbaError(node, "Cannot nest prange")
            if node.orelse:
                raise error.NumbaError(node.orelse,
                                       "Else clause to prange not yet supported")

            prange_node = node.iter

            # Copy the pure AST for compile() during closure type inference
            pure_ast_body = copy.deepcopy(node.body)

            node.target = self.visit(node.target)
            self.assign(node.target, node.iter, Variable(Py_ssize_t))

            # Analyze types before outlining
            node.body = self.visitlist(node.body)

            # Outline function body
            result = outline_prange_body(self.context, self.func, self.symtab,
                                         ast.Suite(body=node.body))
            closure, privates_struct_type, reductions = result
            closure.pure_ast_body = pure_ast_body

            # Pack privates and reductions in a struct and inject function
            # as a closure
            self.prange += 1
            result = self.visit_PrangeNode(
                        prange_node, closure, privates_struct_type,
                        dict(reductions), node.target)
            self.prange -= 1

            return result

        node = super(PrangeTypeInfererMixin, self).visit_For(node)
        return node

    def visit_PrangeNode(self, node, func_def, struct_type,
                         reductions_dict, target):
        templ = template.TemplateContext(self.context, """
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

            #for $i in range({{num_threads}}):
            #    {{invoke_thread}}

            #for $i in range({{num_threads}}):
            #    {{join_thread}}

            # print "performing reductions"
            for $i in range({{num_threads}}):
                $reductions

            # print "unpacking lastprivates"
            $unpack_struct
        """)

        # Allocate context for each thread
        num_threads = 4
        contexts_array_type = minitypes.CArrayType(struct_type,
                                                   num_threads)

        # Create variables for template substitution
        nsteps = templ.temp_var('nsteps')
        temp_i = templ.temp_var('i', int32)
        contexts = templ.temp_var('contexts', contexts_array_type)
        temp_struct = templ.temp_var('temp_struct', struct_type)

        pack_struct, unpack_struct, reductions = templ.code_vars(
                        'pack_struct', 'unpack_struct', 'reductions')
        reductions.sep = "; "

        lastprivates_struct = "%s[{{num_threads}} - 1]" % contexts

        target_name = target.id
        struct_type.add_field(target_name, target.variable.type)

        # Create code for reductions and (last)privates
        for i, (name, type) in enumerate(struct_type.fields):
            if name != target_name and name in reductions_dict:
                reduction_op = reductions_dict[name]
                default = get_reduction_default(reduction_op)
                reductions.codes.append(
                        "%s %s= %s[%s].%s" % (name, reduction_op,
                                              contexts, temp_i, name))
                # reductions.codes.append('print "%s:", %s' % (name, name))
            else:
                if type.is_deferred:
                    self.symtab[name].deleted = True
                    continue
                default = name
                unpack_struct.codes.append(
                        "%s = %s.%s" % (name, lastprivates_struct, name))

            if self.symtab[name].type is not None:
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

        num_threads_node_ = nodes.const(num_threads, Py_ssize_t).cloneable
        num_threads_node = num_threads_node_.clone

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
            num_threads_=num_threads_node_,
            num_threads=num_threads_node,
            start=node.start,
            stop=node.stop,
            step=node.step)

        self.symtab.update(templ.get_vars_symtab())
        tree = templ.template(subs)
        result = self.visit(tree)

        return result


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
        self.func.live_objects.extend((lfunc_wrapper, lfunc_run))

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
            self._dispatch_worker(callback, contexts,  num_threads)
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