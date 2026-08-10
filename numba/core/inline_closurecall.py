import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import (types, errors, ir, config, ir_utils)
from numba.core.ir_utils import (
    next_label,
    add_offset_to_labels,
    replace_vars,
    remove_dels,
    rename_labels,
    find_topo_order,
    require,
    guard,
    get_definition,
    find_callname,
    get_ir_of_code,
    simplify_CFG,
    canonicalize_array_math,
    dead_code_elimination,
)

from numba.core.analysis import (
    compute_cfg_from_blocks)

from numba.core import postproc

"""
Variable enable_inline_arraycall is only used for testing purpose.
"""
enable_inline_arraycall = True


def callee_ir_validator(func_ir):
    """Checks the IR of a callee is supported for inlining
    """
    for blk in func_ir.blocks.values():
        for stmt in blk.find_insts(ir.Assign):
            if isinstance(stmt.value, ir.Yield):
                msg = "The use of yield in a closure is unsupported."
                raise errors.UnsupportedError(msg, loc=stmt.loc)


def _created_inlined_var_name(function_name, var_name):
    """Creates a name for an inlined variable based on the function name and the
    variable name. It does this "safely" to avoid the use of characters that are
    illegal in python variable names as there are occasions when function
    generation needs valid python name tokens."""
    inlined_name = f'{function_name}.{var_name}'
    # Replace angle brackets, e.g. "<locals>" is replaced with "_locals_"
    new_name = inlined_name.replace('<', '_').replace('>', '_')
    # The version "version" of the closure function e.g. foo$2 (id 2) is
    # rewritten as "foo_v2". Further "." is also replaced with "_".
    new_name = new_name.replace('.', '_').replace('$', '_v')
    return new_name


class InlineClosureCallPass(object):
    """InlineClosureCallPass class looks for direct calls to locally defined
    closures, and inlines the body of the closure function to the call site.
    """

    def __init__(self, func_ir, parallel_options, swapped=None, typed=False):
        if swapped is None:
            swapped = {}
        self.func_ir = func_ir
        self.parallel_options = parallel_options
        self.swapped = swapped
        self._processed_stencils = []
        self.typed = typed

    def run(self):
        """Run inline closure call pass.
        """
        # Analysis relies on ir.Del presence, strip out later
        pp = postproc.PostProcessor(self.func_ir)
        pp.run(True)

        modified = False
        work_list = list(self.func_ir.blocks.items())
        debug_print = _make_debug_print("InlineClosureCallPass")
        debug_print(f"START {self.func_ir.func_id.func_qualname}")
        while work_list:
            _label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        call_name = guard(find_callname, self.func_ir, expr)
                        func_def = guard(get_definition, self.func_ir,
                                         expr.func)

                        if guard(self._inline_reduction,
                                 work_list, block, i, expr, call_name):
                            modified = True
                            break # because block structure changed

                        if guard(self._inline_closure,
                                 work_list, block, i, func_def):
                            modified = True
                            break # because block structure changed

                        if guard(self._inline_stencil,
                                 instr, call_name, func_def):
                            modified = True

        if hasattr(self, "_inline_arraycall") and enable_inline_arraycall:
            self._inline_arraycall(modified, debug_print)

        if modified:
            # clean up now dead/unreachable blocks, e.g. unconditionally raising
            # an exception in an inlined function would render some parts of the
            # inliner unreachable
            cfg = compute_cfg_from_blocks(self.func_ir.blocks)
            for dead in cfg.dead_nodes():
                del self.func_ir.blocks[dead]

            # run dead code elimination
            dead_code_elimination(self.func_ir)
            # do label renaming
            self.func_ir.blocks = rename_labels(self.func_ir.blocks)

        # inlining done, strip dels
        remove_dels(self.func_ir.blocks)

        debug_print("END")

    def _inline_reduction(self, work_list, block, i, expr, call_name):
        # only inline reduction in sequential execution, parallel handling
        # is done in ParforPass.
        require(not self.parallel_options.reduction)
        require(call_name == ('reduce', 'builtins') or
                call_name == ('reduce', '_functools'))
        if len(expr.args) not in (2, 3):
            raise TypeError("invalid reduce call, "
                            "two arguments are required (optional initial "
                            "value can also be specified)")
        check_reduce_func(self.func_ir, expr.args[0])

        def reduce_func(f, A, v=None):
            it = iter(A)
            if v is not None:
                s = v
            else:
                s = next(it)
            for a in it:
                s = f(s, a)
            return s

        inline_closure_call(
            self.func_ir, self.func_ir.func_id.func.__globals__,
            block, i, reduce_func, work_list=work_list,
            callee_validator=callee_ir_validator
        )
        return True

    def _inline_stencil(self, instr, call_name, func_def):
        from numba.stencils.stencil import StencilFunc
        lhs = instr.target
        expr = instr.value
        # We keep the escaping variables of the stencil kernel
        # alive by adding them to the actual kernel call as extra
        # keyword arguments, which is ignored anyway.
        if (isinstance(func_def, ir.Global) and
                func_def.name == 'stencil' and
                isinstance(func_def.value, StencilFunc)):
            if expr.kws:
                expr.kws += func_def.value.kws
            else:
                expr.kws = func_def.value.kws
            return True
        # Otherwise we proceed to check if it is a call to numba.stencil
        require(call_name == ('stencil', 'numba.stencils.stencil') or
                call_name == ('stencil', 'numba'))
        require(expr not in self._processed_stencils)
        self._processed_stencils.append(expr)
        if not len(expr.args) == 1:
            raise ValueError("As a minimum Stencil requires"
                             " a kernel as an argument")
        stencil_def = guard(get_definition, self.func_ir, expr.args[0])
        require(isinstance(stencil_def, ir.Expr) and
                stencil_def.op == "make_function")
        kernel_ir = get_ir_of_code(self.func_ir.func_id.func.__globals__,
                                   stencil_def.code)
        options = dict(expr.kws)
        if 'neighborhood' in options:
            fixed = guard(self._fix_stencil_neighborhood, options)
            if not fixed:
                raise ValueError(
                    "stencil neighborhood option should be a tuple"
                    " with constant structure such as ((-w, w),)"
                )
        if 'index_offsets' in options:
            fixed = guard(self._fix_stencil_index_offsets, options)
            if not fixed:
                raise ValueError(
                    "stencil index_offsets option should be a tuple"
                    " with constant structure such as (offset, )"
                )
        sf = StencilFunc(kernel_ir, 'constant', options)
        sf.kws = expr.kws # hack to keep variables live
        sf_global = ir.Global('stencil', sf, expr.loc)
        self.func_ir._definitions[lhs.name] = [sf_global]
        instr.value = sf_global
        return True

    def _fix_stencil_neighborhood(self, options):
        """
        Extract the two-level tuple representing the stencil neighborhood
        from the program IR to provide a tuple to StencilFunc.
        """
        # build_tuple node with neighborhood for each dimension
        dims_build_tuple = get_definition(self.func_ir, options['neighborhood'])
        require(hasattr(dims_build_tuple, 'items'))
        res = []
        for window_var in dims_build_tuple.items:
            win_build_tuple = get_definition(self.func_ir, window_var)
            require(hasattr(win_build_tuple, 'items'))
            res.append(tuple(win_build_tuple.items))
        options['neighborhood'] = tuple(res)
        return True

    def _fix_stencil_index_offsets(self, options):
        """
        Extract the tuple representing the stencil index offsets
        from the program IR to provide to StencilFunc.
        """
        offset_tuple = get_definition(self.func_ir, options['index_offsets'])
        require(hasattr(offset_tuple, 'items'))
        options['index_offsets'] = tuple(offset_tuple.items)
        return True

    def _inline_closure(self, work_list, block, i, func_def):
        require(isinstance(func_def, ir.Expr) and
                func_def.op == "make_function")
        inline_closure_call(self.func_ir,
                            self.func_ir.func_id.func.__globals__,
                            block, i, func_def, work_list=work_list,
                            callee_validator=callee_ir_validator)
        return True


def check_reduce_func(func_ir, func_var):
    """Checks the function at func_var in func_ir to make sure it's amenable
    for inlining. Returns the function itself"""
    reduce_func = guard(get_definition, func_ir, func_var)
    if reduce_func is None:
        raise ValueError("Reduce function cannot be found for njit \
                            analysis")
    if isinstance(reduce_func, (ir.FreeVar, ir.Global)):
        if not isinstance(reduce_func.value,
                          numba.core.registry.CPUDispatcher):
            raise ValueError("Invalid reduction function")
        # pull out the python function for inlining
        reduce_func = reduce_func.value.py_func
    elif not (hasattr(reduce_func, 'code')
              or hasattr(reduce_func, '__code__')):
        raise ValueError("Invalid reduction function")
    f_code = (reduce_func.code
              if hasattr(reduce_func, 'code')
              else reduce_func.__code__)
    if not f_code.co_argcount == 2:
        raise TypeError("Reduction function should take 2 arguments")
    return reduce_func


class InlineWorker(object):
    """ A worker class for inlining, this is a more advanced version of
    `inline_closure_call` in that it permits inlining from function type, Numba
    IR and code object. It also, runs the entire untyped compiler pipeline on
    the inlinee to ensure that it is transformed as though it were compiled
    directly.
    """

    def __init__(self,
                 typingctx=None,
                 targetctx=None,
                 locals=None,
                 pipeline=None,
                 flags=None,
                 validator=callee_ir_validator,
                 typemap=None,
                 calltypes=None):
        """
        Instantiate a new InlineWorker, all arguments are optional though some
        must be supplied together for certain use cases. The methods will refuse
        to run if the object isn't configured in the manner needed. Args are the
        same as those in a numba.core.Compiler.state, except the validator which
        is a function taking Numba IR and validating it for use when inlining
        (this is optional and really to just provide better error messages about
        things which the inliner cannot handle like yield in closure).
        """
        def check(arg, name):
            if arg is None:
                raise TypeError("{} must not be None".format(name))

        from numba.core.compiler import DefaultPassBuilder

        # check the stuff needed to run the more advanced compilation pipeline
        # is valid if any of it is provided
        compiler_args = (targetctx, locals, pipeline, flags)
        compiler_group = [x is not None for x in compiler_args]
        if any(compiler_group) and not all(compiler_group):
            check(targetctx, 'targetctx')
            check(locals, 'locals')
            check(pipeline, 'pipeline')
            check(flags, 'flags')
        elif all(compiler_group):
            check(typingctx, 'typingctx')

        self._compiler_pipeline = DefaultPassBuilder.define_untyped_pipeline

        self.typingctx = typingctx
        self.targetctx = targetctx
        self.locals = locals
        self.pipeline = pipeline
        self.flags = flags
        self.validator = validator
        self.debug_print = _make_debug_print("InlineWorker")

        # check whether this inliner can also support typemap and calltypes
        # update and if what's provided is valid
        pair = (typemap, calltypes)
        pair_is_none = [x is None for x in pair]
        if any(pair_is_none) and not all(pair_is_none):
            msg = ("typemap and calltypes must both be either None or have a "
                   "value, got: %s, %s")
            raise TypeError(msg % pair)
        self._permit_update_type_and_call_maps = not all(pair_is_none)
        self.typemap = typemap
        self.calltypes = calltypes

    def inline_ir(self, caller_ir, block, i, callee_ir, callee_freevars,
                  arg_typs=None):
        """ Inlines the callee_ir in the caller_ir at statement index i of block
        `block`, callee_freevars are the free variables for the callee_ir. If
        the callee_ir is derived from a function `func` then this is
        `func.__code__.co_freevars`. If `arg_typs` is given and the InlineWorker
        instance was initialized with a typemap and calltypes then they will be
        appropriately updated based on the arg_typs.
        """

        # Always copy the callee IR, it gets mutated
        def copy_ir(the_ir):
            kernel_copy = the_ir.copy()
            kernel_copy.blocks = {}
            for block_label, block in the_ir.blocks.items():
                new_block = copy.deepcopy(the_ir.blocks[block_label])
                kernel_copy.blocks[block_label] = new_block
            return kernel_copy

        callee_ir = copy_ir(callee_ir)

        # check that the contents of the callee IR is something that can be
        # inlined if a validator is present
        if self.validator is not None:
            self.validator(callee_ir)

        # save an unmutated copy of the callee_ir to return
        callee_ir_original = copy_ir(callee_ir)
        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        callee_blocks = callee_ir.blocks

        # 1. relabel callee_ir by adding an offset
        max_label = max(
            ir_utils._the_max_label.next(),
            max(caller_ir.blocks.keys()),
        )
        callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
        callee_blocks = simplify_CFG(callee_blocks)
        callee_ir.blocks = callee_blocks
        min_label = min(callee_blocks.keys())
        max_label = max(callee_blocks.keys())
        #    reset globals in ir_utils before we use it
        ir_utils._the_max_label.update(max_label)
        self.debug_print("After relabel")
        _debug_dump(callee_ir)

        # 2. rename all local variables in callee_ir with new locals created in
        # caller_ir
        callee_scopes = _get_all_scopes(callee_blocks)
        self.debug_print("callee_scopes = ", callee_scopes)
        #    one function should only have one local scope
        assert (len(callee_scopes) == 1)
        callee_scope = callee_scopes[0]
        var_dict = {}
        for var in tuple(callee_scope.localvars._con.values()):
            if not (var.name in callee_freevars):
                inlined_name = _created_inlined_var_name(
                    callee_ir.func_id.unique_name, var.name)
                # Update the caller scope with the new names
                new_var = scope.redefine(inlined_name, loc=var.loc)
                # Also update the callee scope with the new names. Should the
                # type and call maps need updating (which requires SSA form) the
                # transformation to SSA is valid as the IR object is internally
                # consistent.
                callee_scope.redefine(inlined_name, loc=var.loc)
                var_dict[var.name] = new_var
        self.debug_print("var_dict = ", var_dict)
        replace_vars(callee_blocks, var_dict)
        self.debug_print("After local var rename")
        _debug_dump(callee_ir)

        # 3. replace formal parameters with actual arguments
        callee_func = callee_ir.func_id.func
        args = _get_callee_args(call_expr, callee_func, block.body[i].loc,
                                caller_ir)

        # 4. Update typemap
        if self._permit_update_type_and_call_maps:
            if arg_typs is None:
                raise TypeError('arg_typs should have a value not None')
            self.update_type_and_call_maps(callee_ir, arg_typs)
            # update_type_and_call_maps replaces blocks
            callee_blocks = callee_ir.blocks

        self.debug_print("After arguments rename: ")
        _debug_dump(callee_ir)

        _replace_args_with(callee_blocks, args)
        # 5. split caller blocks into two
        new_blocks = []
        new_block = ir.Block(scope, block.loc)
        new_block.body = block.body[i + 1:]
        new_label = next_label()
        caller_ir.blocks[new_label] = new_block
        new_blocks.append((new_label, new_block))
        block.body = block.body[:i]
        block.body.append(ir.Jump(min_label, instr.loc))

        # 6. replace Return with assignment to LHS
        topo_order = find_topo_order(callee_blocks)
        _replace_returns(callee_blocks, instr.target, new_label)

        # remove the old definition of instr.target too
        if (instr.target.name in caller_ir._definitions
                and call_expr in caller_ir._definitions[instr.target.name]):
            # NOTE: target can have multiple definitions due to control flow
            caller_ir._definitions[instr.target.name].remove(call_expr)

        # 7. insert all new blocks, and add back definitions
        for label in topo_order:
            # block scope must point to parent's
            block = callee_blocks[label]
            block.scope = scope
            _add_definitions(caller_ir, block)
            caller_ir.blocks[label] = block
            new_blocks.append((label, block))
        self.debug_print("After merge in")
        _debug_dump(caller_ir)

        return callee_ir_original, callee_blocks, var_dict, new_blocks

    def inline_function(self, caller_ir, block, i, function, arg_typs=None):
        """ Inlines the function in the caller_ir at statement index i of block
        `block`. If `arg_typs` is given and the InlineWorker instance was
        initialized with a typemap and calltypes then they will be appropriately
        updated based on the arg_typs.
        """
        callee_ir = self.run_untyped_passes(function)
        freevars = function.__code__.co_freevars
        return self.inline_ir(caller_ir, block, i, callee_ir, freevars,
                              arg_typs=arg_typs)

    def run_untyped_passes(self, func, enable_ssa=False):
        """
        Run the compiler frontend's untyped passes over the given Python
        function, and return the function's canonical Numba IR.

        Disable SSA transformation by default, since the call site won't be in
        SSA form and self.inline_ir depends on this being the case.
        """
        from numba.core.compiler import StateDict, _CompileStatus
        from numba.core.untyped_passes import ExtractByteCode
        from numba.core import bytecode
        from numba.parfors.parfor import ParforDiagnostics
        state = StateDict()
        state.func_ir = None
        state.typingctx = self.typingctx
        state.targetctx = self.targetctx
        state.locals = self.locals
        state.pipeline = self.pipeline
        state.flags = self.flags
        state.flags.enable_ssa = enable_ssa

        state.func_id = bytecode.FunctionIdentity.from_function(func)

        state.typemap = None
        state.calltypes = None
        state.type_annotation = None
        state.status = _CompileStatus(False)
        state.return_type = None
        state.parfor_diagnostics = ParforDiagnostics()
        state.metadata = {}

        ExtractByteCode().run_pass(state)
        # This is a lie, just need *some* args for the case where an obj mode
        # with lift is needed
        state.args = len(state.bc.func_id.pysig.parameters) * (types.pyobject,)

        pm = self._compiler_pipeline(state)

        pm.finalize()
        pm.run(state)
        return state.func_ir

    def update_type_and_call_maps(self, callee_ir, arg_typs):
        """ Updates the type and call maps based on calling callee_ir with
        arguments from arg_typs"""
        from numba.core.ssa import reconstruct_ssa
        from numba.core.typed_passes import PreLowerStripPhis

        if not self._permit_update_type_and_call_maps:
            msg = ("InlineWorker instance not configured correctly, typemap or "
                   "calltypes missing in initialization.")
            raise ValueError(msg)
        from numba.core import typed_passes
        # call branch pruning to simplify IR and avoid inference errors
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        numba.core.analysis.dead_branch_prune(callee_ir, arg_typs)
        # callee's typing may require SSA
        callee_ir = reconstruct_ssa(callee_ir)
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        [f_typemap,
         _f_return_type,
         f_calltypes, _] = typed_passes.type_inference_stage(
            self.typingctx, self.targetctx, callee_ir, arg_typs, None,
        )
        callee_ir = PreLowerStripPhis()._strip_phi_nodes(callee_ir)
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        canonicalize_array_math(callee_ir, f_typemap,
                                f_calltypes, self.typingctx)
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        self.typemap.update(f_typemap)
        self.calltypes.update(f_calltypes)


def inline_closure_call(func_ir, glbls, block, i, callee, typingctx=None,
                        targetctx=None, arg_typs=None, typemap=None,
                        calltypes=None, work_list=None, callee_validator=None,
                        replace_freevars=True):
    """Inline the body of `callee` at its callsite (`i`-th instruction of
    `block`)

    `func_ir` is the func_ir object of the caller function and `glbls` is its
    global variable environment (func_ir.func_id.func.__globals__).
    `block` is the IR block of the callsite and `i` is the index of the
    callsite's node. `callee` is either the called function or a
    make_function node. `typingctx`, `typemap` and `calltypes` are typing
    data structures of the caller, available if we are in a typed pass.
    `arg_typs` includes the types of the arguments at the callsite.
    `callee_validator` is an optional callable which can be used to validate the
    IR of the callee to ensure that it contains IR supported for inlining, it
    takes one argument, the func_ir of the callee

    Returns IR blocks of the callee and the variable renaming dictionary used
    for them to facilitate further processing of new blocks.
    """
    scope = block.scope
    instr = block.body[i]
    call_expr = instr.value
    debug_print = _make_debug_print("inline_closure_call")
    debug_print("Found closure call: ", instr, " with callee = ", callee)
    # support both function object and make_function Expr
    callee_code = callee.code if hasattr(callee, 'code') else callee.__code__
    callee_closure = (callee.closure
                      if hasattr(callee, 'closure') else callee.__closure__)
    # first, get the IR of the callee
    if isinstance(callee, pytypes.FunctionType):
        from numba.core import compiler
        callee_ir = compiler.run_frontend(callee, inline_closures=True)
    else:
        callee_ir = get_ir_of_code(glbls, callee_code)

    # check that the contents of the callee IR is something that can be inlined
    # if a validator is supplied
    if callee_validator is not None:
        callee_validator(callee_ir)

    callee_blocks = callee_ir.blocks

    # 1. relabel callee_ir by adding an offset
    max_label = max(ir_utils._the_max_label.next(), max(func_ir.blocks.keys()))
    callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
    callee_blocks = simplify_CFG(callee_blocks)
    callee_ir.blocks = callee_blocks
    min_label = min(callee_blocks.keys())
    max_label = max(callee_blocks.keys())
    #    reset globals in ir_utils before we use it
    ir_utils._the_max_label.update(max_label)
    debug_print("After relabel")
    _debug_dump(callee_ir)

    # 2. rename all local variables in callee_ir with new locals created in
    #    func_ir
    callee_scopes = _get_all_scopes(callee_blocks)
    debug_print("callee_scopes = ", callee_scopes)
    #    one function should only have one local scope
    assert (len(callee_scopes) == 1)
    callee_scope = callee_scopes[0]
    var_dict = {}
    for var in callee_scope.localvars._con.values():
        if not (var.name in callee_code.co_freevars):
            inlined_name = _created_inlined_var_name(
                callee_ir.func_id.unique_name, var.name)
            new_var = scope.redefine(inlined_name, loc=var.loc)
            var_dict[var.name] = new_var
    debug_print("var_dict = ", var_dict)
    replace_vars(callee_blocks, var_dict)
    debug_print("After local var rename")
    _debug_dump(callee_ir)

    # 3. replace formal parameters with actual arguments
    args = _get_callee_args(call_expr, callee, block.body[i].loc, func_ir)

    debug_print("After arguments rename: ")
    _debug_dump(callee_ir)

    # 4. replace freevar with actual closure var
    if callee_closure and replace_freevars:
        # A nonlocal assigned more than once is SSA-versioned (``x``, ``x.1``);
        # inlining keeps only the base name, dropping later writes (#10379).
        for freevar in callee_code.co_freevars:
            if callee_scope.get_versions_of(freevar):
                msg = ("Reassigning the nonlocal variable '%s' inside an "
                       "inlined closure is unsupported. Assign it once "
                       "instead." % (freevar,))
                raise errors.UnsupportedError(msg, instr.loc)
        closure = func_ir.get_definition(callee_closure)
        debug_print("callee's closure = ", closure)
        if isinstance(closure, tuple):
            cellget = ctypes.pythonapi.PyCell_Get
            cellget.restype = ctypes.py_object
            cellget.argtypes = (ctypes.py_object,)
            items = tuple(cellget(x) for x in closure)
        else:
            assert (isinstance(closure, ir.Expr)
                    and closure.op == 'build_tuple')
            items = closure.items
        assert (len(callee_code.co_freevars) == len(items))
        _replace_freevars(callee_blocks, items)
        debug_print("After closure rename")
        _debug_dump(callee_ir)

    if typingctx:
        from numba.core import typed_passes
        # call branch pruning to simplify IR and avoid inference errors
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        numba.core.analysis.dead_branch_prune(callee_ir, arg_typs)
        try:
            [f_typemap, f_return_type,
             f_calltypes, _] = typed_passes.type_inference_stage(
                typingctx, targetctx, callee_ir, arg_typs, None)
        except Exception:
            [f_typemap, f_return_type,
             f_calltypes, _] = typed_passes.type_inference_stage(
                typingctx, targetctx, callee_ir, arg_typs, None)
        canonicalize_array_math(callee_ir, f_typemap,
                                f_calltypes, typingctx)
        # remove argument entries like arg.a from typemap
        arg_names = [vname for vname in f_typemap if vname.startswith("arg.")]
        for a in arg_names:
            f_typemap.pop(a)
        typemap.update(f_typemap)
        calltypes.update(f_calltypes)

    _replace_args_with(callee_blocks, args)
    # 5. split caller blocks into two
    new_blocks = []
    new_block = ir.Block(scope, block.loc)
    new_block.body = block.body[i + 1:]
    new_label = next_label()
    func_ir.blocks[new_label] = new_block
    new_blocks.append((new_label, new_block))
    block.body = block.body[:i]
    block.body.append(ir.Jump(min_label, instr.loc))

    # 6. replace Return with assignment to LHS
    topo_order = find_topo_order(callee_blocks)
    _replace_returns(callee_blocks, instr.target, new_label)

    # remove the old definition of instr.target too
    if (instr.target.name in func_ir._definitions
            and call_expr in func_ir._definitions[instr.target.name]):
        # NOTE: target can have multiple definitions due to control flow
        func_ir._definitions[instr.target.name].remove(call_expr)

    # 7. insert all new blocks, and add back definitions
    for label in topo_order:
        # block scope must point to parent's
        block = callee_blocks[label]
        block.scope = scope
        _add_definitions(func_ir, block)
        func_ir.blocks[label] = block
        new_blocks.append((label, block))
    debug_print("After merge in")
    _debug_dump(func_ir)

    if work_list is not None:
        for block in new_blocks:
            work_list.append(block)
    return callee_blocks, var_dict


def _get_callee_args(call_expr, callee, loc, func_ir):
    """Get arguments for calling 'callee', including the default arguments.
    keyword arguments are currently only handled when 'callee' is a function.
    """
    if call_expr.op == 'call':
        args = list(call_expr.args)
        if call_expr.vararg:
            msg = "Calling a closure with *args is unsupported."
            raise errors.UnsupportedError(msg, call_expr.loc)
    elif call_expr.op == 'getattr':
        args = [call_expr.value]
    elif ir_utils.is_operator_or_getitem(call_expr):
        args = call_expr.list_vars()
    else:
        raise TypeError("Unsupported ir.Expr.{}".format(call_expr.op))

    debug_print = _make_debug_print("inline_closure_call default handling")

    # handle defaults and kw arguments using pysignature if callee is function
    if isinstance(callee, pytypes.FunctionType):
        pysig = numba.core.utils.pysignature(callee)
        normal_handler = lambda index, param, default: default
        default_handler = lambda index, param, default: ir.Const(default, loc)

        # Throw error for stararg
        # TODO: handle stararg
        def stararg_handler(index, param, default):
            raise NotImplementedError(
                "Stararg not supported in inliner for arg {} {}".format(
                    index, param))
        if call_expr.op == 'call':
            kws = dict(call_expr.kws)
        else:
            kws = {}
        return numba.core.typing.fold_arguments(
            pysig, args, kws, normal_handler, default_handler,
            stararg_handler)
    else:
        # TODO: handle arguments for make_function case similar to function
        # case above
        callee_defaults = (callee.defaults if hasattr(callee, 'defaults')
                           else callee.__defaults__)
        if callee_defaults:
            debug_print("defaults = ", callee_defaults)
            if isinstance(callee_defaults, tuple):  # Python 3.5
                defaults_list = []
                for x in callee_defaults:
                    if isinstance(x, ir.Var):
                        defaults_list.append(x)
                    else:
                        # this branch is predominantly for kwargs from
                        # inlinable functions
                        defaults_list.append(ir.Const(value=x, loc=loc))
                args = args + defaults_list
            elif (isinstance(callee_defaults, ir.Var)
                    or isinstance(callee_defaults, str)):
                default_tuple = func_ir.get_definition(callee_defaults)
                assert (isinstance(default_tuple, ir.Expr))
                assert (default_tuple.op == "build_tuple")
                const_vals = [func_ir.get_definition(x) for
                              x in default_tuple.items]
                args = args + const_vals
            else:
                raise NotImplementedError(
                    "Unsupported defaults to make_function: {}".format(
                        callee_defaults))
        return args


def _make_debug_print(prefix):
    def debug_print(*args):
        if config.DEBUG_INLINE_CLOSURE:
            print(prefix + ": " + "".join(str(x) for x in args))
    return debug_print


def _debug_dump(func_ir):
    if config.DEBUG_INLINE_CLOSURE:
        func_ir.dump()


def _get_all_scopes(blocks):
    """Get all block-local scopes from an IR.
    """
    all_scopes = []
    for label, block in blocks.items():
        if not (block.scope in all_scopes):
            all_scopes.append(block.scope)
    return all_scopes


def _replace_args_with(blocks, args):
    """
    Replace ir.Arg(...) with real arguments from call site
    """
    for label, block in blocks.items():
        assigns = block.find_insts(ir.Assign)
        for stmt in assigns:
            if isinstance(stmt.value, ir.Arg):
                idx = stmt.value.index
                assert (idx < len(args))
                stmt.value = args[idx]


def _replace_freevars(blocks, args):
    """
    Replace ir.FreeVar(...) with real variables from parent function
    """
    for label, block in blocks.items():
        assigns = block.find_insts(ir.Assign)
        for stmt in assigns:
            if isinstance(stmt.value, ir.FreeVar):
                idx = stmt.value.index
                assert (idx < len(args))
                if isinstance(args[idx], ir.Var):
                    stmt.value = args[idx]
                else:
                    stmt.value = ir.Const(args[idx], stmt.loc)


def _replace_returns(blocks, target, return_label):
    """
    Return return statement by assigning directly to target, and a jump.
    """
    for label, block in blocks.items():
        casts = []
        for i in range(len(block.body)):
            stmt = block.body[i]
            if isinstance(stmt, ir.Return):
                assert (i + 1 == len(block.body))
                block.body[i] = ir.Assign(stmt.value, target, stmt.loc)
                block.body.append(ir.Jump(return_label, stmt.loc))
                # remove cast of the returned value
                for cast in casts:
                    if cast.target.name == stmt.value.name:
                        cast.value = cast.value.value
            elif (isinstance(stmt, ir.Assign) and
                    isinstance(stmt.value, ir.Expr) and
                    stmt.value.op == 'cast'):
                casts.append(stmt)


def _add_definitions(func_ir, block):
    """
    Add variable definitions found in a block to parent func_ir.
    """
    definitions = func_ir._definitions
    assigns = block.find_insts(ir.Assign)
    for stmt in assigns:
        definitions[stmt.target.name].append(stmt.value)
