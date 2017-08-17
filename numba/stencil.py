import numpy as np
from numba import compiler, types, ir_utils, ir, typing
from numba.typing.templates import AbstractTemplate, signature
from numba.targets import cpu

def replace_return_with_setitem(blocks, index_vars):
    """
    Find return statements in the IR and replace them with a SetItem
    call of the value "returned" by the kernel into the result array.
    """
    for block in blocks.values():
        scope = block.scope
        loc = block.loc
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Return):
                # If 1D array then avoid the tuple construction.
                if len(index_vars) == 1:
                    rvar = ir.Var(scope, "result", loc)
                    ivar = ir.Var(scope, index_vars[0], loc)
                    new_body.append(ir.SetItem(rvar, ivar, stmt.value, loc))
                else:
                    # Convert the string names of the index variables into
                    # ir.Var's.
                    var_index_vars = []
                    for one_var in index_vars:
                        index_var = ir.Var(scope, one_var, loc)
                        var_index_vars += [index_var]

                    s_index_name = ir_utils.mk_unique_var("stencil_index")
                    s_index_var  = ir.Var(scope, s_index_name, loc)
                    # Build a tuple from the index ir.Var's.
                    tuple_call = ir.Expr.build_tuple(var_index_vars, loc)
                    new_body.append(ir.Assign(tuple_call, s_index_var, loc))
                    rvar = ir.Var(scope, "result", loc)
                    # Write the return statements original value into
                    # the array using the tuple index.
                    si = ir.SetItem(rvar, s_index_var, stmt.value, loc)
                    new_body.append(si)
            else:
                new_body.append(stmt)
        block.body = new_body

def add_indices_to_kernel(kernel, ndim):
    """
    Transforms the stencil kernel as specified by the user into one
    that includes each dimension's index variable as part of the getitem
    calls.  So, in effect array[-1] becomes array[index0-1].
    """
    const_dict = {}
    kernel_consts = []

    for block in kernel.blocks.values():
        scope = block.scope
        loc = block.loc
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Const):
                # Remember consts for use later.
                const_dict[stmt.target.name] = stmt.value.value
            if (isinstance(stmt, ir.Assign) and 
                isinstance(stmt.value, ir.Expr) and 
                stmt.value.op == 'getitem' and 
                stmt.value.value.name in kernel.arg_names):
                # We found a getitem from the input array.
                rhs = stmt.value
                # Store the index used after looking up the variable in
                # the const dictionary.
                if rhs.index.name in const_dict:
                    kernel_consts += [const_dict[rhs.index.name]]
                else:
                    raise ValueError("Non-constant specified for stencil kernel index.")

                if ndim == 0:
                    # Single dimension always has index variable 'index0'.
                    # tmpvar will hold the real index and is computed by
                    # adding the relative offset in stmt.value.index to
                    # the current absolute location in index0.
                    index_var = ir.Var(scope, "index0", loc)
                    tmpname = ir_utils.mk_unique_var("stencil_index")
                    tmpvar  = ir.Var(scope, tmpname, loc)
                    acc_call = ir.Expr.binop('+', stmt.value.index, 
                                             index_var, loc)
                    new_body.append(ir.Assign(acc_call, tmpvar, loc))
                    stmt.value.index = tmpvar
                    new_body.append(stmt)
                else:
                    index_vars = []
                    sum_results = []
                    s_index_name = ir_utils.mk_unique_var("stencil_index")
                    s_index_var  = ir.Var(scope, s_index_name, loc)
                    const_index_vars = []

                    # Same idea as above but you have to extract
                    # individual elements out of the tuple indexing
                    # expression and add the corresponding index variable
                    # to them and then reconstitute as a tuple that can
                    # index the array.
                    for dim in range(ndim):
                        tmpname = ir_utils.mk_unique_var("const_index")
                        tmpvar  = ir.Var(scope, tmpname, loc)
                        new_body.append(ir.Assign(ir.Const(dim, loc), 
                                                  tmpvar, loc))
                        const_index_vars += [tmpvar]
                        index_var = ir.Var(scope, "index" + str(dim), loc)
                        index_vars += [index_var]

                    ind_stencils = []

                    for dim in range(ndim):
                        tmpname = ir_utils.mk_unique_var("ind_stencil_index")
                        tmpvar  = ir.Var(scope, tmpname, loc)
                        ind_stencils += [tmpvar]
                        getitemname = ir_utils.mk_unique_var("getitem")
                        getitemvar  = ir.Var(scope, getitemname, loc)
                        getitemcall = ir.Expr.getitem(stmt.value.index, 
                                                   const_index_vars[dim], loc)
                        new_body.append(ir.Assign(getitemcall, getitemvar, loc))
                        acc_call = ir.Expr.binop('+', getitemvar, 
                                                 index_vars[dim], loc)
                        new_body.append(ir.Assign(acc_call, tmpvar, loc))

                    tuple_call = ir.Expr.build_tuple(ind_stencils, loc)
                    new_body.append(ir.Assign(tuple_call, s_index_var, loc))
                    stmt.value.index = s_index_var
                    new_body.append(stmt)
            else:
                new_body.append(stmt)
        block.body = new_body

    # Find the size of the kernel by finding the maximum absolute value
    # index used in the kernel specification.
    max_const = 0
    for index in kernel_consts:
        if isinstance(index, tuple):
            for te in index:
                max_const = max(max_const, abs(te))    
            index_len = len(index)
        elif isinstance(index, int):
            max_const = max(max_const, abs(index))    
            index_len = 1
        else:
            raise ValueError("Non-tuple or non-integer used as stencil index.")
        if index_len != ndim:
            raise ValueError("Stencil index does not match array dimensionality.")

    return max_const

class StencilFunc(object):
    """
    A special type to hold stencil information for the IR.
    """

    def __init__(self, func, kernel_ir, options):
        from numba import jit
        self.func = func
        self.kernel_ir = kernel_ir
        self.options = options
        def nfunc():
            return None
        self._dispatcher = jit()(nfunc)
        self._typingctx = self._dispatcher.targetdescr.typing_context
        self._install_type(self._typingctx)

    def _install_type(self, typingctx):
        """Constructs and installs a typing class for a StencilFunc object in 
        the input typing context.  If no typing context is given, then
        _install_type() installs into the typing context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
        _ty_cls = type('StencilFuncTyping_' + 
                       str(hex(id(self.func)).replace("-", "_")),
                       (AbstractTemplate,),
                       dict(key=self, generic=self._type_me))
        typingctx.insert_user_function(self, _ty_cls)

    def _type_me(self, argtys, kwtys):
        """
        Implement AbstractTemplate.generic() for the typing class
        built by StencilFunc._install_type().
        Return the call-site signature.
        """
        outtys = [argtys[0]]
        outtys.extend(argtys)
        return signature(*outtys)

    def _stencil_wrapper(self, *args, **kwargs):
        the_array = args[0]
        if 'out' in kwargs:
            result = kwargs['out']
        else:
            result = np.zeros_like(the_array)

        stencil_func_name = "__numba_stencil_%s_%s" % (
                                        hex(id(the_array)).replace("-", "_"), 
                                        hex(id(self.func)).replace("-", "_"))
        
        index_vars = []
        for i in range(the_array.ndim):
            index_var_name = "index" + str(i)
            index_vars += [index_var_name]

        kernel_size = add_indices_to_kernel(self.kernel_ir, the_array.ndim)
        replace_return_with_setitem(self.kernel_ir.blocks, index_vars)

        stencil_func_text = "def " + stencil_func_name + "("
        stencil_func_text += ",".join(self.kernel_ir.arg_names) + ", result):\n"
        stencil_func_text += "    full_shape = "
        stencil_func_text += self.kernel_ir.arg_names[0] + ".shape\n"

        offset = 1
        for i in range(the_array.ndim):
            stri = str(i)
            for j in range(offset):
                stencil_func_text += "    "
            stencil_func_text += "for " + index_vars[i] + " in range("
            stencil_func_text += str(kernel_size) + ", full_shape[" + stri
            stencil_func_text += "] - " + str(kernel_size) + "):\n"
            offset += 1

        for j in range(offset):
            stencil_func_text += "    "
        stencil_func_text += "__sentinel__ = 0\n"
        stencil_func_text += "    return result\n"

        exec(stencil_func_text)
        stencil_func = eval(stencil_func_name)
        stencil_ir = compiler.run_frontend(stencil_func)
        ir_utils.remove_dels(stencil_ir.blocks)

        # rename all variables in stencil_ir afresh
        var_table = ir_utils.get_name_var_table(stencil_ir.blocks)
        new_var_dict = {}
        reserved_names = (["__sentinel__", "result"] + 
                          self.kernel_ir.arg_names + index_vars)
        #  + list(param_dict.values()) + legal_loop_indices
        for name, var in var_table.items():
            if not (name in reserved_names):
                new_var_dict[name] = ir_utils.mk_unique_var(name)
        ir_utils.replace_var_names(stencil_ir.blocks, new_var_dict)

        stencil_stub_last_label = max(stencil_ir.blocks.keys()) + 1

        self.kernel_ir.blocks = ir_utils.add_offset_to_labels(
                                self.kernel_ir.blocks, stencil_stub_last_label)
        new_label = max(self.kernel_ir.blocks.keys()) + 1

        # Search all the block in the stencil outline for the sentinel.
        for label, block in stencil_ir.blocks.items():
            for i, inst in enumerate(block.body):
                if isinstance(
                        inst,
                        ir.Assign) and inst.target.name == "__sentinel__":
                    # We found the sentinel assignment.
                    loc = inst.loc
                    scope = block.scope
                    # split block across __sentinel__
                    # A new block is allocated for the statements prior to the
                    # sentinel but the new block maintains the current block 
                    # label.
                    prev_block = ir.Block(scope, loc)
                    prev_block.body = block.body[:i]
                    # The current block is used for statements after sentinel.
                    block.body = block.body[i + 1:]
                    # But the current block gets a new label.
                    body_first_label = min(self.kernel_ir.blocks.keys())

                    # The previous block jumps to the minimum labelled block of
                    # the parfor body.
                    prev_block.append(ir.Jump(body_first_label, loc))
                    # Add all the parfor loop body blocks to the gufunc 
                    # function's # IR.
                    for (l, b) in self.kernel_ir.blocks.items():
                        stencil_ir.blocks[l] = b
                    body_last_label = max(self.kernel_ir.blocks.keys())
                    stencil_ir.blocks[new_label] = block
                    stencil_ir.blocks[label] = prev_block
                    # Add a jump from the last parfor body block to the block
                    # containing statements after the sentinel.
                    stencil_ir.blocks[body_last_label].append(
                        ir.Jump(new_label, loc))
                    break
            else:
                continue
            break

        stencil_ir.blocks = ir_utils.rename_labels(stencil_ir.blocks)
        ir_utils.remove_dels(stencil_ir.blocks)

        array_npytype = typing.typeof.typeof(the_array)

        new_stencil_param_types = [array_npytype, array_npytype]

        from .targets.registry import cpu_target
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx)
        with cpu_target.nested_context(typingctx, targetctx):
            new_stencil_func = compiler.compile_ir(
                typingctx,
                targetctx,
                stencil_ir,
                new_stencil_param_types,
                array_npytype,
                compiler.DEFAULT_FLAGS,
                {})
            stencil_func = eval(stencil_func_name)
            return new_stencil_func.entry_point(*args, result)

    def __call__(self, *args, **kwargs):
        return self._stencil_wrapper(*args, **kwargs)

def stencil(boundary='skip', **options):
    if boundary != 'skip':
        raise ValueError("Unsupported boundary style " + boundary)

    def decorated(func):
        kernel_ir = compiler.run_frontend(func)
        ir_utils.remove_args(kernel_ir.blocks) 
        return StencilFunc(func, kernel_ir, options)

    return decorated
