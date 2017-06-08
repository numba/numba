from numba import config, ir, ir_utils, utils
import types

from numba.ir_utils import (mk_unique_var, next_label, add_offset_to_labels, 
    replace_vars, remove_dels, remove_dead, rename_labels)
 
class InlineClosureCallPass(object):
    """InlineClosureCallPass class looks for direct calls to locally defined 
    closures, and inlines the body of the closure function to the call site.
    """
    def __init__(self, func_ir, run_frontend):
        self.func_ir = func_ir
        self.run_frontend = run_frontend

    def run(self):
        """Run inline closure call pass.
        """
        modified = False
        work_list = list(self.func_ir.blocks.items())
        _debug_print("START InlineClosureCall")
        while work_list:
            label, block = work_list.pop()
            for i in range(len(block.body)):
                instr = block.body[i]
                if isinstance(instr, ir.Assign):
                    lhs  = instr.target
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        try:
                            func_def = self.func_ir.get_definition(expr.func)
                        except KeyError:
                            func_def = None
                        _debug_print("found call to ", expr.func, " def = ", func_def)
                        if isinstance(func_def, ir.Expr) and func_def.op == "make_function":
                            new_blocks = self.inline_closure_call(block, i, func_def)
                            for block in new_blocks:
                                work_list.append(block)
                            modified = True
                            # current block is modified, skip the rest
                            break
        if modified:
            remove_dels(self.func_ir.blocks)
            # repeat dead code elimintation until nothing can be further removed
            while (remove_dead(self.func_ir.blocks, self.func_ir.arg_names)):
                pass
            self.func_ir.blocks = rename_labels(self.func_ir.blocks)

    def inline_closure_call(self, block, i, callee):
        """Inline the body of `callee` at its callsite (`i`-th instruction of `block`)
        """
        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        _debug_print("Found closure call: ", instr, " with callee = ", callee)
        func_ir = self.func_ir
        # first, get the IR of the callee
        from_ir = self.get_ir_of_code(callee.code)
        from_blocks = from_ir.blocks
        # 1. relabel from_ir by adding an offset
        max_label = max(func_ir.blocks.keys())
        from_blocks = add_offset_to_labels(from_blocks, max_label + 1)
        from_ir.blocks = from_blocks
        min_label = min(from_blocks.keys())
        max_label = max(from_blocks.keys())
        #    reset globals in ir_utils before we use it
        ir_utils._max_label = max_label 
        ir_utils.visit_vars_extensions = {}
        # 2. rename all local variables in from_ir with new locals created in func_ir
        from_scopes = _get_all_scopes(from_blocks)
        _debug_print("obj_IR has scopes: ", from_scopes)
        #    one function should only have one local scope
        assert(len(from_scopes) == 1)
        from_scope = from_scopes[0]
        var_dict = {}
        for var in from_scope.localvars._con.values():
            if not (var.name in callee.code.co_freevars):
                var_dict[var.name] = scope.make_temp(var.loc)
        _debug_print("Before local var rename: var_dict = ", var_dict)
        _debug_dump(from_ir)
        replace_vars(from_blocks, var_dict)
        _debug_print("After local var rename: ")
        _debug_dump(from_ir)
        # 3. replace formal parameters with actual arguments
        args = list(call_expr.args)
        if callee.defaults:
            _debug_print("defaults", callee.defaults)
            if isinstance(callee.defaults, tuple): # Python 3.5
                args = args + list(callee.defaults)
            elif isinstance(callee.defaults, ir.Var) or isinstance(callee.defaults, str):
                defaults = func_ir.get_definition(callee.defaults)
                assert(isinstance(defaults, ir.Const))
                loc = defaults.loc
                args = args + [ ir.Const(value=v, loc=loc) for v in defaults.value ]
            else:
                raise NotImplementedError("Unsupported defaults to make_function: {}".format(defaults))
        _replace_args_with(from_blocks, args)
        _debug_print("After arguments rename: ")
        _debug_dump(from_ir)
        # 4. replace freevar with actual closure var
        if callee.closure:
            closure = func_ir.get_definition(callee.closure)
            assert(isinstance(closure, ir.Expr) and closure.op == 'build_tuple')
            assert(len(callee.code.co_freevars) == len(closure.items))
            _debug_print("callee's closure = ", closure)
            _replace_freevars(from_blocks, closure.items)
            _debug_print("After closure rename: ")
            _debug_dump(from_ir)
        # 5. split caller blocks into two
        new_blocks = []
        new_block = ir.Block(scope, block.loc)
        new_block.body = block.body[i+1:]
        new_label = next_label()
        func_ir.blocks[new_label] = new_block
        new_blocks.append((new_label, new_block))
        block.body = block.body[:i]
        block.body.append(ir.Jump(min_label, instr.loc))
        # 6. replace Return with assignment to LHS
        _replace_returns(from_blocks, instr.target, new_label)
        # 7. insert all new blocks, and add back definitions
        for label, block in from_blocks.items():
            # block scope must point to parent's
            block.scope = scope
            _add_definition(func_ir, block)
            func_ir.blocks[label] = block
            new_blocks.append((label, block))
        _debug_print("After merge: ")
        _debug_dump(func_ir)
        return new_blocks

    def get_ir_of_code(self, fcode):
        """
        Compile a code object to get its IR.
        """
        glbls = self.func_ir.func_id.func.__globals__
        nfree = len(fcode.co_freevars)
        func_env = "\n".join(["  c_%d = None" % i for i in range(nfree)])
        func_clo = ",".join(["c_%d" % i for i in range(nfree)])
        func_arg = ",".join(["x_%d" % i for i in range(fcode.co_argcount)])
        func_text = "def g():\n%s\n  def f(%s):\n    return (%s)\n  return f" % (func_env, func_arg, func_clo)
        loc = {}
        exec(func_text, glbls, loc)

        # hack parameter name .0 for Python 3 versions < 3.6
        if utils.PYVERSION >= (3,) and utils.PYVERSION < (3,6):
            co_varnames = list(fcode.co_varnames)
            if co_varnames[0] == ".0":
                co_varnames[0] = "implicit0"
            fcode = types.CodeType(fcode.co_argcount, fcode.co_kwonlyargcount, 
                        fcode.co_nlocals, fcode.co_stacksize, fcode.co_flags,
                        fcode.co_code, fcode.co_consts, fcode.co_names, tuple(co_varnames),
                        fcode.co_filename, fcode.co_name, fcode.co_firstlineno, fcode.co_lnotab,
                        fcode.co_freevars, fcode.co_cellvars)

        f = loc['g']()
        f.__code__ = fcode
        f.__name__ = fcode.co_name
        ir = self.run_frontend(f)
        return ir

def _debug_print(*args):
    if config.DEBUG_INLINE_CLOSURE:
        print(args)

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
                assert(idx < len(args))
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
                assert(idx < len(args))
                stmt.value = args[idx]

def _replace_returns(blocks, target, return_label):
    """
    Return return statement by assigning directly to target, and a jump.
    """
    for label, block in blocks.items():
        for i in range(len(block.body)):
            stmt = block.body[i]
            if isinstance(stmt, ir.Return):
                assert(i + 1 == len(block.body))
                block.body[i] = ir.Assign(stmt.value, target, stmt.loc)
                block.body.append(ir.Jump(return_label, stmt.loc))

def _add_definition(func_ir, block):
    """
    Add variable definitions to parent func_ir
    """
    definitions = func_ir._definitions
    assigns = block.find_insts(ir.Assign)
    for stmt in assigns:
        definitions[stmt.target.name].append(stmt.value)

