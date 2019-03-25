from numba import ir
from . import register_rewrite, Rewrite
import operator
from operator import truth


@register_rewrite('before-inference')
class RewriteBranches(Rewrite):
    """
    Rewrite IR statements of the kind `branch condition tbr fbr` so that the
    condition is made a valid predicate via wrapping it in a call to
    operator.truth()
    """

    def match(self, func_ir, block, typemap, calltypes):
        self.block = block

        # only care about the terminator
        terminator = block.body[-1]
        if isinstance(terminator, ir.Terminator):
            # if it's a branch and the condition is a call to the `truth` global
            # then this branch is already rewritten/valid in the first place
            # so return false
            if isinstance(terminator, ir.Branch):
                branch = terminator
                defn = func_ir.get_definition(branch.cond)
                if getattr(defn, 'op', None) == 'call':
                    # see if its a call to `truth` from operator
                    called_fn = func_ir.get_definition(defn.func)
                    if called_fn.value == truth:
                        return False
                    else:
                        # see if its a getattr(operator, 'truth')
                        if called_fn.op == 'getattr':
                            attr_truth = called_fn.attr == 'truth'
                            possmod = func_ir.get_definition(called_fn.value)
                            if attr_truth and isinstance(possmod, ir.Global):
                                if possmod.value == operator:
                                    return False
            else:
                # it's a jump, return, raise, etc don't modify
                return False

        return True

    def apply(self):
        from numba.ir_utils import mk_unique_var
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body[:-1]:
            new_block.append(inst)

        # definitions need updating as we go so that the match() function above
        # can track the updated branch conditions via `func_ir.get_definition`
        # as they reference rewritten definitions
        definitions = self.pipeline.func_ir._definitions

        branch = self.block.body[-1]
        loc = branch.loc
        scope = new_block.scope
        _ir_truth_var = ir.Var(scope, mk_unique_var("$truth_glbl"), loc)
        _ir_truth = ir.Global('truth', truth, branch.loc)
        _ir_assigned_truth = ir.Assign(_ir_truth, _ir_truth_var, loc)
        new_block.append(_ir_assigned_truth)
        name = _ir_assigned_truth.target.name
        definitions[name] = [_ir_assigned_truth.value]

        _ir_truth_call = ir.Expr.call(_ir_assigned_truth.target, [
                                      branch.cond], (), branch.loc)
        _ir_truth_eval_var = ir.Var(scope, mk_unique_var("$truth_res"), loc)
        definitions[_ir_truth_eval_var.name] = [_ir_truth_eval_var]
        _ir_assigned_truth_eval = ir.Assign(
            _ir_truth_call, _ir_truth_eval_var, loc)
        new_block.append(_ir_assigned_truth_eval)

        name = _ir_assigned_truth_eval.target.name
        definitions[name] = [_ir_assigned_truth_eval.value]

        # can't delete the predicate now, there's del's in the branch target
        # can delete the global now, call is done
        new_block.append(ir.Del(_ir_truth_var.name, branch.loc))

        new_block.append(ir.Branch(_ir_assigned_truth_eval.target,
                                   branch.truebr,
                                   branch.falsebr,
                                   branch.loc))

        # NOTE: This rewrite leaks a reference to the predicate, closure
        # inlining post processor cleans it up and reinserts it.
        return new_block
