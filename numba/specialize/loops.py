# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast
import textwrap
try:
    import __builtin__ as builtins
except ImportError:
    import builtins

import numba
from numba import missing
from numba import *
from numba import error
from numba import typesystem
from numba import visitors, nodes
from numba.typesystem import get_type
from numba.specialize import loopimpl

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

def unpack_range_args(node):
    start, stop, step = (nodes.const(0, Py_ssize_t),
                         None,
                         nodes.const(1, Py_ssize_t))

    if len(node.args) == 0:
        raise error.NumbaError(node, "Expected at least one argument")
    elif len(node.args) == 1:
        stop, = node.args
    elif len(node.args) == 2:
        start, stop = node.args
    else:
        start, stop, step = node.args

    return [start, stop, step]

def make_while_loop(flow_node):
    "Create a while loop from a flow node (a While or If node)"
    while_node = nodes.While(test=flow_node.test,
                             body=flow_node.body,
                             orelse=flow_node.orelse)
    return ast.copy_location(while_node, flow_node)

def copy_basic_blocks(flow_node_src, flow_node_dst):
    "Copy cfg basic blocks from one flow node to another"
    flow_node_dst.cond_block = flow_node_src.cond_block
    flow_node_dst.if_block   = flow_node_src.if_block
    flow_node_dst.else_block = flow_node_src.else_block
    flow_node_dst.exit_block = flow_node_src.exit_block

def make_while_from_for(for_node):
    "Create a While from a For. The 'test' (loop condition) must still be set."
    while_node = nodes.While(test=None,
                             body=for_node.body,
                             orelse=for_node.orelse)
    copy_basic_blocks(for_node, while_node)
    while_node = nodes.build_while(**vars(while_node))
    return ast.copy_location(while_node, for_node)

def untypedTemp():
    "Temp node with a yet unknown type"
    type = typesystem.DeferredType(None)
    temp = nodes.TempNode(type)
    type.variable = temp.variable
    return temp

#------------------------------------------------------------------------
# Transform for loops
#------------------------------------------------------------------------

class TransformForIterable(visitors.NumbaTransformer):
    """
    This transforms loops over 1D arrays and loops over range().
    """

    def rewrite_range_iteration(self, node):
        """
        Handle range iteration:

            for i in range(start, stop, step):
                ...

        becomes

            nsteps = compute_nsteps(start, stop, step)
            temp = 0

            while temp < nsteps:
                target = start + temp * step
                ...
                temp += 1
        """
        self.generic_visit(node)

        temp = nodes.TempNode(node.target.type, 'target_temp')
        nsteps = nodes.TempNode(Py_ssize_t, 'nsteps')
        start, stop, step = unpack_range_args(node.iter)

        if isinstance(step, nodes.ConstNode):
            have_step = step.pyval != 1
        else:
            have_step = True

        start, stop, step = map(nodes.CloneableNode, (start, stop, step))

        if have_step:
            templ = textwrap.dedent("""
                    {{temp}} = 0
                    {{nsteps}} = ({{stop}} - {{start}} + {{step}} -
                                    (1 if {{step}} >= 0 else -1)) / {{step}}
                    while {{temp_load}} < {{nsteps_load}}:
                        {{target}} = {{start}} + {{temp_load}} * {{step}}
                        {{temp}} = {{temp_load}} + 1
                        {{body}}
                """)
        else:
            templ = textwrap.dedent("""
                    {{temp}} = {{start}}
                    {{nsteps}} = {{stop}}
                    while {{temp_load}} < {{nsteps_load}}:
                        {{target}} = {{temp_load}}
                        {{temp}} = {{temp_load}} + 1
                        {{body}}
                """)

        if node.orelse:
            templ += "\nelse: {{else_body}}"

        # Leave the bodies empty, they are already analyzed
        body = ast.Suite(body=[])
        else_body = ast.Suite(body=[])

        #--------------------------------------------------------------------
        # Substitute template and infer types
        #--------------------------------------------------------------------

        result = self.run_template(
            templ, vars=dict(length=Py_ssize_t),
            start=start, stop=stop, step=step,
            nsteps=nsteps.store(), nsteps_load=nsteps.load(),
            temp=temp.store(), temp_load=temp.load(),
            target=node.target,
            body=body, else_body=else_body)
        ast.copy_location(result, node)
        if hasattr(node, 'lineno'):
            visitor = missing.FixMissingLocations(node.lineno, node.col_offset,
                                              override=True)
            visitor.visit(result)

        #--------------------------------------------------------------------
        # Patch the body and else clause
        #--------------------------------------------------------------------

        body.body.extend(node.body)
        else_body.body.extend(node.orelse)

        while_node = result.body[-1]
        assert isinstance(while_node, ast.While)

        #--------------------------------------------------------------------
        # Create a While with the ForNode's cfg blocks merged in
        #--------------------------------------------------------------------

        while_node = make_while_loop(while_node)
        copy_basic_blocks(node, while_node)
        while_node = nodes.build_while(**vars(while_node))

        # Create the place to jump to for 'continue'
        while_node.continue_block = node.cond_block

        # Set the new while loop in the templated Suite
        result.body[-1] = while_node

        return result

    def rewrite_array_iteration(self, node):
        """
        Convert 1D array iteration to for-range and indexing:

            for value in my_array:
                ...

        becomes

            for i in my_array.shape[0]:
                value = my_array[i]
                ...
        """
        logger.debug(ast.dump(node))

        orig_target = node.target
        orig_iter = node.iter

        #--------------------------------------------------------------------
        # Replace node.target with a temporary
        #--------------------------------------------------------------------

        target_temp = nodes.TempNode(typesystem.Py_ssize_t)
        node.target = target_temp.store()

        #--------------------------------------------------------------------
        # Create range(A.shape[0])
        #--------------------------------------------------------------------

        call_func = ast.Name(id='range', ctx=ast.Load())
        nodes.typednode(call_func, typesystem.range_)

        shape_index = ast.Index(nodes.ConstNode(0, typesystem.Py_ssize_t))
        shape_index.type = typesystem.npy_intp

        stop = ast.Subscript(value=nodes.ShapeAttributeNode(orig_iter),
                             slice=shape_index,
                             ctx=ast.Load())
        nodes.typednode(stop, npy_intp)

        #--------------------------------------------------------------------
        # Create range iterator and replace node.iter
        #--------------------------------------------------------------------

        call_args = [nodes.ConstNode(0, typesystem.Py_ssize_t),
                     nodes.CoercionNode(stop, typesystem.Py_ssize_t),
                     nodes.ConstNode(1, typesystem.Py_ssize_t),]

        node.iter = ast.Call(func=call_func, args=call_args)
        nodes.typednode(node.iter, call_func.type)

        node.index = target_temp.load(invariant=True)

        #--------------------------------------------------------------------
        # Add assignment to new target variable at the start of the body
        #--------------------------------------------------------------------

        index = ast.Index(value=node.index)
        index.type = target_temp.type
        subscript = ast.Subscript(value=orig_iter,
                                  slice=index, ctx=ast.Load())
        nodes.typednode(subscript, get_type(orig_iter).dtype)

        #--------------------------------------------------------------------
        # Add assignment to new target variable at the start of the body
        #--------------------------------------------------------------------

        assign = ast.Assign(targets=[orig_target], value=subscript)
        node.body = [assign] + node.body

        #--------------------------------------------------------------------
        # Specialize new for loop through range iteration
        #--------------------------------------------------------------------

        return self.visit(node)

    def visit_For(self, node):
        if node.iter.type.is_range:
            return self.rewrite_range_iteration(node)
        elif node.iter.type.is_array and node.iter.type.ndim == 1:
            return self.rewrite_array_iteration(node)
        else:
            self.visitchildren(node)
            return node

#------------------------------------------------------------------------
# Transform for loops over builtins
#------------------------------------------------------------------------

class TransformBuiltinLoops(visitors.NumbaTransformer):
    def rewrite_enumerate(self, node):
        """
        Rewrite a loop like

        for i, x in enumerate(array[, start]):
            ...

        into

        _arr = array
        [_s = start]
        for _i in range(len(_arr)):
            i = _i [+ _s]
            x = _arr[_i]
            ...
        """
        call = node.iter
        if (len(call.args) not in (1, 2) or call.keywords or
            call.starargs or call.kwargs):
            self.error(call, 'expected 1 or 2 arguments to enumerate()')

        target = node.target
        if (not isinstance(target, (ast.Tuple, ast.List)) or
            len(target.elts) != 2):
            self.error(call, 'expected 2 iteration variables')

        array = call.args[0]
        start = call.args[1] if len(call.args) > 1 else None
        idx = target.elts[0]
        var = target.elts[1]

        array_temp = untypedTemp()
        if start:
            start_temp = untypedTemp() # TODO: only allow integer start
        idx_temp = nodes.TempNode(typesystem.Py_ssize_t)

        # for _i in range(len(_arr)):
        node.target = idx_temp.store()
        node.iter = ast.Call(ast.Name('range', ast.Load()),
                             [ast.Call(ast.Name('len', ast.Load()),
                                       [array_temp.load(True)],
                                       [], None, None)],
                             [], None, None)

        # i = _i [+ _s]
        new_idx = idx_temp.load()
        if start:
            new_idx = ast.BinOp(new_idx, ast.Add(), start_temp.load(True))
        node.body.insert(0, ast.Assign([idx], new_idx))

        # x = _arr[_i]
        value = ast.Subscript(array_temp.load(True),
                              ast.Index(idx_temp.load()),
                              ast.Load())
        node.body.insert(1, ast.Assign([var], value))

        # _arr = array; [_s = start]; ...
        body = [ ast.Assign([array_temp.store()], array), node ]
        if start:
            body.insert(1, ast.Assign([start_temp.store()], start))
        return map(self.visit, body)

    def rewrite_zip(self, node):
        """
        Rewrite a loop like

        for x, y... in zip(xs, ys...):
            ...

        into

        _xs = xs; _ys = ys...
        for _i in range(min(len(_xs), len(_ys)...)):
            x = _xs[_i]; y = _ys[_i]...
            ...
        """
        call = node.iter
        if not call.args or call.keywords or call.starargs or call.kwargs:
            self.error(call, 'expected at least 1 argument to zip()')

        target = node.target
        if (not isinstance(target, (ast.Tuple, ast.List)) or
            len(target.elts) != len(call.args)):
            self.error(call, 'expected %d iteration variables' % len(call.args))

        temps = [untypedTemp() for _ in xrange(len(call.args))]
        idx_temp = nodes.TempNode(typesystem.Py_ssize_t)

        # min(len(_xs), len(_ys)...)
        len_call = ast.Call(ast.Name('min', ast.Load()),
                            [ast.Call(ast.Name('len', ast.Load()),
                                      [tmp.load(True)], [], None, None)
                             for tmp in temps],
                            [], None, None)

        # for _i in range(...):
        node.target = idx_temp.store()
        node.iter = ast.Call(ast.Name('range', ast.Load()),
                             [len_call], [], None, None)

        # x = _xs[_i]; y = _ys[_i]...
        node.body = [ast.Assign([tgt],
                                ast.Subscript(tmp.load(True),
                                              ast.Index(idx_temp.load()),
                                              ast.Load()))
                     for tgt, tmp in zip(target.elts, temps)] + \
                    node.body

        # _xs = xs; _ys = ys...
        body = [ast.Assign([tmp.store()], arg)
                for tmp, arg in zip(temps, call.args)] + \
               [node]
        return map(self.visit, body)

    HANDLERS = {
        id(enumerate): rewrite_enumerate,
        id(zip):       rewrite_zip,
    }

    def visit_For(self, node):
        if (isinstance(node.iter, ast.Call) and
            isinstance(node.iter.func, ast.Name)):
            name = node.iter.func.id
            if name not in self.symtab:
                obj = (self.func_globals[name]
                       if name in self.func_globals else
                       getattr(builtins, name, None))
                rewriter = self.HANDLERS.get(id(obj))
                if rewriter:
                    return rewriter(self, node)

        self.visitchildren(node)
        return node

#------------------------------------------------------------------------
# Transform for loops over Objects
#------------------------------------------------------------------------

class SpecializeObjectIteration(visitors.NumbaTransformer):
    """
    This transforms for loops over objects.
    """

    def visit_For(self, node):
        while_node = make_while_from_for(node)

        test = nodes.const(True, bool_)
        while_node.test = test

        impl = loopimpl.find_iterator_impl(node)

        # Get the iterator, loop body, and the item
        iter = impl.getiter(self.context, node, self.llvm_module)
        body = impl.body(self.context, node, self.llvm_module)
        item = impl.next(self.context, node, self.llvm_module)

        # Coerce item to LHS and assign
        item = nodes.CoercionNode(item, node.target.type)
        target_assmnt = ast.Assign(targets=[node.target], value=item)

        # Update While node body
        body.insert(0, target_assmnt)
        while_node.body = body

        nodes.merge_cfg_in_while(while_node)

        return ast.Suite(body=[iter, while_node])
