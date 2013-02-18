import ast
import textwrap

import numba
from numba import *
from numba import error, closures, function_util
from numba import macros, utils, typesystem
from numba.symtab import Variable
from numba import visitors, nodes, error, functions
from numba.typesystem import get_type, is_obj, typematch
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
    return while_node

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
    return while_node


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

        start, stop, step = [nodes.CloneableNode(n)
                             for n in (start, stop, step)]

        if have_step:
            compute_nsteps = """
                    $length = {{stop}} - {{start}}
                    {{nsteps}} = $length / {{step}}
                    if {{nsteps_load}} * {{step}} != $length: #$length % {{step}}:
                        # Test for truncation
                        {{nsteps}} = {{nsteps_load}} + 1
                    # print "nsteps", {{nsteps_load}}
                """
        else:
            compute_nsteps = "{{nsteps}} = {{stop}} - {{start}}"

        if node.orelse:
            else_clause = "else: {{else_body}}"
        else:
            else_clause = ""

        templ = textwrap.dedent("""
                %s
                {{temp}} = 0
                while {{temp_load}} < {{nsteps_load}}:
                    {{target}} = {{start}} + {{temp_load}} * {{step}}
                    {{body}}
                    {{temp}} = {{temp_load}} + 1
                %s
            """) % (textwrap.dedent(compute_nsteps), else_clause)

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

        #--------------------------------------------------------------------
        # Patch the body and else clause
        #--------------------------------------------------------------------

        body.body.extend(node.body)
        else_body.body.extend(node.orelse)

        while_node = result.body[-1]
        assert isinstance(while_node, ast.While)

        target_increment = while_node.body[-1]
        assert isinstance(target_increment, ast.Assign)

        # Add target variable increment basic block
        node.incr_block.body = [target_increment]
        while_node.body[-1] = node.incr_block

        #--------------------------------------------------------------------
        # Create a While with the ForNode's cfg blocks merged in
        #--------------------------------------------------------------------

        while_node = make_while_loop(while_node)
        copy_basic_blocks(node, while_node)
        while_node = nodes.build_while(**vars(while_node))

        # Create the place to jump to for 'continue'
        while_node.continue_block = node.incr_block

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

        target_name = orig_target.id + '.idx'
        target_temp = nodes.TempNode(Py_ssize_t)
        node.target = target_temp.store()

        #--------------------------------------------------------------------
        # Create range(A.shape[0])
        #--------------------------------------------------------------------

        call_func = ast.Name(id='range', ctx=ast.Load())
        nodes.typednode(call_func, typesystem.RangeType())

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
