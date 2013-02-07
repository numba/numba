import ast
import textwrap

import numba
from numba import *
from numba import error, closure
from numba import macros, utils, typesystem
from numba.symtab import Variable
from numba import visitors, nodes, error, functions

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


#------------------------------------------------------------------------
# Transform for loops
#------------------------------------------------------------------------

class TransformForIterable(visitors.NumbaTransformer):
    """
    This transforms for loops such as loops over 1D arrays:

            for value in my_array:
                ...

        into

            for i in my_array.shape[0]:
                value = my_array[i]
    """

    def visit_For(self, node):
        if node.iter.type.is_range:
            #
            ### Handle range iteration
            #
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

            # Substitute template and type infer
            result = self.run_template(
                templ, vars=dict(length=Py_ssize_t),
                start=start, stop=stop, step=step,
                nsteps=nsteps.store(), nsteps_load=nsteps.load(),
                temp=temp.store(), temp_load=temp.load(),
                target=node.target,
                body=body, else_body=else_body)

            # Patch the body and else clause
            body.body.extend(node.body)
            else_body.body.extend(node.orelse)

            # Patch cfg block of target variable to the 'body' block of the
            # while
            node.target.variable.block = node.if_block

            # Create the place to jump to for 'continue'
            while_node = result.body[-1]
            assert isinstance(while_node, ast.While)

            target_increment = while_node.body[-1]
            assert isinstance(target_increment, ast.Assign)
            #incr_block = nodes.LowLevelBasicBlockNode(node=target_increment,
            #                                          name='for_increment')
            node.incr_block.body = [target_increment]
            while_node.body[-1] = node.incr_block
            while_node.continue_block = node.incr_block

            # Patch the while with the For nodes cfg blocks
            attrs = dict(vars(node), **vars(while_node))
            while_node = nodes.build_while(**attrs)
            result.body[-1] = while_node

            return result

        elif node.iter.type.is_array and node.iter.type.ndim == 1:
            # Convert 1D array iteration to for-range and indexing
            logger.debug(ast.dump(node))

            orig_target = node.target
            orig_iter = node.iter

            # replace node.target with a temporary
            target_name = orig_target.id + '.idx'
            target_temp = nodes.TempNode(Py_ssize_t)
            node.target = target_temp.store()

            # replace node.iter
            call_func = ast.Name(id='range', ctx=ast.Load())
            call_func.type = typesystem.RangeType()
            call_func.variable = Variable(call_func.type)

            shape_index = ast.Index(nodes.ConstNode(0, typesystem.Py_ssize_t))
            shape_index.type = typesystem.npy_intp
            stop = ast.Subscript(value=nodes.ShapeAttributeNode(orig_iter),
                                 slice=shape_index,
                                 ctx=ast.Load())
            stop.type = typesystem.intp
            stop.variable = Variable(stop.type)
            call_args = [nodes.ConstNode(0, typesystem.Py_ssize_t),
                         nodes.CoercionNode(stop, typesystem.Py_ssize_t),
                         nodes.ConstNode(1, typesystem.Py_ssize_t),]

            node.iter = ast.Call(func=call_func, args=call_args)
            node.iter.type = call_func.type

            node.index = target_temp.load(invariant=True)
            # add assignment to new target variable at the start of the body
            index = ast.Index(value=node.index)
            index.type = target_temp.type
            subscript = ast.Subscript(value=orig_iter,
                                      slice=index, ctx=ast.Load())
            subscript.type = orig_iter.variable.type.dtype
            subscript.variable = Variable(subscript.type)
            coercion = nodes.CoercionNode(subscript, orig_target.type)
            assign = ast.Assign(targets=[orig_target], value=subscript)

            node.body = [assign] + node.body

            return self.visit(node)
        else:
            raise error.NumbaError("Unsupported for loop pattern")

