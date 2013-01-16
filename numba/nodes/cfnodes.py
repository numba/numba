from numba.nodes import *

basic_block_fields = ['cond_block', 'if_block', 'else_block', 'exit_block']

def delete_control_blocks(flow_node, flow):
    """
    Remove all control flow basic blocks from the CFG given a FlowNode
    and the CFG. Also removes Name references from cf_references.
    """
    parent = flow_node.cond_block.idom
    flow_node.exit_block.reparent(parent)
    flow.blocks.remove(flow_node.exit_block)
    flow_node.exit_block = None

    #flow_node.cond_block.delete(flow)
    #flow_node.if_block.delete(flow)

    #if flow_node.orelse:
    #    flow_node.else_block.delete(flow)

    from numba import control_flow
    control_flow.DeleteStatement(flow).visit(flow_node)

class FlowNode(Node):

    cond_block = None
    if_block = None
    else_block = None
    exit_block = None

    def __init__(self, **kwargs):
        super(FlowNode, self).__init__(**kwargs)

        from numba import control_flow
        for field_name in basic_block_fields:
            if not getattr(self, field_name):
                block = control_flow.ControlBlock(-1, is_fabricated=True)
                setattr(self, field_name, block)


class If(ast.If, FlowNode):
    pass

class While(ast.While, FlowNode):

    # Place to jump to when we see a 'continue'. The default is
    # 'the condition block'. For 'for' loops we set this to
    # 'the counter increment block'
    continue_block = None

class For(ast.For, FlowNode):
    pass

def merge_cfg_in_ast(basic_block_fields, bodies, node):
    for bb_name, body_name in zip(basic_block_fields, bodies):
        body = getattr(node, body_name)
        if not body:
            continue
        bb = getattr(node, bb_name)
        if not bb.body:
            if isinstance(body, list):
                bb.body.extend(body)
                bb = [bb]
            else:
                bb.body.append(body)

        setattr(node, body_name, bb)

def build_if(cls=If, **kwargs):
    node = cls(**kwargs)
    bodies = ['test', 'body', 'orelse']
    merge_cfg_in_ast(basic_block_fields, bodies, node)
    return node

def build_while(**kwargs):
    return build_if(cls=While, **kwargs)

def build_for(**kwargs):
    result = For(**kwargs)
    merge_cfg_in_ast(basic_block_fields, ['iter', 'body', 'orelse'], result)
    merge_cfg_in_ast(['target_block'], ['target'], result)
    return result


class LowLevelBasicBlockNode(Node):
    """
    Evaluate a statement or expression in a new LLVM basic block.
    """

    _fields = ['body']

    def __init__(self, body, label='unnamed', **kwargs):
        super(LowLevelBasicBlockNode, self).__init__(**kwargs)
        self.body = body
        self.label = label
        self.entry_block = None

    def create_block(self, translator, label=None):
        if self.entry_block is None:
            self.entry_block = translator.append_basic_block(label or self.label)
        return self.entry_block

class MaybeUnusedNode(Node):
    """
    Wraps an ast.Name() to indicate that the result may be unused.
    """

    _fields = ["name_node"]

    def __init__(self, name_node):
        self.name_node = name_node
