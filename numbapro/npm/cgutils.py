from contextlib import contextmanager
from llvm import core as lc

@contextmanager
def goto_entry_block(builder):
    old = builder.basic_block
    entry = builder.basic_block.function.basic_blocks[0]
    builder.position_at_beginning(entry)
    yield
    builder.position_at_end(old)


@contextmanager
def goto_block(builder, block):
    old = builder.basic_block
    builder.position_at_beginning(block)
    yield
    builder.position_at_end(old)

def append_block(builder, name=''):
    return builder.basic_block.function.append_basic_block(name)

@contextmanager
def if_then(builder, cond):
    then = append_block(builder, 'then')
    orelse = append_block(builder, 'else')
    builder.cbranch(cond, then, orelse)
    with goto_block(builder, then):
        yield orelse
    builder.position_at_end(orelse)

def make_array(builder, elemty, values):
    '''Make a static array out of the given values.
    '''
    n = len(values)
    out = lc.Constant.undef(lc.Type.array(elemty, n))
    for i, v in enumerate(values):
        out = builder.insert_value(out, v, i)
    return out

def explode_array(builder, ary):
    '''Extract all elements of a static array into a list of values.
    '''
    n = ary.type.count
    return [builder.extract_value(ary, i) for i in range(n)]

def get_function(builder, name, return_type, args):
    mod = builder.basic_block.function.module
    functype = lc.Type.function(return_type, args)
    return mod.get_or_insert_function(functype, name)
