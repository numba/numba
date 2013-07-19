from llvm import core as lc
from contextlib import contextmanager

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
