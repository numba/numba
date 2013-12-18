from contextlib import contextmanager
import llvm.core as lc
from .types import auto_intp
from .cgutils import append_block, goto_block


@contextmanager
def loop(builder, begin, end, step):
    '''
    assumes positive steps and intp as index.
    '''
    begin = auto_intp(begin)
    end = auto_intp(end)
    step = auto_intp(step)

    bbentry = builder.basic_block
    bbcond = append_block(builder, 'loop.cond')
    bbbody = append_block(builder, 'loop.body')
    bbnext = append_block(builder, 'loop.next')
    bbend  = append_block(builder, 'loop.end')

    with goto_block(builder, bbcond):
        index = builder.phi(begin.type)
        index.add_incoming(begin, bbentry)

        loop_pred = builder.icmp(lc.ICMP_SLT, index, end)
        builder.cbranch(loop_pred, bbbody, bbend)

    with goto_block(builder, bbbody):
        yield index
        builder.branch(bbnext)

    with goto_block(builder, bbnext):
        index.add_incoming(builder.add(index, step), bbnext)
        builder.branch(bbcond)

    builder.branch(bbcond)
    builder.position_at_end(bbend)


@contextmanager
def loop_nest(builder, begins, ends, steps):
    '''
    Order from the inner most dimension to the outer most dimension.
    Assumes positive steps.
    '''
    ndim = len(begins)
    assert ndim == len(ends), '`ends` must have the same length as `begins`'
    assert ndim == len(steps), '`steps` must have the same length as `begins`'

    @contextmanager
    def nest(old, new):
        with new as newout:
            with old as oldout:
                yield oldout + [newout]

    @contextmanager
    def bottom():
        with loop(builder, begins[0], ends[0], steps[0]) as index:
            yield [index]

    old = bottom()
    for begin, end, step in zip(begins[1:], ends[1:], steps[1:]):
        old = nest(old, loop(builder, begin, end, step))

    with old as out:
        yield out
