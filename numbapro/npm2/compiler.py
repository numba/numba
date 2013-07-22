import os
from contextlib import contextmanager
from collections import defaultdict
from timeit import default_timer as timer
import inspect
from . import symbolic, typing, codegen, execution, fnlib, imlib


global_funclib = fnlib.get_builtin_function_library()
global_implib = imlib.ImpLib(global_funclib)
global_implib.populate_builtin()

def compile(func, retty, argtys):
    with profile((func, tuple(argtys))):
        # preparation
        argspec = inspect.getargspec(func)
        assert not argspec.defaults
        assert not argspec.keywords
        assert not argspec.varargs

        args = dict((arg, typ) for arg, typ in zip(argspec.args, argtys))
        return_type = retty

        # compilation
        blocks =  symbolic_interpret(func)
        type_infer(func, blocks, return_type, args, global_funclib)

        lmod, lfunc = code_generation(func, blocks, return_type, args, global_implib)

        jit = execution.JIT(lfunc = lfunc,
                            retty = retty,
                            argtys = argtys)
        return jit

#----------------------------------------------------------------------------
# Profile

PROFILE_STATS = defaultdict(list)
NPM_PROFILING = int(os.environ.get('NPM_PROFILING', 0))

@contextmanager
def profile(id):
    '''compiler profiler
    '''
    if NPM_PROFILING:
        ts = timer()
        yield
        te = timer()
        PROFILE_STATS[id].append(te - ts)
    else:
        yield

def print_stats():
    '''print profiling stats for compiler
    '''
    cumtimes = []
    for id, nums in PROFILE_STATS.iteritems():
        local_avg = sum(nums)/len(nums)
        cumtimes.append((local_avg, id))

    n = len(cumtimes)
    scumtimes = sorted(cumtimes)
    longest = scumtimes[-1]
    fastest = scumtimes[0]
    median = scumtimes[n // 2]

    print 'longest', longest
    print 'fastest', fastest
    print 'median', median

#----------------------------------------------------------------------------
# Internals

def symbolic_interpret(func):
    se = symbolic.SymbolicExecution(func)
    se.interpret()
    return se.blocks

def type_infer(func, blocks, return_type, args, funclib):
    infer = typing.Infer(func        = func,
                         blocks      = blocks,
                         args        = args,
                         return_type = return_type,
                         funclib     = funclib)
    infer.infer()

def code_generation(func, blocks, return_type, args, implib):
    cg = codegen.CodeGen(func        = func,
                         blocks      = blocks,
                         args        = args,
                         return_type = return_type,
                         implib      = implib)
    cg.codegen()
    return cg.lmod, cg.lfunc
