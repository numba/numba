import os
from contextlib import contextmanager
from collections import defaultdict
from timeit import default_timer as timer
import inspect
from . import (symbolic, typing, codegen, execution, fnlib, imlib, extending,
               arylib)

DEFAULT_FLAGS = 'overflow', 'zerodivision', 'boundcheck', 'wraparound'

def get_builtin_context():
    funclib = fnlib.get_builtin_function_library()
    implib = imlib.ImpLib(funclib)
    implib.populate_builtin()
    libs = funclib, implib
    extending.extends(libs, arylib.extensions)
    return libs

global_builtin_libs = get_builtin_context()

def compile(func, retty, argtys, libs=global_builtin_libs, flags=DEFAULT_FLAGS):
    funclib, implib = libs

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
        type_infer(func, blocks, return_type, args, funclib)

        lmod, lfunc, excs = code_generation(func, blocks, return_type, args,
                                            implib, flags=flags)

        lmod.verify()

        jit = execution.JIT(lfunc = lfunc,
                            retty = retty,
                            argtys = argtys,
                            exceptions = excs)
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

def code_generation(func, blocks, return_type, args, implib, flags):
    cg = codegen.CodeGen(func        = func,
                         blocks      = blocks,
                         args        = args,
                         return_type = return_type,
                         implib      = implib,
                         flags       = flags)
    cg.codegen()
    return cg.lmod, cg.lfunc, cg.exceptions
