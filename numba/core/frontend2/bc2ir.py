from numba.core import bytecode
from . import bcinterp

from numba_rvsdg.core.datastructures.byte_flow import ByteFlow

def run_frontend(func): #, inline_closures=False, emit_dels=False):
    # XXX make this a dedicated Pipeline?
    func_id = bytecode.FunctionIdentity.from_function(func)


    byteflow = ByteFlow.from_bytecode(func.__code__)
    return byteflow
    # bc = bytecode.ByteCode(func_id=func_id)
    # interp = bcinterp.Interpreter(func_id)
    # func_ir = interp.interpret(bc)
    # return func_ir
