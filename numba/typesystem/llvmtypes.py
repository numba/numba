import llvm.core

from numba.typesystem.itypesystem import consing, tyname
from numba.typesystem import universe

domain_name = "llvm"

#------------------------------------------------------------------------
# Helpers
#------------------------------------------------------------------------

def get_target_triple():
    target_machine = llvm.ee.TargetMachine.new()
    is_ppc = target_machine.triple.startswith("ppc")
    is_x86 = target_machine.triple.startswith("x86")
    return is_ppc, is_x86

def lbool():
    return llvm.core.Type.int(1)

def lint(name, itemsize):
    if name == "bool":
        return lbool()

    return llvm.core.Type.int(itemsize * 8)

def lfloat(name, itemsize):
    if itemsize == 4:
        return llvm.core.Type.float()
    elif itemsize == 8:
        return llvm.core.Type.double()
    else:
        is_ppc, is_x86 = get_target_triple()
        if itemsize == 16:
            if is_ppc:
                return llvm.core.Type.ppc_fp128()
            else:
                return llvm.core.Type.fp128()
        else:
            assert itemsize == 10 and is_x86, itemsize
            return llvm.core.Type.x86_fp80()

size = universe.default_type_sizes.__getitem__

unittypes = {}
for typename in universe.int_typenames:
    unittypes[typename] = lint(typename, size(typename))
for typename in universe.float_typenames:
    unittypes[typename] = lfloat(typename, size(typename))
unittypes["void"] = llvm.core.Type.void()

globals().update((tyname(name), ty) for name, ty in unittypes.iteritems())

#------------------------------------------------------------------------
# Exposed types
#------------------------------------------------------------------------

# @consing # llvm types don't hash in python 3 in llvmpy 0.11.2
def struct_(fields, name=None, readonly=False, packed=False):
    if packed:
        struct = llvm.core.Type.packed_struct
    else:
        struct = llvm.core.Type.struct

    return struct([field_type for field_name, field_type in fields])

# @consing
def pointer(base_type):
    if base_type.kind == llvm.core.TYPE_VOID:
        base_type = llvm.core.Type.int(8)
    return llvm.core.Type.pointer(base_type)

# @consing
def function(rettype, argtypes, name=None, is_vararg=False):
    return llvm.core.Type.function(rettype, argtypes, is_vararg)