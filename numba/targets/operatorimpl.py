import operator
import llvm.core as lc
from llvm.core import Type
from numba.targets.imputils import implement, Registry
from numba.targets import builtins
from numba import types, cgutils, utils
from numba.typing import signature

registry = Registry()
register = registry.register


for ty in types.integer_domain:
    register(implement(operator.add, ty, ty)(builtins.int_add_impl))
    register(implement(operator.sub, ty, ty)(builtins.int_sub_impl))
    register(implement(operator.mul, ty, ty)(builtins.int_mul_impl))
    register(implement(operator.eq, ty, ty)(builtins.int_eq_impl))
    register(implement(operator.ne, ty, ty)(builtins.int_ne_impl))
    register(implement(operator.lshift, ty, ty)(builtins.int_shl_impl))
    register(implement(operator.and_, ty, ty)(builtins.int_and_impl))
    register(implement(operator.or_, ty, ty)(builtins.int_or_impl))
    register(implement(operator.xor, ty, ty)(builtins.int_xor_impl))
    register(implement(operator.neg, ty)(builtins.int_negate_impl))
    register(implement(operator.invert, ty)(builtins.int_invert_impl))

for ty in types.unsigned_domain:
    register(implement(operator.div, ty, ty)(builtins.int_udiv_impl))
    register(implement(operator.floordiv, ty, ty)(builtins.int_ufloordiv_impl))
    register(implement(operator.truediv, ty, ty)(builtins.int_utruediv_impl))
    register(implement(operator.mod, ty, ty)(builtins.int_urem_impl))
    register(implement(operator.lt, ty, ty)(builtins.int_ult_impl))
    register(implement(operator.le, ty, ty)(builtins.int_ule_impl))
    register(implement(operator.gt, ty, ty)(builtins.int_ugt_impl))
    register(implement(operator.ge, ty, ty)(builtins.int_uge_impl))
    register(implement(operator.pow, ty, ty)(builtins.int_upower_impl))
    register(implement(operator.rshift, ty, ty)(builtins.int_lshr_impl))

for ty in types.signed_domain:
    register(implement(operator.div, ty, ty)(builtins.int_sdiv_impl))
    register(implement(operator.floordiv, ty, ty)(builtins.int_sfloordiv_impl))
    register(implement(operator.truediv, ty, ty)(builtins.int_struediv_impl))
    register(implement(operator.mod, ty, ty)(builtins.int_srem_impl))
    register(implement(operator.lt, ty, ty)(builtins.int_slt_impl))
    register(implement(operator.le, ty, ty)(builtins.int_sle_impl))
    register(implement(operator.gt, ty, ty)(builtins.int_sgt_impl))
    register(implement(operator.ge, ty, ty)(builtins.int_sge_impl))
    register(implement(operator.pow, ty, ty)(builtins.int_spower_impl))
    register(implement(operator.rshift, ty, ty)(builtins.int_ashr_impl))

for ty in types.real_domain:
    register(implement(operator.add, ty, ty)(builtins.real_add_impl))
    register(implement(operator.sub, ty, ty)(builtins.real_sub_impl))
    register(implement(operator.mul, ty, ty)(builtins.real_mul_impl))
    register(implement(operator.div, ty, ty)(builtins.real_div_impl))
    register(implement(operator.truediv, ty, ty)(builtins.real_div_impl))
    register(implement(operator.mod, ty, ty)(builtins.real_mod_impl))
    register(implement(operator.pow, ty, ty)(builtins.real_power_impl))
    register(implement(operator.eq, ty, ty)(builtins.real_eq_impl))
    register(implement(operator.ne, ty, ty)(builtins.real_ne_impl))
    register(implement(operator.lt, ty, ty)(builtins.real_lt_impl))
    register(implement(operator.le, ty, ty)(builtins.real_le_impl))
    register(implement(operator.gt, ty, ty)(builtins.real_gt_impl))
    register(implement(operator.ge, ty, ty)(builtins.real_ge_impl))
    register(implement(operator.neg, ty)(builtins.real_negate_impl))

