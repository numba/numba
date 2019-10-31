from __future__ import division, print_function

import math
from collections import namedtuple


# GCN architecture specific info
simd_per_cu = 4
wave_size = 64
vector_register_file_size = 64 * 2**10  # 64 kB
byte_per_VGPR = 4
vgpr_per_simd = vector_register_file_size // byte_per_VGPR
sgpr_per_simd = 512
max_wave_count = 10
max_inflight_wave_per_cu = max_wave_count * simd_per_cu

# XXX due to limit in AMDGPU backend
max_group_size = 256


_limits = namedtuple('_limits', ['allowed_wave_due_to_sgpr',
                                 'allowed_wave_due_to_vgpr',
                                 'allowed_wave',
                                 'allowed_vgpr_per_workitem',
                                 'occupancy',
                                 'reasons',
                                 'suggestions'])


def get_limiting_factors(group_size, vgpr_per_workitem, sgpr_per_wave):
    def _ceil(x):
        return int(math.ceil(x))

    # these might be zero, for resource limit treat as 1
    vgpr_per_workitem =  vgpr_per_workitem if vgpr_per_workitem > 0 else 1
    sgpr_per_wave = sgpr_per_wave if sgpr_per_wave > 0 else 1

    workitem_per_simd = group_size / simd_per_cu
    required_wave_count_per_simd = _ceil(workitem_per_simd / wave_size)
    required_vgpr_per_wave = vgpr_per_workitem * wave_size
    # limiting factor
    allowed_wave_due_to_sgpr = sgpr_per_simd // sgpr_per_wave
    allowed_wave_due_to_vgpr = vgpr_per_simd // required_vgpr_per_wave
    allowed_wave = min(allowed_wave_due_to_sgpr, max_wave_count, allowed_wave_due_to_vgpr)
    allowed_vgpr_per_workitem = _ceil(vgpr_per_simd / required_wave_count_per_simd / wave_size)
    # reasons
    reasons = set()
    if allowed_wave_due_to_sgpr < required_wave_count_per_simd:
        reasons.add('allowed_wave_due_to_sgpr')
    if allowed_wave_due_to_vgpr < required_wave_count_per_simd:
        reasons.add('allowed_wave_due_to_vgpr')
    if allowed_wave < required_wave_count_per_simd:
        reasons.add('allowed_wave')
    if group_size > max_group_size:
        reasons.add('group_size')

    suggestions = [_suggestions[r] for r in sorted(reasons)]

    # occupancy
    inflight_wave_per_cu = (0 if reasons else
                            required_wave_count_per_simd * simd_per_cu)
    occupancy = inflight_wave_per_cu / max_inflight_wave_per_cu

    return _limits(allowed_wave_due_to_sgpr=allowed_wave_due_to_sgpr,
                   allowed_wave_due_to_vgpr=allowed_wave_due_to_vgpr,
                   allowed_wave=allowed_wave,
                   allowed_vgpr_per_workitem=allowed_vgpr_per_workitem,
                   occupancy=occupancy,
                   reasons=reasons,
                   suggestions=suggestions)


_suggestions = {}

_suggestions['allowed_wave_due_to_sgpr'] = (
    "* Cannot allocate enough sGPRs for all resident wavefronts."
)

_suggestions['allowed_wave_due_to_vgpr'] = (
    "* Cannot allocate enough vGPRs for all resident wavefronts."
)

_suggestions['allowed_wave'] = (
    "* Launch requires too many wavefronts. Try reducing group-size."
)

_suggestions['group_size'] = (
    "* Exceeds max group size (256)."
)

