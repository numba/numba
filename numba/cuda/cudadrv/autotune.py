"""
- Parse jit compile info
- Compute warp occupany histogram
"""
from __future__ import division, absolute_import, print_function
import math
import re

SMEM16K = 16 * 2 ** 10
SMEM48K = 48 * 2 ** 10
SMEM64K = 64 * 2 ** 10
SMEM96K = 96 * 2 ** 10
SMEM112K = 112 * 2 ** 10

#------------------------------------------------------------------------------
# autotuning


class OccupancyThreadKey(object):
    def __init__(self, item):
        self.occupancy, self.threads = item
        self.comparison = self.occupancy, 1 / self.threads

    def __lt__(self, other):
        return self.comparison < other.comparison

    def __eq__(self, other):
        return self.comparison == other.comparison

    def __ne__(self, other):
        return self.comparison != other.comparison

    def __gt__(self, other):
        return self.comparison > other.comparison

    def __le__(self, other):
        return self.comparison <= other.comparison

    def __ge__(self, other):
        return self.comparison >= other.comparison


class AutoTuner(object):
    """Autotune a kernel based upon the theoretical occupancy.
    """
    def __init__(self, cc, info, smem_config=None, dynsmem=0):
        self.cc = cc
        self.dynsmem = dynsmem
        self._table = warp_occupancy(info=info, cc=cc)
        self._by_occupancy = list(reversed(sorted(((occup, tpb)
                                                   for tpb, (occup, factor)
                                                   in self.table.items()),
                                                  key=OccupancyThreadKey)))

    @property
    def table(self):
        """A dict with thread-per-block as keys and tuple-2 of
        (occupency, limiting factor) as values.
        """
        return self._table

    @property
    def by_occupancy(self):
        """A list of tuple-2 of (occupancy, thread-per-block) sorted in
        descending.

        The first item has the highest occupancy and the lowest number of
        thread-per-block.
        """
        return self._by_occupancy

    def best(self):
        return self.max_occupancy_min_blocks()

    def max_occupancy_min_blocks(self):
        """Returns the thread-per-block that optimizes for
        maximum occupancy and minimum blocks.

        Maximum blocks allows for the best utilization of parallel execution
        because each block can be executed concurrently on different SM.
        """
        return self.by_occupancy[0][1]

    def closest(self, tpb):
        """Find the occupancy of the closest tpb
        """
        # round to the nearest multiple of warpsize
        warpsize = PHYSICAL_LIMITS[self.cc]['thread_per_warp']
        tpb = ceil(tpb, warpsize)
        # search
        return self.table.get(tpb, [0])[0]


    def best_within(self, mintpb, maxtpb):
        """Returns the best tpb in the given range inclusively.
        """
        warpsize = PHYSICAL_LIMITS[self.cc]['thread_per_warp']
        mintpb = int(ceil(mintpb, warpsize))
        maxtpb = int(floor(maxtpb, warpsize))
        return self.prefer(*range(mintpb, maxtpb + 1, warpsize))

    def prefer(self, *tpblist):
        """Prefer the thread-per-block with the highest warp occupancy
        and the lowest thread-per-block.

        May return None if all threads-per-blocks are invalid
        """
        bin = []
        for tpb in tpblist:
            occ = self.closest(tpb)
            if occ > 0:
                bin.append((occ, tpb))
        if bin:
            return sorted(bin, key=OccupancyThreadKey)[-1][1]


#------------------------------------------------------------------------------
# warp occupancy calculator

# Reference: NVIDIA CUDA Toolkit v6.5 Programming Guide, Appendix G.
# URL: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

LIMITS_CC_20 = {
    'thread_per_warp': 32,
    'warp_per_sm': 48,
    'thread_per_sm': 1536,
    'block_per_sm': 8,
    'registers': 32768,
    'reg_alloc_unit': 64,
    'reg_alloc_gran': 'warp',
    'reg_per_thread': 63,
    'smem_per_sm': SMEM48K,
    'smem_alloc_unit': 128,
    'warp_alloc_gran': 2,
    'max_block_size': 1024,
    'default_smem_config': SMEM16K,
}

LIMITS_CC_21 = LIMITS_CC_20

LIMITS_CC_30 = {
    'thread_per_warp': 32,
    'warp_per_sm': 64,
    'thread_per_sm': 2048,
    'block_per_sm': 16,
    'registers': 65536,
    'reg_alloc_unit': 256,
    'reg_alloc_gran': 'warp',
    'reg_per_thread': 63,
    'smem_per_sm': SMEM48K,
    'smem_alloc_unit': 256,
    'warp_alloc_gran': 4,
    'max_block_size': 1024,
    'default_smem_config': SMEM48K,
}

LIMITS_CC_35 = LIMITS_CC_30.copy()
LIMITS_CC_35.update({
    'reg_per_thread': 255,
})

LIMITS_CC_37 = LIMITS_CC_35.copy()

LIMITS_CC_37.update({
    'registers': 131072,
    'default_smem_config': SMEM112K,
})


LIMITS_CC_50 = {
    'thread_per_warp': 32,
    'warp_per_sm': 64,
    'thread_per_sm': 2048,
    'block_per_sm': 32,
    'registers': 65536,
    'reg_alloc_unit': 256,
    'reg_alloc_gran': 'warp',
    'reg_per_thread': 255,
    'smem_per_sm': SMEM64K,
    'smem_per_block': SMEM48K,
    'smem_alloc_unit': 256,
    'warp_alloc_gran': 4,
    'max_block_size': 1024,
    'default_smem_config': SMEM64K,
}

LIMITS_CC_52 = LIMITS_CC_50.copy()
LIMITS_CC_52.update({
    'smem_per_sm': SMEM96K,
    'default_smem_config': SMEM96K,
})
LIMITS_CC_53 = LIMITS_CC_50.copy()
LIMITS_CC_53.update({
    'registers': 32768,
})

LIMITS_CC_60 = LIMITS_CC_50.copy()
LIMITS_CC_60.update({
    'warp_alloc_gran': 2,
})
LIMITS_CC_61 = LIMITS_CC_60.copy()
LIMITS_CC_61.update({
    'smem_per_sm': SMEM96K,
    'default_smem_config': SMEM96K,
    'warp_alloc_gran': 4,
})
LIMITS_CC_62 = LIMITS_CC_60.copy()
LIMITS_CC_62.update({
    'thread_per_sm': 4096,
    'warp_per_sm': 128,
    'warp_alloc_gran': 4,
})

PHYSICAL_LIMITS = {
    (2, 0): LIMITS_CC_20,
    (2, 1): LIMITS_CC_21,
    (3, 0): LIMITS_CC_30,
    (3, 5): LIMITS_CC_35,
    (3, 7): LIMITS_CC_35,
    (5, 0): LIMITS_CC_50,
    (5, 2): LIMITS_CC_52,
    (5, 3): LIMITS_CC_53,
    (6, 0): LIMITS_CC_50,
    (6, 1): LIMITS_CC_61,
    (6, 2): LIMITS_CC_62,
}


def ceil(x, s=1):
    return s * math.ceil(x / s)


def floor(x, s=1):
    return s * math.floor(x / s)


def warp_occupancy(info, cc, smem_config=None):
    """Returns a dictionary of {threadperblock: occupancy, factor}

    Only threadperblock of multiple of warpsize is used.
    Only threadperblock of non-zero occupancy is returned.
    """
    ret = {}
    try:
        limits = PHYSICAL_LIMITS[cc]
    except KeyError:
        raise ValueError("%s is not a supported compute capability"
                             % ".".join(str(c) for c in cc))
    if smem_config is None:
        smem_config = limits['default_smem_config']
    warpsize = limits['thread_per_warp']
    max_thread = info.maxthreads

    for tpb in range(warpsize, max_thread + 1, warpsize):
        result = compute_warp_occupancy(tpb=tpb,
                                        reg=info.regs,
                                        smem=info.shared,
                                        smem_config=smem_config,
                                        limits=limits)
        if result[0]:
            ret[tpb] = result
    return ret


def compute_warp_occupancy(tpb, reg, smem, smem_config, limits):
    assert limits['reg_alloc_gran'] == 'warp', \
        "assume warp register allocation granularity"
    limit_block_per_sm = limits['block_per_sm']
    limit_warp_per_sm = limits['warp_per_sm']
    limit_thread_per_warp = limits['thread_per_warp']
    limit_reg_per_thread = limits['reg_per_thread']
    limit_total_regs = limits['registers']
    limit_total_smem = min(limits['smem_per_sm'], smem_config)
    my_smem_alloc_unit = limits['smem_alloc_unit']
    reg_alloc_unit = limits['reg_alloc_unit']
    warp_alloc_gran = limits['warp_alloc_gran']

    my_warp_per_block = ceil(tpb / limit_thread_per_warp)
    my_reg_count = reg
    my_reg_per_block = my_warp_per_block
    my_smem = smem
    my_smem_per_block = ceil(my_smem, my_smem_alloc_unit)

    # allocated resource
    limit_blocks_due_to_warps = min(limit_block_per_sm,
                                    floor(
                                        limit_warp_per_sm / my_warp_per_block))

    c39 = floor(limit_total_regs / ceil(my_reg_count * limit_thread_per_warp,
                                        reg_alloc_unit),
                warp_alloc_gran)

    limit_blocks_due_to_regs = (0
                                if my_reg_count > limit_reg_per_thread
                                else (floor(c39 / my_reg_per_block)
                                      if my_reg_count > 0
                                      else limit_block_per_sm))

    limit_blocks_due_to_smem = (floor(limit_total_smem /
                                      my_smem_per_block)
                                if my_smem_per_block > 0
                                else limit_block_per_sm)

    # occupancy
    active_block_per_sm = min(limit_blocks_due_to_smem,
                              limit_blocks_due_to_warps,
                              limit_blocks_due_to_regs)

    if active_block_per_sm == limit_blocks_due_to_warps:
        factor = 'warps'
    elif active_block_per_sm == limit_blocks_due_to_regs:
        factor = 'regs'
    else:
        factor = 'smem'

    active_warps_per_sm = active_block_per_sm * my_warp_per_block
    #active_threads_per_sm = active_warps_per_sm * limit_thread_per_warp

    occupancy = active_warps_per_sm / limit_warp_per_sm
    return occupancy, factor

