from .support import testcase, main, assertTrue
from numbapro.cudadrv import autotune


SAMPLE1 = '''
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'foo' for 'sm_30'
ptxas info    : Function properties for foo
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, 336 bytes cmem[0]
'''
SAMPLE2 = '''
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'foo' for 'sm_30'
ptxas info    : Function properties for foo
    8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, 336 bytes cmem[0]
ptxas info    : 0 bytes gmem
ptxas info    : Function properties for bar
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
info    : 0 bytes gmem
info    : Function properties for 'foo':
info    : used 19 registers, 8 stack, 0 bytes smem, 336 bytes cmem[0], 0 bytes lmem
    '''


@testcase
def test_autotune_parsing_1():
    info = autotune.parse_compile_info(SAMPLE1)
    assertTrue(info['foo']['stack'] == 0)
    assertTrue(info['foo']['reg'] == 14)

@testcase
def test_autotune_parsing_2():
    info = autotune.parse_compile_info(SAMPLE2)
    assertTrue(info['foo']['stack'] == 8)
    assertTrue(info['foo']['reg'] == 19)
    assertTrue(info['foo']['shared'] == 0)
    assertTrue(info['foo']['local'] == 0)

    assertTrue(info['bar']['stack'] == 0)


def round100(x):
    return round(x * 100)

@testcase
def test_autotune_occupancy_1():
    info = autotune.parse_compile_info(SAMPLE1)
    result = autotune.warp_occupancy(info['foo'], cc=(2, 0))
    assertTrue(round100(result[256][0]) == 100)
    assertTrue(round100(result[288][0]) == 94)
    assertTrue(round100(result[320][0]) == 83)
    assertTrue(round100(result[352][0]) == 92)
    assertTrue(round100(result[416][0]) == 81)
    assertTrue(round100(result[448][0]) == 88)
    assertTrue(round100(result[480][0]) == 94)
    assertTrue(round100(result[512][0]) == 100)
    assertTrue(round100(result[544][0]) == 71)

@testcase
def test_autotune_occupancy_2():
    info = autotune.parse_compile_info(SAMPLE1)
    result = autotune.warp_occupancy(info['foo'], cc=(3, 5))
    assertTrue(round100(result[256][0]) == 100)
    assertTrue(round100(result[288][0]) == 98)
    assertTrue(round100(result[320][0]) == 94)
    assertTrue(round100(result[352][0]) == 86)
    assertTrue(round100(result[416][0]) == 81)
    assertTrue(round100(result[448][0]) == 88)
    assertTrue(round100(result[480][0]) == 94)
    assertTrue(round100(result[512][0]) == 100)
    assertTrue(round100(result[544][0]) == 80)

@testcase
def test_autotune():
    at = autotune.AutoTuner.parse('foo', SAMPLE1, cc=(2, 0))
    assertTrue(192 == at.max_occupancy_max_blocks())
    assertTrue(352 == at.prefer(320, 352, 416))
    assertTrue(512 == at.prefer(320, 352, 416, 512))
    assertTrue(256 == at.prefer(320, 352, 416, 512, 256))
    assertTrue(192 == at.best_within(100, 300))

@testcase
def test_calc_occupancy():
    from numbapro import cuda
    autotuner= cuda.calc_occupancy(cc=(2, 0), reg=32, smem=1200)
    assertTrue(autotuner.best() == 128)


    
if __name__ == '__main__':
    main()

