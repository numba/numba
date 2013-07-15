from .support import testcase, main
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
    assert info['foo']['stack'] == 0
    assert info['foo']['reg'] == 14

@testcase
def test_autotune_parsing_2():
    sample=    info = autotune.parse_compile_info(SAMPLE2)
    assert info['foo']['stack'] == 8
    assert info['foo']['reg'] == 19
    assert info['foo']['shared'] == 0
    assert info['foo']['local'] == 0

    assert info['bar']['stack'] == 0


def round100(x):
    return round(x * 100)

@testcase
def test_autotune_occupancy_1():
    info = autotune.parse_compile_info(SAMPLE1)
    result = autotune.warp_occupancy(info['foo'], cc=(2, 0))
    assert round100(result[256][0]) == 100
    assert round100(result[288][0]) == 94
    assert round100(result[320][0]) == 83
    assert round100(result[352][0]) == 92
    assert round100(result[416][0]) == 81
    assert round100(result[448][0]) == 88
    assert round100(result[480][0]) == 94
    assert round100(result[512][0]) == 100
    assert round100(result[544][0]) == 71

@testcase
def test_autotune_occupancy_2():
    info = autotune.parse_compile_info(SAMPLE1)
    result = autotune.warp_occupancy(info['foo'], cc=(3, 5))
    assert round100(result[256][0]) == 100
    assert round100(result[288][0]) == 98
    assert round100(result[320][0]) == 94
    assert round100(result[352][0]) == 86
    assert round100(result[416][0]) == 81
    assert round100(result[448][0]) == 88
    assert round100(result[480][0]) == 94
    assert round100(result[512][0]) == 100
    assert round100(result[544][0]) == 80

@testcase
def test_autotune():
    at = autotune.AutoTuner.parse('foo', SAMPLE1, cc=(2, 0))
    assert 192 == at.max_occupancy_max_blocks()
    assert 352 == at.prefer(320, 352, 416)
    assert 512 == at.prefer(320, 352, 416, 512)
    assert 256 == at.prefer(320, 352, 416, 512, 256)
    assert 192 == at.best_within(100, 300)

@testcase
def test_calc_occupancy():
    from numbapro import cuda
    autotuner= cuda.calc_occupancy(cc=(2, 0), reg=32, smem=1200)
    assert autotuner.best() == 128


    
if __name__ == '__main__':
    main()

