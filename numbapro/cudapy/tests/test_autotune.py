from .support import testcase, main
from numbapro.cudapy import autotune

@testcase
def test_autotune_parsing_1():
    sample = '''
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'foo' for 'sm_30'
ptxas info    : Function properties for foo
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, 336 bytes cmem[0]
'''
    info = autotune.parse_compile_info(sample)
    assert info['foo']['stack'] == 0
    assert info['foo']['reg'] == 14

@testcase
def test_autotune_parsing_2():
    sample='''
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
    info = autotune.parse_compile_info(sample)
    assert info['foo']['stack'] == 8
    assert info['foo']['reg'] == 19
    assert info['foo']['shared'] == 0
    assert info['foo']['local'] == 0

    assert info['bar']['stack'] == 0


if __name__ == '__main__':
    main()

