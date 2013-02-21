import numba

def test_nosource():
    source = '''
@numba.autojit
def foo (): return 99

'''
    exec source
    assert foo() == 99

if __name__ == "__main__":
    test_nosource()
