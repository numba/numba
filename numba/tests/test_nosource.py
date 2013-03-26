import sys
import numba

def test_nosource():
    source = '''
@numba.autojit
def foo (): return 99

'''
    new_locals = {}
    exec(source, globals(), new_locals)
    foo = new_locals['foo']
    assert foo() == 99

if sys.version_info[:2] < (2, 7):
    del test_nosource
elif __name__ == "__main__":
    test_nosource()
