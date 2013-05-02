'''
python test_issue_28.py 
--------------------- Numba Encountered Errors or Warnings ---------------------
def test(x):
^
Warning 13:0: Unreachable code
--------------------------------------------------------------------------------

'''
from numbapro import jit

@jit('int32(int32)')
def test(x):
    if x <= 0:
        return 1
    else:
        return  x

print test


