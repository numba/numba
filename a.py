import numpy as np
from numba import njit, types
from numba.typed.typedlist import List

@njit
def foo(l):
	try:
		print('try block')
		l.append(0)
	except:
		print('except block')
		l.append(1)
	else:
		print('else block')
		l.append(2)
	finally:
		print('finally block')
		l.append(3)
		return l

print('-- Test foo --')
l = List.empty_list(types.int64)
try:
	print('foo:', foo(l)) # works
except:
	print('foo: UnsupportedError')

##########

@njit
def bar(l):
	try:
		print('try block')
		l.append(0)
		raise ValueError('Error')
	except Exception:
		print('except block')
		l.append(1)
	else:
		print('else block')
		l.append(2)
	finally:
		print('finally block')
		l.append(3)
		return l

print('\n-- Test bar --')
l2 = List.empty_list(types.int64)
try:
	print('bar:', bar(l2))
except:
	print('bar: UnsupportedError')


##########

class CustomError(Exception):
	pass

@njit
def baz(l):
	try:
		print('try block')
		l.append(0)
		raise CustomError('Error')
	except Exception:
		print('except CustomError block')
		l.append(1)
	except: 
		print('except block')
		l.append(1)
	else:
		print('else block')
		l.append(2)
	finally:
		print('finally block')
		l.append(3)
		return l

print('\n-- Test baz --')
l3 = List.empty_list(types.int64)
try:
	print('baz:', baz(l3))
except:
	print('baz: UnsupportedError')


#######

@njit
def qux(l):
	try:
		print('try block')
		l.append(0)
	finally:
		print('finally block')
		l.append(3)
		return l

print('\n-- Test qux --')
l3 = List.empty_list(types.int64)
try:
	print('qux:', qux(l3))
except:
	print('qux: UnsupportedError')
