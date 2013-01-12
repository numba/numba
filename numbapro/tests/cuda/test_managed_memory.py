from numbapro._cuda.ndarray import SmallMemoryManager

smm = SmallMemoryManager(nbytes=256)
val0 = 1, 2, 3
mm0 = smm.obtain(val0)
mm1 = smm.obtain(val0)
val1 = 3, 2, 1
mm2 = smm.obtain(val1)

del mm0
del mm1
del mm2

assert len(smm.valuemap) == 0
