import numpy as np

numpy_version = tuple(map(int, np.__version__.split('.')[:2]))

sizeof_unicode_char = np.dtype('U1').itemsize
