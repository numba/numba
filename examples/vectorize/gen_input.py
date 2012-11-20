# Generate random values between 0 to 1
# and store it in a file

import sys
import numpy as np

nstring, filename = sys.argv[1:]
n = int(float(nstring))
print 'N', n
array = np.array(np.random.sample(n), dtype=np.float32)
with open(filename, 'wb') as fout:
    array.tofile(fout)