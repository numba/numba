from matplotlib import pyplot as plt
import pickle, sys

with open(sys.argv[1], 'rb') as fin:
    record = pickle.load(fin)

for name in 'cython numba basic stream parallel'.split():
    plt.plot(record[name], label=name)

plt.ylabel('peak bandwidth (word/second)')
plt.xlabel('word count (log2)')

plt.legend()
plt.show()
