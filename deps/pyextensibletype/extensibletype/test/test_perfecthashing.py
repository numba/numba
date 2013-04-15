import time
import itertools

from nose.tools import eq_, ok_
import numpy as np
from .. import extensibletype, methodtable


def test_binsort():
    nbins = 64

    p = np.zeros(nbins, dtype=np.uint16)
    binsizes = np.random.randint(0, 7, size=nbins).astype(np.uint8)

    num_by_size = np.zeros(8, dtype=np.uint16)
    x = np.bincount(binsizes).astype(np.uint16)

    num_by_size[:x.shape[0]] = x
    extensibletype.bucket_argsort(p, binsizes, num_by_size)
    assert np.all(sorted(binsizes) == binsizes[p][::-1])

def test_basic():
    n=64
    prehashes = extensibletype.draw_hashes(np.random, n)
    assert len(prehashes) == len(set(prehashes))
    p, r, m_f, m_g, d = extensibletype.perfect_hash(prehashes, repeat=10**5)
    hashes = ((prehashes >> r) & m_f) ^ d[prehashes & m_g]
    print(p)
    print(d)
    hashes.sort()
    print(hashes)
    assert len(hashes) == len(np.unique(hashes))

# ---
# Test methodtable

def make_signature(type_permutation):
    return "".join(type_permutation[:-1]) + '->' + type_permutation[-1]

def make_ids():
    types = ['f', 'd', 'i', 'l', 'O']
    power = 5
    return map(make_signature, itertools.product(*(types,) * power))

def build_and_verify_methodtable(ids, flags, funcs):
    table = methodtable.PerfectHashMethodTable(methodtable.Hasher())
    table.generate_table(len(ids), ids, flags, funcs)

    for (signature, flag, func) in zip(ids, flags, funcs):
        result = table.find_method(signature)
        assert result is not None

        got_func, got_flag = result
        assert func == got_func, (func, got_func)
        # assert flag == got_flag, (flag, got_flag)

def test_methodtable():
    # ids = ["ff->f", "dd->d", "ii->i", "ll->l", "OO->O"]

    ids = make_ids()
    flags = range(1, len(ids) + 1)
    funcs = range(len(ids))

    step = 100

    i = len(ids)
    for i in range(1, len(ids), step):
        t = time.time()
        build_and_verify_methodtable(ids[:i], flags[:i], funcs[:i])
        t = time.time() - t
        print i, "table building took", t, "seconds."

if __name__ == '__main__':
    test_methodtable()