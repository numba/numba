from .. import intern

def test_global_interning():
    # Can't really test for this with nose...
    # try:
    #     intern.global_intern("hello")
    # except AssertionError as e:
    #     pass
    # else:
    #     raise Exception("Expects complaint about uninitialized table")

    intern.global_intern_initialize()
    id1 = intern.global_intern("hello")
    id2 = intern.global_intern("hello")
    id3 = intern.global_intern("hallo")
    assert id1 == id2
    assert id1 != id3

def test_interning():
    table = intern.InternTable()

    id1 = intern.global_intern("hello")
    id2 = intern.global_intern("hello")
    id3 = intern.global_intern("hallo")
    assert id1 == id2
    assert id1 != id3

def test_intern_many():
    table = intern.InternTable()

    itoid = {}
    for i in range(1000000):
        id = table.intern("my randrom string %d" % i)
        itoid[i] = id

        id1 = table.intern("my randrom string %d" % (i // 2))
        id2 = table.intern("my randrom string %d" % (i // 4))

        assert id1 == itoid[i//2]
        assert id2 == itoid[i//4]

if __name__ == '__main__':
    test_intern_many()