from .. import intern

def test_global_interning():
    try:
        intern.global_intern("hello")
    except AssertionError as e:
        pass
    else:
        raise Exception("Expects complaint about uninitialized table")

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
