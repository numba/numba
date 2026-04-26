import rumba


def test_second_call_reuses_compiled_artifact():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    first_key = next(iter(add._compiled.values())).key
    assert add(3, 4) == 7
    assert len(add._compiled) == 1
    assert next(iter(add._compiled.values())).key == first_key


def test_different_signatures_create_distinct_artifacts():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add(1.5, 2.5) == 4.0
    assert len(add._compiled) == 2
