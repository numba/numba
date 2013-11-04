from .support import testcase, main, assertTrue
from numbapro.vectorizers._common import parse_signature

@testcase
def test_signature():
    assertTrue(parse_signature('(m, n) -> (m, n)') == \
                ([('m', 'n')], [('m', 'n')]))
    assertTrue(parse_signature('(m, n), (n, p) -> (m, p)') == \
                ([('m', 'n'), ('n', 'p')], [('m', 'p')]))
    assertTrue(parse_signature('(m,) -> ()') == \
                ([('m',)], [()]))
    assertTrue(parse_signature('(m) -> ()') == \
                ([('m',)], [()]))
    assertTrue(parse_signature('(m), -> ()') == \
                ([('m',)], [()]))
    assertTrue(parse_signature('(m, n), -> (m), (n,)') == \
                ([('m', 'n')], [('m',), ('n',)]))

    try:
        parse_signature('(m, n) -> (m, p)')
    except NameError, e:
        print 'expected exception:', e
    else:
        raise AssertionError('expected raise')

if __name__ == '__main__':
    main()
