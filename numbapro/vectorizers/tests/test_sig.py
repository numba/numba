from .support import testcase, main
from numbapro.vectorizers._common import parse_signature

@testcase
def test_signature():
    assert parse_signature('(m, n) -> (m, n)') == \
                ([('m', 'n')], [('m', 'n')])
    assert parse_signature('(m, n), (n, p) -> (m, p)') == \
                ([('m', 'n'), ('n', 'p')], [('m', 'p')])
    assert parse_signature('(m,) -> ()') == \
                ([('m',)], [()])
    assert parse_signature('(m) -> ()') == \
                ([('m',)], [()])
    assert parse_signature('(m), -> ()') == \
                ([('m',)], [()])
    assert parse_signature('(m, n), -> (m), (n,)') == \
                ([('m', 'n')], [('m',), ('n',)])

    try:
        parse_signature('(m, n) -> (m, p)')
    except NameError, e:
        print 'expected exception:', e
    else:
        raise AssertionError('expected raise')

if __name__ == '__main__':
    main()
