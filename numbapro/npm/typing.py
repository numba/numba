import __builtin__
import itertools, inspect
from pprint import pprint
from collections import namedtuple, defaultdict, deque, Set, Mapping
from .symbolic import OP_MAP
from .utils import cache
from .errors import CompileError

class TypeInferError(CompileError):
    def __init__(self, value, msg):
        super(TypeInferError, self).__init__(value.lineno, msg)

class KeywordNotSupported(TypeInferError):
    def __init__(self, value, msg="kwargs is not supported"):
        super(KeywordNotSupported, self).__init__(value, msgs)

class ScalarType(namedtuple('ScalarType', ['code'])):
    is_array = False
    is_scalar = True
    is_tuple = False

    @property
    def is_int(self):
        return self.kind in 'iu'

    @property
    def is_unsigned(self):
        assert self.is_int
        return self.kind == 'u'

    @property
    def is_signed(self):
        assert self.is_int
        return self.kind == 'i'

    @property
    def is_float(self):
        return self.kind == 'f'

    @property
    def is_complex(self):
        return self.kind == 'c'

    @property
    @cache
    def bitwidth(self):
        return int(self.code[1:])

    @property
    def kind(self):
        return self.code[0]

    @property
    def complex_element(self):
        assert self.is_complex
        return ScalarType('f%d' % (self.bitwidth//2))

    def coerce(self, other):
        def bitwidth_of(x):
            return x.bitwidth
        if not isinstance(other, ScalarType):
            return

        mykind, itskind = self.kind, other.kind
        if mykind == itskind:
            return max(self, other, key=bitwidth_of)
        elif mykind in 'iu':
            if itskind in 'fc':
                return other
            elif itskind in 'iu':
                if self.bitwidth == other.bitwidth:
                    if other.is_signed: return other
                    else: return self
                else:
                    return max(self, other, key=bitwidth_of)
            else:
                return self
        elif mykind in 'f':
            if itskind == 'c': return other
            else: return self
        elif mykind in 'c':
            return self

        raise TypeError(self, other)

    def __repr__(self):
        return str(self.code)

class ArrayType(namedtuple('ArrayType', ['element', 'ndim', 'order'])):
    is_array = True
    is_scalar = False
    is_tuple = False

    def coerce(self, other):
        if isinstance(other, ArrayType) and self == other:
            return self
        else:
            return None

    def __repr__(self):
        return '[%s x %s %s]' % (self.element, self.ndim, self.order)

class TupleType(namedtuple('TupleType', ['element', 'count'])):
    is_tuple = True
    is_scalar = False
    is_array = False

def coerce(*args):
    if len(args) == 1:
        return args[0]
    else:
        base = args[0]
        for ty in args[1:]:
            base = base.coerce(ty)
            if base is None:
                return None
        return base

def can_coerce(*args):
    return coerce(*args) is not None

###############################################################################
# Types

boolean = ScalarType('i1')

int8 = ScalarType('i8')
int16 = ScalarType('i16')
int32 = ScalarType('i32')
int64 = ScalarType('i64')

uint8 = ScalarType('u8')
uint16 = ScalarType('u16')
uint32 = ScalarType('u32')
uint64 = ScalarType('u64')

float32 = ScalarType('f32')
float64 = ScalarType('f64')

complex64 = ScalarType('c64')
complex128 = ScalarType('c128')

signed_set   = frozenset([int8, int16, int32, int64])
unsigned_set = frozenset([uint8, uint16, uint32, uint64])
int_set      = signed_set | unsigned_set
float_set    = frozenset([float32, float64])
complex_set  = frozenset([complex64, complex128])
bool_set     = frozenset([boolean])
numeric_set  = int_set | float_set | complex_set
scalar_set   = numeric_set | bool_set

def arraytype(elemty, ndim, order):
    assert order in 'CFA'
    return ArrayType(elemty, ndim, order)


def tupletype(elemty, count):
    assert elemty.is_scalar
    return TupleType(elemty, count)

###############################################################################
# Infer

PENALTY_MAP = {
    'ii': 1,
    'uu': 1,
    'ff': 1,
    'cc': 1,
    'iu': 3,
    'ui': 2,
    'if': 4,
    'uf': 5,
    'fi': 10,
    'fu': 11,
}

def calc_cast_penalty(fromty, toty):

    if fromty == toty:
        return 1

    szdiff = 1
    if fromty.bitwidth > toty.bitwidth:
        # large penalty so that this is highly discouraged
        szdiff = (fromty.bitwidth - toty.bitwidth) * 100
    elif fromty.bitwidth < toty.bitwidth:
        szdiff = (toty.bitwidth - fromty.bitwidth)

    code = '%s%s' % (fromty.kind, toty.kind)
    factor = PENALTY_MAP.get(code)
    if factor:
        return factor * szdiff

CAST_PENALTY = {}

for pairs in itertools.product(scalar_set, scalar_set):
    CAST_PENALTY[pairs] = calc_cast_penalty(*pairs)

def cast_penalty(fromty, toty):
    return CAST_PENALTY.get((fromty, toty))

class Infer(object):
    '''Provide type inference and freeze the type of global values.
    '''
    def __init__(self, blocks, known, globals, intp, extended_globals={},
                 extended_calls={}):
        self.blocks = blocks
        self.argtys = known.copy()
        self.retty = self.argtys.pop('')
        self.globals = globals.copy()
        self.globals.update(vars(__builtin__))
        self.intp = ScalarType('i%d' % intp)
        self.possible_types = scalar_set | set(self.argtys.itervalues())
        self.extended_globals = extended_globals

        self.callrules = {}
        self.callrules.update(BUILTIN_RULES)
        self.callrules.update(extended_calls)

        self.rules = defaultdict(set)
        self.phis = defaultdict(set)

    def infer(self):
        for blk in self.blocks.itervalues():
            for v in blk.body:
                self.visit(v)
            if blk.terminator.kind == 'Ret':
                self.visit_Ret(blk.terminator)

        self.propagate_rules()
        possibles, conditions = self.deduce_possible_types()
        return self.find_solutions(possibles, conditions)

    def propagate_rules(self):
        # propagate rules on PHIs
        changed = True
        while changed:
            changed = False
            for phi, incomings in self.phis.iteritems():
                for i in incomings:
                    phiold = self.rules[phi]
                    iold = self.rules[i]
                    new = phiold | iold
                    if phiold != new or iold != new:
                        self.rules[phi] = self.rules[i] = new
                        changed = True

    def deduce_possible_types(self):
        # list possible types for each values
        possibles = {}
        conditions = defaultdict(list)
        for v, rs in self.rules.iteritems():
            if v not in self.phis:
                tset = self.possible_types
                for r in rs:
                    if r.require:
                        tset = set(filter(r, tset))
                    else:
                        conditions[v].append(r)
                possibles[v] = tset
        return possibles, conditions

    def find_solutions(self, possibles, conditions):
        # TODO: this function can use some refactoring
        # propagate phis
        def propagate_phis(possibles):
            changed = True
            while changed:
                changed = False
                for phi, incs in self.phis.iteritems():
                    if phi not in possibles:
                        # default to all possible types
                        possibles[phi] = self.possible_types
                    else:
                        old = possibles[phi]
                        tyset = possibles[phi]
                        for i in incs:
                            if i in possibles:
                                orig = tyset
                                tyset &= possibles[i]
                                assert tyset, (orig, possibles[i])
                        if tyset != old:
                            possibles[phi] = tyset
                            changed = True

        def iterate():
            for blk in self.blocks.itervalues():
                for v in blk.body:
                    yield v

        def find_good_solutions(clist, deps, deptysets):
            for tys in itertools.product(*deptysets):
                depmap = dict(zip(deps, tys))

                penalty = 0
                for cond in clist:
                    args = [depmap[val]]
                    for dep in cond.deps:
                        args.append(depmap[dep])
                    score = cond(*args)
                    if not score:
                        break
                    else:
                        penalty += score
                else:
                    yield (penalty, tys)

        def build_depsets(clist, val, conditions):
            depset = set([val])
            for cond in clist:
                for dep in cond.deps:
                    depset.add(dep)
            deps = list(depset)
            deptysets = [possibles[d] for d in deps]
            return deps, deptysets

        propagate_phis(possibles)
        changed = True
        while changed:
            changed = False

            for val in iterate():
                if val not in conditions: continue
                clist = conditions[val]
                deps, deptysets = build_depsets(clist, val, conditions)
                good = list(sorted(find_good_solutions(clist, deps, deptysets)))
                if not good:
                    raise TypeInferError(val, "cannot infer type")
                threshold = good[0][0]
                best = itertools.takewhile(lambda x: x[0] == threshold, good)

                resmap = defaultdict(set)
                for _score, tys in best:
                    for v, t in zip(deps, tys):
                        resmap[v].add(t)

                for v, ts in resmap.iteritems():
                    if not ts:
                        raise TypeInferError(v, "typeset is empty")
                    elif possibles[v] != ts:
                        changed = True
                        possibles[v] &= ts

                propagate_phis(possibles)

        # pick the most generic type if the final type set has multiple types
        soln = {}
        for v, ts in possibles.iteritems():
            r = coerce(*ts)
            if not r:
                msg = "cannot determine type: possible %s"
                raise TypeInferError(v, msg % list(ts))
            soln[v] = r
        return soln

    def visit(self, value):
        kind = value.kind
        fname = OP_MAP.get(value.kind, value.kind)

        fn = getattr(self, 'visit_' + fname, self.generic_visit)
        fn(value)

    def generic_visit(self, value):
        raise NotImplementedError(value)

    # ------ specialized visit ------- #

    def visit_Ret(self, term):
        val = term.args.value.value
        def rule(value):
            if not self.retty:
                raise TypeInferError(value, 'function has no return value')
            return cast_penalty(value, self.retty)
        self.rules[val].add(Conditional(rule))

    def visit_Arg(self, value):
        pos, name = value.args.num, value.args.name
        expect = self.argtys[name]
        self.rules[value].add(MustBe(expect))

    def visit_Undef(self, value):
        pass

    def visit_Global(self, value):
        gname = value.args.name
        parts = gname.split('.')
        obj = self.globals[parts[0]]
        for part in parts[1:]:
            obj = getattr(obj, part)
        assert not hasattr(obj, '_npm_func_')
        if isinstance(obj, (int, float, complex)):
            tymap = {int:       self.intp,
                     float:     float64,
                     complex:   complex128}
            self.rules[value].add(MustBe(tymap[type(obj)]))
        elif obj in self.extended_globals:
            moretypes = self.extended_globals[obj](self.rules, value)
            if moretypes:
                self.possible_types |= set(moretypes)
        else:
            msg = "only support global value of int, float or complex"
            raise TypeInferError(value, msg)


    def visit_Const(self, value):
        const = value.args.value
        if isinstance(const, (int, long)):
            expect = self.intp
        elif isinstance(const, float):
            expect = float64
        elif isinstance(const, complex):
            expect = complex128
        else:
            msg = 'invalid constant value of %s' % type(const)
            raise TypeInferError(value, msg)

        self.rules[value].add(MustBe(expect))

    def visit_BinOp(self, value):
        lhs, rhs = value.args.lhs.value, value.args.rhs.value

        if value.kind == '/':
            def rule(value, lhs, rhs):
                rty = coerce(lhs, rhs)
                if rty.is_float or rty.is_int:
                    target = ScalarType('f%d' % max(rty.bitwidth, 32))
                    return cast_penalty(value, target)

            self.rules[value].add(Conditional(rule, lhs, rhs))
            self.rules[value].add(Restrict(float_set))
            self.rules[lhs].add(Restrict(int_set|float_set))
            self.rules[rhs].add(Restrict(int_set|float_set))

        
        elif value.kind == '//':
        
            def rule(value, lhs, rhs):
                rty = coerce(lhs, rhs)
                if rty.is_int or rty.is_float:
                    target = ScalarType('i%d' % max(rty.bitwidth, 32))
                    return cast_penalty(value, target)

            self.rules[value].add(Conditional(rule, lhs, rhs))
            self.rules[value].add(Restrict(int_set))
            self.rules[lhs].add(Restrict(int_set|float_set))
            self.rules[rhs].add(Restrict(int_set|float_set))
            
        elif value.kind in ['&', '|', '^']:
            
            def rule(value, lhs, rhs):
                rty = coerce(lhs, rhs)
                if rty.is_int:
                    return cast_penalty(value, rty)

            self.rules[value].add(Conditional(rule, lhs, rhs))
            self.rules[value].add(Restrict(int_set))
            self.rules[lhs].add(Restrict(int_set))
            self.rules[rhs].add(Restrict(int_set))
        else:
            
            def rule(value, lhs, rhs):
                return cast_penalty(coerce(lhs, rhs), value)

            self.rules[value].add(Restrict(numeric_set))
            self.rules[value].add(Conditional(rule, lhs, rhs))
            self.rules[lhs].add(Restrict(numeric_set))
            self.rules[rhs].add(Restrict(numeric_set))

    def visit_BoolOp(self, value):
        lhs, rhs = value.args.lhs.value, value.args.rhs.value

        operand_set = int_set | float_set | bool_set
        self.rules[lhs].add(Restrict(operand_set))
        self.rules[rhs].add(Restrict(operand_set))
        
        self.rules[value].add(MustBe(boolean))


    def visit_UnaryOp(self, value):
        operand = value.args.value.value
        self.depmap[value] |= set([operand])
        if value.kind == '~':
            self.rules[lhs].add(Restrict(int_set))
            self.rules[rhs].add(Restrict(int_set))
            self.rules[value].add(Restrict(int_set))
        else:
            operand_set = int_set | float_set
            self.rules[lhs].add(Restrict(operand_set))
            self.rules[rhs].add(Restrict(operand_set))
            self.rules[value].add(Restrict(operand_set))


    def visit_Phi(self, value):
        incs = [inc.value for bb, inc in value.args.incomings]
        self.phis[value] |= set(incs)

    def visit_For(self, value):
        index, stop, step = (value.args.index.value,
                             value.args.stop.value,
                             value.args.step.value)

        self.rules[index].add(MustBe(self.intp))
        self.rules[stop].add(Restrict(int_set))
        self.rules[step].add(Restrict(int_set))
        self.rules[value].add(MustBe(boolean))

    def visit_GetItem(self, value):
        obj = value.args.obj.value
        self.rules[obj].add(Restrict(filter_array(self.possible_types)))
        for i in value.args.idx:
            iv = i.value
            self.rules[iv].add(Restrict(int_set))

        self.rules[value].add(Restrict(numeric_set))

        def rule(value, obj):
            if obj.is_array:
                return cast_penalty(obj.element, value)
        self.rules[value].add(Conditional(rule, obj))

    def visit_SetItem(self, value):
        obj = value.args.obj.value
        self.rules[obj].add(Restrict(filter_array(self.possible_types)))
        for i in value.args.idx:
            iv = i.value
            self.rules[iv].add(Restrict(int_set))

        val = value.args.value.value

        self.rules[val].add(Restrict(numeric_set))

        def rule(value, obj):
            if obj.is_array:
                return cast_penalty(value, obj.element)

        self.rules[val].add(Conditional(rule, obj))

    def visit_ArrayAttr(self, value):
        obj = value.args.obj.value
        self.rules[obj].add(Restrict(filter_array(self.possible_types)))
        idx = value.args.idx
        assert value.args.attr in ['size', 'ndim', 'shape', 'strides']
        if idx:
            assert value.args.attr in ['shape', 'strides']
            self.rules[idx.value].add(Restrict(int_set))

        self.rules[value].add(MustBe(self.intp))

    def visit_ComplexGetAttr(self, value):
        obj = value.args.obj.value
        self.rules[obj].add(Restrict(complex_set))
        self.rules[value].add(Restrict(float_set))
        def rule(value, obj):
            return cast_penalty(value, obj.complex_element)
        self.rules[value].add(Conditional(rule, obj))

    def visit_Call(self, value):
        funcname = value.args.func
        parts = funcname.split('.')
        obj = self.globals[parts[0]]
        for attr in parts[1:]:
            obj = getattr(obj, attr)
        if obj not in self.callrules:
            msg = "%s is not a regconized builtins"
            raise TypeInferError(value, msg % funcname)
        callrule = self.callrules[obj]
        moretypes = callrule(self.rules, value)
        if moretypes:
            self.possible_types |= set(moretypes)

    def visit_Unpack(self, value):
        obj = value.args.obj

        def must_be_tuple(value):
            return value.is_tuple
        self.rules[obj].add(Conditional(must_be_tuple))

        def match_element(value, obj):
            if obj.is_tuple:
                return value == obj.element
        self.rules[value].add(Conditional(match_element, obj))

###############################################################################
# Rules

class Rule(object):
    require = False
    conditional = False

    def __init__(self, checker):
        assert callable(callable)
        self.checker = checker

class Require(Rule):
    require = True
    def __call__(self, t):
        return self.checker(t)

class Conditional(Rule):
    conditional = True
    def __init__(self, checker, *deps):
        super(Conditional, self).__init__(checker)
        self.deps = deps

    def __call__(self, *args):
        return self.checker(*args)

def MustBe(expect):
    def _mustbe(t):
        return t == expect
    return Require(_mustbe)


def Restrict(tset):
    assert isinstance(tset, Set)
    def _restrict(t):
        return t in tset
    return Require(_restrict)

def filter_array(ts):
    return set(t for t in ts if t.is_array)


###############################################################################
# Rules for builtin functions

def call_complex_rule(rules, call):
    def cond_to_float(value, call):
        return cast_penalty(value, call.complex_element)

    args = call.args.args
    kws = call.args.kws
    if kws:
        raise KeywordNotSupported(call)
    nargs = len(args)
    if nargs == 1:
        (real,) = args
        rules[real.value].add(Restrict(int_set|float_set))
        rules[real.value].add(Conditional(cond_to_float, call))
    elif nargs == 2:
        (real, imag) = args
        rules[real.value].add(Restrict(int_set|float_set))
        rules[imag.value].add(Restrict(int_set|float_set))
        rules[real.value].add(Conditional(cond_to_float, call))
        rules[imag.value].add(Conditional(cond_to_float, call))
    else:
        raise TypeInferError(call, "invalid use of complex(real[, imag])")
    rules[call].add(Restrict(complex_set))
    call.replace(func=complex)

def call_int_rule(rules, call):
    args = call.args.args
    kws = call.args.kws
    if kws:
        raise KeywordNotSupported(call)
    nargs = len(args)
    if nargs == 1:
        (num,) = args
        rules[num.value].add(Restrict(int_set|float_set))
    else:
        raise TypeInferError(call, "invalid use of int(x)")
    rules[call].add(Restrict(int_set))
    call.replace(func=int)

def call_float_rule(rules, call):
    args = call.args.args
    kws = call.args.kws
    if kws:
        raise KeywordNotSupported(call)
    nargs = len(args)
    if nargs == 1:
        (num,) = args
        rules[num.value].add(Restrict(int_set|float_set))
    else:
        raise TypeInferError(call, "invalid use of float(x)")
    rules[call].add(Restrict(float_set))
    call.replace(func=float)

def call_maxmin_rule(fname):
    def inner(rules, call):
        args = call.args.args
        kws = call.args.kws
        if kws:
            raise KeywordNotSupported(call)
        nargs = len(args)
        if nargs < 2:
            msg = "%s() must have at least two arguments"
            raise TypeInferError(call, msg % fname)

        argvals =[a.value for a in args]

        for a in argvals:
            rules[a].add(Restrict(int_set|bool_set|float_set))

        def rule(value, *tys):
            rty = coerce(*tys)
            return cast_penalty(rty, value)
        rules[call].add(Conditional(rule, *argvals))

        call.replace(func={'min': min, 'max': max}[fname])
    return inner

def call_abs_rule(rules, call):
    args = call.args.args
    kws = call.args.kws
    if kws:
        raise KeywordNotSupported(call)
    nargs = len(args)
    if nargs != 1:
        raise TypeInferError(call, "abs(number) takes one argument only")

    arg = args[0].value
    rules[arg].add(Restrict(signed_set|float_set))

    def prefer_same_as_arg(value, arg):
        return cast_penalty(value, arg)

    rules[call].add(Conditional(prefer_same_as_arg, arg))
    call.replace(func=abs)

BUILTIN_RULES = {
    complex:    call_complex_rule,
    int:        call_int_rule,
    float:      call_float_rule,
    min:        call_maxmin_rule('min'),
    max:        call_maxmin_rule('max'),
    abs:        call_abs_rule,
}

