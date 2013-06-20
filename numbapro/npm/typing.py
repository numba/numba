import __builtin__
import itertools, inspect
from pprint import pprint
from collections import namedtuple, defaultdict, deque, Set, Mapping
from .symbolic import OP_MAP, find_dominators
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
    is_user = False
    
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
    is_user = False

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
    is_user = False


    def coerce(self, other):
        if self == other: return self

    def __repr__(self):
        return '(%s x %d)' % (self.element, self.count)

class UserType(namedtuple('UserType', ['name', 'object'])):
    is_tuple = False
    is_scalar = False
    is_array = False
    is_user = True

    def coerce(self, other):
        if self == other: return self

    def __repr__(self):
        return 'type(%s, %s)' % (self.name, self.object)

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
        self.possible_types = set(scalar_set | set(self.argtys.itervalues()))
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

#        self.propagate_rules()
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
                tset = frozenset(self.possible_types)
                for r in rs:
                    if r.require:
                        tset = set(filter(r, tset))
                    else:
                        conditions[v].append(r)
                possibles[v] = tset
        return possibles, conditions

    def find_solutions(self, possibles, conditions):
        '''
        IDom dictates the type of values when they participate in a PHI.
        '''
        # find dominators
        doms = find_dominators(self.blocks)
        # topsort
        def topsort_blocks(doms):
            # make copy of doms
            doms = dict((k, set(v)) for k, v in doms.iteritems())
            topsorted = []
            pending = deque(self.blocks.keys())
            while pending:
                tos = pending.popleft()
                domset = doms[tos]
                domset.discard(tos)
                for x in list(domset):
                    if x in topsorted:
                        domset.remove(x)
                if not domset:
                    topsorted.append(tos)
                else:
                    pending.append(tos)
            return topsorted

        topsorted = topsort_blocks(doms)

        # infer block by block
        processed_blocks = set()
        for blknum in topsorted:
            blk = self.blocks[blknum]
            values = []
            for value in blk.body:
                if value.kind == 'Phi':
                    # pickup the type from the previously inferred set
                    for ib, iv in value.args.incomings:
                        if ib in processed_blocks:
                            possibles[value] = possibles[iv.value]
                            break
                elif value.ref.uses:
                    # only process the values that are used.
                    depset = set([value])
                    clist = conditions[value]
                    for cond in clist:
                        depset |= set(cond.deps)

                    variables = list(depset)
                    # earlier instruction has more heavier weight
                    pool = []
                    for v in variables:
                        pool.append(possibles.get(v, self.possible_types))

                    chosen = []
                    for selected in itertools.product(*pool):
                        localmap = dict(zip(variables, selected))
                        penalty = 1
                        for cond in clist:
                            deps = [value] + list(cond.deps)
                            args = [localmap[x] for x in deps]
                            res = cond(*args)
                            if res and isinstance(res, int):
                                penalty += res
                            else:
                                break
                        else:
                            if penalty:
                                chosen.append((penalty, localmap))

                    chosen_sorted = sorted(chosen)
                    best_score = chosen_sorted[0][0]
                    take_filter = lambda x: x[0] == best_score
                    filterout = itertools.takewhile(take_filter, chosen_sorted)
                    bests = [x for _, x in filterout]


                    if len(bests) > 1:
                        # find the most generic solution
                        temp = defaultdict(set)
                        for b in bests:
                            for v, t in b.iteritems():
                                temp[v].add(t)
                        soln = {}
                        for v, ts in temp.iteritems():
                            soln[v] = coerce(*ts)

                    elif len(bests) == 1:
                        soln = bests[0]
                    else:
                        raise TypeInferError(value, "cannot infer value/operation not supported on input types")
                    for k, v in soln.iteritems():
                        possibles[k] = frozenset([v])
            processed_blocks.add(blknum)

        soln = dict((k, iter(vs).next()) for k, vs in possibles.iteritems())
        pprint(soln)
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
        key = parts[0]
        try:
            obj = self.globals[key]
        except KeyError:
            raise TypeInferError(value, "global %s is not defined" % key)
        for part in parts[1:]:
            obj = getattr(obj, part)

        if isinstance(obj, (int, float, complex)):
            tymap = {int:       self.intp,
                     float:     float64,
                     complex:   complex128}
            self.rules[value].add(MustBe(tymap[type(obj)]))
        elif obj in self.extended_globals:
            self.extended_globals[obj](self, value, obj)
        else:
            msg = "only support global value of int, float or complex"
            raise TypeInferError(value, msg)


    def visit_Const(self, value):
        const = value.args.value
        if isinstance(const, (int, long)):
            self.rules[value].add(MustBe(self.intp))
        elif isinstance(const, float):
            self.rules[value].add(MustBe(float64))
        elif isinstance(const, complex):
            self.rules[value].add(MustBe(complex128))
        elif isinstance(const, tuple) and all(isinstance(x, (int, long))
                                              for x in const):
            tuty = tupletype(self.intp, len(const))
            self.possible_types.add(tuty)
            self.rules[value].add(MustBe(tuty))
        else:
            msg = 'invalid constant value of %s' % type(const)
            raise TypeInferError(value, msg)

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

        elif value.kind == 'ForInit':
            value.kind = '-'

            self.rules[value].add(MustBe(self.intp))

            def rule(value):
                return cast_penalty(value, self.intp)

            self.rules[lhs].add(Conditional(rule))
            self.rules[rhs].add(Conditional(rule))
            self.rules[lhs].add(Restrict(int_set))
            self.rules[rhs].add(Restrict(int_set))

        else:
            assert value.kind in '+-*%'
            
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
                return obj.element == value
                #return cast_penalty(obj.element, value)
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
            if not hasattr(obj, '_npm_context_'):
                msg = "%s is not a regconized builtins"
                raise TypeInferError(value, msg % funcname)
            else:
                _lmod, _lfunc, retty, argtys = obj._npm_context_
                args = map(lambda x: x.value, value.args.args)
                def can_cast_to(t):
                    def wrapped(v):
                        return cast_penalty(v, t)
                    return wrapped

                if len(args) != len(argtys):
                    msg = "call to %s takes %d args but %d given"
                    msgargs = obj, len(argtys), len(args)
                    raise TypeInferError(value, msg % msgargs)

                for aval, aty in zip(args, argtys):
                    self.rules[aval].add(Conditional(can_cast_to(aty)))

                self.rules[value].add(MustBe(retty))
            value.replace(func=obj)
        else:
            callrule = self.callrules[obj]
            callrule(self, value)

    def visit_Unpack(self, value):
        obj = value.args.obj.value

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

def call_complex_rule(infer, call):
    def cond_to_float(value, call):
        return cast_penalty(value, call.complex_element)

    args = call.args.args
    kws = call.args.kws
    if kws:
        raise KeywordNotSupported(call)
    nargs = len(args)
    if nargs == 1:
        (real,) = args
        infer.rules[real.value].add(Restrict(int_set|float_set))
        infer.rules[real.value].add(Conditional(cond_to_float, call))
    elif nargs == 2:
        (real, imag) = args
        infer.rules[real.value].add(Restrict(int_set|float_set))
        infer.rules[imag.value].add(Restrict(int_set|float_set))
        infer.rules[real.value].add(Conditional(cond_to_float, call))
        infer.rules[imag.value].add(Conditional(cond_to_float, call))
    else:
        raise TypeInferError(call, "invalid use of complex(real[, imag])")
    infer.rules[call].add(Restrict(complex_set))
    call.replace(func=complex)

def call_int_rule(infer, call):
    args = call.args.args
    kws = call.args.kws
    if kws:
        raise KeywordNotSupported(call)
    nargs = len(args)
    if nargs == 1:
        (num,) = args
        infer.rules[num.value].add(Restrict(int_set|float_set))
    else:
        raise TypeInferError(call, "invalid use of int(x)")
    infer.rules[call].add(Restrict(int_set))
    call.replace(func=int)

def call_float_rule(infer, call):
    args = call.args.args
    kws = call.args.kws
    if kws:
        raise KeywordNotSupported(call)
    nargs = len(args)
    if nargs == 1:
        (num,) = args
        infer.rules[num.value].add(Restrict(int_set|float_set))
    else:
        raise TypeInferError(call, "invalid use of float(x)")
    infer.rules[call].add(Restrict(float_set))
    call.replace(func=float)

def call_maxmin_rule(fname):
    def inner(infer, call):
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
            infer.rules[a].add(Restrict(int_set|bool_set|float_set))

        def rule(value, *tys):
            rty = coerce(*tys)
            return cast_penalty(rty, value)
        infer.rules[call].add(Conditional(rule, *argvals))

        call.replace(func={'min': min, 'max': max}[fname])
    return inner

def call_abs_rule(infer, call):
    args = call.args.args
    kws = call.args.kws
    if kws:
        raise KeywordNotSupported(call)
    nargs = len(args)
    if nargs != 1:
        raise TypeInferError(call, "abs(number) takes one argument only")

    arg = args[0].value
    infer.rules[arg].add(Restrict(signed_set|float_set))

    def prefer_same_as_arg(value, arg):
        return cast_penalty(value, arg)

    infer.rules[call].add(Conditional(prefer_same_as_arg, arg))
    call.replace(func=abs)

BUILTIN_RULES = {
    complex:    call_complex_rule,
    int:        call_int_rule,
    float:      call_float_rule,
    min:        call_maxmin_rule('min'),
    max:        call_maxmin_rule('max'),
    abs:        call_abs_rule,
}

