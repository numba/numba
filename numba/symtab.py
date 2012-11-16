from . import utils

class Variable(object):
    """
    Variables placed on the stack. They allow an indirection
    so, that when used in an operation, the correct LLVM type can be inserted.

    Attributes:
        type: the Numba type (see _numba_types and minivect/minitypes)
        is_local/is_global/is_constant
        name: name of local or global
        lvalue: LLVM Value
        state: state passed from one stage to the next
    """

    _type = None

    def __init__(self, type, is_constant=False, is_local=False,
                 name=None, lvalue=None, constant_value=None,
                 promotable_type=True, is_arg=False):
        self.type = type
        self.name = name
        self.renamed_name = None
        self.is_constant = is_constant
        self.constant_value = constant_value
        self.lvalue = lvalue
        self.promotable_type = promotable_type
        self.deleted = False

        self.parent_var = None
        self.block = None

        self.is_local = is_local
        self.is_arg = is_arg
        self.is_cellvar = False
        self.is_freevar = False
        self.need_arg_copy = True

        # The control_flow.NameAssignment that defines this
        # variable (or PhiNode if a phi)
        self.name_assignment = None

        self.cf_assignments = []
        self.cf_references = [] # def-use chain

        # position of first definition
        self.lineno = -1
        self.col_offset = -1

    @classmethod
    def make_shared_property(cls, name):
        def _get(self):
            if self.parent_var:
                return getattr(self.parent_var, name)
            return getattr(self, '_' + name)

        def _set(self, value):
            if self.parent_var:
                setattr(self.parent_var, name, value)
            else:
                setattr(self, '_' + name, value)

        setattr(cls, '_' + name, None)
        setattr(cls, name, property(_get, _set))

    def _type_get(self):
        return self._type

    def _type_set(self, type):
        assert not (self.type and type is None)
        if type is None:
            print 'Setting None type!', self.name
        self._type = type

    #type = property(_type_get, _type_set)

    @classmethod
    def from_variable(cls, variable, **kwds):
        result = cls(variable.type)
        vars(result).update(dict(kwds, **vars(variable)))
        return result

    @property
    def is_global(self):
        return self.type and self.type.is_global

    @property
    def ltype(self):
        """
        The LLVM type for the type of this variable or LLVM Value.
        """
        if self.lvalue is not None:
            return self.lvalue.type
        return self.type.to_llvm(utils.context)

    @property
    def ctypes_type(self):
        """
        The ctypes type for the type of this variable.
        """

    @property
    def unmangled_name(self):
        name = self.renamed_name.lstrip("__numba_renamed_")
        counter, sep, var_name = name.partition('_')
        name = '%s_%s' % (var_name, counter)
        return name

    def __repr__(self):
        args = []
        if self.is_local:
            args.append("is_local=True")
        if self.is_global:
            args.append("is_global=True")
        if self.is_constant:
            args.append("is_constant=True")
        if self.is_freevar:
            args.append("is_freevar=True")
        if self.is_cellvar:
            args.append("is_cellvar=True")
        if self.block:
            args.append("block=%d" % self.block.id)
        if self.lvalue:
            args.append("llvm=%s" % (self.lvalue,))

        if args:
            extra_info = " " + ", ".join(args)
        else:
            extra_info = ""

        if self.name:
            if self.renamed_name:
                name = self.unmangled_name
            else:
                name = self.name
            return "<Variable(name=%r, type=%s%s)>" % (name, self.type,
                                                       extra_info)
        else:
            return "<Variable(type=%s%s)>" % (self.type, extra_info)


Variable.make_shared_property('is_cellvar')
Variable.make_shared_property('is_freevar')
Variable.make_shared_property('need_arg_copy')


class Symtab(object):
    def __init__(self, symtab_dict=None, parent=None):
        self.symtab = symtab_dict or {}
        self.parent = parent
        if parent:
            self.counters = parent.counters
        else:
            self.counters = None

        # Last counter values local to this block after renaming finishes
        self._counters = None

    def lookup(self, name):
        result = self.symtab.get(name, None)
        if result is None and self.parent is not None:
            result = self.parent.lookup(name)
        return result

    def lookup_most_recent(self, name):
        """
        Look up the most recent definition of a variable.
        This is used during the renaming process.
        """
        last_count = self.counters[name]
        renamed_name = self.renamed_name(name, last_count)
        return self[renamed_name]

    def lookup_last(self, name):
        """
        Look up the last definition in the block for a variable.
        This is used after the renaming process.
        """
        last_count = self._counters[name]
        renamed_name = self.renamed_name(name, last_count)
        return self[renamed_name]

    def renamed_name(self, name, count):
        return '__numba_renamed_%d_%s' % (count, name)

    def rename(self, var, block):
        """
        Create a new renamed variable linked to the given variable, which
        becomes its parent.
        """
        new_var = Variable.from_variable(var)
        new_var.block = block
        self.counters[var.name] += 1
        new_var.renamed_name = self.renamed_name(var.name,
                                                 self.counters[var.name])

        new_var.parent_var = var
        self.symtab[new_var.renamed_name] = new_var

        return new_var

    def __repr__(self):
        return "symtab(%s)" % self.symtab

    def __getitem__(self, name):
        result = self.lookup(name)
        assert result is not None
        return result

    def __setitem__(self, name, variable):
        self.symtab[name] = variable

    def __iter__(self):
        return iter(self.symtab)

    def __getattr__(self, attr):
        return getattr(self.symtab, attr)