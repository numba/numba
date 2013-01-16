from . import utils
import collections
import llvm.core

class Variable(object):
    """
    Variables placed on the stack. They allow an indirection
    so, that when used in an operation, the correct LLVM type can be inserted.

    Attributes:
        type: the Numba type (see numba.typesystem and minivect/minitypes)
        is_local/is_global/is_constant
        name: name of local or global
        lvalue: LLVM Value
        state: state passed from one stage to the next
    """

    _type = None
    warn_unused = True

    def __init__(self, type, is_constant=False, is_local=False,
                 name=None, lvalue=None, constant_value=None,
                 promotable_type=True, is_arg=False):
        self.type = type
        self.name = name

        self.renameable = not is_constant
        self.renamed_name = None

        self.is_constant = is_constant
        self.constant_value = constant_value
        self.lvalue = lvalue
        self.promotable_type = promotable_type
        self.deleted = False

        self.set_uninitialized = False
        self.uninitialized_value = None

        self.killing_def = None # The definition that kills us, or None
        self.killed_def = None  # The definition that we killed

        self.parent_var = None
        self.block = None
        self.is_phi = False

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

        self._deferred_type = None

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

    @property
    def deferred_type(self):
        if self._deferred_type:
            return self._deferred_type

        from numba import typesystem
        self._deferred_type = typesystem.DeferredType(self)
        return self._deferred_type

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
        if not self.renamed_name:
            return self.name or "<unnamed>"
        name = self.renamed_name.lstrip("__numba_renamed_")
        counter, sep, var_name = name.partition('_')
        name = '%s_%s' % (var_name, counter)
        return name

    def __deepcopy__(self, memo):
        return self

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
        if self.is_phi:
            args.append("is_phi=True")
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
        self.local_counters = {}
        self.promotions = {}
        if parent:
            self.counters = parent.counters
            self.local_counters.update(parent.local_counters)
        else:
            self.counters = None

    def lookup(self, name):
        result = self.symtab.get(name, None)
        if result is None and self.parent is not None:
            result = self.parent.lookup(name)
        return result

    def lookup_most_recent(self, name):
        """
        Look up the most recent definition of a variable in this block.
        """
        if name in self.local_counters:
            last_count = self.local_counters[name]
        else:
            assert self.parent
            return self.parent.lookup_most_recent(name)

        return self.lookup_renamed(name, last_count)

    def lookup_promotion(self, var_name, dst_type):
        if (var_name, dst_type) in self.promotions:
            return self.promotions[var_name, dst_type]

        assert self.parent
        return self.parent.lookup_promotion(var_name, dst_type)

    def renamed_name(self, name, count):
        return '__numba_renamed_%d_%s' % (count, name)

    def lookup_renamed(self, name, version):
        renamed_name = self.renamed_name(name, version)
        return self[renamed_name]

    def rename(self, var, block, kills_previous_def=True):
        """
        Create a new renamed variable linked to the given variable, which
        becomes its parent.
        """
        new_var = Variable.from_variable(var)
        new_var.block = block
        new_var.cf_references = []
        self.counters[var.name] += 1
        if self.counters[var.name] and kills_previous_def:
            previous_var = self.lookup_most_recent(var.name)
            previous_var.killing_def = new_var
            new_var.killed_def = previous_var

        self.local_counters[var.name] = self.counters[var.name]
        new_var.renamed_name = self.renamed_name(var.name,
                                                 self.counters[var.name])

        new_var.parent_var = var
        self.symtab[new_var.renamed_name] = new_var

        # print "renaming %s to %s" % (var, new_var)
        return new_var

    def __repr__(self):
        return "symtab(%s)" % self.symtab

    def __getitem__(self, name):
        result = self.lookup(name)
        if result is None:
            raise KeyError(name)
        return result

    def __setitem__(self, name, variable):
        self.symtab[name] = variable

    def __iter__(self):
        return iter(self.symtab)

    def __getattr__(self, attr):
        return getattr(self.symtab, attr)