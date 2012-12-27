"""
Extension types
"""

from numba.typesystem import *

class ExtensionType(NumbaType, minitypes.ObjectType):

    is_extension = True
    is_final = False

    _need_tp_dealloc = None

    def __init__(self, py_class, **kwds):
        super(ExtensionType, self).__init__(**kwds)
        assert isinstance(py_class, type), "Must be a new-style class"
        self.name = py_class.__name__
        self.py_class = py_class
        self.symtab = {}  # attr_name -> attr_type
        self.methods = [] # (method_name, func_signature)
        self.methoddict = {} # method_name -> (func_signature, vtab_index)

        self.vtab_offset = extension_types.compute_vtab_offset(py_class)
        self.attr_offset = extension_types.compute_attrs_offset(py_class)
        self.attribute_struct = None
        self.vtab_type = None

        self.parent_attr_struct = None
        self.parent_vtab_type = None
        self.parent_type = getattr(py_class, "__numba_ext_type", None)

    @property
    def need_tp_dealloc(self):
        """
        Returns whether this extension type needs a tp_dealloc, tp_traverse
        and tp_clear filled out.

        This needs to be computed on demand since the attributes are mutated
        after creation.
        """
        if self._need_tp_dealloc is not None:
            result = self._need_tp_dealloc
        if self.parent_type is not None and self.parent_type.need_tp_dealloc:
            result = False
        else:
            field_types = self.attribute_struct.fielddict.itervalues()
            result = any(map(is_obj, field_types))

        self._need_tp_dealloc = result
        return result

    def add_method(self, method_name, method_signature):
        if method_name in self.methoddict:
            # Patch current signature after type inference
            signature = self.get_signature(method_name)
            assert method_signature.args == signature.args
            if signature.return_type is None:
                signature.return_type = method_signature.return_type
            else:
                assert signature.return_type == method_signature.return_type, \
                                                            method_signature
        else:
            self.methoddict[method_name] = (method_signature, len(self.methods))
            self.methods.append((method_name, method_signature))

    def get_signature(self, method_name):
        signature, vtab_offset = self.methoddict[method_name]
        return signature

    def set_attributes(self, attribute_list):
        """
        Create the symbol table and attribute struct from a list of
        (varname, attribute_type)
        """
        import numba.symtab

        self.attribute_struct = numba.struct(attribute_list)
        self.symtab.update([(name, numba.symtab.Variable(type))
                               for name, type in attribute_list])

    def __repr__(self):
        return "<Extension %s>" % self.name

    def __str__(self):
        if self.attribute_struct:
            return "<Extension %s(%s)>" % (self.name,
                                           self.attribute_struct.fielddict)
        return repr(self)

class ExtMethodType(NumbaType, minitypes.FunctionType):
    """
    Extension method type used for vtab purposes.

    is_class: is classmethod?
    is_static: is staticmethod?
    """

    def __init__(self, return_type, args, name=None,
                 is_class=False, is_static=False, **kwds):
        super(ExtMethodType, self).__init__(return_type, args, name, **kwds)
        self.is_class = is_class
        self.is_static = is_static