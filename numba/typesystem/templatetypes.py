"""
Autojit template types.
"""

from numba.typesystem import *

# type_attribute => [type_assertions]
VALID_TYPE_ATTRIBUTES = {
    "dtype": ["is_array"],
    "base_type": ["is_pointer", "is_carray", "is_complex"],
    "args": ["is_function"],
    "return_type": ["is_function"],
    # "fields": ["is_struct"],
    "fielddict": ["is_struct"],
}

class TemplateType(NumbaType):

    is_template = True
    template_count = 0

    def __init__(self, name=None, **kwds):
        if name is None:
            name = "T%d" % self.template_count
            TemplateType.template_count += 1

        self.name = name

    def resolve_template(self, template_context):
        if self not in template_context:
            raise error.InvalidTemplateError("Unknown template type: %s" % self)
        return template_context[self]

    def __getitem__(self, index):
        if isinstance(index, (tuple, slice)):
            return super(TemplateType, self).__getitem__(index)

        return TemplateIndexType(self, index)

    def __getattr__(self, attr):
        if attr in VALID_TYPE_ATTRIBUTES:
            return TemplateAttributeType(self, attr)

        return super(TemplateType, self).__getattr__(attr)

    def __repr__(self):
        return "template(%s)" % self.name

    def __str__(self):
        return self.name


class TemplateAttributeType(TemplateType):

    is_template_attribute = True
    subtypes = ['template_type']

    def __init__(self, template_type, attribute_name, **kwds):
        self.template_type = template_type
        self.attribute_name = attribute_name

        assert attribute_name in VALID_TYPE_ATTRIBUTES

    def resolve_template(self, template_context):
        resolved_type = self.template_type.resolve_template(template_context)

        assertions = VALID_TYPE_ATTRIBUTES[self.attribute_name]
        valid_attribute = any(getattr(resolved_type, a) for a in assertions)
        if not valid_attribute:
            raise error.InvalidTemplateError(
                    "%s has no attribute %s" % (self.template_type,
                                                self.attribute_name))

        return getattr(resolved_type, self.attribute_name)

    def __repr__(self):
        return "%r.%s" % (self.template_type, self.attribute_name)

    def __str__(self):
        return "%s.%s" % (self.template_type, self.attribute_name)

class TemplateIndexType(TemplateType):

    is_template_index = True
    subtypes = ['template_type']

    def __init__(self, template_type, index, **kwds):
        self.template_type = template_type
        self.index = index

    def resolve_template(self, template_context):
        attrib = self.template_type.resolve_template(template_context)
        assert isinstance(attrib, (list, tuple, dict))
        return attrib[self.index]

    def __repr__(self):
        return "%r[%d]" % (self.template_type, self.index)

    def __str__(self):
        return "%s[%d]" % (self.template_type, self.index)



template = TemplateType

def validate_template(concrete_type, template_type):
    if type(template_type) != type(concrete_type):
        raise error.InvalidTemplateError(
                "Type argument does not match template type: %s and %s" % (
                concrete_type, template_type))

    if concrete_type.is_array:
        if template_type.ndim != concrete_type.ndim:
            raise error.InvalidTemplateError(
                "Template expects %d dimensions, got %d" % (template_type.ndim,
                                                            concrete_type.ndim))

def match_template(template_type, concrete_type, template_context):
    """
    This function matches up T in the example below with a concrete type
    like double when a double pointer is passed in as argument:

        def f(T.pointer() pointer):
            scalar = T(...)

    We can go two ways with this, e.g.

        def f(T.base_type scalar):
            pointer = T(...)

    Which could work for things like pointers, though not for things like
    arrays, since we can't infer the dimensionality.

    We mandate that each Template type be resolved through a concrete type,
    i.e.:

        def f(T scalar):
            pointer = T.pointer(...)


    template_context:

        Dict mapping template types to concrete types:

            T1 -> double *
            T2 -> float[:]
    """
    if template_type.is_template_attribute:
        # As noted in the description, we don't handle this
        pass
    elif template_type.is_template:
        if template_type in template_context:
            prev_type = template_context[template_type]
            if prev_type != concrete_type:
                raise error.InvalidTemplateError(
                        "Inconsistent types found for template: %s and %s" % (
                                                    prev_type, concrete_type))
        else:
            template_context[template_type] = concrete_type
    else:
        validate_template(concrete_type, template_type)

        for t1, t2 in zip(template_type.subtype_list,
                          concrete_type.subtype_list):
            if not isinstance(t1, (list, tuple)):
                t1, t2 = [t1], [t2]

            for t1, t2 in zip(t1, t2):
                match_template(t1, t2, template_context)

def resolve_template_type(template_type, template_context):
    """
    After the template context is known, resolve functions on template types
    E.g.

        T[:]                -> ArrayType(dtype=T)
        void(T)             -> FunctionType(args=[T])
        Struct { T arg }    -> struct(fields={'arg': T})
        T *                 -> PointerType(base_type=T)

    Any other compound types?
    """
    r = lambda t: resolve_template_type(t, template_context)

    if template_type.is_template:
        template_type = template_type.resolve_template(template_context)
    elif template_type.is_array:
        template_type = template_type.copy()
        template_type.dtype = r(template_type.dtype)
    elif template_type.is_function:
        template_type = r(template_type.return_type)(*map(r, template_type.args))
    elif template_type.is_struct:
        S = template_type
        fields = []
        for field_name, field_type in S.fields:
            fields.append((field_name, r(field_type)))
        template_type = numba.struct(fields, name=S.name, readonly=S.readonly,
                                     packed=S.packed)
    elif template_type.is_pointer:
        template_type = r(template_type.base_type).pointer()

    return template_type

def is_template_list(types):
    return any(is_template(type) for type in types)

def is_template(T):
    if isinstance(T, (list, tuple)):
        return is_template_list(T)

    return T.is_template or is_template_list(T.subtype_list)

def resolve_templates(locals, template_signature, arg_names, arg_types):
    """
    Resolve template types given a signature with concrete types.
    """
    template_context = {}
    locals = locals or {}

    # Resolve the template context with the types we have
    for i, (arg_name, arg_type) in enumerate(zip(arg_names, arg_types)):
        T = template_signature.args[i]
        if is_template(T):
            # Resolve template type
            if arg_name in locals:
                # Locals trump inferred argument types
                arg_type = locals[arg_name]
            match_template(T, arg_type, template_context)
        else:
            # Concrete type, patch argtypes. This is valid since templates
            # are only supported for autojit functions
            arg_types[i] = T

    # Resolve types of local variables and functions on templates
    # (T.dtype, T.pointer(), etc)
    for local_name, local_type in locals.iteritems():
        locals[local_name] = resolve_template_type(local_type,
                                                   template_context)

    return_type = resolve_template_type(template_signature.return_type,
                                        template_context)
    signature = return_type(*arg_types)
    return template_context, signature

