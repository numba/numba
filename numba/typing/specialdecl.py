from numba import stack_array, types
from numba.typing.templates import (AttributeTemplate, MacroTemplate,
                                    Registry)

registry = Registry()
register = registry.register
register_attr = registry.register_attr

class StackArray_new(MacroTemplate):
    key = stack_array.new

@register_attr
class StackArrayModuleTemplate(AttributeTemplate):
    key = types.Module(stack_array)

    def resolve_new(self, mod):
        return types.Macro(StackArray_new)
