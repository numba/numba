
class Formatter(object):

    def format_stats(self, pattern, indent, stats):
        pattern = pattern + " " * indent
        return pattern.join(stats)

    def get_fields(self, fields, obj="self"):
        fieldnames = (str(field.name) for field in fields)
        return ["%s.%s" % (obj, name) for name in fieldnames]

class PythonFormatter(Formatter):

    def format_type(self, asdl_type):
        """
        ASDL's five builtin types are identifier, int, string, object, bool
        """
        type = str(asdl_type)

        defaults = {
            'identifier': 'str',
            'string': 'str',
        }

        return defaults.get(type, type)

class CythonFormatter(Formatter):

    def format_type(self, asdl_type):
        """
        ASDL's five builtin types are identifier, int, string, object, bool
        """
        type = str(asdl_type)

        defaults = {
            'identifier': 'str',
            'string': 'str',
            'bool': 'bint',
        }

        return defaults.get(type, type)

def format_fields(fields):
    return ", ".join("%s %s" % (f.type, f.name) for f in fields)

py_formatter = PythonFormatter()
cy_formatter = CythonFormatter()