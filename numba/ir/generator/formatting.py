
def format_stats(pattern, indent, stats):
    pattern = pattern + " " * indent
    return pattern.join(stats)

def get_fields(fields, obj="self"):
    fieldnames = (str(field.name) for field in fields)
    return ["%s.%s" % (obj, name) for name in fieldnames]