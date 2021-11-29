"""gdb printing extension for Numba types.
"""
import re
import sys
_PYVERSION = sys.version_info[:2]

try:
    import gdb.printing
    import gdb
except ImportError:
    raise ImportError("GDB python support is not available.")


class NumbaArrayPrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            import numpy as np
            HAVE_NUMPY = True
        except ImportError:
            HAVE_NUMPY = False

        try:
            NULL = 0x0

            # raw data refs
            nitems = int(self.val["nitems"])
            data = self.val['data']
            rshp = self.val["shape"]
            itemsize = self.val["itemsize"]

            # type information decode, simple type:
            ty_str = str(self.val.type)
            if HAVE_NUMPY and ('unaligned' in ty_str or 'Record' in ty_str):
                ty_str = ty_str.lstrip('unaligned').strip()
                matcher = re.compile(r"array\((Record.*), (.*), (.*)\)\ \(.*")
                arr_info = [x.strip() for x in matcher.match(ty_str).groups()]
                dtype_str, ndim_str, order_str = arr_info
                field_dts = re.match(
                    'Record\\((.*\\[.*\\])', dtype_str).groups()[0].split(',')
                struct_entries = []
                for f in field_dts:
                    name, stuff = f.split('[')
                    dt_as_str = stuff.split(';')[0].split('=')[1]
                    if "unichr" in dt_as_str:
                        raise ValueError
                    else:
                        dtype = np.dtype(dt_as_str)
                    struct_entries.append((name, dtype))
                    # The dtype is actually a record of some sort
                dtype_str = struct_entries

            else:  # simple type
                matcher = re.compile(r"array\((.*),(.*),(.*)\)\ \(.*")
                arr_info = [x.strip() for x in matcher.match(ty_str).groups()]
                dtype_str, ndim_str, order_str = arr_info

            # shape extraction
            fields = rshp.type.fields()
            lo, hi = fields[0].type.range()
            shape = tuple([int(rshp[x]) for x in range(lo, hi + 1)])

            # if data is not NULL
            if data != NULL:
                if HAVE_NUMPY:
                    # TODO: Deal with order and non-contiguous data
                    dtype_clazz = np.dtype(dtype_str)
                    dtype = dtype_clazz  # .type
                    this_proc = gdb.selected_inferior()
                    mem = this_proc.read_memory(int(data), nitems * itemsize)
                    new_arr = np.frombuffer(mem, dtype=dtype).reshape(shape)
                    return str(new_arr)
                # Catch all for no NumPy
                return "array([...], dtype=%s, shape=%s)" % (dtype_str, shape)
            else:
                # Not yet initialized or NULLed out data
                buf = list(["NULL/Uninitialized"])
                return "array([" + ', '.join(buf) + "]" + ")"
        except Exception as e:
            return 'Failed to parse. %s' % e


class NumbaComplexPrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        return "%s+%sj" % (self.val['real'], self.val['imag'])


class NumbaTuplePrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        buf = []
        fields = self.val.type.fields()
        for f in fields:
            buf.append(str(self.val[f.name]))
        return "(%s)" % ', '.join(buf)


class NumbaUniTuplePrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        # unituples are arrays
        fields = self.val.type.fields()
        lo, hi = fields[0].type.range()
        buf = []
        for i in range(lo, hi + 1):
            buf.append(str(self.val[i]))
        return "(%s)" % ', '.join(buf)


class NumbaUnicodeTypePrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        NULL = 0x0
        data = self.val["data"]
        nitems = self.val["length"]
        kind = self.val["kind"]
        if data != NULL:
            # This needs sorting out, encoding is wrong
            this_proc = gdb.selected_inferior()
            mem = this_proc.read_memory(int(data), nitems * kind)
            if _PYVERSION < (3, 0):
                try:
                    buf = unicode(mem, 'utf-8') # noqa F821
                except UnicodeDecodeError as e:
                    buf = "ERROR: %s" % str(e)
            else:
                buf = mem.decode('utf-8')
        else:
            buf = str(data)
        return "'%s'" % buf


def _create_printers():
    printer = gdb.printing.RegexpCollectionPrettyPrinter("Numba")
    printer.add_printer('Numba unaligned array printer', '^unaligned array\\(',
                        NumbaArrayPrinter)
    printer.add_printer('Numba array printer', '^array\\(', NumbaArrayPrinter)
    printer.add_printer('Numba complex printer', '^complex[0-9]+\\ ',
                        NumbaComplexPrinter)
    printer.add_printer('Numba Tuple printer', '^Tuple\\(',
                        NumbaTuplePrinter)
    printer.add_printer('Numba UniTuple printer', '^UniTuple\\(',
                        NumbaUniTuplePrinter)
    printer.add_printer('Numba unicode_type printer', '^unicode_type\\s+\\(',
                        NumbaUnicodeTypePrinter)
    return printer


# register the Numba pretty printers for the current object
gdb.printing.register_pretty_printer(gdb.current_objfile(), _create_printers())
