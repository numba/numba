def specialized_mangle(func_name, types):
    type_strings = "_".join(str(t).replace(" ", "_") for t in types)
    return "__numba_specialized_%s_%s" % (func_name, type_strings)