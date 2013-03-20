import llvm.core

class LLVMConstantsManager(object):
    """
    Manage global constants. The result should be linked into the consumer
    LLVM module.
    """

    def __init__(self):
        self.module = llvm.core.Module.new("numba_constants")

        # py_constant -> llvm value
        self.constant_values = {}

    def link(self, dst_module):
        dst_module.link_in(self.module, preserve=True)

    def get_string_constant(self, const_str):
        if const_str in self.constant_values:
            ret_val = self.constant_values[const_str]
        else:
            lconst_str = llvm.core.Constant.stringz(const_str)
            ret_val = self.module.add_global_variable(
                lconst_str.type, "__STR_%d" % (len(self.constant_values),))

            ret_val.linkage = llvm.core.LINKAGE_LINKONCE_ODR
            ret_val.initializer = lconst_str
            ret_val.is_global_constant = True

            self.constant_values[const_str] = ret_val

        return ret_val
