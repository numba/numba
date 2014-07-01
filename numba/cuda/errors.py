from __future__ import print_function, absolute_import


class KernelRuntimeError(RuntimeError):
    def __init__(self, msg, tid=None, ctaid=None):
        self.tid = tid
        self.ctaid = ctaid
        self.msg = msg
        t = ("An exception was raised in thread=%s block=%s\n"
             "\t%s")
        msg = t % (self.tid, self.ctaid, self.msg)
        super(KernelRuntimeError, self).__init__(msg)
