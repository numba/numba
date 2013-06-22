cimport cpuscheduler as _sch

ctypedef unsigned long long ptr_t

cdef class WorkGang(object):
    cdef void *gang
    def __init__(self, int ncpu, ptr_t kernel, int ntid, ptr_t args, int arglen,
                 ptr_t atomic_add):
        self.gang = _sch.start_workers(ncpu, <void*>kernel, ntid, <void*>args, 
                                       arglen, <void*>atomic_add)
    
    def join(self):
        _sch.join_workers(self.gang)
