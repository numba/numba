cdef extern from "cpuscheduler.h":
    void* start_workers(int ncpu, void *kernel, int ntid, void *args,
                        int arglen, void *atomic_add)
    void join_workers(void *gang)
