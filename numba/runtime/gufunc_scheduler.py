import numba 
import numpy as np
from operator import mul
from functools import reduce
from numba import config

class RangeActual():
    def __init__(self, start, end):
        if isinstance(start, list):
            assert isinstance(end, list)
            assert len(start) == len(end)
            self.start = start
            self.end   = end
        else:
            self.start = [start]
            self.end   = [end]

    def __str__(self):
        return '(' + str(self.start) + ',' + str(self.end) + ')'

    def __repr__(self):
        return self.__str__()

    def ndim(self):
        return len(self.start)

    def iters_per_dim(self):
        return [self.end[x] - self.start[x] + 1 for x in range(len(self.start))]

def create_full_iteration_from_array(real_array):
    return RangeActual([0] * real_array.ndim, [x-1 for x in np.shape(real_array)])

def create_full_iteration(loop_ranges):
    return RangeActual([0] * len(loop_ranges), [x-1 for x in loop_ranges])

class dimlength():
    def __init__(self, dim, length):
        self.dim    = dim
        self.length = length

    def __str__(self):
        return "(" + str(self.dim) + "," + str(self.length) + ")"

    def __repr__(self):
        return self.__str__()

class isf_range():
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        return "(" + str(self.dim) + "," + str(self.lower_bound) + "," + str(self.upper_bound) + ")"

    def __repr__(self):
        return self.__str__()


def chunk(rs, re, divisions):
    assert divisions >= 1
    total = (re - rs) + 1
    if divisions == 1:
        return (rs, re, re + 1)
    else:
        len = total // divisions
        rem = total %  divisions
        res_end = rs + len - 1
        return (rs, res_end, res_end + 1)

def isfRangeToActual(build):
    bunsort = sorted(build, key=lambda x: x.dim)
    if config.DEBUG_ARRAY_OPT:
        print("isfRangeToActual: build = ", build, " bunsort = ", bunsort)
    return RangeActual([x.lower_bound for x in bunsort],[x.upper_bound for x in bunsort])
    
def divide_work(full_iteration_space, assignments, build, start_thread, end_thread, dims, index):
    num_threads = (end_thread - start_thread) + 1

    if config.DEBUG_ARRAY_OPT:
        print("divide_work num_threads = ", num_threads)
        print("fis = ", full_iteration_space, " build = ", build, " start_thread = ", start_thread, " end_thread = ", end_thread, " dims = ", dims, " index = ", index)
            
    assert num_threads >= 1
    if num_threads == 1:
        assert len(build) <= len(dims)
            
        if len(build) == len(dims):
            if config.DEBUG_ARRAY_OPT:
                print("build == len, ", len(build))
            pra = isfRangeToActual(build) 
            assignments[start_thread] = pra
        else:
            new_build = build[0:(index-1)] + [isf_range(dims[index].dim, full_iteration_space.start[dims[index].dim], full_iteration_space.end[dims[index].dim])]
            divide_work(full_iteration_space, assignments, new_build, start_thread, end_thread, dims, index+1)
    else:
        assert index <= len(dims)
        total_len = sum(map(lambda x: x.length, dims[index:]))                                         
        if config.DEBUG_ARRAY_OPT:
            print("total_len = ", total_len)
        if total_len == 0: 
            divisions_for_this_dim = num_threads                                                  
        else: 
            percent_dims = [x.length / total_len for x in dims[index:]]
            dim_prod = reduce(mul, percent_dims, 1)
            if config.DEBUG_ARRAY_OPT:
                print("percent_dims = ", percent_dims)
                print("dim_prod = ", dim_prod)
            #divisions_for_this_dim = (num_threads * dims[index].length) // total_len
            divisions_for_this_dim = int(round(((num_threads / dim_prod) ** (1.0 / len(percent_dims))) * percent_dims[0]))
              
        if config.DEBUG_ARRAY_OPT:
            print("divisions for this dim = ", divisions_for_this_dim)

        chunkstart = full_iteration_space.start[dims[index].dim]                           
        chunkend   = full_iteration_space.end[dims[index].dim]                           
                                                                                                  
        threadstart = start_thread
        threadend   = end_thread
              
        if config.DEBUG_ARRAY_OPT:
            print(chunkstart, " ", chunkend, " ", threadstart, " ", threadend)

        for i in range(divisions_for_this_dim):
            (ls, le, chunkstart)  = chunk(chunkstart,  chunkend,  divisions_for_this_dim - i) 
            (ts, te, threadstart) = chunk(threadstart, threadend, divisions_for_this_dim - i) 
            if config.DEBUG_ARRAY_OPT:
                print("i = ", i)
                print("ls = ", ls, " le = ", le, " ts = ", ts, " te = ", te)
                print("build = ", build, " build[0:index+1] = ", build[0:(index+1)])
            divide_work(full_iteration_space, assignments, build[0:(index+1)] + [isf_range(dims[index].dim, ls, le)], ts, te, dims, index+1) 

def create_schedule(full_space, num_sched):
    ipd = full_space.iters_per_dim()
    if config.DEBUG_ARRAY_OPT:
        print("create_schedule ipd = ", ipd)
    if full_space.ndim() == 1:
        ra_len = ipd[0]
        if ra_len <= num_sched:
            return flatten_schedule(np.array([RangeActual(full_space.start[0]+x,full_space.start[0]+x) if x < ra_len else RangeActual(-1,-2) for x in range(num_sched)]))
        else:
            ilen = ra_len // num_sched
            imod = ra_len %  num_sched 
            return flatten_schedule(np.array([RangeActual(full_space.start[0] + (ilen * x), full_space.start[0] + ((ilen * (x+1))) - 1 if x < (num_sched-1) else full_space.end[0]) for x in range(num_sched)]))
    else:
        dims = [dimlength(x, ipd[x]) for x in range(len(ipd))]
        if config.DEBUG_ARRAY_OPT:
            print("unsorted dims = ", dims)
        dims.sort(key=lambda x: x.length, reverse=True)
        if config.DEBUG_ARRAY_OPT:
            print("sorted dims = ", dims)
        assignments = np.array([RangeActual(-1,-2) for x in range(num_sched)])
        if config.DEBUG_ARRAY_OPT:
            print("init assignments = ", assignments)
        divide_work(full_space, assignments, [], 0, num_sched-1, dims, 0)
        return flatten_schedule(assignments)
    
def flatten_schedule(sched):
    return np.array([(x.start + x.end) for x in sched] )

def print_schedule(sched):
    for ix in range(np.size(sched)):
        print(ix, " -> ", sched[ix])

def arg_to_gufunc_sig(arg):
    if config.DEBUG_ARRAY_OPT:
        print("arg_to_gufunc arg = ", arg, " type = ", type(arg))
    if isinstance(arg, types.ArrayCompatible):
        return '(' + (",".join(map(lambda x: 'x' + str(x), range(arg.ndim)))) + ')'
    else:
        return '()'

# Below is an example of creating and executing a schedule.
#@guvectorize([(int64[:],float64[:,:],float64[:,:],float64[:,:],float64[:])], '(n),(o,p),(o,p)->(o,p),()', target='parallel')
#def guex_schedule2(sched, x, y, res, red):
#    for i in range(sched[0], sched[2] + 1):
#        for j in range(sched[1], sched[3] + 1):
#            res[i,j] = np.power(x[i,j], y[i,j])
#            red[0] += res[i,j]
#
#    ra1000_2 = np.full((20,50), 5.7)
#    ra1000_2_2 = np.full((20,50), 5.7)
#    ra1000_2_3 = np.full((20,50), 0.0)
#    fi1000_2 = create_full_iteration(ra1000_2)
#    print("fi1000_2 = ", fi1000_2)
#    scheduling_array1000_2 = create_schedule(fi1000_2, 100)
#    print_schedule(scheduling_array1000_2)
#    flat_sched = flatten_schedule(scheduling_array1000_2)  # (100, 4)
#    print("flat_sched = ", flat_sched)
#    gsres = np.full((100,), 10.0)
#    guex_schedule2(flat_sched, ra1000_2, ra1000_2_2, ra1000_2_3, gsres)
#    final_reduction = np.sum(gsres)
