#include <vector>
#include <assert.h>
#include <algorithm>
#include <cmath>

#include <iostream>

class RangeActual {
public:
    std::vector<int> start, end;

    RangeActual() {}

    RangeActual(int s, int e) {
        start.push_back(s);
        end.push_back(e);
    }

    RangeActual(const std::vector<int> &s, const std::vector<int> &e) {
        assert(s.size() == e.size());
        start = s;
        end = e;
    }

    RangeActual(const std::vector<int> &lens) {
        for(int i = 0; i < lens.size(); ++i) {
            start.push_back(0);
            end.push_back(lens[i] - 1); 
        }
    }

    RangeActual(int num_dims, int *lens) {
        for(int i = 0; i < num_dims; ++i) {
            start.push_back(0);
            end.push_back(lens[i] - 1); 
        }
    }

    int ndim() const {
        return start.size();
    }

    std::vector<int> iters_per_dim() const {
        std::vector<int> ret;
        for(int i = 0; i < start.size(); ++i) {
            ret.push_back(end[i] - start[i] + 1);
        } 
        return ret;
    }
};

class dimlength {
public:
    unsigned dim, length;
    dimlength(unsigned d, unsigned l) : dim(d), length(l) {}
};

struct dimlength_by_dim {
    bool operator()(const dimlength &a, const dimlength &b) const {
        return a.dim < b.dim;
    }
};

struct dimlength_by_length_reverse {
    bool operator()(const dimlength &a, const dimlength &b) const {
        return a.length > b.length;
    }
};

class isf_range {
public:
    int dim, lower_bound, upper_bound;
    isf_range(int d, int l, int u) : dim(d), lower_bound(l), upper_bound(u) {}
};

struct isf_range_by_dim {
    bool operator()(const isf_range &a, const isf_range &b) const {
        return a.dim < b.dim;
    }
};

class chunk_info {
public:
    int m_a, m_b, m_c;
    chunk_info(int a, int b, int c) : m_a(a), m_b(b), m_c(c) {}
};

chunk_info chunk(int rs, int re, int divisions) {
    assert(divisions >= 1);
    int total = (re - rs) + 1;
    if( divisions == 1) {
        return chunk_info(rs, re, re + 1);
    } else {
        int len = total / divisions;
        int rem = total % divisions;
        int res_end = rs + len - 1;
        return chunk_info(rs, res_end, res_end + 1);
    }
}

RangeActual isfRangeToActual(const std::vector<isf_range> &build) {
    std::vector<isf_range> bunsort(build);
    std::sort(bunsort.begin(), bunsort.end(), isf_range_by_dim());
    std::vector<int> lower_bounds(bunsort.size()), upper_bounds(bunsort.size());
    for(unsigned i = 0; i < bunsort.size(); ++i) {
        lower_bounds[i] = bunsort[i].lower_bound;
        upper_bounds[i] = bunsort[i].upper_bound;
    }
    return RangeActual(lower_bounds, upper_bounds);
}
    
void divide_work(const RangeActual &full_iteration_space, 
                 std::vector<RangeActual> &assignments, 
                 std::vector<isf_range> &build, 
                 unsigned start_thread, 
                 unsigned end_thread, 
                 const std::vector<dimlength> &dims, 
                 unsigned index) {
    unsigned num_threads = (end_thread - start_thread) + 1;

    assert(num_threads >= 1);
    if(num_threads == 1) {
        assert(build.size() <= dims.size());
            
        if(build.size() == dims.size()) {
            assignments[start_thread] = isfRangeToActual(build);
        } else {
            std::vector<isf_range> new_build(&build[0], &build[index]);
            new_build.push_back(isf_range(dims[index].dim, full_iteration_space.start[dims[index].dim], full_iteration_space.end[dims[index].dim]));
            divide_work(full_iteration_space, assignments, new_build, start_thread, end_thread, dims, index+1);
        }
    } else {
        assert(index < dims.size());
        unsigned total_len = 0;
        for(unsigned i = index; i < dims.size(); ++i) total_len += dims[i].length;
        unsigned divisions_for_this_dim;
        if(total_len == 0) {
            divisions_for_this_dim = num_threads; 
        } else {
            std::vector<float> percent_dims;
            float dim_prod = 1;
            for(unsigned i = index; i < dims.size(); ++i) {
                float temp = (float)dims[i].length / total_len;
                percent_dims.push_back(temp);
                dim_prod *= temp;
            }
            divisions_for_this_dim = int(round(std::pow((num_threads / dim_prod), (1.0 / percent_dims.size())) * percent_dims[0]));
        }
              
        unsigned chunkstart = full_iteration_space.start[dims[index].dim];
        unsigned chunkend   = full_iteration_space.end[dims[index].dim];
                                                                                                  
        unsigned threadstart = start_thread;
        unsigned threadend   = end_thread;
              
        for(unsigned i = 0; i < divisions_for_this_dim; ++i) {
            chunk_info chunk_index = chunk(chunkstart,  chunkend,  divisions_for_this_dim - i);
            chunk_info chunk_thread = chunk(threadstart, threadend, divisions_for_this_dim - i);
            chunkstart = chunk_index.m_c;
            threadstart = chunk_thread.m_c;
            std::vector<isf_range> new_build(&build[0], &build[index]);
            new_build.push_back(isf_range(dims[index].dim, chunk_index.m_a, chunk_index.m_b)); 
            divide_work(full_iteration_space, assignments, new_build, chunk_thread.m_a, chunk_thread.m_b, dims, index+1);
        }
    }
}

void flatten_schedule(const std::vector<RangeActual> &sched, int *out_sched) {
    unsigned outer = sched.size();
    unsigned inner = sched[0].start.size() * 2; // start and end are the same size so multiply by 2
    unsigned end_offset = sched[0].start.size();
    for(unsigned i = 0; i < outer; ++i) {
        for(unsigned j = 0; j < inner; ++j) {
            out_sched[(i*inner) + j] = sched[i].start[j];
        }
        for(unsigned j = 0; j < inner; ++j) {
            out_sched[(i*inner) + j + end_offset] = sched[i].end[j];
        }
    }
}

void create_schedule(const RangeActual &full_space, unsigned num_sched, int *sched) {
    std::vector<int> ipd = full_space.iters_per_dim();
    if(full_space.ndim() == 1) {
        unsigned ra_len = ipd[0];
        if(ra_len <= num_sched) {
            std::vector<RangeActual> ret;
            for(unsigned i = 0; i < num_sched; ++i) {
                if(i < ra_len) {
                    ret.push_back(RangeActual(full_space.start[0] + i, full_space.start[0] + i));
                } else {
                    ret.push_back(RangeActual(-1, -2));
                }
            }
            flatten_schedule(ret, sched);
            return;
        } else {
            unsigned ilen = ra_len; 
            unsigned imod = ra_len % num_sched;

            std::vector<RangeActual> ret;
            for(unsigned i = 0; i < num_sched; ++i) {
                int start = full_space.start[0] + (ilen * i);
                int end;
                if(i < num_sched-1) {
                    end = full_space.start[0] + (ilen * (i+1)) - 1;
                } else {
                    end = full_space.end[0];
                }
                ret.push_back(RangeActual(start, end));
            }
            flatten_schedule(ret, sched);
            return;
        }
    } else {
        std::vector<dimlength> dims;
        for(unsigned i = 0; i < ipd.size(); ++i) dims.push_back(dimlength(i, ipd[i]));
       
        std::sort(dims.begin(), dims.end(), dimlength_by_length_reverse());
        std::vector<RangeActual> assignments(num_sched, RangeActual(-1,-2));
        std::vector<isf_range> build;
        divide_work(full_space, assignments, build, 0, num_sched-1, dims, 0);
        flatten_schedule(assignments, sched);
        return;
    }
}
    
/*
    num_dim (D) is the number of dimensions of the iteration space.
    dims is the length of each of those dimensions.
    num_threads is the number (N) of chunks to break the iteration space into
    sched is pre-allocated memory for the schedule to be stored in and is of size NxD.
*/
extern "C" void do_scheduling(int num_dim, int *dims, unsigned num_threads, int *sched) {
    RangeActual full_space(num_dim, dims);
    create_schedule(full_space, num_threads, sched);
}

