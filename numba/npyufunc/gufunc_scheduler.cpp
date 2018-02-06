/*
 * Copyright (c) 2017 Intel Corporation
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <vector>
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include "gufunc_scheduler.h"

// round not available on VS2010.
double guround (double number) {
	return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

class RangeActual {
public:
    std::vector<intp> start, end;

    RangeActual() {}

    RangeActual(intp s, intp e) {
        start.push_back(s);
        end.push_back(e);
    }

    RangeActual(const std::vector<intp> &s, const std::vector<intp> &e) {
        assert(s.size() == e.size());
        start = s;
        end = e;
    }

    RangeActual(const std::vector<intp> &lens) {
        for(uintp i = 0; i < lens.size(); ++i) {
            start.push_back(0);
            end.push_back(lens[i] - 1);
        }
    }

    RangeActual(uintp num_dims, intp *lens) {
        for(uintp i = 0; i < num_dims; ++i) {
            start.push_back(0);
            end.push_back(lens[i] - 1);
        }
    }

    RangeActual(uintp num_dims, intp *starts, intp *ends) {
        for(uintp i = 0; i < num_dims; ++i) {
            start.push_back(starts[i]);
            end.push_back(ends[i]);
        }
    }

    uintp ndim() const {
        return start.size();
    }

    std::vector<intp> iters_per_dim() const {
        std::vector<intp> ret;
        for(uintp i = 0; i < start.size(); ++i) {
			intp ret_val = end[i] - start[i] + 1;
			if(end[i] < start[i])
				ret_val = 0;
            ret.push_back(ret_val);
        }
        return ret;
    }
};

class dimlength {
public:
    uintp dim;
    intp length;
    dimlength(uintp d, intp l) : dim(d), length(l) {}
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
    uintp dim;
    intp lower_bound, upper_bound;
    isf_range(uintp d, intp l, intp u) : dim(d), lower_bound(l), upper_bound(u) {}
};

struct isf_range_by_dim {
    bool operator()(const isf_range &a, const isf_range &b) const {
        return a.dim < b.dim;
    }
};

class chunk_info {
public:
    intp m_a, m_b, m_c;
    chunk_info(intp a, intp b, intp c) : m_a(a), m_b(b), m_c(c) {}
};

chunk_info chunk(intp rs, intp re, intp divisions) {
    assert(divisions >= 1);
    intp total = (re - rs) + 1;
    if( divisions == 1) {
        return chunk_info(rs, re, re + 1);
    } else {
        intp len = total / divisions;
        //intp rem = total % divisions;
        intp res_end = rs + len - 1;
        return chunk_info(rs, res_end, res_end + 1);
    }
}

chunk_info equalizing_chunk(intp rs, intp re, intp divisions, float thread_percent) {
    assert(divisions >= 1);
    intp total = (re - rs) + 1;
    if (divisions == 1) {
        return chunk_info(rs, re, re + 1);
    }
    else {
        //intp len = total / divisions;
        intp len = total * thread_percent;
        intp res_end = rs + len - 1;
        return chunk_info(rs, res_end, res_end + 1);
    }
}

RangeActual isfRangeToActual(const std::vector<isf_range> &build) {
    std::vector<isf_range> bunsort(build);
    std::sort(bunsort.begin(), bunsort.end(), isf_range_by_dim());
    std::vector<intp> lower_bounds(bunsort.size()), upper_bounds(bunsort.size());
    for(uintp i = 0; i < bunsort.size(); ++i) {
        lower_bounds[i] = bunsort[i].lower_bound;
        upper_bounds[i] = bunsort[i].upper_bound;
    }
    return RangeActual(lower_bounds, upper_bounds);
}

void divide_work(const RangeActual &full_iteration_space,
                 std::vector<RangeActual> &assignments,
                 std::vector<isf_range> &build,
                 uintp start_thread,
                 uintp end_thread,
                 const std::vector<dimlength> &dims,
                 uintp index) {
    uintp num_threads = (end_thread - start_thread) + 1;

    assert(num_threads >= 1);
    if(num_threads == 1) {
        assert(build.size() <= dims.size());

        if(build.size() == dims.size()) {
            assignments[start_thread] = isfRangeToActual(build);
        } else {
            std::vector<isf_range> new_build(build.begin()+0, build.begin()+index);
            new_build.push_back(isf_range(dims[index].dim, full_iteration_space.start[dims[index].dim], full_iteration_space.end[dims[index].dim]));
            divide_work(full_iteration_space, assignments, new_build, start_thread, end_thread, dims, index+1);
        }
    } else {
        assert(index < dims.size());
        intp total_len = 0;
        for(uintp i = index; i < dims.size(); ++i) total_len += dims[i].length > 1 ? dims[i].length : 0;
        intp divisions_for_this_dim;
        if(total_len == 0) {
            divisions_for_this_dim = num_threads;
        } else {
            divisions_for_this_dim = intp(guround(num_threads * ((float)dims[index].length / total_len)));
        }

        intp chunkstart = full_iteration_space.start[dims[index].dim];
        intp chunkend   = full_iteration_space.end[dims[index].dim];

        intp threadstart = start_thread;
        intp threadend   = end_thread;

        for(uintp i = 0; i < divisions_for_this_dim; ++i) {
            chunk_info chunk_thread = chunk(threadstart, threadend, divisions_for_this_dim - i);
            uintp threads_used_here = (1 + (chunk_thread.m_b - chunk_thread.m_a));
            chunk_info chunk_index = equalizing_chunk(chunkstart, chunkend, divisions_for_this_dim - i, threads_used_here / (float)num_threads);
            num_threads -= threads_used_here;
            chunkstart = chunk_index.m_c;
            threadstart = chunk_thread.m_c;
            std::vector<isf_range> new_build(build.begin()+0, build.begin()+index);
            new_build.push_back(isf_range(dims[index].dim, chunk_index.m_a, chunk_index.m_b));
            divide_work(full_iteration_space, assignments, new_build, chunk_thread.m_a, chunk_thread.m_b, dims, index+1);
        }
    }
}

void flatten_schedule(const std::vector<RangeActual> &sched, uintp *out_sched) {
    uintp outer = sched.size();
    uintp inner = sched[0].start.size();
    for(uintp i = 0; i < outer; ++i) {
        for(uintp j = 0; j < inner; ++j) {
            out_sched[(i*inner*2) + j] = sched[i].start[j];
        }
        for(uintp j = 0; j < inner; ++j) {
            out_sched[(i*inner*2) + j + inner] = sched[i].end[j];
        }
    }
}

void create_schedule(const RangeActual &full_space, uintp num_sched, uintp *sched) {
    std::vector<intp> ipd = full_space.iters_per_dim();
    if(full_space.ndim() == 1) {
        intp ra_len = ipd[0];
        if(ra_len <= num_sched) {
            std::vector<RangeActual> ret;
            for(uintp i = 0; i < num_sched; ++i) {
                if(i < ra_len) {
                    ret.push_back(RangeActual(full_space.start[0] + i, full_space.start[0] + i));
                } else {
                    ret.push_back(RangeActual((intp)1, (intp)0));
                }
            }
            flatten_schedule(ret, sched);
            return;
        } else {
            intp ilen = ra_len / num_sched;
            //uintp imod = ra_len % num_sched;

            std::vector<RangeActual> ret;
            for(uintp i = 0; i < num_sched; ++i) {
                intp start = full_space.start[0] + (ilen * i);
                intp end;
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
        for(uintp i = 0; i < ipd.size(); ++i) dims.push_back(dimlength(i, ipd[i]));

        std::sort(dims.begin(), dims.end(), dimlength_by_length_reverse());
        std::vector<RangeActual> assignments(num_sched, RangeActual((intp)1,(intp)0));
        std::vector<isf_range> build;
        divide_work(full_space, assignments, build, 0, num_sched-1, dims, 0);
        flatten_schedule(assignments, sched);
        return;
    }
}

/*
    num_dim (D) is the number of dimensions of the iteration space.
    starts is the range-start of each of those dimensions, inclusive.
    ends is the range-end of each of those dimensions, inclusive.
    num_threads is the number (N) of chunks to break the iteration space into
    sched is pre-allocated memory for the schedule to be stored in and is of size NxD.
    debug is non-zero if DEBUG_ARRAY_OPT is turned on.
*/
extern "C" void do_scheduling(uintp num_dim, intp *starts, intp *ends, uintp num_threads, uintp *sched, intp debug) {
    if (debug) {
        printf("num_dim = %d\n", (int)num_dim);
        printf("ranges = (");
        for (unsigned i = 0; i < num_dim; i++) {
            printf("[%d, %d], ", (int)starts[i], (int)ends[i]);
        }
        printf(")\n");
        printf("num_threads = %d\n", (int)num_threads);
    }

    if (num_threads == 0) return;

    RangeActual full_space(num_dim, starts, ends);
    create_schedule(full_space, num_threads, sched);
    static int count = 0;
    ++count;
    if(count >= 2) {
//        exit(-1);
    }
}
