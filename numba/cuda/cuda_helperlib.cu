#define NPY_MAXDIMS 32

typedef long int npy_intp;
/*
 * Handle reshaping of zero-sized array.
 * See numba_attempt_nocopy_reshape() below.
 */
extern "C" __device__ int
nocopy_empty_reshape(npy_intp nd, const npy_intp *dims, const npy_intp *strides,
                     npy_intp newnd, const npy_intp *newdims,
                     npy_intp *newstrides, npy_intp itemsize,
                     int is_f_order)
{
    int i;
    /* Just make the strides vaguely reasonable
     * (they can have any value in theory).
     */
    for (i = 0; i < newnd; i++)
        newstrides[i] = itemsize;
    return 1;  /* reshape successful */
}

/*
 * Straight from Numpy's _attempt_nocopy_reshape()
 * (np/core/src/multiarray/shape.c).
 * Attempt to reshape an array without copying data
 *
 * This function should correctly handle all reshapes, including
 * axes of length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills `npy_intp *newstrides`
 *     with appropriate strides
 */

extern "C" __device__ int
numba_attempt_nocopy_reshape(npy_intp nd, const npy_intp *dims, const npy_intp *strides,
                             npy_intp newnd, const npy_intp *newdims,
                             npy_intp *newstrides, npy_intp itemsize,
                             int is_f_order)
{
    int oldnd;
    npy_intp olddims[NPY_MAXDIMS];
    npy_intp oldstrides[NPY_MAXDIMS];
    npy_intp np, op, last_stride;
    int oi, oj, ok, ni, nj, nk;

    oldnd = 0;
    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
    for (oi = 0; oi < nd; oi++) {
        if (dims[oi]!= 1) {
            olddims[oldnd] = dims[oi];
            oldstrides[oldnd] = strides[oi];
            oldnd++;
        }
    }

    np = 1;
    for (ni = 0; ni < newnd; ni++) {
        np *= newdims[ni];
    }
    op = 1;
    for (oi = 0; oi < oldnd; oi++) {
        op *= olddims[oi];
    }
    if (np != op) {
        /* different total sizes; no hope */
        return 0;
    }

    if (np == 0) {
        /* the Numpy code does not handle 0-sized arrays */
        return nocopy_empty_reshape(nd, dims, strides,
                                    newnd, newdims, newstrides,
                                    itemsize, is_f_order);
    }

    /* oi to oj and ni to nj give the axis ranges currently worked with */
    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while (ni < newnd && oi < oldnd) {
        np = newdims[ni];
        op = olddims[oi];

        while (np != op) {
            if (np < op) {
                /* Misses trailing 1s, these are handled later */
                np *= newdims[nj++];
            } else {
                op *= olddims[oj++];
            }
        }

        /* Check whether the original axes can be combined */
        for (ok = oi; ok < oj - 1; ok++) {
            if (is_f_order) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                     /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        /* Calculate new strides for all axes currently worked with */
        if (is_f_order) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
            }
        }
        ni = nj++;
        oi = oj++;
    }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
    if (ni >= 1) {
        last_stride = newstrides[ni - 1];
    }
    else {
        last_stride = itemsize;
    }
    if (is_f_order) {
        last_stride *= newdims[ni - 1];
    }
    for (nk = ni; nk < newnd; nk++) {
        newstrides[nk] = last_stride;
    }

    return 1;
}