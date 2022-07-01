#ifndef FAST_FUNC_UNARY_H
#define FAST_FUNC_UNARY_H

#include "vcl/vcl.h"

#ifndef RP(T)
#define RP(T) T * __restrict
#endif

template <typename VTYPE, typename DTYPE, bool parallel = false>
inline size_t vclBaseUnaryFunc(VTYPE(*func)(VTYPE), const RP(DTYPE) x, RP(DTYPE) y, const size_t n)
{
    if constexpr (is_complex_vector<VTYPE>()) {
        size_t i;
        VTYPE xv, yv;
        using RVTYPE = decltype(xv.to_vector());
        using RTYPE = decltype(RVTYPE()[0]);
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x, y) private(i, xv, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                xv.load((RTYPE *)(&x[i]));
                yv = func(xv);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                xv.load((RTYPE *)(&x[i]));
                yv = func(xv);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        return r;
    }
    else {
        size_t i;
        VTYPE xv, yv;
        const size_t m = n & (-xv.size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x, y) private(i, xv, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                xv.load(&x[i]);
                yv = func(xv);
                yv.store(&y[i]);
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                xv.load(&x[i]);
                yv = func(xv);
                yv.store(&y[i]);
            }
        }
        return r;
    }
}

#endif // FAST_FUNC_UNARY_H