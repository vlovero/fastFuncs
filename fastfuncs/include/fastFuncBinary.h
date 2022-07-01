#ifndef FAST_FUNC_BINARY_H
#define FAST_FUNC_BINARY_H

#include "vcl/vcl.h"

#ifndef RP(T)
#define RP(T) T * __restrict
#endif

template <typename VTYPE, typename DTYPE, bool parallel = false>
inline size_t vclBaseBinaryFunc(VTYPE(*func)(VTYPE, VTYPE), const RP(DTYPE) x1, const RP(DTYPE) x2, RP(DTYPE) y, const size_t n)
{
    if constexpr (is_complex_vector<VTYPE>()) {
        size_t i;
        VTYPE x1v, x2v, yv;
        using RVTYPE = decltype(x1v.to_vector());
        using RTYPE = decltype(RVTYPE()[0]);
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x1, x2, y) private(i, x1v, x2v, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load((RTYPE *)(&x1[i]));
                x2v.load((RTYPE *)(&x2[i]));
                yv = func(x1v, x2v);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load((RTYPE *)(&x1[i]));
                x2v.load((RTYPE *)(&x2[i]));
                yv = func(x1v, x2v);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        return r;
    }
    else {
        size_t i;
        VTYPE x1v, x2v, yv;
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x1, x2, y) private(i, x1v, x2v, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load(&x1[i]);
                x2v.load(&x2[i]);
                yv = func(x1v, x2v);
                yv.store(&y[i]);
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load(&x1[i]);
                x2v.load(&x2[i]);
                yv = func(x1v, x2v);
                yv.store(&y[i]);
            }
        }
        return r;
    }
}

// compute y = f((scalar) x1, x2)
template <typename VTYPE, typename DTYPE, bool parallel = false>
inline size_t vclBaseBinaryFunc(VTYPE(*func)(VTYPE, VTYPE), const DTYPE x1, const RP(DTYPE) x2, RP(DTYPE) y, const size_t n)
{
    if constexpr (is_complex_vector<VTYPE>()) {
        size_t i;
        VTYPE x1v(x1), x2v, yv;
        using RVTYPE = decltype(x1v.to_vector());
        using RTYPE = decltype(RVTYPE()[0]);
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x2, y, x1v) private(i, x2v, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                x2v.load((RTYPE *)(&x2[i]));
                yv = func(x1v, x2v);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                x2v.load((RTYPE *)(&x2[i]));
                yv = func(x1v, x2v);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        return r;
    }
    else {
        size_t i;
        VTYPE x1v(x1), x2v, yv;
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x2, y, x1v) private(i, x2v, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                x2v.load(&x2[i]);
                yv = func(x1v, x2v);
                yv.store(&y[i]);
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                x2v.load(&x2[i]);
                yv = func(x1v, x2v);
                yv.store(&y[i]);
            }
        }
        return r;
    }
}

// compute y = f(x1, (scalar) x2)
template <typename VTYPE, typename DTYPE, bool parallel = false>
inline size_t vclBaseBinaryFunc(VTYPE(*func)(VTYPE, VTYPE), const RP(DTYPE) x1, const DTYPE x2, RP(DTYPE) y, const size_t n)
{
    if constexpr (is_complex_vector<VTYPE>()) {
        size_t i;
        VTYPE x1v, x2v(x2), yv;
        using RVTYPE = decltype(x1v.to_vector());
        using RTYPE = decltype(RVTYPE()[0]);
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x1, y, x2v) private(i, x1v, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load((RTYPE *)(&x1[i]));
                yv = func(x1v, x2v);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load((RTYPE *)(&x1[i]));
                yv = func(x1v, x2v);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        return r;
    }
    else {
        size_t i;
        VTYPE x1v, x2v(x2), yv;
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x1, y, x2v) private(i, x1v, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load(&x1[i]);
                yv = func(x1v, x2v);
                yv.store(&y[i]);
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load(&x1[i]);
                yv = func(x1v, x2v);
                yv.store(&y[i]);
            }
        }
        return r;
    }
}

// compute y = f(x, x)
template <typename VTYPE, typename DTYPE, bool parallel = false>
inline size_t vclBaseBinaryFunc(VTYPE(*func)(VTYPE, VTYPE), const RP(DTYPE) x1, RP(DTYPE) y, const size_t n)
{
    if constexpr (is_complex_vector<VTYPE>()) {
        size_t i;
        VTYPE x1v, yv;
        using RVTYPE = decltype(x1v.to_vector());
        using RTYPE = decltype(RVTYPE()[0]);
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x1, y) private(i, x1v, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load((RTYPE *)(&x1[i]));
                yv = func(x1v, x1v);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load((RTYPE *)(&x1[i]));
                yv = func(x1v, x1v);
                yv.store((RTYPE *)(&y[i]));
            }
        }
        return r;
    }
    else {
        size_t i;
        VTYPE x1v, yv;
        const size_t m = n & (-VTYPE::size());
        const size_t r = n - m;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(m, x1, y) private(i, x1v, yv)
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load(&x1[i]);
                yv = func(x1v, x1v);
                yv.store(&y[i]);
            }
        }
        else {
            for (i = 0; i < m; i += VTYPE::size()) {
                x1v.load(&x1[i]);
                yv = func(x1v, x1v);
                yv.store(&y[i]);
            }
        }
        return r;
    }
}

#endif // FAST_FUNC_BINARY_H