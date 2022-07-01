#ifndef FAST_UFUNC_H
#define FAST_UFUNC_H

#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "fastFuncUnary.h"
#include "fastFuncBinary.h"

#include <complex>
#include <cmath>
#include <vector>

#ifndef RP(T)
#define RP(T) T * __restrict
#endif


template <typename T>
using unary_func_t = T(*)(T);

template <typename T>
using binary_func_t = T(*)(T, T);


namespace misc
{
    template<typename T>
    static inline T polynomial_5(const T x, T c0, T c1, T c2, T c3, T c4, T c5)
    {
        T x2 = x * x;
        T x4 = x2 * x2;
        return std::fma(std::fma(c3, x, c2), x2, std::fma(std::fma(c5, x, c4), x4, std::fma(c1, x, c0)));
    }

    template <typename T>
    static inline T polynomial_13m(T const x, T c2, T c3, T c4, T c5, T c6, T c7, T c8, T c9, T c10, T c11, T c12, T c13)
    {
        const T x2 = x * x;
        const T x4 = x2 * x2;
        const T x8 = x4 * x4;
        return std::fma(
            std::fma(std::fma(c13, x, c12), x4, std::fma(std::fma(c11, x, c10), x2, std::fma(c9, x, c8))), x8,
            std::fma(std::fma(std::fma(c7, x, c6), x2, std::fma(c5, x, c4)), x4, std::fma(std::fma(c3, x, c2), x2, x)));
    }

    static inline double pow2n(const double n)
    {
        constexpr double pow2_52 = 4503599627370496.0;
        constexpr double bias = 1023.0;
        const double a = n + (bias + pow2_52);
        int64_t b = *(int64_t *)(&a);
        int64_t c = b << 52;
        double d = *(double *)(&c);
        return d;
    }

    static inline float pow2n(const float n)
    {
        constexpr float pow2_23 = 8388608.0;
        constexpr float bias = 127.0;
        float a = n + (bias + pow2_23);
        int32_t b = *(int32_t *)(&a);
        int32_t c = b << 23;
        float d = *(float *)(&c);
        return d;
    }

    static inline bool isnan(double x)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const int64_t y = *(int64_t *)(&x);
        const int64_t z = *(int64_t *)(&nan);
        return z <= y;
    }
}

namespace std
{
    double exp10(double x)
    {
        return exp(x * M_LN10);
    }

    float exp10(float x)
    {
        return exp(x * (float)M_LN10);
    }

    std::complex<double> exp10(std::complex<double> x)
    {
        return exp(x * M_LN10);
    }

    std::complex<float> exp10(std::complex<float> x)
    {
        return exp(x * (float)M_LN10);
    }

    std::complex<double> exp2(std::complex<double> x)
    {
        return exp(x * M_LN2);
    }

    std::complex<float> exp2(std::complex<float> x)
    {
        return exp(x * (float)M_LN2);
    }

    std::complex<double> log2(std::complex<double> x)
    {
        return VM_LOG2E * log(x);
    }

    std::complex<float> log2(std::complex<float> x)
    {
        return (float)VM_LOG2E * log(x);
    }

    std::complex<double> log1p(std::complex<double> x)
    {
        return log(x + 1.0);
    }

    std::complex<float> log1p(std::complex<float> x)
    {
        return log(x + 1.0f);
    }

    std::complex<double> expm1(std::complex<double> x0)
    {
        constexpr double p2 = 1.0 / 2.0;
        constexpr double p3 = 1.0 / 6.0;
        constexpr double p4 = 1.0 / 24.0;
        constexpr double p5 = 1.0 / 120.0;
        constexpr double p6 = 1.0 / 720.0;
        constexpr double p7 = 1.0 / 5040.0;
        constexpr double p8 = 1.0 / 40320.0;
        constexpr double p9 = 1.0 / 362880.0;
        constexpr double p10 = 1.0 / 3628800.0;
        constexpr double p11 = 1.0 / 39916800.0;
        constexpr double p12 = 1.0 / 479001600.0;
        constexpr double p13 = 1.0 / 6227020800.0;
        constexpr double max_x = 708.39;
        constexpr double ln2d_hi = 0.693145751953125;
        constexpr double ln2d_lo = 1.42860682030941723212E-6;

        double re, im, s, c, exp_x, z, x, r, n2;
        re = x0.real();
        im = x0.imag();
        c = std::cos(im); // cos(b)
        s = std::sin(im); // sin(b);

        x = re;
        r = std::round(re * VM_LOG2E);
        x = std::fma(r, -ln2d_hi, x);
        x = std::fma(r, -ln2d_lo, x);

        z = misc::polynomial_13m(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
        n2 = misc::pow2n(r);
        exp_x = (z + 1.0) * n2;
        n2 *= c;
        z = std::fma(z, n2, n2 - 1.0); // exp(a) * cos(b) - 1

        bool inrange = std::abs(re) < max_x;
        inrange &= std::isfinite(re);
        if (!inrange) {
            r = std::signbit(re) ? -1.0 : std::numeric_limits<double>::infinity();
            z = r;
            z = misc::isnan(re) ? re : r;
            r = std::signbit(re) ? 0.0 : std::numeric_limits<double>::infinity();
            exp_x = r;
            exp_x = misc::isnan(re) ? re : exp_x;
        }

        exp_x *= s;
        return { z, exp_x };

    }

    std::complex<float> expm1(std::complex<float> x0)
    {
        constexpr float P0expf = 1.0f / 2.0f;
        constexpr float P1expf = 1.0f / 6.0f;
        constexpr float P2expf = 1.0f / 24.0f;
        constexpr float P3expf = 1.0f / 120.0f;
        constexpr float P4expf = 1.0f / 720.0f;
        constexpr float P5expf = 1.0f / 5040.0f;
        constexpr float ln2f_hi = 0.693359375f;
        constexpr float ln2f_lo = -2.12194440e-4f;
        constexpr float max_x = 87.3f;

        bool inrange;
        float x, r, x2, z, n2, re, im, s, c, exp_x;

        re = x0.real();
        im = x0.imag();
        s = std::sin(im);
        c = std::cos(im);

        x = re;
        r = std::round(re * (float)VM_LOG2E);
        x = std::fma(r, -ln2f_hi, x);
        x = std::fma(r, -ln2f_lo, x);

        x2 = x * x;
        z = misc::polynomial_5(x, P0expf, P1expf, P2expf, P3expf, P4expf, P5expf);
        z = std::fma(z, x2, x);

        n2 = misc::pow2n(r);
        exp_x = (z + 1.0f) * n2;
        n2 *= c;
        z = std::fma(z, n2, n2 - 1.0f);

        inrange = std::abs(re) < max_x;
        inrange &= std::isfinite(re);
        if (!inrange) {

        }
        exp_x *= s;
        return { z, exp_x };
    }

    std::complex<float> rint(std::complex<float> x)
    {
        return { std::rint(x.real()), std::rint(x.imag()) };
    }

    std::complex<double> rint(std::complex<double> x)
    {
        return { std::rint(x.real()), std::rint(x.imag()) };
    }

    std::complex<float> floor(std::complex<float> x)
    {
        return { std::floor(x.real()), std::floor(x.imag()) };
    }

    std::complex<double> floor(std::complex<double> x)
    {
        return { std::floor(x.real()), std::floor(x.imag()) };
    }

    std::complex<float> round(std::complex<float> x)
    {
        return { std::round(x.real()), std::round(x.imag()) };
    }

    std::complex<double> round(std::complex<double> x)
    {
        return { std::round(x.real()), std::round(x.imag()) };
    }

    std::complex<float> ceil(std::complex<float> x)
    {
        return { std::ceil(x.real()), std::ceil(x.imag()) };
    }

    std::complex<double> ceil(std::complex<double> x)
    {
        return { std::ceil(x.real()), std::ceil(x.imag()) };
    }

    std::complex<float> trunc(std::complex<float> x)
    {
        return { std::ceil(x.real()), std::ceil(x.imag()) };
    }

    std::complex<double> trunc(std::complex<double> x)
    {
        return { std::trunc(x.real()), std::trunc(x.imag()) };
    }

    template <typename T>
    inline T truncate(T x)
    {
        return std::trunc(x);
    }
}

namespace unary
{
    template <typename DTYPE, typename VTYPE, typename FTYPE>
    __attribute__((always_inline)) void _doRemainding(FTYPE std_func, const RP(DTYPE) x, RP(DTYPE) y, const ptrdiff_t m, ptrdiff_t r)
    {
        if (r) {
            ptrdiff_t k, q, i;
            constexpr ptrdiff_t s = VTYPE::size();  // constexpr will let compiler get rid of loops
            k = s >> 1;
            x += m;
            y += m;
            while (k) {
                q = r & -k;
                r -= q;
                if (q) {
                    for (i = 0; i < k; i++) {
                        y[i] = std_func(x[i]);
                    }
                    x += k;
                    y += k;
                }
                k >>= 1;
            }
        }
    }

    template <typename DTYPE, typename VTYPE, DTYPE(*std_func)(DTYPE)>
    static void inline doRemainding(const RP(DTYPE) x, RP(DTYPE) y, const ptrdiff_t m, ptrdiff_t r)
    {
        _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, x, y, m, r);
    }

    template <typename DTYPE, typename VTYPE, DTYPE(*std_func)(const DTYPE &)>
    static void inline doRemainding(const RP(DTYPE) x, RP(DTYPE) y, const ptrdiff_t m, ptrdiff_t r)
    {
        _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, x, y, m, r);
    }

    template <typename DTYPE, typename VTYPE, typename VFTYPE, typename SFTYPE, const bool parallel = false>
    __attribute__((always_inline)) void _fastUFuncBase(VFTYPE vcl_func, SFTYPE std_func, char **args, npy_intp *dimensions, npy_intp *steps, void *data)
    {
        const size_t n = dimensions[0];
        const size_t m = n & -VTYPE::size();
        const DTYPE *in = (DTYPE *)(args[0]);
        DTYPE *out = (DTYPE *)(args[1]);
        const size_t step_in = steps[0];
        const size_t step_out = steps[1];
        size_t i;

        if (n == 1) {
            out[0] = std_func(in[0]);
            return;
        }

        const bool b_in = step_in == sizeof(DTYPE);
        const bool b_out = step_out == sizeof(DTYPE);

        if (b_in & b_out) {
            _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, in, out, m, vclBaseUnaryFunc<VTYPE, DTYPE, parallel>(vcl_func, in, out, n));
        }
        else if (!b_in && b_out) {
            // x needs copy
            char *x = args[0];
            DTYPE *x_tmp = (DTYPE *)std::malloc(n * sizeof(DTYPE));
            for (i = 0; i < n; i++) {
                x_tmp[i] = *((DTYPE *)x);
                x += step_in;
            }
            _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, x_tmp, out, m, vclBaseUnaryFunc<VTYPE, DTYPE, parallel>(vcl_func, x_tmp, out, n));
            std::free(x_tmp);
        }
        else if (b_in && !b_out) {
            // y needs copy
            char *y = args[1];
            DTYPE *y_tmp = (DTYPE *)std::malloc(n * sizeof(DTYPE));
            for (i = 0; i < n; i++) {
                y_tmp[i] = *((DTYPE *)y);
                y += step_out;
            }
            _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, in, y_tmp, m, vclBaseUnaryFunc<VTYPE, DTYPE, parallel>(vcl_func, in, y_tmp, n));
            std::free(y_tmp);
        }
        else {
            // x and y need copy
            char *x = args[0];
            char *y = args[1];
            DTYPE *x_tmp = (DTYPE *)std::malloc(n * sizeof(DTYPE));
            DTYPE *y_tmp = (DTYPE *)std::malloc(n * sizeof(DTYPE));
            for (i = 0; i < n; i++) {
                x_tmp[i] = *((DTYPE *)x);
                x += step_in;
            }
            _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, x_tmp, y_tmp, m, vclBaseUnaryFunc<VTYPE, DTYPE, parallel>(vcl_func, x_tmp, y_tmp, n));
            for (i = 0; i < n; i++) {
                *((DTYPE *)y) = y_tmp[i];
                y += step_out;
            }
            std::free(x_tmp);
            std::free(y_tmp);
        }
    }

    template <typename DTYPE, typename VTYPE, VTYPE(*vcl_func)(VTYPE), DTYPE(*std_func)(DTYPE), const bool parallel = false>
    static void fastUFuncBase(char **args, npy_intp *dimensions, npy_intp *steps, void *data)
    {
        _fastUFuncBase<DTYPE, VTYPE, decltype(vcl_func), decltype(std_func), parallel>(vcl_func, std_func, args, dimensions, steps, data);
    }

    template <typename DTYPE, typename VTYPE, VTYPE(*vcl_func)(VTYPE), DTYPE(*std_func)(const DTYPE &), const bool parallel = false>
    static void fastUFuncBase(char **args, npy_intp *dimensions, npy_intp *steps, void *data)
    {
        _fastUFuncBase<DTYPE, VTYPE, decltype(vcl_func), decltype(std_func), parallel>(vcl_func, std_func, args, dimensions, steps, data);
    }

    template <typename DTYPE_IN, typename DTYPE_OUT, DTYPE_OUT(*std_func)(DTYPE_IN), const bool parallel = false>
    static void fastUFuncBase_std(char **args, npy_intp *dimensions, npy_intp *step, void *data)
    {
        const size_t n = dimensions[0];
        const DTYPE_IN *in = (DTYPE_IN *)(args[0]);
        DTYPE_OUT *out = (DTYPE_OUT *)(args[1]);
        size_t i;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(in, out, n) private(i)
            for (i = 0; i < n; i++) {
                out[i] = std_func(in[i]);
            }
        }
        else {
            for (i = 0; i < n; i++) {
                out[i] = std_func(in[i]);
            }
        }
    }

    template <typename DTYPE_IN, typename DTYPE_OUT, DTYPE_OUT(*std_func)(const DTYPE_IN &), const bool parallel = false>
    static void fastUFuncBase_std(char **args, npy_intp *dimensions, npy_intp *step, void *data)
    {
        const size_t n = dimensions[0];
        const DTYPE_IN *in = (DTYPE_IN *)(args[0]);
        DTYPE_OUT *out = (DTYPE_OUT *)(args[1]);
        size_t i;

        if constexpr (parallel) {
#pragma omp parallel for firstprivate(in, out, n) private(i)
            for (i = 0; i < n; i++) {
                out[i] = std_func(in[i]);
            }
        }
        else {
            for (i = 0; i < n; i++) {
                out[i] = std_func(in[i]);
            }
        }
    }
}

namespace binary
{
    template <typename DTYPE, typename VTYPE, typename FTYPE>
    __attribute__((always_inline)) void _doRemainding(FTYPE std_func, const RP(DTYPE) x1, const RP(DTYPE) x2, RP(DTYPE) y, const ptrdiff_t m, ptrdiff_t r)
    {
        if (r) {
            ptrdiff_t k, q, i;
            constexpr ptrdiff_t s = VTYPE::size();  // constexpr will let compiler get rid of loops
            k = s >> 1;
            x1 += m;
            x2 += m;
            y += m;
            while (k) {
                q = r & -k;
                r -= q;
                if (q) {
                    for (i = 0; i < k; i++) {
                        y[i] = std_func(x1[i], x2[i]);
                    }
                    x1 += k;
                    x2 += k;
                    y += k;
                }
                k >>= 1;
            }
        }
    }

    template <typename DTYPE, typename VTYPE, typename FTYPE>
    __attribute__((always_inline)) void _doRemainding(FTYPE std_func, const RP(DTYPE) x1, const DTYPE x2, RP(DTYPE) y, const ptrdiff_t m, ptrdiff_t r)
    {
        if (r) {
            ptrdiff_t k, q, i;
            constexpr ptrdiff_t s = VTYPE::size();  // constexpr will let compiler get rid of loops
            k = s >> 1;
            x1 += m;
            y += m;
            while (k) {
                q = r & -k;
                r -= q;
                if (q) {
                    for (i = 0; i < k; i++) {
                        y[i] = std_func(x1[i], x2);
                    }
                    x1 += k;
                    y += k;
                }
                k >>= 1;
            }
        }
    }

    template <typename DTYPE, typename VTYPE, typename FTYPE>
    __attribute__((always_inline)) void _doRemainding(FTYPE std_func, const DTYPE x1, const RP(DTYPE) x2, RP(DTYPE) y, const ptrdiff_t m, ptrdiff_t r)
    {
        if (r) {
            ptrdiff_t k, q, i;
            constexpr ptrdiff_t s = VTYPE::size();  // constexpr will let compiler get rid of loops
            k = s >> 1;
            x2 += m;
            y += m;
            while (k) {
                q = r & -k;
                r -= q;
                if (q) {
                    for (i = 0; i < k; i++) {
                        y[i] = std_func(x1, x2[i]);
                    }
                    x2 += k;
                    y += k;
                }
                k >>= 1;
            }
        }
    }

    template <typename DTYPE, typename VTYPE, typename FTYPE>
    __attribute__((always_inline)) void _doRemainding(FTYPE std_func, const RP(DTYPE) x1, RP(DTYPE) y, const ptrdiff_t m, ptrdiff_t r)
    {
        if (r) {
            ptrdiff_t k, q, i;
            constexpr ptrdiff_t s = VTYPE::size();  // constexpr will let compiler get rid of loops
            k = s >> 1;
            x1 += m;
            y += m;
            while (k) {
                q = r & -k;
                r -= q;
                if (q) {
                    for (i = 0; i < k; i++) {
                        y[i] = std_func(x1[i], x1[i]);
                    }
                    x1 += k;
                    y += k;
                }
                k >>= 1;
            }
        }
    }

    template <typename DTYPE, typename VTYPE, typename VFTYPE, typename SFTYPE, const bool parallel = false>
    __attribute__((always_inline)) void _fastUFuncBase(VFTYPE vcl_func, SFTYPE std_func, char **args, npy_intp *dimensions, npy_intp *steps, void *data)
    {
        const size_t n = dimensions[0];
        const size_t m = n & -VTYPE::size();
        const size_t step_in1 = steps[0];
        const size_t step_in2 = steps[1];
        const size_t step_out = steps[2];

        size_t i;
        char *x1 = args[0];
        char *x2 = args[1];
        char *y = args[2];
        DTYPE *x1_tmp;
        DTYPE *x2_tmp;
        DTYPE *y_tmp;

        if (n == 1) {
            *((DTYPE *)y) = std_func(*((DTYPE *)x1), *((DTYPE *)x2));
            // out[0] = std_func(in1[0], in2[0]);
            return;
        }

        // the ~step_in? is for when one of the inputs is scalar
        const bool b_in1 = (step_in1 == sizeof(DTYPE)) | (step_in1 == 0);
        const bool b_in2 = (step_in2 == sizeof(DTYPE)) | (step_in2 == 0);
        const bool b_out = step_out == sizeof(DTYPE);
        const bool s12 = x1 == x2;

        // check if argument is contiguous, if not make contiguous copy
        // this gives performance improvements with certain functions
        // need to make compile time flag to check
        if (!b_in1) {
            x1_tmp = (DTYPE *)std::malloc(n * sizeof(DTYPE));
            for (i = 0; i < n; i++) {
                x1_tmp[i] = *((DTYPE *)x1);
                x1 += step_in1;
            }
        }
        else {
            x1_tmp = (DTYPE *)x1;
        }
        if (!(b_in2 | s12)) {
            x2_tmp = (DTYPE *)std::malloc(n * sizeof(DTYPE));
            for (i = 0; i < n; i++) {
                x2_tmp[i] = *((DTYPE *)x2);
                x2 += step_in2;
            }
        }
        else {
            x2_tmp = s12 ? x1_tmp : (DTYPE *)x2;
        }
        if (!b_out) {
            y_tmp = (DTYPE *)std::malloc(n * sizeof(DTYPE));
        }
        else {
            y_tmp = (DTYPE *)y;
        }

        if (step_in1 == 0) {
            _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, *x1_tmp, x2_tmp, y_tmp, m, vclBaseBinaryFunc<VTYPE, DTYPE, parallel>(vcl_func, *x1_tmp, x2_tmp, y_tmp, n));
        }
        else if (step_in2 == 0) {
            _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, x1_tmp, *x2_tmp, y_tmp, m, vclBaseBinaryFunc<VTYPE, DTYPE, parallel>(vcl_func, x1_tmp, *x2_tmp, y_tmp, n));
        }
        else if (s12) {
            _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, x1_tmp, y_tmp, m, vclBaseBinaryFunc<VTYPE, DTYPE, parallel>(vcl_func, x1_tmp, y_tmp, n));
        }
        else {
            _doRemainding<DTYPE, VTYPE, decltype(std_func)>(std_func, x1_tmp, x2_tmp, y_tmp, m, vclBaseBinaryFunc<VTYPE, DTYPE, parallel>(vcl_func, x1_tmp, x2_tmp, y_tmp, n));
        }


        // free memory if allocated
        if (!b_in1) {
            std::free(x1_tmp);
        }
        if (!b_in2) {
            std::free(x2_tmp);
        }
        if (!b_out) {
            for (i = 0; i < n; i++) {
                *((DTYPE *)y) = y_tmp[i];
                y += step_out;
            }
            std::free(y_tmp);
        }
    }

    template <typename DTYPE, typename VTYPE, VTYPE(*vcl_func)(VTYPE, VTYPE), DTYPE(*std_func)(DTYPE, DTYPE), const bool parallel = false>
    static void fastUFuncBase(char **args, npy_intp *dimensions, npy_intp *steps, void *data)
    {
        _fastUFuncBase<DTYPE, VTYPE, decltype(vcl_func), decltype(std_func), parallel>(vcl_func, std_func, args, dimensions, steps, data);
    }

    template <typename DTYPE, typename VTYPE, VTYPE(*vcl_func)(VTYPE, VTYPE), DTYPE(*std_func)(const DTYPE &, const DTYPE &), const bool parallel = false>
    static void fastUFuncBase(char **args, npy_intp *dimensions, npy_intp *steps, void *data)
    {
        _fastUFuncBase<DTYPE, VTYPE, decltype(vcl_func), decltype(std_func), parallel>(vcl_func, std_func, args, dimensions, steps, data);
    }
}


// template <bool parallel = false>
struct fastFuncObject
{
    const char *name;
    std::vector<PyUFuncGenericFunction> funcs;
    std::vector<char> types;
    std::vector<void *> data;
};

// macros for creating new unary ufunc objects
#define unary_IN_OUT_ONLY_REAL { NPY_DOUBLE, NPY_DOUBLE, NPY_FLOAT, NPY_FLOAT }
#define unary_IN_OUT_REAL_AND_COMPLEX { NPY_DOUBLE, NPY_DOUBLE, NPY_FLOAT, NPY_FLOAT, NPY_CDOUBLE, NPY_CDOUBLE, NPY_CFLOAT, NPY_CFLOAT }

// macros for creating new binary ufunc objects
#define binary_IN_OUT_ONLY_REAL { NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_FLOAT, NPY_FLOAT, NPY_FLOAT }
#define binary_IN_OUT_REAL_AND_COMPLEX { NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, NPY_CDOUBLE, NPY_CDOUBLE,  NPY_CDOUBLE, NPY_CFLOAT, NPY_CFLOAT, NPY_CFLOAT }

// generic for creating unary or binary
#define BASE(ftype, dtype, vtype, name, par) (PyUFuncGenericFunction)(ftype::fastUFuncBase<dtype, vtype, name, std::name, par>)
#define BASE_STD(ftype, dtype1, dtype2, name, par) (PyUFuncGenericFunction)(ftype::fastUFuncBase_std<dtype1, dtype2, std::name, par>)

#define FUNCS_ONLY_REAL(ftype, name, par) { BASE(ftype, double, Vec8d, name, par), BASE(ftype, float, Vec16f, name, par) }
#define FUNCS_REAL_AND_COMPLEX(ftype, name, par) { BASE(ftype, double, Vec8d, name, par), BASE(ftype, float, Vec16f, name, par), BASE(ftype, std::complex<double>, Complex4d, name, par), BASE(ftype, std::complex<float>, Complex8f, name, par) }

#define DATA_ONLY_REAL { NULL, NULL }
#define DATA_REAL_AND_COMPLEX { NULL, NULL, NULL, NULL }

#define fastFuncObject_new(ftype, name, par, dataTypes) { "uf_" #name, FUNCS_ ##dataTypes(ftype, name, par), ftype## _IN_OUT_ ##dataTypes, DATA_ ##dataTypes }

#endif // FAST_UFUNC_H