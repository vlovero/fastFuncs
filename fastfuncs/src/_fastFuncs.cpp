#include "fastUFunc.h"
#include <omp.h>

namespace serial
{
#define runParallel false
    static fastFuncObject fastFuncObjectsUnary[] = {
        fastFuncObject_new(unary, sin, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, cos, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, tan, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, exp, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, exp2, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, exp10, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, sqrt, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, sinh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, cosh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, tanh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, asin, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, acos, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, atan, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, log, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, asinh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, acosh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, atanh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, log2, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, log10, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, log1p, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, expm1, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, rint, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, round, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, ceil, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, floor, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, truncate, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, cbrt, runParallel, ONLY_REAL),
        { "uf_abs",
          { BASE_STD(unary, double, double, abs, runParallel),
            BASE_STD(unary, float, float, abs, runParallel),
            BASE_STD(unary, long double, long double, abs, runParallel),
            BASE_STD(unary, std::complex<double>, double, abs, runParallel),
            BASE_STD(unary, std::complex<float>, float, abs, runParallel),
            BASE_STD(unary, std::complex<long double>, long double, abs, runParallel) },
        { NPY_DOUBLE, NPY_DOUBLE, NPY_FLOAT, NPY_FLOAT, NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_CFLOAT, NPY_FLOAT, NPY_CLONGDOUBLE, NPY_LONGDOUBLE },
        { NULL, NULL, NULL, NULL, NULL, NULL, NULL } }
    };

    static fastFuncObject fastFuncObjectsBinary[] = {
        fastFuncObject_new(binary, fmod, runParallel, ONLY_REAL),
        fastFuncObject_new(binary, hypot, runParallel, ONLY_REAL),
        fastFuncObject_new(binary, atan2, runParallel, ONLY_REAL),
        fastFuncObject_new(binary, pow, runParallel, ONLY_REAL)
    };
#undef runParallel

    static PyMethodDef moduleMethods[] = {
        {NULL, NULL, 0, NULL} /* Sentinel */
    };

    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "fastUFuncs",
        NULL,
        -1,
        moduleMethods
    };
}

namespace parallel
{
#define runParallel true
    static fastFuncObject fastFuncObjectsUnary[] = {
        fastFuncObject_new(unary, sin, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, cos, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, tan, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, exp, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, exp2, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, exp10, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, sqrt, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, sinh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, cosh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, tanh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, asin, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, acos, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, atan, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, log, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, asinh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, acosh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, atanh, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, log2, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, log10, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, log1p, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, expm1, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, rint, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, round, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, ceil, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, floor, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, truncate, runParallel, REAL_AND_COMPLEX),
        fastFuncObject_new(unary, cbrt, runParallel, ONLY_REAL),
        { "uf_abs",
          { BASE_STD(unary, double, double, abs, runParallel),
            BASE_STD(unary, float, float, abs, runParallel),
            BASE_STD(unary, long double, long double, abs, runParallel),
            BASE_STD(unary, std::complex<double>, double, abs, runParallel),
            BASE_STD(unary, std::complex<float>, float, abs, runParallel),
            BASE_STD(unary, std::complex<long double>, long double, abs, runParallel) },
        { NPY_DOUBLE, NPY_DOUBLE, NPY_FLOAT, NPY_FLOAT, NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_CFLOAT, NPY_FLOAT, NPY_CLONGDOUBLE, NPY_LONGDOUBLE },
        { NULL, NULL, NULL, NULL, NULL, NULL, NULL } }
    };

    static fastFuncObject fastFuncObjectsBinary[] = {
        fastFuncObject_new(binary, fmod, runParallel, ONLY_REAL),
        fastFuncObject_new(binary, hypot, runParallel, ONLY_REAL),
        fastFuncObject_new(binary, atan2, runParallel, ONLY_REAL),
        fastFuncObject_new(binary, pow, runParallel, ONLY_REAL)
    };
#undef runParallel

    PyObject *set_num_threads(PyObject *self, PyObject *args)
    {
        int num_threads = 0;
        if (!PyArg_ParseTuple(args, "i", &num_threads)) {
            return NULL;
        }

        omp_set_num_threads(num_threads);

        Py_RETURN_NONE;
    }

    static PyMethodDef moduleMethods[] = {
        {"set_num_threads", set_num_threads, METH_VARARGS, "Set the number of threads for parallel ufuncs."},
        {NULL, NULL, 0, NULL} /* Sentinel */
    };

    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ufunc_test",
        NULL,
        -1,
        moduleMethods
    };
}


extern "C" PyObject * PyInit__fastfuncs(void)
{
    size_t i = 0;
    PyObject *moduleSerial, *moduleParallel, *temp;
    constexpr size_t num_declared_ufuncs = 2 * (sizeof(serial::fastFuncObjectsUnary) / sizeof(serial::fastFuncObjectsUnary[0]) + sizeof(serial::fastFuncObjectsBinary) / sizeof(serial::fastFuncObjectsBinary[0]));
    PyObject *ufunc_list[num_declared_ufuncs] = { NULL };

    moduleSerial = PyModule_Create(&serial::moduledef);
    moduleParallel = PyModule_Create(&parallel::moduledef);
    import_array();
    import_umath();

    if (moduleSerial == NULL) {
        return NULL;
    }
    if (moduleParallel == NULL) {
        Py_XDECREF(moduleSerial);
        return NULL;
    }

    // add serial functions to serial module
    for (auto &fastFunc : serial::fastFuncObjectsUnary) {
        temp = PyUFunc_FromFuncAndData(fastFunc.funcs.data(), fastFunc.data.data(), fastFunc.types.data(), fastFunc.funcs.size(), 1, 1, PyUFunc_None, fastFunc.name, "test ufunc api", 0);
        ufunc_list[i] = temp;
        if (PyModule_AddObject(moduleSerial, fastFunc.name + 3, temp) < 0) {
            for (auto ufunc : ufunc_list) {
                Py_XDECREF(ufunc);
            }
            Py_XDECREF(moduleSerial);
            return NULL;
        }
        i++;
    }
    for (auto &fastFunc : serial::fastFuncObjectsBinary) {
        temp = PyUFunc_FromFuncAndData(fastFunc.funcs.data(), fastFunc.data.data(), fastFunc.types.data(), fastFunc.funcs.size(), 2, 1, PyUFunc_None, fastFunc.name, "test ufunc api", 0);
        ufunc_list[i] = temp;
        if (PyModule_AddObject(moduleSerial, fastFunc.name + 3, temp) < 0) {
            for (auto ufunc : ufunc_list) {
                Py_XDECREF(ufunc);
            }
            Py_XDECREF(moduleSerial);
            return NULL;
        }
        i++;
    }

    // add parallel functions to parallel module
    for (auto &fastFunc : parallel::fastFuncObjectsUnary) {
        temp = PyUFunc_FromFuncAndData(fastFunc.funcs.data(), fastFunc.data.data(), fastFunc.types.data(), fastFunc.funcs.size(), 1, 1, PyUFunc_None, fastFunc.name, "test ufunc api", 0);
        ufunc_list[i] = temp;
        if (PyModule_AddObject(moduleParallel, fastFunc.name + 3, temp) < 0) {
            for (auto ufunc : ufunc_list) {
                Py_XDECREF(ufunc);
            }
            Py_XDECREF(moduleSerial);
            Py_XDECREF(moduleParallel);
            return NULL;
        }
        i++;
    }
    for (auto &fastFunc : parallel::fastFuncObjectsBinary) {
        temp = PyUFunc_FromFuncAndData(fastFunc.funcs.data(), fastFunc.data.data(), fastFunc.types.data(), fastFunc.funcs.size(), 2, 1, PyUFunc_None, fastFunc.name, "test ufunc api", 0);
        ufunc_list[i] = temp;
        if (PyModule_AddObject(moduleParallel, fastFunc.name + 3, temp) < 0) {
            for (auto ufunc : ufunc_list) {
                Py_XDECREF(ufunc);
            }
            Py_XDECREF(moduleSerial);
            Py_XDECREF(moduleParallel);
            return NULL;
        }
        i++;
    }

    // add parallel module to serial module
    if (PyModule_AddObject(moduleSerial, "_parallel", moduleParallel) < 0) {
        for (auto ufunc : ufunc_list) {
            Py_XDECREF(ufunc);
        }
        Py_XDECREF(moduleSerial);
        Py_XDECREF(moduleParallel);
        return NULL;
    }

    return moduleSerial;
}