import pytest
import numpy as np
import fastfuncs as ff
import fastfuncs.parallel as ffp


UFUNC_NAMES = ('abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'cbrt',
               'ceil', 'cos', 'cosh', 'exp', 'exp10', 'exp2', 'expm1',
               'floor', 'fmod', 'hypot', 'log', 'log10', 'log1p', 'log2', 'pow',
               'rint', 'round', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'truncate')

np.pow = np.power
np.exp10 = lambda x, *_: np.power(10.0, x)
np.truncate = np.trunc
np.atanh = np.arctanh
np.asinh = np.arcsinh
np.acosh = np.arccosh
np.asin = np.arcsin
np.acos = np.arccos
np.atan = np.arctan
np.atan2 = np.arctan2


TEST_DATA = [(name, size, dtype) for name in UFUNC_NAMES for size in (1, 2, 3, 4, 7, 16, 17, 32) for dtype in (np.float32, np.float64)]
np.random.seed(0)


@pytest.mark.parametrize("ufunc, n, dtype", TEST_DATA)
def test_same_as_numpy(ufunc, n, dtype):
    f_ff = getattr(ff, ufunc)
    f_np = getattr(np, ufunc)
    f_ffp = getattr(ffp, ufunc)

    if ufunc == "acosh":
        x = np.full(n, 2.5, dtype=dtype)
    else:
        x = np.random.random(n).astype(dtype)

    y = np.full_like(x, 2.0)

    if ufunc == "round":
        args = (x,)
    else:
        args = (x, y)

    ans = f_np(*args)

    assert np.allclose(f_ff(*args), ans)
    assert np.allclose(f_ffp(*args), ans)
