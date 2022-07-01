"""Faster ufuncs that have a parallel option.

Examples
--------------------------
>>> import numpy as np
>>> import fastfuncs as ff
>>> import fastfuncs.parallel as ffp
>>>
>>> x = np.arange(5.0)
>>> ff.sin(x)
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ])
>>> ffp.sin(x)
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ])
>>> np.allclose(np.sin(x), ff.sin(x))
True
"""

from ._fastfuncs import (
    abs, acos, acosh, asin, asinh, atan, atan2, atanh, cbrt,
    ceil, cos, cosh, exp, exp10, exp2, expm1, floor, fmod,
    hypot, log, log10, log1p, log2, pow, rint,
    round, sin, sinh, sqrt, tan, tanh, truncate
)
from . import parallel


__all__ = ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'cbrt',
           'ceil', 'cos', 'cosh', 'exp', 'exp10', 'exp2', 'expm1',
           'floor', 'fmod', 'hypot', 'log', 'log10', 'log1p', 'log2', 'pow',
           'rint', 'round', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'truncate', 'parallel']
