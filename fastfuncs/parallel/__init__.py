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

from .._fastfuncs import _parallel


abs = _parallel.abs
acos = _parallel.acos
acosh = _parallel.acosh
asin = _parallel.asin
asinh = _parallel.asinh
atan = _parallel.atan
atan2 = _parallel.atan2
atanh = _parallel.atanh
cbrt = _parallel.cbrt
ceil = _parallel.ceil
cos = _parallel.cos
cosh = _parallel.cosh
exp = _parallel.exp
exp10 = _parallel.exp10
exp2 = _parallel.exp2
expm1 = _parallel.expm1
floor = _parallel.floor
fmod = _parallel.fmod
hypot = _parallel.hypot
log = _parallel.log
log10 = _parallel.log10
log1p = _parallel.log1p
log2 = _parallel.log2
pow = _parallel.pow
rint = _parallel.rint
round = _parallel.round
sin = _parallel.sin
sinh = _parallel.sinh
sqrt = _parallel.sqrt
tan = _parallel.tan
tanh = _parallel.tanh
truncate = _parallel.truncate
set_num_threads = _parallel.set_num_threads

__all__ = ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'cbrt',
           'ceil', 'cos', 'cosh', 'exp', 'exp10', 'exp2', 'expm1',
           'floor', 'fmod', 'hypot', 'log', 'log10', 'log1p', 'log2', 'pow',
           'rint', 'round', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'truncate', 'set_num_threads']
