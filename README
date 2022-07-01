# fastFuncs

Faster versions of most of numpy's ufuncs with a parallel version of each function.

### Benifit
* For contiguous data, `fastfuncs` can be upto 20x faster and for non-contiguous data, there is no difference in performance.

## Example Usage
```Python
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
```