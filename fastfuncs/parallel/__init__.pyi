from typing import overload
from numpy import ndarray


@overload
def abs(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.abs."""
    ...


@overload
def acos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.acos."""
    ...


@overload
def acosh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.acosh."""
    ...


@overload
def asin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.asin."""
    ...


@overload
def asinh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.asinh."""
    ...


@overload
def atan(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.atan."""
    ...


@overload
def atan2(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.atan2."""
    ...


@overload
def atanh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.atanh."""
    ...


@overload
def cbrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.cbrt."""
    ...


@overload
def ceil(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.ceil."""
    ...


@overload
def cos(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.cos."""
    ...


@overload
def cosh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.cosh."""
    ...


@overload
def exp(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.exp."""
    ...


@overload
def exp10(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.exp10."""
    ...


@overload
def exp2(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.exp2."""
    ...


@overload
def expm1(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.expm1."""
    ...


@overload
def floor(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.floor."""
    ...


@overload
def fmod(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.fmod."""
    ...


@overload
def hypot(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.hypot."""
    ...


@overload
def log(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.log."""
    ...


@overload
def log10(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.log10."""
    ...


@overload
def log1p(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.log1p."""
    ...


@overload
def log2(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.log2."""
    ...


@overload
def pow(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.pow."""
    ...


@overload
def rint(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.rint."""
    ...


@overload
def round(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.round."""
    ...


@overload
def sin(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.sin."""
    ...


@overload
def sinh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.sinh."""
    ...


@overload
def sqrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.sqrt."""
    ...


@overload
def tan(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.tan."""
    ...


@overload
def tanh(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.tanh."""
    ...


@overload
def truncate(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True) -> ndarray:
    """See docs for numpy.truncate."""
    ...


@overload
def set_num_threads(n: int) -> None:
    """Set the number of threads for parallel ufuncs in fastfuncs.parllel module.

    Parameters
    ----------
    n : int
        Number of threads.
    """
