"""Try to import numba."""

import os

# try to enable use without numba.  Those with guvectorize will not work.
try:
    from numba import guvectorize, jit, njit, prange

    # if numba is diabled, redfine the jit decorator to do nothing
    # so that coverage testing will work
    if os.getenv("NUMBA_DISABLE_JIT", "0") == "1":  # pragma: no cover
        raise ImportError

except ImportError:  # pragma: no cover
    from inspect import currentframe, getframeinfo
    from warnings import warn_explicit

    _WARNING = (
        "Numba disabled. Mutual inductance calculations "
        + "will not be accelerated and some API will not be available."
    )
    _finfo = getframeinfo(currentframe())  # type: ignore
    warn_explicit(_WARNING, RuntimeWarning, _finfo.filename, _finfo.lineno)

    def _jit(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # called as @decorator
            return args[0]
        else:
            # called as @decorator(*args, **kwargs)
            return lambda f: f

    def _guvectorize(*args, **kwds):
        def fake_decorator(f):
            warning = f"{f.__name__} requires Numba JIT."
            finfo = getframeinfo(currentframe().f_back)  # type: ignore
            warn_explicit(warning, RuntimeWarning, finfo.filename, finfo.lineno)
            return lambda f: None

        return fake_decorator

    guvectorize = _guvectorize
    njit = _jit
    prange = range
    jit = _jit
