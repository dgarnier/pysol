"""Disable Numba JIT when running coverage."""
import os

COVERAGE = os.getenv("COVERAGE_RUN", "")
if COVERAGE:
    os.environ["NUMBA_DISABLE_JIT"] = "1"
