import contextlib
import logging
import timeit

mlogger = logging.getLogger('dl4to4ocp')


@contextlib.contextmanager
def log_timing(desc: str):
    """A context manager to time a block of code."""
    start_time = timeit.default_timer()
    yield
    end_time = timeit.default_timer()
    mlogger.info(f"{desc} took {end_time - start_time:.2f} seconds")
