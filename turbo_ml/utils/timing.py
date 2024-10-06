import time
from functools import wraps

TIMING_ENABLED = False


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TIMING_ENABLED:
            return func(*args, **kwargs)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function {func.__name__} took {end - start} for execution")
        return result
    return wrapper
