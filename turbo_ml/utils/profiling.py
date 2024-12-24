import cProfile
import pstats


def profiling(func, *args, **kwargs):
    with cProfile.Profile() as pr:
        result = func(*args, **kwargs)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(f"{func.__name__}.prof")
    return result
