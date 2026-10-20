import numpy as np


def cubic_plan(final, start, end, every, initial=0.0):
    # s_t = s_f + (s_i - s_f) * (1 - (t - t0)/(t_end - t0))^3
    # fast early (lots of redundancy), slow near the end (every weight left matters)
    steps = list(range(start, end, every)) + [end]
    plan = {}
    for t in steps:
        frac = (t - start) / (end - start)
        plan[t] = final + (initial - final) * (1 - frac) ** 3
    plan[end] = final
    return plan


def oneshot_plan(final, at):
    return {at: final}
