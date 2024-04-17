import numpy as np
import math
from mpmath import *
mp.dps = 25; mp.pretty = True

# Reimplement the maximum random baseline using mpmath for high precision
# This probability mass function is written in explicit terms of hypergeometric functions so we can increase the precision
# Allows us to calculate just the pmf for k without calculating all values of the pmf
# Only works for datasets with the same number of choices across all examples
# (i.e. the underlying distribution is binomial, not Poisson binomial)

def max_order_statistic_binomial_pmf(k, n, p, t):
    return - power((1 - power(1 - p, n-k) * power(p, k) * binomial(n,k) * hyp2f1(1, -n + k, 1 + k, p / (-1 + p), maxterms=10**6, asymp_tol=1e-4)), t) \
    + power((1 - power(1 - p, -1 + n - k) * power(p, 1 + k) * binomial(n, 1 + k) * hyp2f1(1, 1 - n + k, 2 + k, p / (-1 + p), maxterms=10**6, asymp_tol=1e-4)), t)

def approximate_max_random_baseline(n, p, t):
    total = mpf(0)
    start = 0
    if n > 1500:
        if t < 10:
            # almost just a binomial
            start = math.floor(n*p - 5 * np.sqrt(n*p*(1-p)))
        else:
            start = math.floor(n*p)
    for k in range(start, n+1):
        prob = max_order_statistic_binomial_pmf(k, n, p, t)
        if total > 1 and prob < 1e-20:
            break
        total += k * prob
    return 1/n * float(nstr(total, 15))