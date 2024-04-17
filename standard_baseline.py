import math
import numpy as np
from scipy.special import binom, betainc

def binomial_f(k, n, p):
    return binom(n, math.floor(k)) * np.power(p, math.floor(k)) * np.power(1 - p, n - math.floor(k))

def binomial_F(k, n, p):
    return betainc(n - math.floor(k), 1 + math.floor(k), 1 - p)

def binomial_p_value(k, n, p):
    return 1 - binomial_F(k, n, p) + binomial_f(k, n, p)
