"""
max_random_baseline.py

This file implements the expected maximum random baseline for classification
tasks described in "Stronger Random Baselines for In-Context Learning."

Example basic usage:

max_random_baseline(100, 0.5, 10)
    
    This returns the expected maximum accuracy among classifiers that guess
    randomly on a binary classification task with n = 100 examples.


max_random_baseline(100, {2: 50, 5: 50}, 10)
    
    This returns the expected maximum accuracy among classifiers that guess
    randomly on a classification task with n = 100 examples, where 50 of the
    examples have 2 possible labels, and the other 50 examples have 5 possible
    labels.


max_random_baseline(100, [0.5] * 50 + [0.2] * 50, 10)
    
    Same as above example. This returns the expected maximum accuracy among
    classifiers that guess randomly on a classification task with n = 100
    examples, where 50 of the examples have 2 possible labels (0.5 probability
    of guessing correctly), and the other 50 examples have 5 possible labels
    (0.2 probability of guessing correctly).

"""

import sys
import math
import numpy as np
from poibin.poibin import PoiBin

def max_random_baseline(n, arg, t):
    """
    Calculates the expected maximum random baseline.
    
    Inputs:
        n: number of examples in the dataset
        t: number of random classifiers
        arg: specifies the probability of success in one of three ways:
            float: the probability of success p for correctly guessing on each
                   example (same probability across examples)
            list: a list of n floats that specifies the probability of correctly
                  guessing each example (can be different across examples)
            dict: maps a number of labels to the number of examples with that
                  many labels
            
    Outputs: the expected maximum accuracy from among t different classifiers
             that guess uniformly at random
    """
    order = MaxOrderStatisticPoissonBinomial(n, arg)
    return order.max_random_baseline(t)


def max_random_p_value(acc, n, arg, t):
    """
    Calculates the proportion of maximum accuracies among t different 
    classifiers that are greater than acc.
    
    Inputs:
        acc: the accuracy to evaluate the p-value for
        n: number of examples in the dataset
        t: number of random classifiers
        arg: the probability of success formatted as in max_random_baseline()
            
    Output: the proportion of maximum accuracies among t different classifiers
            that are greater than acc
    """
    order = MaxOrderStatisticPoissonBinomial(n, arg)
    return order.p_value(acc, t)


def max_random_F(num_correct, n, arg, t):
    """
    Calculates the distribution function for the maximum order statistic of the
    Poisson binomial with given parameters.
    
    Inputs:
        num_correct: number of examples classified correctly
        n: number of examples in the dataset
        t: number of random classifiers
        arg: the probability of success formatted as in max_random_baseline()
            
    Output: the distribution function evaluated at num_correct
    """
    order = MaxOrderStatisticPoissonBinomial(n, arg)
    return order.F(num_correct, t)


def max_random_pmf(num_correct, n, arg, t):
    """
    Evaluates the probability mass function for the maximum order statistic of
    the Poisson binomial with given parameters.
    
    Inputs:
        num_correct: number of examples classified correctly
        n: number of examples in the dataset
        t: number of random classifiers
        arg: the probability of success formatted as in max_random_baseline()
            
    Output: the probability mass function evaluated at num_correct
    """
    order = MaxOrderStatisticPoissonBinomial(n, arg)
    return order.pmf(num_correct, t)


class MaxOrderStatisticPoissonBinomial():
    """
    Class for the maximum order statistic of the Poisson binomial distribution.
    
    Allows for faster calculations of the baseline when the parameters
    n (number of examples) and ps (list of probabilities of success for each of
    the n trials) are fixed. The parameter t can take different values.
    """

    def _convert_arg_to_ps(self, n, arg):
        """
        Helper function to parse the probabilities of success.
        
        Inputs:
            n: number of examples in the dataset
            arg: specifies the probability of success in one of three ways:
                float: the probability of success p for correctly guessing on
                       each example (same probability across examples)
                list: a list of n floats that specifies the probability of
                      correctly guessing each example (can be different across
                      examples)
                dict: maps a number of labels to the number of examples with
                      that many labels
        Output: list of n probabilities of success, one for each trial in the
                Poisson binomial distribution        
        """
        if isinstance(arg, float):
            return [arg] * n
        if isinstance(arg, dict):
            ps = [p for num_labels, num_examples in arg.items() \
                    for p in [1/num_labels] * num_examples]
            assert len(ps) == n
            return ps
        if isinstance(arg, list):
            return arg
        print('Unrecognized type for probability of correct guesses.')
        print('Please see the README for proper use.')
        sys.exit()

    def __init__(self, n, arg):
        self.n = n
        ps = self._convert_arg_to_ps(n, arg)
        self._pb = PoiBin(ps)

    # The probability mass function
    def pmf(self, k, t):
        k = math.floor(k)
        F_k = self._pb.cdf(k)
        f_k = self._pb.pmf(k)
        return np.power(F_k, t) - np.power(F_k - f_k, t)

    # The distribution function
    def F(self, k, t):
        k = math.floor(k)
        if k < 0:
            return 0
        F_k = self._pb.cdf(k)
        return np.power(F_k, t)

    # The expected maximum number of correct guesses among t different
    # classifiers that guess uniformly at random
    def expectation(self, t):
        total = 0
        for k in range(self.n+1):
            total += k * self.pmf(k, t)
        # clamp to [0, n]
        return max(min(total, self.n), 0.)

    # Converts the expected maximum number of correct guesses to an accuracy
    def max_random_baseline(self, t):
        return 1/self.n * self.expectation(t)

    # Evaluates the p-value of the provided accuracy
    def p_value(self, acc, t):
        p_val = 1 - self.F(self.n * acc - 1, t)
        # clamp to [0, 1]
        return max(min(p_val, 1.), 0.)
