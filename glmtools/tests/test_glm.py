""" py.test test for glmtools code

Run with:

    py.test glmtools
"""

import numpy as np
import scipy.stats as stats
import numpy.linalg as npl

from glmtools import glm, t_test

from numpy.testing import assert_almost_equal


def test_glm_t_test():
    # Test glm and t_test against scipy
    # Your test code here

    #creating an independent set of random numbers to use for testing
    np.random.seed(2016)
    n = 20
    X = np.ones((n,2))
    X[:,0] = np.random.normal(10, 2, size=n)
    Y = np.random.normal(20, 1, size=n)

    """GLM"""
    #scipy linear regression
    reg = stats.linregress(X[:,0], Y)

    #linear regression from function we defined
    B, sigma_2, df = glm(Y,X)

    #asserting to see if the slopes, intercepts, and df are the same
    # between glm function and that of scipy
    assert_almost_equal(B[0], reg.slope)
    assert_almost_equal(B[1], reg.intercept)
    #calculating and asserting degrees of freedom
    p = npl.matrix_rank(X)
    df_e = len(Y) - p
    assert_almost_equal(df_e, df)

    """t-test"""
    #creating dummy variable as we did in the dummies exercise
    n_mid = n/2
    X_dummies = np.zeros((n,2))
    X_dummies[:n_mid,0] = 1
    X_dummies[n_mid:,1] = 1

    #define contrast vector for testing purposes
    c = np.array([1,-1])
    #getting the other values for the t-test function from running the GLM function
    #this should be OK because this step will only occur if the above tests pass...
    B, sigma_2, df_e = glm(Y,X_dummies)
    #t and p values from function we defined
    t, p = t_test(c, X_dummies, B, sigma_2, df_e)
    #scipy t-test
    t_scipy, p_scipy = stats.ttest_ind(Y[:n_mid], Y[n_mid:])

    assert_almost_equal(t, t_scipy)
    assert_almost_equal(p, p_scipy)

    return
