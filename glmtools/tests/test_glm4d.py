""" Test glm4d module

Run with:

    py.test glmtools
"""

import numpy as np

from glmtools import glm_4d, t_test_3d, glm, t_test

from numpy.testing import assert_almost_equal, assert_equal


def test_glm4d():
    # Test GLM model on 4D data
    # +++your code here+++

    #creating an independent set of random numbers to use for testing
    np.random.seed(2016)
    Y = np.random.normal(20, 1, size=(3,5,7,12)) #I,J,K,T is size
    X = np.random.normal(10,2,size = (Y.shape[-1],3))
    X[:,0] = np.ones(Y.shape[-1])

    """4D GLM"""

    #linear regression from function we defined
    B, sigma_2, df = glm_4d(Y,X)

    #turn into 2D matrix to test against scipy
    n_voxels = Y.shape[0]*Y.shape[1]*Y.shape[2]
    n_timepoints = Y.shape[3]

    #- Transpose to give time by voxel 2D
    Y_2D = np.reshape(Y,(n_voxels,n_timepoints)).T

    """#reg = stats.linregress(X[:,i], Y_2D)
    #can't check against scipy because won't accept 2D matrix
    #using the glm2d function instead"""

    #this took a while to figure out (or rather, to struggle with an idea),
    # maybe not the best way to test because
    # it's amost identical to how I have to do the glm_4d function itself...
    B_test, s_2_test, df_test = glm(Y_2D, X)
    B_test = B_test.T
    B_test_3D = B_test.reshape((Y.shape[0], Y.shape[1], Y.shape[2], X.shape[-1]))
    s_2_test_3D = s_2_test.reshape((Y.shape[0], Y.shape[1], Y.shape[2]))

    #asserting to see if beta and sigma matrices are the same from glm_4d and glm(using reshaped 4d)
    assert_almost_equal(B_test_3D, B)
    assert_almost_equal(s_2_test_3D, sigma_2)
    assert_almost_equal(df_test, df)

    """3d t-test"""
    c = np.array([0, -1, 1])

    #3d t-test from function we defined
    t, p = t_test_3d(c, X, B, sigma_2, df)

    #regular t-test with reshaped (2d) data to compare for testing
    t_2, p_2 = t_test(c, X, B_test.T, s_2_test, df)
    t_2_3D = t_2.reshape((Y.shape[:3]))
    assert_almost_equal(t_2_3D, t)
    assert_almost_equal(p_2, p)

    return
