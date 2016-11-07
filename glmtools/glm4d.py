""" Run GLM on final dimension of 4D arrays
"""

import numpy as np

from .glm import glm, t_test


def glm_4d(Y, X):
    """ Run GLM on on 4D data `Y` and design `X`

    Parameters
    ----------
    Y : array shape (I, J, K, T)
        4D array to fit to model with design `X`.  Column vectors are vectors
        over the final length T dimension.
    X : array ahape (T, P)
        2D design matrix to fit to data `Y`.

    Returns
    -------
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.
    """
    # +++your code here+++

    n_voxels = Y.shape[0]*Y.shape[1]*Y.shape[2]
    n_timepoints = Y.shape[3]

    #- Transpose to give time by voxel 2D
    Y_2D = np.reshape(Y,(n_voxels,n_timepoints)).T

    #get betas for 2d reshaped array
    B, sigma_2, df = glm(Y_2D, X)

    #reshape betas to return correctly in their original shape
    B = B.T
    B = np.reshape(B, (Y.shape[0], Y.shape[1], Y.shape[2], X.shape[-1]))
    sigma_2 = sigma_2.reshape((Y.shape[0], Y.shape[1], Y.shape[2]))

    return B, sigma_2, df


def t_test_3d(c, X, B, sigma_2, df):
    """ Two-tailed t-test on 3D estimates given contrast `c`, design `X`

    Parameters
    ----------
    c : array shape (P,)
        contrast specifying conbination of parameters to test.
    X : array shape (N, P)
        design matrix.
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.

    Returns
    -------
    t : array shape (I, J, K)
        t statistics for each data vector.
    p : array shape (V,)
        two-tailed probability value for each t statistic.
    """
    # Your code code here

    n_voxels = B.shape[0]*B.shape[1]*B.shape[2]
    B_2D = B.reshape(n_voxels, X.shape[-1])
    B_2D = B_2D.T
    s_2_1D = sigma_2.reshape(n_voxels)

    t, p = t_test(c, X, B_2D, s_2_1D, df)

    t_3D = t.reshape((B.shape[0], B.shape[1], B.shape[2]))

    return t_3D, p
