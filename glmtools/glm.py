""" Functions for running GLM on 2D and 3D data
"""

import numpy as np
import numpy.linalg as npl
import scipy.stats as stats


def glm(Y, X):
    """ Run GLM on on data `Y` and design `X`

    Parameters
    ----------
    Y : array shape (N, V)
        1D or 2D array to fit to model with design `X`.  `Y` is column
        concatenation of V data vectors.
    X : array ahape (N, P)
        2D design matrix to fit to data `Y`.

    Returns
    -------
    B : array shape (P, V)
        parameter matrix, one column for each column in `Y`.
    sigma_2 : array shape (V,)
        unbiased estimate of variance for each column of `Y`.
    df : int
        degrees of freedom due to error.
    """
    # +++your code here+++
    B = npl.pinv(X).dot(Y)
    E = Y - X.dot(B) #residuals (Y - fitted data)
    n = Y.shape[0] #number of observations per column
    df = n - npl.matrix_rank(X) #df_error = n - p
    sigma_2 = np.sum(E ** 2, axis = 0) / df

    return B, sigma_2, df

def t_test(c, X, B, sigma_2, df):
    """ Two-tailed t-test given contrast `c`, design `X`

    Parameters
    ----------
    c : array shape (P,)
        contrast specifying conbination of parameters to test.
    X : array shape (N, P)
        design matrix.
    B : array shape (P, V)
        parameter estimates for V vectors of data.
    sigma_2 : float
        estimate for residual variance.
    df : int
        degrees of freedom due to error.

    Returns
    -------
    t : array shape (V,)
        t statistics for each data vector.
    p : array shape (V,)
        two-tailed probability value for each t statistic.
    """
    # Your code code here
    n = X.shape[0] #number of observations in each column of X

    c_b_cov = c.dot(npl.pinv(X.T.dot(X))).dot(c) #from in-class exercise on Monday
    t = c.dot(B) / np.sqrt(sigma_2 * c_b_cov)

    t_dist = stats.t(df=df)
    p_value = 1 - t_dist.cdf(abs(t)) #have to use absolute value to account for different possible contrasts
    p = 2*p_value #two-tailed probability value

    return t, p
