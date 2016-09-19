#
# Copyright (c) 2016 KAMADA Ken'ichi.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#

"""FIML estimation of the mean/covariance of data with missing values.

This is an implementation of full information maximum likelihood (FIML)
method to estimate the mean and the covariance of data with missing
values.
"""

import numpy as np
import scipy as sp
import scipy.optimize

_log2pi = np.log(2 * np.pi)

def fiml(data, bias=False):
    """FIML estimation of the mean/covariance of data with missing values.

    Estimate the mean and the covariance of data with missing values by
    full information maximum likelihood (FIML) method.

    Parameters
    ----------
    data : ndarray
        A 2-D array containing variables and observations.
        Each row is an observation and each column is a variable.
        A missing value is represented by `np.nan`.
    bias : bool, optional
        Must be True for now.

    Returns
    -------
    mean : ndarray
        Estimated means of the variables.
    cov : ndarray
        Estimated covariance of the variables.
    """

    if not bias:
        raise NotImplementedError("unbiased estimator is not yet implemented")

    size, dim = data.shape
    mean0 = np.zeros(dim)
    cov0 = np.eye(dim)
    params0 = _pack_params(dim, mean0, cov0)
    data_blocks = _sort_missing(data)
    result = sp.optimize.fmin_slsqp(
        _obj_func, params0, args=(dim, data_blocks), disp=False)
    mean, cov = _unpack_params(dim, result)
    return mean, cov

# Sort data by the missing patterns.
# The return value is in the following format.
# Missing variables (columns) are removed from the data blocks.
#  [(observation_pattern1, data_block1),
#   (observation_pattern2, data_block2),
#   ...]
def _sort_missing(data):
    # Convert them to lists so that it can be sorted by the standard
    # comparator.
    obsmap = ~np.isnan(data)
    obsmap_list = map(list, obsmap)
    # argsort.
    sortedidx = sorted(range(data.shape[0]), key=obsmap_list.__getitem__)
    # Split row indexes into blocks.
    blocks = [[sortedidx[0]]]
    for idx, prev in zip(sortedidx[1:], sortedidx[:-1]):
        if (obsmap[prev] == obsmap[idx]).all():
            blocks[-1].append(idx)
        else:
            blocks.append([idx])
    return [(obsmap[b[0]], data[b][:, obsmap[b[0]]]) for b in blocks]

# Pack the mean and the covariance into a 1-dimensional array.
def _pack_params(dim, mean, cov):
    params = np.zeros(dim + dim * (dim + 1) / 2)
    params[:dim] = mean
    for p, i, j in zip(range(dim * (dim + 1) / 2), *np.tril_indices(dim)):
        params[dim + p] = cov[i, j]
    return params

# Unpack the mean and the covariance from a 1-dimensional array.
def _unpack_params(dim, params):
    mean = params[0:dim]
    cov = np.zeros((dim, dim))
    for v, i, j in zip(params[dim:], *np.tril_indices(dim)):
        cov[i, j] = v
        cov[j, i] = v
    return mean, cov

def _obj_func(params, dim, data_blocks):
    mean, cov = _unpack_params(dim, params)
    # Check if cov is positive semidefinite.
    # A matrix has a Cholesky decomposition iff it is symmetric and
    # positive semidefinite.  It is said that Cholesky decomposition is
    # faster and more numerically stable than finding eigenvalues.
    # However, numpy.linalg.cholesky() rejects singular matrices (i.e.,
    # strictly "semi"-definite ones).
    # try:
    #     _ = np.linalg.cholesky(cov)
    # except np.linalg.LinAlgError:
    #     return np.inf
    if (np.linalg.eigvalsh(cov) < 0).any():
        # XXX Returning inf is not a good idea, because many solvers
        # cannot cope with it.
        return np.inf
    objval = 0.0
    for obs, obs_data in data_blocks:
        obs_mean = mean[obs]
        obs_cov = cov[obs][:, obs]
        objval += _log_likelihood_composed(obs_data, obs_mean, obs_cov)
    return -objval

def _obj_func_1d(params, dim, data):
    mean, cov = _unpack_params(dim, params)
    objval = 0.0
    for x in data:
        obs = ~np.isnan(x)
        objval += _log_likelihood_1d(x[obs], mean[obs], cov[obs][:, obs])
    return -objval

# Composite function of _log_likelihood() and _pdf_normal().
def _log_likelihood_composed(x, mean, cov):
    xshift = x - mean
    t1 = x.shape[-1] * _log2pi
    sign, logdet = np.linalg.slogdet(cov)
    t2 = logdet
    t3 = xshift.dot(np.linalg.inv(cov)) * xshift
    size = x.shape[0] if x.ndim == 2 else 1
    return -0.5 * ((t1 + t2) * size + t3.sum())

# Log likelihood function.
# The input x can be one- or two-dimensional.
def _log_likelihood(x, mean, cov):
    return np.log(_pdf_normal(x, mean, cov)).sum()

# Log likelihood function.
def _log_likelihood_1d(x, mean, cov):
    return np.log(_pdf_normal_1d(x, mean, cov))

# Probability density function of multivariate normal distribution.
# The input x can be one- or two-dimensional.
def _pdf_normal(x, mean, cov):
    xshift = x - mean
    t1 = (2 * np.pi) ** (-0.5 * x.shape[-1])
    t2 = np.linalg.det(cov) ** (-0.5)
    t3 = -0.5 * (xshift.dot(np.linalg.inv(cov)) * xshift).sum(axis=-1)
    return t1 * t2 * np.exp(t3)

# Probability density function of multivariate normal distribution.
def _pdf_normal_1d(x, mean, cov):
    xshift = x - mean
    t1 = (2 * np.pi) ** (-0.5 * len(x))
    t2 = np.linalg.det(cov) ** (-0.5)
    t3 = -0.5 * xshift.dot(np.linalg.inv(cov)).dot(xshift)
    return t1 * t2 * np.exp(t3)
