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

"""Tests for fiml module.
"""

import unittest

import fiml
import numpy as np

class TestFIML(unittest.TestCase):
    def test_sort_missing(self):
        data = np.array(
            [[1, 2, 3],
             [np.nan, 5, 6],
             [7, np.nan, np.nan],
             [np.nan, 11, 12],
             [13, np.nan, np.nan]])
        ans = [(np.array([False, True, True]),
                np.array([[5., 6.],
                          [11., 12.]])),
               (np.array([True, False, False]),
                np.array([[7.],
                          [13.]])),
               (np.array([True, True, True]),
                np.array([[1., 2., 3.]]))]
        data_blocks = fiml._sort_missing(data)
        self.assertNpSeqEqual(ans, data_blocks)

    def test_pack_params(self):
        dim = 5
        template = fiml._pack_params(dim, np.zeros(dim), np.eye(dim))
        params = np.random.randn(len(template))
        mean, cov = fiml._unpack_params(dim, params)
        self.assertNpEqual(cov, cov.T)
        params2 = fiml._pack_params(dim, mean, cov)
        self.assertClose(params, params2)

    def test_missing_2d(self):
        data = np.array(
            ((0, 0.4, 0.5, 0.6, 1),
             (0, 0.6, np.nan, 0.4, 1))).T
        ans_mean = np.array((0.5, 0.5))
        ans_cov = np.array(((0.104, 0.096), (0.096, 701.0 / 6500)))
        mean, cov = fiml.fiml(data, bias=True)
        self.assertModestlyClose(ans_mean, mean)
        self.assertModestlyClose(ans_cov, cov)

    def test_not_missing_1d(self):
        self._test_not_missing(100, 1)

    def test_not_missing_2d(self):
        self._test_not_missing(100, 2)

    def test_not_missing_3d(self):
        self._test_not_missing(100, 3)

    def _test_not_missing(self, size, dim):
        data = np.random.randn(size, dim)
        mean1 = data.mean(axis=0)
        cov1 = np.cov(data, rowvar=False, bias=True)
        mean2, cov2 = fiml.fiml(data, bias=True)
        self.assertModestlyClose(mean1, mean2)
        self.assertModestlyClose(cov1, cov2)

    # Test if _pdf_normal() and _log_likelihood() accepts
    # both a 2-D ndarray (multiple observations) and a 1-D ndarray.
    def test_1d_and_2d(self):
        for dim in range(2, 10):
            data = np.random.randn(dim * 2, dim)
            m = data.mean(axis=0)
            c = np.cov(data, rowvar=False)

            x = np.random.randn(dim)
            r1 = fiml._pdf_normal_1d(x, m, c)
            r2 = fiml._pdf_normal(x, m, c)
            self.assertClose(r1, r2)
            r1 = fiml._log_likelihood_1d(x, m, c)
            r2 = fiml._log_likelihood(x, m, c)
            r3 = fiml._log_likelihood_composed(x, m, c)
            self.assertClose(r1, r2)
            self.assertClose(r1, r3)

            xx = np.random.randn(3, dim)
            r1 = np.array([fiml._pdf_normal_1d(x, m, c) for x in xx])
            r2 = fiml._pdf_normal(xx, m, c)
            self.assertClose(r1, r2)
            r1 = sum([fiml._log_likelihood_1d(x, m, c) for x in xx])
            r2 = fiml._log_likelihood(xx, m, c)
            r3 = fiml._log_likelihood_composed(xx, m, c)
            self.assertClose(r1, r2)
            self.assertClose(r1, r3)

    def assertNpEqual(self, expected, actual):
        if (expected != actual).any():
            self.fail("{} != {}".format(expected, actual))

    def assertClose(self, expected, actual):
        #self.assertTrue(np.allclose(expected, actual))
        if not np.allclose(expected, actual):
            self.fail("{} != {}".format(expected, actual))

    def assertModestlyClose(self, expected, actual):
        # The default xtol of scipy.optimize.fmin is 1e-4.
        if not np.allclose(expected, actual, atol=1e-4):
            self.fail("{} != {}".format(expected, actual))

    def assertNpSeqEqual(self, expected, actual):
        def recursive(seq1, seq2):
            if type(seq1) is not type(seq2):
                return False
            if isinstance(seq1, np.ndarray):
                return np.array_equal(seq1, seq2)
            if isinstance(seq1, (list, tuple)):
                if len(seq1) != len(seq2):
                    return False
                for sub1, sub2 in zip(seq1, seq2):
                    if not recursive(sub1, sub2):
                        return False
                return True
            return seq1 == seq2
        if not recursive(expected, actual):
            self.fail("{} != {}".format(expected, actual))

if __name__ == "__main__":
    unittest.main()
