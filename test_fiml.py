"""Tests for fiml module.
"""

import unittest

import fiml
import numpy as np

class TestFIML(unittest.TestCase):
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

    def assertClose(self, expected, actual):
        #self.assertTrue(np.allclose(expected, actual))
        if not np.allclose(expected, actual):
            raise AssertionError("{} != {}".format(expected, actual))

    def assertModestlyClose(self, expected, actual):
        # The default xtol of scipy.optimize.fmin is 1e-4.
        if not np.allclose(expected, actual, atol=1e-4):
            raise AssertionError("{} != {}".format(expected, actual))

if __name__ == "__main__":
    unittest.main()
