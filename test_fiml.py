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
        cov1 = np.cov(data, rowvar=False)
        mean2, cov2 = fiml.fiml(data)
        self.assertModestlyClose(mean1, mean2)
        self.assertModestlyClose(cov1, cov2)

    def assertModestlyClose(self, expected, actual):
        # The default xtol of scipy.optimize.fmin is 1e-4.
        if not np.allclose(expected, actual, atol=1e-4):
            raise AssertionError("{} != {}".format(expected, actual))

if __name__ == "__main__":
    unittest.main()
