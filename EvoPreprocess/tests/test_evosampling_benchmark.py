import unittest

import numpy as np

from evopreprocess.data_sampling.SamplingBenchmark import SamplingBenchmark


class MyTestCase(unittest.TestCase):

    def test_phenotype(self):
        raw_gene = [0.123, 0.57, 0, 0.78, 1, 0.7, 0.4, 0.5]
        appearances_gene = [0.1, 0.2, 0.4, 0.6, 0.8]
        result = SamplingBenchmark.to_phenotype(np.concatenate([raw_gene, appearances_gene]))
        expected = np.array([1, 3, 0, 4, 5, 4, 3, 3])
        np.testing.assert_array_equal(result, expected)

    def test_phenotype_all_appearances_0(self):
        raw_gene = [0.123, 0.57, 0, 0.78, 1, 0.7, 0.4, 0.5]
        appearances_gene = [0, 0, 0, 0, 0]
        result = SamplingBenchmark.to_phenotype(np.concatenate([raw_gene, appearances_gene]))
        expected = np.array([5, 5, 5, 5, 5, 5, 5, 5])
        np.testing.assert_array_equal(result, expected)

    def test_map_to_phenotype(self):
        occurances = [2, 0, 1, 3, 0, 4, 0, 5]
        result = SamplingBenchmark.map_to_phenotype(occurances)
        expected = np.array([0, 0, 2, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 7])
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
