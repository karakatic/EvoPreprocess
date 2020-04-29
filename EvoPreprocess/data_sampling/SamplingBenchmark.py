"""
Helper class which evaluates the sampling and transforms the genotype of evolutionary and nature inspired algorithms
from NiaPy to actual dataset.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: GNU General Public License v3.0

import math
import random

import numpy as np
import pandas as pd
from NiaPy.benchmarks import Benchmark
from sklearn.base import ClassifierMixin
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import safe_indexing


class SamplingBenchmark(Benchmark):
    """
    Helper benchmark class for sampling data.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Matrix containing the data which have to be sampled.

    y : array-like, shape (n_samples)
        Corresponding label for each instance in X.

    train_indices : array-like, shape (n_samples)
        Corresponding indices for training instances from X.

    valid_indices : array-like, shape (n_samples)
        Corresponding indices for validation instances from X.

    random_seed : int or None, optional (default=1234)
        It used as seed for the random number generator.

    evaluator : classifier or regressor, optional (default=None)
        The classification or regression object from scikit-learn framework.
        If None, the GausianNB for classification is used.
    """

    # _________________0____1_____2______3_______4___
    mapping = np.array([0.5, 0.75, 0.875, 0.9375, 1])

    def __init__(self,
                 X, y,
                 train_indices=None, valid_indices=None,
                 random_seed=1234,
                 evaluator=None):
        self.Lower = 0
        self.Upper = 1
        super().__init__(self.Lower, self.Upper)

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X_train, self.X_valid = X[train_indices, :], X[valid_indices, :]
        self.y_train, self.y_valid = y[train_indices], y[valid_indices]

        self.evaluator = GaussianNB() if evaluator is None else evaluator
        self.evaluator.random_state = random_seed
        self.metric = f1_score if issubclass(type(self.evaluator), ClassifierMixin) else mean_squared_error

        self.random_seed = random_seed
        random.seed(random_seed)

    def function(self):
        def evaluate(D, sol):
            phenotype = SamplingBenchmark.map_to_phenotype(SamplingBenchmark.to_phenotype(sol))
            X_sampled = safe_indexing(self.X_train, phenotype)
            y_sampled = safe_indexing(self.y_train, phenotype)

            if X_sampled.shape[0] > 0:  # Check if no instances were selected
                cls = self.evaluator.fit(X_sampled, y_sampled)
                y_predicted = cls.predict(self.X_valid)
                acc = self.metric(self.y_valid, y_predicted)
                # used_percentage = len(y_sampled) / len(sol)

                # Check if classifier or regressor
                acc = (1 - acc) if issubclass(type(self.evaluator), ClassifierMixin) else acc
                # return acc + used_percentage
                return acc
            else:
                return math.inf

        return evaluate

    @staticmethod
    def to_phenotype(genotype):
        setting = np.cumsum(genotype[-5:])
        appearances = genotype[:-5]

        return np.digitize(appearances, setting / setting[-1])

    @staticmethod
    def genotype_to_map(genotype):
        return np.digitize(genotype, SamplingBenchmark.mapping)

    @staticmethod
    def map_to_phenotype(mapping):
        return np.repeat(range(len(mapping)), mapping)


if __name__ == '__main__':
    gene = np.array([0.123, 0.57, 0, 0.78, 1, 0.7, 0.4, 0.5])
    print(gene)
    phenotype = SamplingBenchmark.to_phenotype(gene)
    print(phenotype)
