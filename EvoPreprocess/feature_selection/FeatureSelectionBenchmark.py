"""
Helper class which evaluates the feature selection and transforms the genotype of evolutionary and nature inspired
algorithms from NiaPy to actual dataset.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: MIT

import math
import random

import numpy as np
from NiaPy.benchmarks import Benchmark
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB


class FeatureSelectionBenchmark(Benchmark):
    """
    Helper benchmark class for feature selection.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Matrix containing the data which have to have features selected.

    y : array-like, shape (n_samples)
        Corresponding target values for each instance in X.

    train_indices : array-like, shape (n_samples)
        Corresponding indices for training instances from X.

    valid_indices : array-like, shape (n_samples)
        Corresponding indices for validation instances from X.

    random_seed : int or None, optional (default=1234)
        It used as seed by the random number generator.

    evaluator : classifier or regressor, optional (default=None)
        The classification or regression object from scikit-learn framework.
        If None, the GausianNB for classification is used.
    """

    # _________________0_|_1
    mapping = np.array([0.5])

    def __init__(self,
                 X, y,
                 train_indices=None, valid_indices=None,
                 random_seed=1234,
                 evaluator=None):
        self.Lower = 0
        self.Upper = 1
        super().__init__(self.Lower, self.Upper)

        self.X_train, self.X_valid = X[train_indices], X[valid_indices]
        self.y_train, self.y_valid = y[train_indices], y[valid_indices]

        self.evaluator = GaussianNB() if evaluator is None else evaluator
        self.evaluator.random_state = random_seed
        self.metric = accuracy_score if self.evaluator is ClassifierMixin else mean_squared_error

        self.random_seed = random_seed
        random.seed(random_seed)

    def function(self):
        def evaluate(D, sol):
            phenotype = FeatureSelectionBenchmark.to_phenotype(sol)
            X_train_new = self.X_train[:, phenotype]
            X_valid_new = self.X_valid[:, phenotype]

            if X_train_new.shape[1] > 0:  # Check if no features were selected
                cls = self.evaluator.fit(X_train_new, self.y_train)
                y_predicted = cls.predict(X_valid_new)
                acc = self.metric(self.y_valid, y_predicted)
                # used_percentage = X_train_new.shape[1] / len(sol)

                # Check if classifier or regressor
                acc = (1 - acc) if self.evaluator is ClassifierMixin else acc
                return acc
            else:
                return math.inf

        return evaluate

    @staticmethod
    def to_phenotype(genotype):
        return FeatureSelectionBenchmark.map_to_phenotype(FeatureSelectionBenchmark.genotype_to_map(genotype))

    @staticmethod
    def genotype_to_map(genotype):
        return np.digitize(genotype, FeatureSelectionBenchmark.mapping)

    @staticmethod
    def map_to_phenotype(mapping):
        return np.where(mapping == 1)[0]


if __name__ == '__main__':
    gene = np.array([0.123, 0.57, 0, 0.78, 1])
    print(gene)
    phenotype = FeatureSelectionBenchmark.to_phenotype(gene)
    print(phenotype)
