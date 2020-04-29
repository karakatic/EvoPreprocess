"""
Helper class which evaluates the feature selection and transforms the genotype of evolutionary and nature inspired
algorithms from NiaPy to actual dataset.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: GNU General Public License v3.0

import math
import random

import numpy as np
import pandas as pd
from NiaPy.benchmarks import Benchmark
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score
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
                 evaluator=None,
                 split=None):
        self.Lower = 0
        self.Upper = 1
        super().__init__(self.Lower, self.Upper)

        self.split = split
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
            phenotype = FeatureSelectionBenchmark.to_phenotype(sol, self.split)
            X_train_new = self.X_train[:, phenotype]
            X_valid_new = self.X_valid[:, phenotype]

            if X_train_new.shape[1] > 0:  # Check if no features were selected
                cls = self.evaluator.fit(X_train_new, self.y_train)
                y_predicted = cls.predict(X_valid_new)
                acc = self.metric(self.y_valid, y_predicted)
                # used_percentage = X_train_new.shape[1] / len(sol)

                # Check if classifier or regressor
                acc = (1 - acc) if issubclass(type(self.evaluator), ClassifierMixin) else acc
                return acc
            else:
                return math.inf

        return evaluate

    @staticmethod
    def to_phenotype(genotype, split=None):
        if split is None:
            s = genotype[-1] if split is None else split
            features = genotype[:-1]
        else:
            s = split
            features = genotype
        return features >= s

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
