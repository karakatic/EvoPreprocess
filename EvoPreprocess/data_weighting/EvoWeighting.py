"""
Class to perform instance weighting with evolutionary and nature inspired algorithms.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: MIT

import logging
import time

import numpy as np
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_random_state

from EvoPreprocess.data_sampling import EvoSampling
from EvoPreprocess.data_weighting.WeightingBenchmark import WeightingBenchmark

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


class EvoWeighting(object):
    """
    Wight instances from the dataset with evolutionary and nature-inspired methods.

    Parameters
    ----------
    random_seed : int or None, optional (default=None)
        It used as seed by the random number generator.
        If None, the current system time is used for the seed.

    evaluator : classifier or regressor, optional (default=None)
        The classification or regression object from scikit-learn framework.
        If None, the GausianNB for classification is used.

    optimizer : evolutionary or nature-inspired optimization method, optional (default=GeneticAlgorithm)
        The evolutionary or or nature-inspired optimization method from NiaPy framework.

    n_runs : int, optional (default=10)
        The number of runs on each fold. Only the best performing result of all runs is used.

    n_folds : int, optional (default=3)
        The number of folds for cross-validation split into the training and validation sets.

    benchmark : object, optional (default=WeightingBenchmark)
        The benchmark object with mapping and fitness value calculation.

    n_jobs : int, optional (default=None)
        The number of jobs to run in parallel.
        If None, then the number of jobs is set to the number of cores.
    """

    def __init__(self,
                 random_seed=None,
                 evaluator=None,
                 optimizer=GeneticAlgorithm,
                 n_runs=10,
                 n_folds=3,
                 benchmark=WeightingBenchmark,
                 n_jobs=None):

        self.evaluator = GaussianNB() if evaluator is None else evaluator
        self.random_seed = int(time.time()) if random_seed is None else random_seed
        self.random_state = check_random_state(self.random_seed)
        self.optimizer = optimizer
        self.n_runs = n_runs
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.benchmark = benchmark

    def reweight(self, X, y):
        """Reweight the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be weighted.

        y : array-like, shape (n_samples)
            Corresponding label for each instance in X.

        Returns
        -------
        weights : ndarray, shape (n_samples)
            The corresponding instance weights of `X` set.

        """

        weights = np.empty((len(y), self.n_folds))  # Columns are weights in one run
        weights.fill(np.nan)

        if self.evaluator is ClassifierMixin:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        else:
            skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        i = 0
        for train_index, val_index in skf.split(X, y):
            benchmark = self.benchmark(X=X, y=y,
                                       train_indices=train_index, valid_indices=val_index,
                                       random_seed=self.random_state)
            optimization = self.optimizer(**EvoSampling._get_args(self.optimizer, benchmark))

            best = optimization.run()
            weights[train_index, i] = best[0]
            i = i + 1

        weights = np.ma.masked_array(weights, np.isnan(weights))
        weights = np.mean(weights, axis=1)
        return weights
