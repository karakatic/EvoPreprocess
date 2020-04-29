"""
Class to perform instance weighting with evolutionary and nature inspired algorithms.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: GNU General Public License v3.0

import logging
import sys
import time
from multiprocessing.pool import Pool

import numpy as np
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from NiaPy.task import OptimizationType, StoppingTask
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_random_state

import EvoPreprocess.utils.EvoSettings as es
from EvoPreprocess.data_weighting.WeightingBenchmark import WeightingBenchmark

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


class EvoWeighting(object):
    """
    Weight instances from the dataset with evolutionary and nature-inspired methods.

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

    optimizer_settings : dict, optional (default={})
        Custom settings for the optimizer.
    """

    def __init__(self,
                 random_seed=None,
                 evaluator=None,
                 optimizer=GeneticAlgorithm,
                 n_runs=10,
                 n_folds=2,
                 benchmark=WeightingBenchmark,
                 n_jobs=None,
                 optimizer_settings={}):

        self.evaluator = GaussianNB() if evaluator is None else evaluator
        self.random_seed = int(time.time()) if random_seed is None else random_seed
        self.random_state = check_random_state(self.random_seed)
        self.optimizer = optimizer
        self.n_runs = n_runs
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.benchmark = benchmark
        self.optimizer_settings = optimizer_settings

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

        mask = []
        evos = []  # Parameters for parallel threaded evolution run

        for train_index, val_index in skf.split(X, y):
            mask.append(train_index)
            for j in range(self.n_runs):
                evos.append(
                    (X, y, train_index, val_index, self.random_seed + j + 1, self.optimizer, self.evaluator,
                     self.benchmark, self.optimizer_settings))

        with Pool(processes=self.n_jobs) as pool:
            results = pool.starmap(EvoWeighting._run, evos)
        weights = EvoWeighting._reduce(mask, results, self.n_runs, self.n_folds, len(y))
        return weights

    @staticmethod
    def _run(X, y, train_index, val_index, random_seed, optimizer, evaluator, benchmark, optimizer_settings):
        opt_settings = es.get_args(optimizer)
        opt_settings.update(optimizer_settings)
        benchm = benchmark(X=X, y=y,
                           train_indices=train_index, valid_indices=val_index,
                           random_seed=random_seed,
                           evaluator=evaluator)
        task = StoppingTask(D=len(train_index) + 1,
                            nFES=opt_settings.pop('nFES', 1000),
                            optType=OptimizationType.MINIMIZATION,
                            benchmark=benchm)

        evo = optimizer(seed=random_seed, **opt_settings)
        r = evo.run(task=task)
        if isinstance(r[0], np.ndarray):
            return benchmark.to_phenotype(r[0], benchm.Upper), r[1]
        else:
            return benchmark.to_phenotype(r[0].x, benchm.Upper), r[1]

    @staticmethod
    def _reduce(mask, results, runs, cv, len_y=10):
        weights = np.full((len_y, cv), np.nan)  # Columns are number of occurrences in one run

        result_list = [results[x:x + runs] for x in range(0, cv * runs, runs)]
        i = 0
        for cv_one in result_list:
            best_fitness = sys.float_info.max
            best_solution = None
            for result_one in cv_one:
                if (best_solution is None) or (best_fitness > result_one[1]):
                    best_solution, best_fitness = result_one[0], result_one[1]

            weights[mask[i], i] = best_solution
            i = i + 1

        weights = np.nanmean(weights, axis=1)

        return weights
