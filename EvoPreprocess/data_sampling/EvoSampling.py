"""
Class to perform sampling with evolutionary and nature inspired algorithms.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: GNU General Public License v3.0

import logging
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
from NiaPy.algorithms import Individual
from NiaPy.algorithms.basic.ga import GeneticAlgorithm
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.util import objects2array
from imblearn.base import BaseSampler
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_random_state, safe_indexing

from EvoPreprocess.data_sampling.SamplingBenchmark import SamplingBenchmark
from EvoPreprocess.utils import EvoSettings as es

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


class EvoSampling(BaseSampler):
    _sampling_type = 'clean-sampling'

    """
    Sample data with evolutionary and nature-inspired methods.

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

    benchmark : object, optional (default=SamplingBenchmark)
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
                 n_folds=2,
                 benchmark=SamplingBenchmark,
                 n_jobs=None,
                 optimizer_settings={}):
        super(EvoSampling, self).__init__()

        if optimizer_settings is None:
            optimizer_settings = {}

        self.evaluator = GaussianNB() if evaluator is None else evaluator
        self.random_seed = int(time.time()) if random_seed is None else random_seed
        self.random_state = check_random_state(self.random_seed)
        self.optimizer = optimizer
        self.n_runs = n_runs
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.benchmark = benchmark
        self.optimizer_settings = optimizer_settings

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)

    def _fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
        """

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
            results = pool.starmap(EvoSampling._run, evos)
        occurrences = EvoSampling._reduce(mask, results, self.n_runs, self.n_folds, self.benchmark, len(y))
        phenotype = self.benchmark.map_to_phenotype(occurrences)
        return (safe_indexing(X, phenotype),
                safe_indexing(y, phenotype))

    @staticmethod
    def _run(X, y, train_index, val_index, random_seed, optimizer, evaluator, benchmark, optimizer_settings):
        opt_settings = es.get_args(optimizer)
        opt_settings.update(optimizer_settings)
        benchm = benchmark(X=X, y=y,
                           train_indices=train_index, valid_indices=val_index,
                           random_seed=random_seed,
                           evaluator=evaluator)
        task = StoppingTask(D=len(train_index) + 5,
                            nFES=opt_settings.pop('nFES', 1000),
                            optType=OptimizationType.MINIMIZATION,
                            benchmark=benchm)

        # if evaluator is ClassifierMixin:
        #     evo = optimizer(seed=random_seed, InitPopFunc=EvoSampling.heuristicInit, **opt_settings)
        # else:
        evo = optimizer(seed=random_seed, **opt_settings)
        r = evo.run(task=task)
        if isinstance(r[0], np.ndarray):
            return benchmark.to_phenotype(r[0]), r[1]
        else:
            return benchmark.to_phenotype(r[0].x), r[1]

    @staticmethod
    def _reduce(mask, results, runs, cv, benchmark, len_y=10):
        occurrences = np.full((len_y, cv), np.nan)  # Columns are number of occurrences in one run

        result_list = [results[x:x + runs] for x in range(0, cv * runs, runs)]
        i = 0
        for cv_one in result_list:
            best_fitness = sys.float_info.max
            best_solution = None
            for result_one in cv_one:
                if (best_solution is None) or (best_fitness > result_one[1]):
                    best_solution, best_fitness = result_one[0], result_one[1]

            occurrences[mask[i], i] = best_solution
            i = i + 1

        # occurrences = stats.mode(occurrences, axis=1, nan_policy='omit')[0].flatten()
        occurrences = np.ceil(np.nanmean(occurrences, axis=1))

        return occurrences.astype(np.int8)

    @staticmethod
    def heuristicInit(task, NP, rnd, **kwargs):
        target_stats = Counter(task.benchmark.y_train)
        class_perc = 1 - np.array(list(target_stats.values())) / sum(target_stats.values())
        instance_perc = class_perc.take(task.benchmark.y_train)

        pop = np.random.normal(0, 0.25, (NP, task.D))
        pop = pop + instance_perc
        out_of_range = (pop < 0) | (pop > 1)
        while np.any(out_of_range):
            pop[out_of_range] = np.random.normal(0, 0.25, len(pop[out_of_range])) + np.tile(instance_perc, (NP, 1))[
                out_of_range]
            out_of_range = (pop < 0) | (pop > 1)

        pop = objects2array([Individual(task=task, rnd=rnd, e=True, x=pop[i]) for i in range(NP)])
        return pop, np.asarray([x.f for x in pop])
