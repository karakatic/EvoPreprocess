import numpy as np
from niapy.algorithms import Algorithm
from numpy import apply_along_axis, math
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.utils import _safe_indexing

from evopreprocess1.data_sampling import EvoSampling
from evopreprocess1.data_sampling.SamplingBenchmark import SamplingBenchmark


class RandomSearch(Algorithm):
    Name = ['RandomSearch', 'RS']

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        try:
            population = task.Lower + self.rng.rand(self.population_size, task.D) * task.bRange
            population_fitness = apply_along_axis(task.eval, 1, population)
            best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
            return population, population_fitness, best_x, best_fitness, {}
        except Exception as x:
            print(x)
            return None, None, None


class CustomSamplingBenchmark(SamplingBenchmark):
    # _________________0____1_____2______3_______4___
    mapping = np.array([0.5, 0.75, 0.875, 0.9375, 1])

    def function(self):
        def evaluate(D, sol):
            phenotype = SamplingBenchmark.map_to_phenotype(CustomSamplingBenchmark.to_phenotype(sol))
            X_sampled = _safe_indexing(self.X_train, phenotype)
            y_sampled = _safe_indexing(self.y_train, phenotype)

            if X_sampled.shape[0] > 0:
                cls = self.evaluator.fit(X_sampled, y_sampled)
                y_predicted = cls.predict(self.X_valid)
                quality = accuracy_score(self.y_valid, y_predicted)
                size_percentage = len(y_sampled) / len(sol)

                return (1 - quality) * size_percentage
            else:
                return math.inf

        return evaluate

    @staticmethod
    def to_phenotype(genotype):
        return np.digitize(genotype[:-5], CustomSamplingBenchmark.mapping)


if __name__ == '__main__':
    # Load classification data
    dataset = load_breast_cancer()

    # Print the size of dataset
    print(dataset.data.shape, len(dataset.target))

    # Sample instances of dataset with default settings with EvoSampling
    X_resampled, y_resampled = EvoSampling(optimizer=RandomSearch,
                                           benchmark=CustomSamplingBenchmark
                                           ).fit_resample(dataset.data, dataset.target)

    # Print the size of dataset after sampling
    print(X_resampled.shape, len(y_resampled))
