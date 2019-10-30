from NiaPy.algorithms import Algorithm
from numpy import apply_along_axis, math
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.utils import safe_indexing

from EvoPreprocess.data_sampling import EvoSampling
from EvoPreprocess.data_sampling.SamplingBenchmark \
    import SamplingBenchmark


class RandomSearch(Algorithm):
    Name = ['RandomSearch', 'RS']

    def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
        pop = task.Lower + self.Rand.rand(self.NP, task.D) * task.bRange
        fpop = apply_along_axis(task.eval, 1, pop)
        return pop, fpop, {}


class CustomSamplingBenchmark(SamplingBenchmark):
    def function(self):
        def evaluate(D, sol):
            phenotype = SamplingBenchmark.to_phenotype(sol)
            X_sampled = safe_indexing(self.X_train, phenotype)
            y_sampled = safe_indexing(self.y_train, phenotype)

            if X_sampled.shape[0] > 0:
                cls = self.evaluator.fit(X_sampled, y_sampled)
                y_predicted = cls.predict(self.X_valid)
                quality = self.metric(self.y_valid, y_predicted)
                size_percentage = len(y_sampled) / len(sol)

                if self.evaluator is ClassifierMixin:
                    return (1 - quality) * size_percentage
                else:
                    return quality * size_percentage
            else:
                return math.inf

        return evaluate


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
