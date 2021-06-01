import niapy.algorithms.basic as nia
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor

from evopreprocess1.data_sampling import EvoSampling

if __name__ == '__main__':
    # Load regression data
    dataset = load_boston()

    # Print the size of dataset
    print(dataset.data.shape, len(dataset.target))

    # Sample instances of dataset with custom settings and regression with EvoSampling
    X_resampled, y_resampled = EvoSampling(
        evaluator=DecisionTreeRegressor(),
        optimizer=nia.EvolutionStrategyMpL,
        n_folds=5,
        n_runs=5,
        n_jobs=4
    ).fit_resample(dataset.data, dataset.target)

    # Print the size of dataset after sampling
    print(X_resampled.shape, len(y_resampled))
