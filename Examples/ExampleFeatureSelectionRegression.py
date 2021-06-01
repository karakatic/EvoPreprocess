import niapy.algorithms.basic as nia
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor

from evopreprocess1.feature_selection import EvoFeatureSelection

if __name__ == '__main__':
    # Load regression data
    dataset = load_boston()

    # Print the size of dataset
    print(dataset.data.shape)

    # Run feature selection with custom settings and regression with EvoFeatureSelection
    X_new = EvoFeatureSelection(
        evaluator=DecisionTreeRegressor(max_depth=2),
        optimizer=nia.DifferentialEvolution,
        random_seed=1,
        n_runs=5,
        n_folds=5,
        n_jobs=4
    ).fit_transform(dataset.data, dataset.target)

    # Print the size of dataset after feature selection
    print(X_new.shape)
