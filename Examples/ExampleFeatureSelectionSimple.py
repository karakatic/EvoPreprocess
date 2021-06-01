from sklearn.datasets import load_breast_cancer

from evopreprocess1.feature_selection import EvoFeatureSelection

if __name__ == '__main__':
    # Load classification data
    dataset = load_breast_cancer()

    # Print the size of dataset
    print(dataset.data.shape)

    # Run feature selection with EvoFeatureSelection
    X_new = EvoFeatureSelection().fit_transform(
                                        dataset.data,
                                        dataset.target)

    # Print the size of dataset after feature selection
    print(X_new.shape)
