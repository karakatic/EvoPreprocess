from sklearn.datasets import load_breast_cancer
from EvoPreprocess.data_sampling import EvoSampling

if __name__ == '__main__':
    # Load classification data
    dataset = load_breast_cancer()

    # Print the size of dataset
    print(dataset.data.shape, len(dataset.target))

    # Sample instances of dataset with default settings with EvoSampling
    X_resampled, y_resampled = EvoSampling().fit_resample(dataset.data, dataset.target)

    # Print the size of dataset after sampling
    print(X_resampled.shape, len(y_resampled))
