import niapy.algorithms.basic as nia
from sklearn.datasets import load_breast_cancer

from evopreprocess.data_sampling import EvoSampling

if __name__ == '__main__':
    # Load classification data
    dataset = load_breast_cancer()

    # Print the size of dataset
    print(dataset.data.shape, len(dataset.target))

    settings = {'population_size': 1000,
                'loudness': 0.5,
                'pulse_rate': 0.5,
                'min_frequency': 0.0,
                'max_frequency': 2.0}

    # Sample instances of dataset with custom optimizer and its settings
    X_resampled, y_resampled = EvoSampling(optimizer=nia.BatAlgorithm,
                                           optimizer_settings=settings
                                           ).fit_resample(dataset.data, dataset.target)

    # Print the size of dataset after sampling
    print(X_resampled.shape, len(y_resampled))
