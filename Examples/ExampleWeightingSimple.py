from sklearn.datasets import load_breast_cancer
from EvoPreprocess.data_weighting import EvoWeighting

if __name__ == '__main__':
    # Load classification data
    dataset = load_breast_cancer()

    # Get weights for the instances
    instance_weights = EvoWeighting().reweight(dataset.data, dataset.target)

    # Print the weights for instances
    print(instance_weights)
