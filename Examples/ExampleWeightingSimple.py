from sklearn.datasets import load_breast_cancer

from evopreprocess1.data_weighting import EvoWeighting

if __name__ == '__main__':
    # Load classification data
    dataset = load_breast_cancer()

    # Get weights for the instances
    instance_weights = EvoWeighting(random_seed=123).reweight(dataset.data, dataset.target)

    # Print the weights for instances
    print(instance_weights)
