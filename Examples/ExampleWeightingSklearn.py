from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from EvoPreprocess.data_weighting import EvoWeighting

if __name__ == '__main__':
    # Set the random seed for the reproducibility
    random_seed = 1234

    # Load classification data
    dataset = load_breast_cancer()

    # Split the dataset to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                                                        test_size=0.33,
                                                        random_state=random_seed)

    # Train the decision tree model with custom instance weights
    cls = DecisionTreeClassifier(random_state=random_seed)
    cls.fit(X_train, y_train)

    # Print the results: shape of the original dataset and the accuracy of decision tree classifier on original data
    print(X_train.shape, accuracy_score(y_test, cls.predict(X_test)), sep=': ')

    # Get weights for the instances
    instance_weights = EvoWeighting(random_seed=random_seed).reweight(X_train, y_train)

    # Fit the decision tree model
    cls.fit(X_train, y_train, sample_weight=instance_weights)

    # Print the results: shape of the original dataset and the accuracy of decision tree classifier on original data
    print(X_train.shape, accuracy_score(y_test, cls.predict(X_test)), sep=': ')
