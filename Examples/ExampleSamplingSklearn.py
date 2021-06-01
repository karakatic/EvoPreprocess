from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from evopreprocess.data_sampling import EvoSampling

if __name__ == '__main__':
    # Set the random seed for the reproducibility
    random_seed = 1111

    # Load classification data
    dataset = load_breast_cancer()

    # Split the dataset to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                                                        test_size=0.33,
                                                        random_state=random_seed)

    # Train the decision tree model
    cls = DecisionTreeClassifier(random_state=random_seed)
    cls.fit(X_train, y_train)

    # Print the results: shape of the original dataset and the accuracy of decision tree classifier on original data
    print(X_train.shape, accuracy_score(y_test, cls.predict(X_test)), sep=': ')

    # Sample the data with random_seed set
    evo = EvoSampling(n_folds=3, random_seed=random_seed)
    X_resampled, y_resampled = evo.fit_resample(X_train, y_train)

    # Fit the decision tree model
    cls.fit(X_resampled, y_resampled)

    # Print the results: shape of the original dataset and the accuracy of decision tree classifier on original data
    print(X_resampled.shape, accuracy_score(y_test, cls.predict(X_test)), sep=': ')
