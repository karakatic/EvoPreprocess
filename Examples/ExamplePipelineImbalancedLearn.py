from imblearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from evopreprocess.data_sampling import EvoSampling
from evopreprocess.feature_selection import EvoFeatureSelection

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

    # Make scikit-learn pipeline with feature selection and data sampling
    pipeline = Pipeline(steps=[
        ('feature_selection', EvoFeatureSelection(n_folds=10, random_seed=random_seed)),
        ('data_sampling', EvoSampling(n_folds=10, random_seed=random_seed)),
        ('classifier', DecisionTreeClassifier(random_state=random_seed))])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Print the results: the accuracy of the pipeline
    print(accuracy_score(y_test, pipeline.predict(X_test)))
