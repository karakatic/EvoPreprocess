from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from evopreprocess.feature_selection import EvoFeatureSelection

if __name__ == '__main__':
    # Set the random seed for the reproducibility
    random_seed = 654

    # Load regression data
    dataset = load_boston()

    # Split the dataset to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                                                        test_size=0.33,
                                                        random_state=random_seed)

    # Train the decision tree model
    model = DecisionTreeRegressor(random_state=random_seed)
    model.fit(X_train, y_train)

    # Print the results: shape of the original dataset and the accuracy of decision tree regressor on original data
    print(X_train.shape, mean_squared_error(y_test, model.predict(X_test)), sep=': ')

    # Sample the data with random_seed set
    evo = EvoFeatureSelection(evaluator=model, random_seed=random_seed)
    X_train_new = evo.fit_transform(X_train, y_train)

    # Fit the decision tree model
    model.fit(X_train_new, y_train)

    # Keep only selected feature on test set
    X_test_new = evo.transform(X_test)

    # Print the results: shape of the original dataset and the MSE of decision tree regressor on original data
    print(X_train_new.shape, mean_squared_error(y_test, model.predict(X_test_new)), sep=': ')
