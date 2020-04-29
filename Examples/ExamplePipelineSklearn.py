from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from EvoPreprocess.feature_selection import EvoFeatureSelection

if __name__ == '__main__':
    # Set the random seed for the reproducibility
    random_seed = 987

    # Load classification data
    dataset = load_boston()

    # Split the dataset to training and testing set
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                                                        test_size=0.33,
                                                        random_state=random_seed)

    # Train the decision tree model
    model = DecisionTreeRegressor(random_state=random_seed)
    model.fit(X_train, y_train)

    # Print the results: shape of the original dataset and the MSE of decision tree regressor on original data
    print(X_train.shape, mean_squared_error(y_test, model.predict(X_test)), sep=': ')

    # Make scikit-learn pipeline with feature selection and data sampling
    pipeline = Pipeline(steps=[
        ('feature_selection', EvoFeatureSelection(
            evaluator=LinearRegression(),
            n_folds=4,
            n_runs=8,
            random_seed=random_seed)),
        ('regressor', DecisionTreeRegressor(random_state=random_seed))])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Print the results: the MSE of the pipeline
    print(mean_squared_error(y_test, pipeline.predict(X_test)))
