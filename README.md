# EvoPreprocess

EvoPreprocess is a Python toolkit for sampling datasets, instance weighting, and feature selection. It is compatible with [scikit-learn](http://scikit-learn.org/stable/) and [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/). It is based on [NiaPy](https://github.com/NiaOrg/NiaPy) library for the implementation of nature-inspired algorithms and is distributed under GNU General Public License v3.0 license.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Requirements
- Python 3.6+
- PIP

### Dependencies
EvoSampling requires:

- numpy
- scipy
- scikit-learn
- imbalanced-learn
- NiaPy(==2.0.0rc16)

### Installation
Install EvoPreprocess with pip:

```sh
$ pip install evopreprocess
```
To install EvoPreprocess on Fedora:

```sh
$ dnf install python-EvoPreprocess
```

Or directly from the source code:

```sh
$ git clone https://github.com/karakatic/EvoPreprocess.git
$ cd evopreprocess
$ python setup.py install
```

# Usage

After installation, the package can be imported:

```sh
$ python
>>> import evopreprocess
>>> evopreprocess.__version__
```

## Data sampling

### Simple data sampling example

```python
from sklearn.datasets import load_breast_cancer
from evopreprocess.data_sampling import EvoSampling

# Load classification data
dataset = load_breast_cancer()

# Print the size of dataset
print(dataset.data.shape, len(dataset.target))

# Sample instances of dataset with default settings with EvoSampling
X_resampled, y_resampled = EvoSampling().fit_resample(dataset.data, dataset.target)

# Print the size of dataset after sampling
print(X_resampled.shape, len(y_resampled))
```

### Data sampling for regression with custom nature-inspired algorithm and other custom settings

```python
import niapy.algorithms.basic as nia
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from evopreprocess.data_sampling import EvoSampling

# Load regression data
dataset = load_boston()

# Print the size of dataset
print(dataset.data.shape, len(dataset.target))

# Sample instances of dataset with custom settings and regression with EvoSampling
X_resampled, y_resampled = EvoSampling(
    evaluator=DecisionTreeRegressor(),
    optimizer=nia.EvolutionStrategyMpL,
    n_folds=5,
    n_runs=5,
    n_jobs=4
).fit_resample(dataset.data, dataset.target)

# Print the size of dataset after sampling
print(X_resampled.shape, len(y_resampled))
```

### Data sampling with scikit-learn

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from evopreprocess.data_sampling import EvoSampling

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
```

## Instance weighting

### Simple instance weighting example

```python
from sklearn.datasets import load_breast_cancer
from evopreprocess.data_weighting import EvoWeighting

# Load classification data
dataset = load_breast_cancer()

# Get weights for the instances
instance_weights = EvoWeighting(random_seed=123).reweight(dataset.data, dataset.target)

# Print the weights for instances
print(instance_weights)
```

### Instance weighting with scikit-learn

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from evopreprocess.data_weighting import EvoWeighting

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
```

## Feature selection

### Simple feature selection example

```python
from sklearn.datasets import load_breast_cancer
from evopreprocess.feature_selection import EvoFeatureSelection

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
```

### Feature selection for regression with custom nature-inspired algorithm and other custom settings

```python
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
import niapy.algorithms.basic as nia
from evopreprocess.feature_selection import EvoFeatureSelection

# Load regression data
dataset = load_boston()

# Print the size of dataset
print(dataset.data.shape)

# Run feature selection with custom settings and regression with EvoFeatureSelection
X_new = EvoFeatureSelection(
    evaluator=DecisionTreeRegressor(max_depth=2),
    optimizer=nia.DifferentialEvolution,
    random_seed=1,
    n_runs=5,
    n_folds=5,
    n_jobs=4
).fit_transform(dataset.data, dataset.target)

# Print the size of dataset after feature selection
print(X_new.shape)
```

### Feature selection with scikit-learn

```python
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from evopreprocess.feature_selection import EvoFeatureSelection

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
```

## EvoPreprocess as a part of the pipeline (from imbalanced-learn)

```python
from imblearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from evopreprocess.data_sampling import EvoSampling
from evopreprocess.feature_selection import EvoFeatureSelection

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
```

For more examples please look at **Examples** folder.

# Author

EvoPreprocess was programmed and is maintained by Saso Karakatic from University of Maribor.

# Citing

## Plain format
```
Karakatic, S., (2020).
EvoPreprocess - Data Preprocessing Framework with Nature-Inspired Optimization Algorithms.
Mathematics, 8(6), p.900. <https://doi.org/10.3390/math8060900>
```

## Bibtex format
```
@article{karakativc2020evopreprocess,
  title={EvoPreprocess—Data Preprocessing Framework with Nature-Inspired Optimization Algorithms},
  author={Karakati{\v{c}}, Sa{\v{s}}o},
  journal={Mathematics},
  volume={8},
  number={6},
  pages={900},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## License

This project is licensed under the GNU General Public License v3.0 License - see <http://www.opensource.org/licenses/GNU General Public License v3.0>.
