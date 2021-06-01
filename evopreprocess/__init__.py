"""``evopreprocess``: Data preprocessing Python toolkit with evolutionary and nature inspired algorithms.
The main problems tackled by the toolkit are **data sampling** (over- and under-sampling data instances
and mixture of both), **instance weighting** and **feature selection**.

Subpackages
-----------
feature_selection
    Module for feature selection with evolutionary and nature inspired algorithms.
data_sampling
    Module for sampling datasets with evolutionary and nature inspired algorithms.
data_weighting
    Module for weighting instances with evolutionary and nature inspired algorithms.
"""

from . import data_sampling
from . import data_weighting
from . import feature_selection

name = 'evopreprocess'
__project__ = 'evopreprocess'
__version__ = '0.4.1'
__all__ = ['feature_selection', 'data_sampling', 'data_weighting']
