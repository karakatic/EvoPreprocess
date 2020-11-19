import logging
import os

import setuptools

PACKAGE_NAME = 'EvoPreprocess'


def read_package_variable(key, filename='__init__.py'):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, filename)
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ', 2)
            if parts[:-1] == [key, '=']:
                return parts[-1].strip("'")
    logging.warning("'%s' not found in '%s'", key, module_path)
    return None


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name=read_package_variable('__project__'),
    version=read_package_variable('__version__'),
    author='Sašo Karakatič',
    author_email='karakatic@gmail.com',
    description='Data Preprocessing with Evolutionary and Nature Inspired Algorithms.',
    license='GPLv3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/karakatic/EvoPreprocess',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    install_requires=[
        'numpy>=1.8.2',
        'pandas',
        'scipy>=0.17',
        'scikit-learn>=0.19.0'
        'imbalanced-learn>=0.3.1'
        'NiaPy>=2.0.0rc5'
    ],
    keywords=[
        'Evolutionary Algorithms',
        'Nature Inspired Algorithms',
        'Data Sampling',
        'Instance Weighting',
        'Feature Selection',
        'Preprocessing',
        'Machine Learning'
    ]
)
