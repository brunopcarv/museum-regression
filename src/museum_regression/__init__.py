"""
Museum Population Regression Package

A package to correlate museum visitor attendance with city populations
using data from Wikipedia.
"""

__version__ = "0.1.0"

from museum_regression.data.wikipedia_client import WikipediaClient
from museum_regression.db.database import Database
from museum_regression.ml.regression import MuseumRegressionModel

__all__ = ["WikipediaClient", "Database", "MuseumRegressionModel"]
