"""This module provides functionality for automatically tuning hyper parameters.
"""

from .searchspace import SearchSpace, Uniform, LogUniform, OneOf
from .searcher import RandomSearcher
