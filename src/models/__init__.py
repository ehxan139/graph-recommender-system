"""
Graph-based Recommendation Models

This module provides production-ready implementations of state-of-the-art
graph neural network models for collaborative filtering.
"""

from .ngcf import NGCF, save_model as save_ngcf, load_model as load_ngcf
from .lightgcn import LightGCN, save_model as save_lightgcn, load_model as load_lightgcn

__all__ = [
    'NGCF',
    'LightGCN',
    'save_ngcf',
    'load_ngcf',
    'save_lightgcn',
    'load_lightgcn'
]
