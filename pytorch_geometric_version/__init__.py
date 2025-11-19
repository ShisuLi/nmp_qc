"""
PyTorch Geometric MPNN Implementation

A clean, modern implementation of Message Passing Neural Networks
for molecular property prediction.
"""

from .mpnn_model import MPNN, create_mpnn_model
from .data_utils import (
    load_qm9_pyg,
    QM9Properties,
    molecule_to_pyg_data,
    get_qm9_statistics,
    create_data_splits
)

__version__ = '1.0.0'
__author__ = 'Refactored for PyTorch Geometric'

__all__ = [
    'MPNN',
    'create_mpnn_model',
    'load_qm9_pyg',
    'QM9Properties',
    'molecule_to_pyg_data',
    'get_qm9_statistics',
    'create_data_splits'
]
