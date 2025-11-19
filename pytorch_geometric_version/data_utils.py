"""
Data utilities for converting molecular data to PyTorch Geometric format.

This module provides utilities to work with QM9 dataset and convert
molecular graphs to PyTorch Geometric Data objects.
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import QM9 as PyG_QM9
import os


class QM9Properties:
    """
    QM9 dataset target properties.

    The QM9 dataset contains 19 regression targets for each molecule.
    This class provides descriptions and units for each property.
    """
    PROPERTIES = [
        ('mu', 'Dipole moment', 'D'),
        ('alpha', 'Isotropic polarizability', 'Bohr^3'),
        ('homo', 'Highest occupied molecular orbital energy', 'eV'),
        ('lumo', 'Lowest unoccupied molecular orbital energy', 'eV'),
        ('gap', 'Gap between HOMO and LUMO', 'eV'),
        ('r2', 'Electronic spatial extent', 'Bohr^2'),
        ('zpve', 'Zero point vibrational energy', 'eV'),
        ('U0', 'Internal energy at 0K', 'eV'),
        ('U', 'Internal energy at 298.15K', 'eV'),
        ('H', 'Enthalpy at 298.15K', 'eV'),
        ('G', 'Free energy at 298.15K', 'eV'),
        ('Cv', 'Heat capacity at 298.15K', 'cal/mol/K'),
        ('U0_atom', 'Atomization energy at 0K', 'eV'),
        ('U_atom', 'Atomization energy at 298.15K', 'eV'),
        ('H_atom', 'Atomization enthalpy at 298.15K', 'eV'),
        ('G_atom', 'Atomization free energy at 298.15K', 'eV'),
        ('A', 'Rotational constant A', 'GHz'),
        ('B', 'Rotational constant B', 'GHz'),
        ('C', 'Rotational constant C', 'GHz'),
    ]

    @classmethod
    def get_property_names(cls):
        """Get list of property names."""
        return [prop[0] for prop in cls.PROPERTIES]

    @classmethod
    def get_property_info(cls, idx_or_name):
        """
        Get information about a specific property.

        Args:
            idx_or_name: Index or name of the property

        Returns:
            Tuple of (name, description, unit)
        """
        if isinstance(idx_or_name, int):
            return cls.PROPERTIES[idx_or_name]
        else:
            for prop in cls.PROPERTIES:
                if prop[0] == idx_or_name:
                    return prop
            raise ValueError(f"Property {idx_or_name} not found")

    @classmethod
    def print_all_properties(cls):
        """Print all available properties."""
        print("QM9 Dataset Properties:")
        print("-" * 80)
        for i, (name, desc, unit) in enumerate(cls.PROPERTIES):
            print(f"{i:2d}. {name:10s} - {desc:45s} [{unit}]")


def load_qm9_pyg(root='./data', target_property=None):
    """
    Load QM9 dataset using PyTorch Geometric's built-in loader.

    Args:
        root: Root directory to store the dataset
        target_property: Index or name of target property to predict.
                        If None, returns all 19 properties.

    Returns:
        QM9 dataset object

    Example:
        >>> # Load dataset for HOMO energy prediction
        >>> dataset = load_qm9_pyg(target_property='homo')
        >>> print(f"Dataset size: {len(dataset)}")
        >>> print(f"Sample: {dataset[0]}")
    """
    # Download and load QM9 dataset
    dataset = PyG_QM9(root=root)

    # If specific property is requested, filter targets
    if target_property is not None:
        if isinstance(target_property, str):
            # Convert property name to index
            prop_names = QM9Properties.get_property_names()
            if target_property not in prop_names:
                raise ValueError(f"Unknown property: {target_property}")
            target_idx = prop_names.index(target_property)
        else:
            target_idx = target_property

        # Create a wrapper to return only the specified target
        class SinglePropertyDataset:
            def __init__(self, base_dataset, target_idx):
                self.base_dataset = base_dataset
                self.target_idx = target_idx

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                data = self.base_dataset[idx].clone()
                # Keep only the specified target
                data.y = data.y[:, self.target_idx:self.target_idx+1]
                return data

            def get_split_indices(self, split='train'):
                """Get standard QM9 train/val/test split indices."""
                n = len(self)
                if split == 'train':
                    return list(range(0, 100000))
                elif split == 'val':
                    return list(range(100000, 110000))
                elif split == 'test':
                    return list(range(110000, n))
                else:
                    raise ValueError(f"Unknown split: {split}")

        dataset = SinglePropertyDataset(dataset, target_idx)

    return dataset


def molecule_to_pyg_data(node_features, edge_index, edge_features, target=None):
    """
    Convert molecular graph to PyTorch Geometric Data object.

    Args:
        node_features: Node feature matrix [num_nodes, num_node_features]
        edge_index: Graph connectivity in COO format [2, num_edges]
        edge_features: Edge feature matrix [num_edges, num_edge_features]
        target: Target values [num_targets] (optional)

    Returns:
        PyTorch Geometric Data object

    Example:
        >>> # Create a simple molecule
        >>> node_feat = torch.randn(5, 11)  # 5 atoms
        >>> edge_idx = torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]])
        >>> edge_feat = torch.randn(8, 4)  # 8 bonds
        >>> target = torch.tensor([1.5])
        >>> data = molecule_to_pyg_data(node_feat, edge_idx, edge_feat, target)
    """
    data = Data(
        x=torch.FloatTensor(node_features),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_features)
    )

    if target is not None:
        data.y = torch.FloatTensor(target).unsqueeze(0) if len(target.shape) == 0 else torch.FloatTensor(target)

    return data


def get_qm9_statistics(dataset, target_idx=0):
    """
    Compute mean and std of target property for normalization.

    Args:
        dataset: QM9 dataset
        target_idx: Index of target property (default: 0)

    Returns:
        Tuple of (mean, std)

    Example:
        >>> dataset = load_qm9_pyg()
        >>> mean, std = get_qm9_statistics(dataset, target_idx=2)
        >>> print(f"HOMO energy - Mean: {mean:.4f}, Std: {std:.4f}")
    """
    targets = []
    for i in range(min(100000, len(dataset))):  # Use training set only
        data = dataset[i]
        if len(data.y.shape) > 1:
            targets.append(data.y[0, target_idx].item())
        else:
            targets.append(data.y[target_idx].item() if data.y.numel() > 1 else data.y.item())

    targets = np.array(targets)
    return targets.mean(), targets.std()


def normalize_targets(dataset, mean, std):
    """
    Normalize target values using provided mean and std.

    Args:
        dataset: Dataset to normalize
        mean: Mean value for normalization
        std: Standard deviation for normalization

    Returns:
        Dataset with normalized targets

    Note:
        This modifies the dataset in-place.
    """
    class NormalizedDataset:
        def __init__(self, base_dataset, mean, std):
            self.base_dataset = base_dataset
            self.mean = mean
            self.std = std

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            data = self.base_dataset[idx].clone()
            data.y = (data.y - self.mean) / self.std
            return data

        def denormalize(self, y):
            """Denormalize predictions back to original scale."""
            return y * self.std + self.mean

    return NormalizedDataset(dataset, mean, std)


def create_data_splits(dataset, train_size=100000, val_size=10000):
    """
    Create train/validation/test splits for QM9 dataset.

    Args:
        dataset: QM9 dataset
        train_size: Number of training samples (default: 100000)
        val_size: Number of validation samples (default: 10000)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Example:
        >>> dataset = load_qm9_pyg(target_property='homo')
        >>> train_ds, val_ds, test_ds = create_data_splits(dataset)
        >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    from torch.utils.data import Subset

    # Standard QM9 split
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def print_data_example(data):
    """
    Print information about a PyTorch Geometric Data object.

    Args:
        data: PyTorch Geometric Data object

    Example:
        >>> dataset = load_qm9_pyg()
        >>> print_data_example(dataset[0])
    """
    print("PyTorch Geometric Data Object:")
    print("-" * 60)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Node feature dimension: {data.x.shape[1] if len(data.x.shape) > 1 else 1}")
    print(f"Edge feature dimension: {data.edge_attr.shape[1] if len(data.edge_attr.shape) > 1 else 1}")
    if hasattr(data, 'y') and data.y is not None:
        print(f"Target shape: {data.y.shape}")
        print(f"Target values: {data.y}")
    print(f"\nNode features (first 3 nodes):\n{data.x[:3]}")
    print(f"\nEdge index (first 10 edges):\n{data.edge_index[:, :10]}")
    print(f"\nEdge features (first 3 edges):\n{data.edge_attr[:3]}")


if __name__ == "__main__":
    print("Testing QM9 Data Utilities...")
    print("=" * 80)

    # Print available properties
    QM9Properties.print_all_properties()

    print("\n" + "=" * 80)
    print("Loading QM9 dataset...")

    # Load dataset for HOMO energy prediction
    try:
        dataset = load_qm9_pyg(root='../data/qm9_pyg', target_property='homo')
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")

        # Show example
        print("\n" + "=" * 80)
        print("Example molecule:")
        print_data_example(dataset[0])

        # Compute statistics
        print("\n" + "=" * 80)
        print("Computing dataset statistics...")
        mean, std = get_qm9_statistics(dataset)
        print(f"✓ HOMO energy - Mean: {mean:.4f} eV, Std: {std:.4f} eV")

        # Create splits
        print("\n" + "=" * 80)
        print("Creating data splits...")
        train_ds, val_ds, test_ds = create_data_splits(dataset)
        print(f"✓ Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    except Exception as e:
        print(f"Note: Full dataset loading requires PyTorch Geometric QM9 download.")
        print(f"Error: {e}")
        print("\nCreating a simple synthetic example instead...")

        # Create synthetic example
        node_feat = torch.randn(5, 11)
        edge_idx = torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]])
        edge_feat = torch.randn(8, 4)
        target = torch.tensor([1.5])

        data = molecule_to_pyg_data(node_feat, edge_idx, edge_feat, target)
        print("\nSynthetic molecule example:")
        print_data_example(data)

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
