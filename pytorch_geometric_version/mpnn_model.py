"""
Message Passing Neural Network (MPNN) for Molecular Property Prediction

This is a clean, modern implementation using PyTorch Geometric.
Based on "Neural Message Passing for Quantum Chemistry" by Gilmer et al. (2017)

Author: Refactored for clarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


class EdgeNetwork(nn.Module):
    """
    Edge Network: Transforms edge features into edge weights for message passing.

    This network learns to weight the importance of messages based on edge attributes
    (e.g., bond type, distance between atoms).
    """
    def __init__(self, edge_dim, hidden_dim):
        """
        Args:
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for the network
        """
        super(EdgeNetwork, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: Edge features [num_edges, edge_dim]
        Returns:
            Edge weights [num_edges, hidden_dim]
        """
        return self.edge_mlp(edge_attr)


class MPNNLayer(MessagePassing):
    """
    Single Message Passing Layer.

    This layer performs one iteration of message passing:
    1. Messages are computed from source nodes and edge features
    2. Messages are aggregated at each target node
    3. Node representations are updated using a GRU
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, aggr='add'):
        """
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for messages and updates
            aggr: Aggregation method ('add', 'mean', 'max')
        """
        super(MPNNLayer, self).__init__(aggr=aggr)

        # Edge network: transforms edge features
        self.edge_network = EdgeNetwork(edge_dim, hidden_dim)

        # Message network: combines source node and edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update network: GRU for updating node states
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Project input nodes to hidden dimension if needed
        self.node_projection = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Project nodes to hidden dimension
        h = self.node_projection(x)

        # Propagate messages
        out = self.propagate(edge_index, x=x, h=h, edge_attr=edge_attr)

        # Update node states with GRU
        h_new = self.gru(out, h)

        return h_new

    def message(self, x_j, edge_attr):
        """
        Construct messages from source nodes to target nodes.

        Args:
            x_j: Source node features [num_edges, node_dim]
            edge_attr: Edge features [num_edges, edge_dim]
        Returns:
            Messages [num_edges, hidden_dim]
        """
        # Transform edge features
        edge_weights = self.edge_network(edge_attr)

        # Concatenate source node features with edge weights
        combined = torch.cat([x_j, edge_weights], dim=-1)

        # Apply message network
        message = self.message_mlp(combined)

        return message


class Readout(nn.Module):
    """
    Readout Network: Aggregates node-level representations to graph-level.

    This network converts the node embeddings from all message passing layers
    into a single graph-level representation for property prediction.
    """
    def __init__(self, hidden_dim, output_dim, num_layers=3, pool_type='add'):
        """
        Args:
            hidden_dim: Dimension of node embeddings
            output_dim: Dimension of final prediction
            num_layers: Number of message passing layers (for concatenation)
            pool_type: Pooling method ('add' or 'mean')
        """
        super(Readout, self).__init__()

        self.pool_type = pool_type
        self.num_layers = num_layers

        # MLP to process concatenated layer representations
        # Input: concatenated representations from all layers
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node_representations_list, batch):
        """
        Args:
            node_representations_list: List of node embeddings from each layer
                                      Each element: [num_nodes, hidden_dim]
            batch: Batch assignment vector [num_nodes]
        Returns:
            Graph-level predictions [batch_size, output_dim]
        """
        # Pool each layer's representations
        pooled_layers = []
        for node_rep in node_representations_list:
            if self.pool_type == 'add':
                pooled = global_add_pool(node_rep, batch)
            else:
                pooled = global_mean_pool(node_rep, batch)
            pooled_layers.append(pooled)

        # Concatenate representations from all layers
        # This allows the model to use information from different depths
        graph_rep = torch.cat(pooled_layers, dim=-1)

        # Apply readout MLP to get final prediction
        out = self.readout_mlp(graph_rep)

        return out


class MPNN(nn.Module):
    """
    Message Passing Neural Network for Molecular Property Prediction.

    Architecture:
    1. Node feature embedding
    2. Multiple message passing layers
    3. Readout network for graph-level prediction

    This is a clean implementation using PyTorch Geometric that's easier to
    understand and extend compared to the original implementation.
    """
    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        hidden_dim=64,
        output_dim=1,
        num_layers=3,
        pool_type='add',
        aggr='add'
    ):
        """
        Args:
            node_input_dim: Dimension of input node features
            edge_input_dim: Dimension of input edge features
            hidden_dim: Hidden dimension for message passing
            output_dim: Dimension of final prediction
            num_layers: Number of message passing iterations
            pool_type: Graph pooling method ('add' or 'mean')
            aggr: Message aggregation method ('add', 'mean', 'max')
        """
        super(MPNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Initial node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MPNNLayer(
                node_dim=hidden_dim,
                edge_dim=edge_input_dim,
                hidden_dim=hidden_dim,
                aggr=aggr
            )
            for _ in range(num_layers)
        ])

        # Readout network
        self.readout = Readout(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            pool_type=pool_type
        )

    def forward(self, data):
        """
        Forward pass through the MPNN.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, node_input_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_input_dim]
                - batch: Batch assignment [num_nodes]
        Returns:
            Predictions [batch_size, output_dim]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Initial node embedding
        h = self.node_embedding(x)

        # Store representations from each layer for readout
        layer_representations = []

        # Message passing iterations
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_attr)
            layer_representations.append(h)

        # Readout to graph-level prediction
        out = self.readout(layer_representations, batch)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_dim={self.hidden_dim}, '
                f'num_layers={self.num_layers})')


def create_mpnn_model(node_dim, edge_dim, output_dim, hidden_dim=64, num_layers=3):
    """
    Convenience function to create an MPNN model.

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        output_dim: Number of properties to predict
        hidden_dim: Hidden dimension (default: 64)
        num_layers: Number of message passing layers (default: 3)

    Returns:
        MPNN model instance

    Example:
        >>> model = create_mpnn_model(node_dim=11, edge_dim=4, output_dim=12)
        >>> print(model)
    """
    return MPNN(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers
    )


if __name__ == "__main__":
    # Simple test
    print("Testing MPNN Model...")
    from torch_geometric.data import Data, Batch

    # Create a simple molecular graph
    # 4 atoms (nodes) with 3-dimensional features
    x = torch.randn(4, 11)  # 4 nodes, 11 features each

    # 4 bonds (edges, bidirectional = 8 directed edges)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],  # source nodes
        [1, 0, 2, 1, 3, 2, 0, 3]   # target nodes
    ], dtype=torch.long)

    # Edge features (e.g., bond type, distance)
    edge_attr = torch.randn(8, 4)  # 8 edges, 4 features each

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Create a batch with this single graph
    batch = Batch.from_data_list([data])

    # Create model
    model = create_mpnn_model(node_dim=11, edge_dim=4, output_dim=12, hidden_dim=64, num_layers=3)

    print(model)
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")

    # Forward pass
    with torch.no_grad():
        output = model(batch)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    print("\n✓ Model test passed!")
