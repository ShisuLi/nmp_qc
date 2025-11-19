"""
Simple training script for MPNN on QM9 dataset.

This script trains an MPNN model to predict molecular properties
from the QM9 dataset using PyTorch Geometric.

Usage:
    python train.py --property homo --epochs 50 --batch-size 64
"""

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import time

from mpnn_model import create_mpnn_model
from data_utils import (
    load_qm9_pyg,
    QM9Properties,
    get_qm9_statistics,
    create_data_splits
)


def train_epoch(model, loader, optimizer, criterion, device, mean, std):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)

        # Normalize targets
        targets = (batch.y - mean) / std

        # Forward pass
        optimizer.zero_grad()
        output = model(batch)

        # Compute loss
        loss = criterion(output, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, mean, std):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            batch = batch.to(device)

            # Normalize targets
            targets = (batch.y - mean) / std

            # Forward pass
            output = model(batch)

            # Compute loss
            loss = criterion(output, targets)
            total_loss += loss.item() * batch.num_graphs

            # Denormalize for MAE calculation
            pred_denorm = output * std + mean
            target_denorm = batch.y

            all_predictions.append(pred_denorm.cpu())
            all_targets.append(target_denorm.cpu())

    # Calculate metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    mae = torch.mean(torch.abs(all_predictions - all_targets)).item()

    return total_loss / len(loader.dataset), mae


def main():
    parser = argparse.ArgumentParser(description='Train MPNN on QM9 dataset')

    # Data arguments
    parser.add_argument('--data-root', type=str, default='../data/qm9_pyg',
                        help='Root directory for QM9 data')
    parser.add_argument('--property', type=str, default='homo',
                        choices=QM9Properties.get_property_names(),
                        help='Property to predict')

    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of message passing layers')
    parser.add_argument('--pool-type', type=str, default='add',
                        choices=['add', 'mean'],
                        help='Graph pooling type')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-path', type=str, default='mpnn_best.pth',
                        help='Path to save the best model')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading QM9 dataset for property: {args.property}")
    dataset = load_qm9_pyg(root=args.data_root, target_property=args.property)
    print(f"Total samples: {len(dataset)}")

    # Create splits
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Compute normalization statistics
    print("\nComputing normalization statistics...")
    mean, std = get_qm9_statistics(dataset, target_idx=0)
    mean, std = torch.tensor(mean).to(device), torch.tensor(std).to(device)
    print(f"Mean: {mean.item():.4f}, Std: {std.item():.4f}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Get feature dimensions
    sample = next(iter(train_loader))
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]

    print(f"\nNode features: {node_dim}, Edge features: {edge_dim}")

    # Create model
    print(f"\nCreating MPNN model...")
    model = create_mpnn_model(
        node_dim=node_dim,
        edge_dim=edge_dim,
        output_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, mean, std)

        # Validate
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, mean, std)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            epochs_no_improve = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'mean': mean.item(),
                'std': std.item(),
                'args': vars(args)
            }, args.save_path)

            indicator = '🌟'
        else:
            epochs_no_improve += 1
            indicator = ''

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Time: {epoch_time:.1f}s {indicator}")

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation MAE: {best_val_mae:.4f}")

    # Load best model and evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_mae = evaluate(model, test_loader, criterion, device, mean, std)
    print(f"\nTest Results:")
    print(f"  MSE Loss: {test_loss:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  RMSE: {np.sqrt(test_loss * std.item()**2):.4f}")

    print(f"\n✓ Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
