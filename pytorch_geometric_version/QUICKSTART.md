# Quick Start Guide

Get up and running with PyTorch Geometric MPNN in 5 minutes!

## 🚀 Installation

```bash
# 1. Install PyTorch (check pytorch.org for your specific setup)
pip install torch torchvision

# 2. Install PyTorch Geometric
pip install torch-geometric

# 3. Install other dependencies
pip install numpy matplotlib seaborn tqdm jupyter

# Or install everything at once:
pip install -r requirements.txt
```

## 📖 Three Ways to Get Started

### Option 1: Interactive Tutorial (Recommended for Learning)

Perfect for understanding how MPNNs work:

```bash
jupyter notebook mpnn_tutorial.ipynb
```

The tutorial includes:
- 📚 Introduction to Graph Neural Networks
- 🏗️ Architecture explanation with diagrams
- 💻 Step-by-step code walkthrough
- 📊 Training and evaluation
- 📈 Visualization of results
- 🎯 Exercises to practice

**Time:** 30-60 minutes

### Option 2: Command Line Training (Quickest)

Train a model with a single command:

```bash
# Basic usage (trains on HOMO energy for 50 epochs)
python train.py

# Train on different property
python train.py --property lumo --epochs 100

# Customize architecture
python train.py --hidden-dim 128 --num-layers 4 --batch-size 128

# See all options
python train.py --help
```

**Time:** 2-3 minutes to start training

### Option 3: Custom Python Script

Use the model in your own code:

```python
import torch
from torch_geometric.loader import DataLoader
from mpnn_model import create_mpnn_model
from data_utils import load_qm9_pyg, create_data_splits

# 1. Load data
print("Loading data...")
dataset = load_qm9_pyg(root='./data', target_property='homo')

# 2. Create splits
train_ds, val_ds, test_ds = create_data_splits(dataset)

# 3. Create dataloaders
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# 4. Create model
model = create_mpnn_model(
    node_dim=11,      # QM9 node features
    edge_dim=4,       # QM9 edge features
    output_dim=1,     # Single property
    hidden_dim=64,    # Hidden dimension
    num_layers=3      # Message passing iterations
)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# 5. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    model.train()
    for batch in train_loader:
        output = model(batch)
        loss = criterion(output, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete!")
```

**Time:** 5 minutes to write and run

## 🎯 What to Predict?

The QM9 dataset has 19 molecular properties. Here are the most interesting ones:

```python
from data_utils import QM9Properties

# See all properties
QM9Properties.print_all_properties()

# Common choices:
# - 'homo': Energy of highest occupied molecular orbital (reactivity)
# - 'lumo': Energy of lowest unoccupied molecular orbital (reactivity)
# - 'gap': HOMO-LUMO gap (optical properties)
# - 'mu': Dipole moment (polarity)
# - 'alpha': Polarizability (response to electric field)
# - 'U0': Internal energy at 0K (stability)
```

## 🔧 Common Customizations

### Change Target Property

```python
dataset = load_qm9_pyg(target_property='lumo')  # LUMO energy
dataset = load_qm9_pyg(target_property='mu')    # Dipole moment
dataset = load_qm9_pyg(target_property='gap')   # HOMO-LUMO gap
```

### Adjust Model Size

```python
# Smaller, faster model
model = create_mpnn_model(..., hidden_dim=32, num_layers=2)

# Larger, more powerful model
model = create_mpnn_model(..., hidden_dim=128, num_layers=5)

# Multi-task: predict all properties
model = create_mpnn_model(..., output_dim=19)
```

### Change Aggregation

```python
from mpnn_model import MPNNLayer

# Sum aggregation (default)
layer = MPNNLayer(..., aggr='add')

# Mean aggregation
layer = MPNNLayer(..., aggr='mean')

# Max aggregation
layer = MPNNLayer(..., aggr='max')
```

## 📊 Expected Results

Training on HOMO energy (default settings):

| Metric | Value | Time |
|--------|-------|------|
| **MAE** | ~0.04 eV | - |
| **RMSE** | ~0.06 eV | - |
| **Training** | - | ~2-3 hours (GPU) |
| **Training** | - | ~8-10 hours (CPU) |

*GPU: NVIDIA RTX 3080 or similar*

## 🐛 Troubleshooting

### PyTorch Geometric Installation Issues

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# For Apple Silicon (M1/M2)
pip install torch torchvision torchaudio
pip install torch-geometric
```

### Out of Memory Error

```python
# Reduce batch size
python train.py --batch-size 32  # or 16

# Or reduce model size
python train.py --hidden-dim 32 --num-layers 2
```

### Slow Training

```python
# Use GPU if available
python train.py --device cuda

# Increase batch size (if you have memory)
python train.py --batch-size 128

# Reduce dataset size for testing
train_dataset = Subset(train_dataset, range(10000))  # First 10k samples
```

### Dataset Download Issues

The first run will download the QM9 dataset (~300 MB). If download fails:

```python
# Manually specify cache location
dataset = load_qm9_pyg(root='/path/to/writable/directory')
```

## 📚 Next Steps

After getting started:

1. **Understand the model**: Read through `mpnn_model.py` - it's well commented!
2. **Try the tutorial**: `jupyter notebook mpnn_tutorial.ipynb`
3. **Experiment**: Try different properties, architectures, hyperparameters
4. **Extend**: Add your own molecular datasets or model modifications
5. **Read the comparison**: See `COMPARISON.md` to understand improvements over original

## 💡 Tips

- Start with the **tutorial notebook** if you're new to GNNs
- Use **default settings** first to verify everything works
- **GPU training** is 3-4x faster than CPU
- Check `README.md` for detailed documentation
- Look at `COMPARISON.md` to understand design decisions

## 🎓 Learning Resources

Inside this implementation:
- `mpnn_tutorial.ipynb` - Complete interactive tutorial
- `README.md` - Comprehensive documentation
- `COMPARISON.md` - Design rationale
- Code comments - Every class and function explained

External resources:
- [Neural Message Passing for Quantum Chemistry (Paper)](https://arxiv.org/abs/1704.01212)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [QM9 Dataset Paper](https://www.nature.com/articles/sdata201422)

## ✅ Checklist

Before starting, make sure you have:

- [ ] Python 3.7+ installed
- [ ] PyTorch installed
- [ ] PyTorch Geometric installed
- [ ] CUDA (optional, for GPU)
- [ ] 4GB+ RAM (8GB+ recommended)
- [ ] ~2GB disk space for QM9 dataset

## 🤔 Still Stuck?

1. Read the error message carefully
2. Check `TROUBLESHOOTING` section above
3. Review the tutorial notebook
4. Make sure all dependencies are installed: `pip install -r requirements.txt`
5. Try with a smaller model first: `--hidden-dim 32 --num-layers 2`

## 🚀 Ready to Go!

Pick your starting point:

```bash
# For learning:
jupyter notebook mpnn_tutorial.ipynb

# For quick training:
python train.py

# For custom code:
python your_script.py
```

Happy coding! 🎉
