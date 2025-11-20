# QM9 MPNN Study: From Math to Code

**Goal:** Master Message Passing Neural Networks (MPNNs) for quantum chemistry through rigorous implementation and analysis.

## Project Structure

This is a pedagogical "from-scratch" implementation based on:
> **Neural Message Passing for Quantum Chemistry**
> Gilmer et al., ICML 2017
> https://arxiv.org/abs/1704.01212

---

## Curriculum (4 Notebooks)

### ✅ Step 1: Data Anatomy & The Message Passing Framework
**File:** `01_data_and_theory.ipynb`

- **Theory:** General MPNN framework (Message, Update, Readout functions)
- **Practice:** Load QM9, inspect data structures, visualize molecules, compute edge distances
- **Key Insight:** Establish the Math ↔ Code mapping (e.g., $h_v^t$ → `node_feats`)

---

### ✅ Step 2: The Baseline - Topological GCN
**File:** `02_topology_gcn.ipynb`

- **Hypothesis:** "Chemistry requires geometry. Topology alone will fail."
- **Theory:** Graph Convolutional Networks (GCN) - Kipf & Welling
  - LaTeX: $\mathbf{h}_i^{(l+1)} = \sigma(\sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{\deg(i)\deg(j)}} \mathbf{W} \mathbf{h}_j^{(l)})$
  - Critical Analysis: Missing $e_{vw}$ (edge features) compared to MPNN
- **Practice:** Implement using `torch_geometric.nn.GCNConv`
- **Training:** Full training loop with MSE loss, validation, and test evaluation
- **Visualization:** Loss curves + Prediction vs Ground Truth scatter plot
- **Expected Result:** High error (~0.5-1.0 eV MAE for HOMO-LUMO gap)

---

### ✅ Step 3: The Gilmer Model - Continuous Filter Convolution
**File:** `03_geometry_mpnn_pyg.ipynb`

- **Theory:** Edge-Conditioned Convolution (NNConv) - Continuous Filter concept
  - LaTeX: $\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \sum_{j} \text{MLP}(e_{ij}) \cdot \mathbf{h}_j^{(l)}$
  - Explains why RBF expansion helps (smooth distance encoding)
  - Contrast with GCN: Dynamic vs. static weight matrices
- **Practice:** Implement using `torch_geometric.nn.NNConv`
  - Edge network: MLP maps RBF features → weight matrices
  - GRU updates for stable training
  - Set2Set attention-based readout
- **Training:** Same hyperparameters as Step 2 for fair comparison
- **Visualization:** Direct GCN vs MPNN comparison plots
- **Expected Result:** Massive error reduction (16× improvement: 0.8 eV → 0.05 eV MAE)

---

### 🔲 Step 4: Under the Hood - Pure PyTorch Implementation
**File:** `04_mpnn_from_scratch.ipynb`

- **Goal:** Remove the PyG abstraction; implement message passing manually
- **Theory:** Batching strategies (padding vs. disjoint union)
- **Practice:** Dense tensors, distance matrices, manual aggregation
- **Verification:** Assert identical outputs to Step 3

---

## Engineering Standards

All notebooks follow:
- ✅ **Docstrings:** Google-style with Input/Output shapes
- ✅ **Type Hints:** `def forward(self, data: Data) -> torch.Tensor`
- ✅ **Logging:** Use `logger` instead of `print()`
- ✅ **Visualization:** Prediction vs. Ground Truth scatter plots

---

## Dependencies

```bash
pip install torch torch-geometric rdkit matplotlib
```

## Target Property

**Primary:** HOMO-LUMO gap ($\Delta \epsilon = \epsilon_{\text{LUMO}} - \epsilon_{\text{HOMO}}$)
**Alternative:** Internal energy at 0K ($U_0$)

---

**Status:** Step 1 completed. Awaiting user confirmation to proceed to Step 2.
