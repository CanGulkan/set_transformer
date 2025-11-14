# Getting Started with Set Transformer - Quick Guide

This is a quick-start guide to help you understand and use the Set Transformer repository in 5 minutes.

## What Is This?

**Set Transformer** is a neural network that processes **unordered sets** of data (where order doesn't matter). It uses attention mechanisms to let elements interact with each other.

**Example use cases:**
- 3D point clouds (classify objects from point sets)
- Clustering sets of points
- Finding patterns in unordered collections

---

## Installation

```bash
# Install dependencies
pip install torch>=1.0 matplotlib scipy tqdm

# Clone repository (if not already done)
git clone https://github.com/CanGulkan/set_transformer.git
cd set_transformer
```

---

## Quick Start: Run an Experiment

### Option 1: Maximum Value Regression (Easiest)

Open and run the Jupyter notebook:
```bash
jupyter notebook max_regression_demo.ipynb
```

This demonstrates finding the maximum value in a set of numbers.

---

### Option 2: Clustering Experiment

Train a Set Transformer to cluster data:

```bash
# Train with Set Transformer
python run.py --net=set_transformer

# Or train with simpler DeepSet baseline
python run.py --net=deepset
```

**What it does:**
- Takes sets of 2D points from multiple Gaussian distributions
- Learns to predict cluster centers and distributions
- Saves results in `results/set_transformer/` or `results/deepset/`

**Monitor training:**
- Check the console for training progress
- Log files are saved in the results directory

**Visualize results:**
```bash
python run.py --mode=plot --net=set_transformer
```

---

### Option 3: Point Cloud Classification (Advanced)

**⚠️ Requires ModelNet40 dataset** (not included - must obtain separately)

```bash
python main_pointcloud.py --batch_size 256 --num_pts 100
```

---

## Understanding the Code Structure

### Core Files

1. **`modules.py`** - Building blocks:
   - `MAB`: Multihead Attention Block
   - `SAB`: Set Attention Block (self-attention)
   - `ISAB`: Induced Set Attention Block (efficient version)
   - `PMA`: Pooling by Multihead Attention

2. **`models.py`** - Complete models:
   - `SetTransformer`: The main model
   - `DeepSet`: Baseline comparison

3. **`run.py`** - Training script for clustering experiment

---

## Using Set Transformer in Your Code

```python
import torch
from models import SetTransformer

# Create a Set Transformer
model = SetTransformer(
    dim_input=10,      # Input feature dimension
    num_outputs=1,     # Number of output vectors
    dim_output=5,      # Output dimension
    num_inds=32,       # Inducing points (for efficiency)
    dim_hidden=128,    # Hidden layer size
    num_heads=4        # Number of attention heads
)

# Process a batch of sets (variable size is OK!)
batch = torch.randn(16, 100, 10)  # 16 sets, 100 elements each, 10 features
output = model(batch)  # Shape: (16, 1, 5)
```

---

## Key Concepts (Simple Explanation)

### 1. Sets vs Sequences
- **Sequence**: `[a, b, c]` is different from `[c, a, b]` (order matters)
- **Set**: `{a, b, c}` is the same as `{c, a, b}` (order doesn't matter)

Set Transformer handles **sets** naturally.

### 2. Attention
Elements in the set can "look at" other elements to understand context, similar to how Transformers work in language models.

### 3. Inducing Points (ISAB)
To make computation faster, instead of every element attending to every other element (slow for large sets), elements attend to a small number of learned "inducing points" (fast!).

---

## Common Commands

### Training
```bash
# Basic training
python run.py --net=set_transformer

# Custom settings
python run.py --net=set_transformer --K=4 --num_steps=50000 --lr=1e-3
```

### Testing
```bash
python run.py --mode=test --net=set_transformer
```

### Visualization
```bash
python run.py --mode=plot --net=set_transformer
```

### Generate Benchmark
```bash
python run.py --mode=bench
```

---

## Project Structure at a Glance

```
set_transformer/
├── modules.py              # Core attention blocks (MAB, SAB, ISAB, PMA)
├── models.py               # SetTransformer & DeepSet models
├── run.py                  # Clustering experiment script
├── max_regression_demo.ipynb   # Interactive demo
├── main_pointcloud.py      # 3D point cloud experiment
└── README.md               # Basic info
```

---

## Where to Go Next?

1. **For detailed explanation**: Read `EXPLANATION.md`
2. **For the research paper**: http://proceedings.mlr.press/v97/lee19d.html
3. **For hands-on learning**: Run `max_regression_demo.ipynb`
4. **For custom tasks**: Check the example code above and modify `models.py`

---

## Need Help?

- **Understanding Set Transformer**: Read `EXPLANATION.md`
- **Code issues**: Check Python/PyTorch versions match requirements
- **CUDA errors**: Reduce batch size or use CPU
- **Dataset issues**: For point clouds, obtain ModelNet40 separately

---

## Tips

1. **Start simple**: Try the Jupyter notebook first
2. **GPU recommended**: Training is much faster with CUDA
3. **Experiment with hyperparameters**: Adjust `num_inds`, `dim_hidden`, `num_heads`
4. **Variable set sizes**: The model handles different sizes naturally

---

**You're ready to go! Start with the Jupyter notebook and explore from there. 🎯**
