# Set Transformer Repository - Quick Summary

## What Is This? 🤔

A PyTorch implementation of **Set Transformer** - a neural network that processes **unordered sets** of data using attention mechanisms.

**Key Feature**: Order doesn't matter! `{a, b, c}` = `{c, a, b}` = `{b, a, c}`

---

## Repository at a Glance 📊

```
┌─────────────────────────────────────────────────────────┐
│                    SET TRANSFORMER                      │
│                                                         │
│  Input: Unordered Set {x₁, x₂, ..., xₙ}               │
│              ↓                                          │
│  Encoder (ISAB layers)  ← Efficient attention          │
│              ↓                                          │
│  Decoder (PMA + SAB)    ← Pooling + attention          │
│              ↓                                          │
│  Output: Fixed-size representation                      │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure 📁

| File | Purpose |
|------|---------|
| **`modules.py`** | Core building blocks (MAB, SAB, ISAB, PMA) |
| **`models.py`** | Complete models (SetTransformer, DeepSet) |
| **`run.py`** | Training script for clustering experiments |
| **`max_regression_demo.ipynb`** | Interactive demo notebook |
| **`main_pointcloud.py`** | 3D point cloud classification |
| **`GETTING_STARTED.md`** | ⭐ Start here! Quick guide |
| **`EXPLANATION.md`** | ⭐ Complete detailed guide |

---

## Quick Commands 🚀

### Run the Interactive Demo
```bash
jupyter notebook max_regression_demo.ipynb
```

### Train on Clustering Task
```bash
python run.py --net=set_transformer
```

### Visualize Results
```bash
python run.py --mode=plot --net=set_transformer
```

---

## Core Components 🧩

### 1. MAB (Multihead Attention Block)
The fundamental building block - implements attention mechanism.

### 2. SAB (Set Attention Block)
Self-attention - elements attend to each other.

### 3. ISAB (Induced Set Attention Block) ⚡
**Efficient version** - uses inducing points to reduce complexity from O(n²) to O(n).

### 4. PMA (Pooling by Multihead Attention)
Pools variable-size sets into fixed-size outputs.

---

## Three Experiments 🧪

### 1. Maximum Value Regression
**Task**: Find the max value in a set  
**Difficulty**: ⭐ Easy  
**Run**: Open `max_regression_demo.ipynb`

### 2. Amortized Clustering
**Task**: Cluster points from mixture of Gaussians  
**Difficulty**: ⭐⭐ Medium  
**Run**: `python run.py --net=set_transformer`

### 3. Point Cloud Classification
**Task**: Classify 3D objects from point clouds  
**Difficulty**: ⭐⭐⭐ Advanced (requires dataset)  
**Run**: `python main_pointcloud.py --batch_size 256 --num_pts 100`

---

## Key Advantages ✨

1. **Permutation Invariant**: Order doesn't matter
2. **Efficient**: Linear complexity with ISAB (vs quadratic)
3. **Expressive**: Attention captures element interactions
4. **Flexible**: Handles variable-size sets
5. **General**: Works for many set-based tasks

---

## Use Cases 💡

- 🎯 Point cloud processing (3D vision)
- 🎯 Graph node classification
- 🎯 Multiple instance learning
- 🎯 Set anomaly detection
- 🎯 Particle physics
- 🎯 Few-shot learning

---

## Performance Comparison 📈

```
Task: Clustering (from paper)

DeepSet Baseline:     -1.52 log likelihood
Set Transformer:      -1.48 log likelihood  ✓ Better!
```

Set Transformer achieves state-of-the-art results on multiple benchmarks.

---

## Installation ⚙️

```bash
pip install torch>=1.0 matplotlib scipy tqdm
```

---

## Quick Code Example 💻

```python
from models import SetTransformer
import torch

# Create model
model = SetTransformer(
    dim_input=10,
    num_outputs=1,
    dim_output=5
)

# Process a batch of sets
batch = torch.randn(16, 100, 10)  # 16 sets, 100 elements each
output = model(batch)  # Works regardless of ordering!
```

---

## Where to Go Next? 🗺️

1. **New user?** → Read `GETTING_STARTED.md`
2. **Want details?** → Read `EXPLANATION.md`
3. **Want to try it?** → Open `max_regression_demo.ipynb`
4. **Want the theory?** → Read the [paper](http://proceedings.mlr.press/v97/lee19d.html)

---

## Key Insight 💡

**Set Transformer = Transformer for Sets**

Just like Transformers revolutionized NLP by processing sequences with attention, Set Transformers process **unordered sets** with attention!

---

## Citation 📝

```bibtex
@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based 
           Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and 
            Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={ICML},
    year={2019}
}
```

---

**Happy exploring! 🎉**
