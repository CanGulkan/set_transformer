# Set Transformer Repository - Complete Explanation

## Table of Contents
1. [What is This Repository?](#what-is-this-repository)
2. [Key Concepts](#key-concepts)
3. [Repository Structure](#repository-structure)
4. [How to Get Started](#how-to-get-started)
5. [Running Experiments](#running-experiments)
6. [Understanding the Code](#understanding-the-code)
7. [Advanced Usage](#advanced-usage)

---

## What is This Repository?

This is the **official PyTorch implementation** of the research paper ["Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"](http://proceedings.mlr.press/v97/lee19d.html) published at ICML 2019.

### The Problem It Solves

Many machine learning tasks work with **sets of data** where the order doesn't matter:
- **Multiple instance learning**: A bag of instances where you need to make predictions
- **3D shape recognition**: Point clouds representing 3D objects
- **Few-shot classification**: Small sets of example images

The challenge: Traditional neural networks care about input order, but sets are inherently **permutation-invariant** (order doesn't matter).

### The Solution

**Set Transformer** uses attention mechanisms to:
1. Model interactions among elements in a set
2. Maintain permutation invariance (changing order doesn't affect output)
3. Scale efficiently using **Inducing Point methods** (reduces complexity from O(n²) to O(n))

---

## Key Concepts

### 1. Permutation Invariance
If you have a set `{a, b, c}`, it should produce the same result as `{c, a, b}` or any other ordering.

### 2. Attention Mechanisms
The model uses attention to let elements in a set "look at" and learn from other elements, similar to Transformers in NLP.

### 3. Set Encoder-Decoder Architecture
- **Encoder**: Processes the input set and captures relationships
- **Decoder**: Produces the desired output (e.g., classification, clustering)

### 4. Key Building Blocks

#### MAB (Multihead Attention Block)
- The fundamental attention module
- Takes Query (Q) and Key-Value (K) inputs
- Returns attention-weighted features

#### SAB (Set Attention Block)
- Self-attention: `SAB(X) = MAB(X, X)`
- Elements attend to all other elements in the set

#### ISAB (Induced Set Attention Block)
- **Efficiency optimization**: Uses "inducing points" to reduce computation
- Instead of O(n²) complexity, achieves O(n) by using a fixed number of learned inducing points
- Key innovation for scaling to large sets

#### PMA (Pooling by Multihead Attention)
- Pools set information into a fixed-size output
- Uses learned "seed" vectors that attend to the entire set

---

## Repository Structure

```
set_transformer/
├── README.md                  # Basic project information
├── EXPLANATION.md            # This file - detailed explanation
├── LICENSE                   # MIT License
│
├── modules.py                # Core building blocks (MAB, SAB, ISAB, PMA)
├── models.py                 # Complete models (SetTransformer, DeepSet)
│
├── run.py                    # Amortized clustering experiment
├── mixture_of_mvns.py        # Mixture of Gaussians utilities
├── mvn_diag.py              # Diagonal multivariate normal distribution
├── plots.py                  # Visualization utilities
│
├── max_regression_demo.ipynb # Maximum value regression experiment
├── main_pointcloud.py        # Point cloud classification experiment
└── data_modelnet40.py        # Data loader for ModelNet40 dataset
```

---

## How to Get Started

### 1. Prerequisites

Install required packages:
```bash
pip install torch>=1.0 matplotlib scipy tqdm
```

**Requirements:**
- Python 3
- PyTorch >= 1.0
- matplotlib (for visualization)
- scipy (for scientific computing)
- tqdm (for progress bars)

### 2. Quick Test

Try the maximum value regression demo:
```bash
jupyter notebook max_regression_demo.ipynb
```

This notebook demonstrates a simple task: given a set of numbers, predict the maximum value.

---

## Running Experiments

The repository implements three experiments from the paper:

### Experiment 1: Maximum Value Regression (Section 5.1)

**Task**: Given a set of numbers, predict the maximum value.

**How to run**:
```bash
jupyter notebook max_regression_demo.ipynb
```

**What it demonstrates**: Basic set processing where the model learns to find the maximum without being explicitly programmed for it.

---

### Experiment 2: Amortized Clustering (Section 5.3)

**Task**: Cluster data points from a mixture of Gaussian distributions.

**How to run with Set Transformer**:
```bash
python run.py --net=set_transformer
```

**How to run with DeepSets baseline**:
```bash
python run.py --net=deepset
```

**Optional arguments**:
```bash
python run.py \
  --net=set_transformer \
  --B=10 \              # Batch size
  --N_min=300 \         # Minimum number of points
  --N_max=600 \         # Maximum number of points
  --K=4 \               # Number of clusters
  --lr=1e-3 \           # Learning rate
  --num_steps=50000     # Training steps
```

**What happens**:
1. The model receives a set of 2D points sampled from K Gaussian distributions
2. It learns to predict the parameters (means and variances) of these distributions
3. Results are saved in `results/set_transformer/` or `results/deepset/`

**Modes**:
- `--mode=bench`: Generate benchmark dataset
- `--mode=train`: Train the model (default)
- `--mode=test`: Evaluate trained model
- `--mode=plot`: Visualize clustering results

---

### Experiment 3: Point Cloud Classification (Section 5.5)

**Task**: Classify 3D objects from point clouds (ModelNet40 dataset).

**⚠️ Important**: You need to obtain the preprocessed dataset "ModelNet40_cloud.h5" separately (not included due to copyright).

**How to run**:
```bash
# For 100 points per cloud
python main_pointcloud.py --batch_size 256 --num_pts 100

# For 1000 points per cloud  
python main_pointcloud.py --batch_size 256 --num_pts 1000

# For 5000 points per cloud
python main_pointcloud.py --batch_size 256 --num_pts 5000
```

**Hardware requirements**: Multiple GPUs recommended (paper used 8 Tesla P40s)

---

## Understanding the Code

### Core Modules (`modules.py`)

#### 1. MAB (Multihead Attention Block)
```python
MAB(dim_Q, dim_K, dim_V, num_heads, ln=False)
```
- `dim_Q`: Dimension of query
- `dim_K`: Dimension of key
- `dim_V`: Dimension of value (output)
- `num_heads`: Number of attention heads
- `ln`: Whether to use Layer Normalization

**Usage**: Foundation for all other blocks

#### 2. SAB (Set Attention Block)
```python
SAB(dim_in, dim_out, num_heads, ln=False)
```
- Self-attention: queries and keys come from the same set
- Allows elements to interact with each other

**Usage**: Build deeper encoders with multiple SAB layers

#### 3. ISAB (Induced Set Attention Block)
```python
ISAB(dim_in, dim_out, num_heads, num_inds, ln=False)
```
- `num_inds`: Number of inducing points (typically 32)
- More efficient than SAB for large sets
- Uses learned inducing points to reduce complexity

**Usage**: Preferred over SAB for large sets (>100 elements)

#### 4. PMA (Pooling by Multihead Attention)
```python
PMA(dim, num_heads, num_seeds, ln=False)
```
- `num_seeds`: Number of seed vectors (determines output size)
- Pools variable-size set into fixed-size output

**Usage**: Decoder input or final pooling layer

---

### Complete Models (`models.py`)

#### 1. SetTransformer
```python
SetTransformer(dim_input, num_outputs, dim_output, 
               num_inds=32, dim_hidden=128, num_heads=4, ln=False)
```

**Architecture**:
```
Input (B, N, dim_input)
    ↓
Encoder:
  - ISAB(dim_input → dim_hidden)
  - ISAB(dim_hidden → dim_hidden)
    ↓
Decoder:
  - PMA(num_outputs seeds)
  - SAB(dim_hidden → dim_hidden)
  - SAB(dim_hidden → dim_hidden)
  - Linear(dim_hidden → dim_output)
    ↓
Output (B, num_outputs, dim_output)
```

**Parameters**:
- `dim_input`: Input feature dimension
- `num_outputs`: Number of output vectors
- `dim_output`: Output feature dimension
- `num_inds`: Number of inducing points (affects speed/accuracy tradeoff)
- `dim_hidden`: Hidden layer size
- `num_heads`: Number of attention heads

#### 2. DeepSet (Baseline)
```python
DeepSet(dim_input, num_outputs, dim_output, dim_hidden=128)
```

**Architecture**:
```
Input → Encoder (MLP) → Mean Pooling → Decoder (MLP) → Output
```

Simpler baseline that uses mean pooling instead of attention.

---

## Advanced Usage

### Custom Set Processing Task

Here's how to use Set Transformer for your own task:

```python
import torch
from models import SetTransformer

# Create model
model = SetTransformer(
    dim_input=10,      # Your input feature dimension
    num_outputs=1,     # Number of outputs (e.g., 1 for classification)
    dim_output=5,      # Output dimension (e.g., 5 classes)
    num_inds=32,       # Inducing points (tune for speed/accuracy)
    dim_hidden=128,    # Hidden dimension
    num_heads=4        # Attention heads
)

# Example: Process a batch of sets
batch_size = 16
set_size = 100  # Can be variable!
X = torch.randn(batch_size, set_size, 10)  # Random input

output = model(X)  # Shape: (16, 1, 5)
```

### Key Design Choices

1. **ISAB vs SAB**: 
   - Use ISAB for sets with >100 elements (faster)
   - Use SAB for smaller sets (potentially more accurate)

2. **Number of inducing points (`num_inds`)**: 
   - More points = more capacity but slower
   - Typical values: 16-64

3. **Number of attention heads**: 
   - More heads = more expressive but slower
   - Typical values: 4-8

4. **Hidden dimension**: 
   - Larger = more capacity
   - Typical values: 128-512

---

## Common Questions

### Q: What makes this different from regular Transformers?
**A**: Set Transformers are designed for sets (unordered), while regular Transformers use positional encodings because they process sequences (ordered).

### Q: When should I use Set Transformer vs DeepSet?
**A**: 
- **Set Transformer**: When element interactions matter (better performance, more complex)
- **DeepSet**: When you need a simple, fast baseline (just mean pooling)

### Q: How does ISAB improve efficiency?
**A**: Instead of every element attending to every other element (O(n²)), elements attend to a small set of learned inducing points (O(n×num_inds)), then aggregate back.

### Q: Can the input set size vary?
**A**: Yes! That's a key feature. The model handles variable-size inputs naturally.

### Q: What tasks can I use this for?
**A**: Any task with set-structured data:
- Point clouds
- Graph node sets
- Multiple instance learning
- Particle physics
- Set anomaly detection
- Many more!

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}
```

---

## Additional Resources

- **Paper**: http://proceedings.mlr.press/v97/lee19d.html
- **Related Work**: 
  - [Deep Sets](https://papers.nips.cc/paper/6931-deep-sets) (simpler baseline)
  - [Attention is All You Need](https://arxiv.org/abs/1706.03762) (original Transformer)
- **Applications**:
  - Point cloud processing (3D vision)
  - Graph neural networks
  - Multiple instance learning

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use ISAB with fewer inducing points

### Issue: Training is slow
**Solution**: 
- Use ISAB instead of SAB
- Reduce number of inducing points
- Reduce hidden dimension or number of heads

### Issue: Poor performance
**Solution**: 
- Increase hidden dimension
- Add more ISAB/SAB layers
- Increase number of inducing points
- Try layer normalization (`ln=True`)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy experimenting with Set Transformers! 🚀**
