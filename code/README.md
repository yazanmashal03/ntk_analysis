# Neural Tangent Kernel (NTK) and Physics-Informed Neural Networks (PINN) Analysis

This repository contains Jupyter notebooks for analyzing Neural Tangent Kernel (NTK) behavior and Physics-Informed Neural Networks (PINN) in both finite and infinite width regimes.

## Setup Instructions

### 1. Create and Activate Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n ntk_pinn python=3.10
conda activate ntk_pinn
```

### 2. Install Dependencies

```bash
# Navigate to the code directory
cd code

# Install required packages
pip install -r requirements.txt
```

## Notebook Descriptions

### NTK Analysis Notebooks

1. `infinite_ntk.ipynb`
   - Analyzes the Neural Tangent Kernel in the infinite width regime
   - Explores theoretical properties and behavior of NTK in the limit of infinite network width

2. `finite_width_analysis.ipynb`
   - Investigates NTK behavior in finite-width neural networks
   - Compares empirical results with theoretical predictions
   - Analyzes how network width affects NTK properties

### PINN Analysis Notebooks

1. `pinn_infinite.ipynb`
   - Studies Physics-Informed Neural Networks in the infinite width regime
   - Analyzes the relationship between PINNs and NTK theory
   - Explores theoretical guarantees and limitations

2. `pinn_finite_width_analysis.ipynb`
   - Examines PINN performance with finite-width networks
   - Investigates practical implementation considerations
   - Analyzes the impact of network architecture on PINN performance

### Supplementary Analysis 

- `/supplementary/full_training_analysis.ipynb`
  - Comprehensive analysis of training dynamics over multiple SGD steps
  - Compares different network architectures and training regimes
  - Visualizes and analyzes training metrics and convergence properties

## Requirements

The project uses the following main dependencies:
- JAX and Flax for neural network implementation
- Neural Tangents for NTK computations
- Pandas for data manipulation
- Matplotlib for visualization

See `requirements.txt` for specific version requirements.

## Note

The notebooks are designed to be run with Python 3.10 and the specified package versions to ensure reproducibility.
