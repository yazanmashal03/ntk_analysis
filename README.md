# Neural Tangent Kernel (NTK) Analysis Project

This repository contains a comprehensive analysis of Neural Tangent Kernel (NTK) behavior in both finite and infinite width regimes, with a focus on Physics-Informed Neural Networks (PINNs). This project was completed for the course **WI4450: Special Topics in Computational Science and Engineering (2024/2025 Q3–Q4)**.

## 🎯 Project Overview

This project investigates the scaling properties of Neural Tangent Kernels in finite-width and finite-depth neural networks, inspired by the paper ["Finite Depth and Width Corrections to the Neural Tangent Kernel"](https://arxiv.org/abs/1909.05989). The research explores:

- **NTK scaling behavior** with respect to network depth and width
- **Activation function effects** on NTK properties (ReLU, GELU, Sigmoid)
- **Physics-Informed Neural Networks (PINNs)** in both infinite and finite width regimes
- **Training dynamics** and convergence properties

## 📁 Repository Structure

### `code/` - Implementation and Experiments
The main implementation directory containing all computational experiments and analysis.

**Key Components:**
- **`experiments/`** - Jupyter notebooks with all numerical experiments:
  - `infinite_ntk.ipynb` - NTK analysis in infinite width regime
  - `finite_width_analysis.ipynb` - NTK behavior in finite-width networks
  - `pinn_infinite.ipynb` - PINN analysis in infinite width regime
  - `pinn_finite_width_analysis.ipynb` - PINN performance with finite-width networks
  - `supplementary/full_training_analysis.ipynb` - Comprehensive training dynamics analysis
- **`util/`** - Utility functions and helper modules
- **`requirements.txt`** - Python dependencies
- **`README.md`** - Detailed setup and usage instructions

**Quick Start:**
```bash
cd code
conda create -n ntk_pinn python=3.10
conda activate ntk_pinn
pip install -r requirements.txt
```

### `data/` - Data Storage
Contains all experimental data, including:
- Pre-computed NTK matrices for different activation functions
- Training data and results
- PINN-specific datasets

### `report/` - Final Project Report
- **`report.pdf`** - Complete project report with findings and analysis
- LaTeX source files for the report

### `presentation/` - Project Presentation
Contains materials used for the final project presentation, including slides and supporting documents.

### `project-proposal/` - Initial Research Plan
- **`draft.md`** - Original project proposal with research objectives
- **`feedback.md`** - Feedback received on the proposal
- **`README.md`** - Proposal documentation

### `1909.05989v1.pdf` - Reference Paper
The foundational paper that inspired this research project.

## 🚀 Getting Started

### For Researchers/Students
1. **Read the report** (`report/report.pdf`) to understand the findings
2. **Explore the code** in `code/experiments/` to see the implementation
3. **Run experiments** by following the setup instructions in `code/README.md`

### For Code Reviewers
1. Start with `code/experiments/finite_width_analysis.ipynb` for core NTK analysis
2. Check `code/experiments/pinn_finite_width_analysis.ipynb` for PINN-specific results
3. Review `code/util/` for implementation details

### For Presentation Reviewers
1. Check `presentation/` for slides and materials
2. Review the main findings in `report/report.pdf`

## 🔬 Key Findings

The project provides insights into:
- **NTK scaling laws** in practical finite-width settings
- **Activation function impact** on NTK behavior
- **PINN performance** across different network architectures
- **Training dynamics** and convergence properties

## 📋 Requirements

- Python 3.10
- JAX and Flax for neural network implementation
- Neural Tangents for NTK computations
- See `code/requirements.txt` for complete dependencies

## 📚 References

- [Finite Depth and Width Corrections to the Neural Tangent Kernel](https://arxiv.org/abs/1909.05989) - Primary reference paper
- Additional references available in the final report

## 🤝 Contributing

This is a completed academic project. For questions or discussions about the methodology or results, please refer to the report and code documentation.

---

**Note:** This project was completed as part of WI4450: Special Topics in Computational Science and Engineering (2024/2025 Q3–Q4). All code, analysis, and findings are documented for reproducibility and educational purposes.