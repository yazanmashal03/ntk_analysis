# Neural Tangent Kernel Analysis with Finite Width and Depth 

## Project Overview
This project aims to conduct a comprehensive numerical analysis of Neural Tangent Kernel (NTK) behavior of finite width and finite depth of an Artificial Neural Network (ANN), inspired by this [paper](https://arxiv.org/abs/1909.05989). The study will investigate the scaling properties of NTK on detailed numerical analysis, we will also explore different activation function than the one prosed by the authors.

## Research Objectives

1. **NTK Scaling Analysis**
   - [x] Investigate the asymptotic behavior of NTK with respect to **finite** depth (d) and width (n)
   - [x] Validate theoretical predictions regarding NTK scaling laws
   - [x] Analyze the relationship between the ratio (d/n) and the behavior of NTK
        - used $\beta$ and ratio

2. **Architecture Optimization**
   - [] Determine optimal d/n ratio as a "rule of thumb"
   - [] Study the impact of different activation functions on NTK behavior
        - only able to analyze emperically as theoritical results are not present

3. **Data-Dependent Analysis**
   - [] Investigate authors conjecture regarding data-dependent aspects of NTK
   - [] Analyze how input data distribution affects NTK properties

### Possible research questions (might change):
1. Do we observe stochaticity of NTK with finite d and n? Yes
2. How does NTK scale with finite d and n on different networks than ReLU? Yet to be formalized
3. What ratio d/n excibit data-dependent learning?

## Methodology

1. **Numerical Implementation**
   - Implement a neural network solver for the Poisson equation
   - Develop tools for computing and analyzing NTK matrices
   - Create visualization tools for NTK spectrum analysis

2. **Experimental Framework**
   - Conduct systematic experiments varying network depth and width
   - Test different activation functions (ReLU, tanh, sigmoid, etc.)
   - Analyze NTK behavior

3. **Analysis Tools**
   - Develop metrics for quantifying NTK properties
   - Implement statistical analysis of NTK spectrum
   - Create visualization tools for NTK evolution during training
   - Compare with results in literature

## Expected Outcomes

1. **Theoretical Insights**
   - Validation of NTK scaling laws in practical settings
   - Guidelines for optimal network architecture design
   - Understanding of data-dependent NTK behavior

2. **Practical Contributions**
   - Recommendations for network architecture selection
   - Insights into activation function choice

## Implementation Plan

1. **Phase 1: Setup and Basic Implementation**
   - Implement basic neural network PDE solver
   - Develop NTK computation tools
   - Create initial visualization framework

2. **Phase 2: Scaling Analysis**
   - Conduct systematic experiments on depth/width scaling
   - Analyze NTK behavior across different architectures
   - Study activation function impact

3. **Phase 3: Data-Dependent Analysis**
   - Investigate data-dependent aspects of NTK
   - Analyze generalization properties
   - Validate theoretical conjectures

4. **Phase 4: Synthesis and Documentation**
   - Compile results and findings
   - Create comprehensive documentation
   - Develop practical guidelines

## References

1. FINITE DEPTH AND WIDTH CORRECTIONS TO THE NEURAL TANGENT KERNEL - https://arxiv.org/pdf/1909.05989

## Deliverables

1. Code repository with implementation
2. Comprehensive analysis notebooks
3. Final report with findings and recommendations
4. Presentation materials
5. Documentation and usage guidelines
