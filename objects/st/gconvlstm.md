# Graph Convolutional LSTM (GConvLSTM)

**Year**: 2016  
**Publication**: ICLR Conference  
**Paper Link**: [arXiv:1612.07659](https://arxiv.org/abs/1612.07659)

#### Task:
- Spatio-Temporal Sequence Modeling (e.g., video prediction, language modeling)

#### Architecture:
- Graph Convolutional LSTM (GConvLSTM)

#### Spatial Module:
- Graph Convolutional Network (GCN) with Chebyshev polynomial filters

#### Temporal Module:
- Long Short-Term Memory (LSTM)

#### Missing Values:
- Not specifically addressed; uses graph and temporal sequences without imputation methods

#### Input Graph:
- Required with k-nearest neighbor or pre-defined relationships (e.g., word embeddings in language tasks)

#### Learned Graph Relations:
- Static with predefined spatial structure (e.g., 2D grid for moving-MNIST, k-nearest neighbors for language data)

#### Graph Heuristics:
- Graph Fourier transform for convolution; rotationally invariant filters to capture isotropic spatial dependencies

# GCRN: Graph Convolutional Recurrent Network for Structured Sequence Modeling

## 1. Introduction
**GCRN** introduces a structured sequence modeling approach that combines the spatial capabilities of graph CNNs with the dynamic sequence learning of LSTMs. This model is designed for graph-structured data, such as sensor networks or natural language sequences, where traditional grid-based convolutions are insufficient.

## 2. Methodology
The model is designed to:
- **Spatial Dependency**: Captured through GCN layers with Chebyshev polynomial filters to identify spatial structures in the graph.
- **Temporal Dependency**: Modeled with LSTM layers that handle time-varying patterns.
- **Rotational Invariance**: Convolutional filters in GCRN are isotropic, allowing the model to capture rotation-invariant features, useful for structured but non-grid data.

## 3. Model Components
- **Spatial Aggregation**: GCN layers with Chebyshev polynomial filters of varying support \(K\), enabling localized information propagation.
- **Temporal Aggregation**: LSTM layers handle sequential dependency, with graph convolution applied to hidden and cell states.
- **Graph Structure**: Built using k-nearest neighbor graphs for non-grid data (e.g., word embeddings) or pre-defined structures for grid data.

## 4. Hyperparameters
- **Learning Rate**: 0.001 with RMSProp optimizer
- **Filter Support (K)**: Tuned per dataset, typically 3-9 for synthetic experiments
- **Batch Size**: Dataset-specific
- **Evaluation Metrics**: Cross-entropy (moving-MNIST), Perplexity (Penn Treebank)

## 5. Performance Metrics

| Dataset               | Model                | Filter Support | Parameters  | Runtime (s) | Test Metric (Lower is Better) |
|-----------------------|----------------------|----------------|-------------|-------------|-------------------------------|
| Moving-MNIST          | LSTM+CNN (5x5)      | -              | 13,524,496  | 2.10        | 3851 (Cross-entropy)          |
| Moving-MNIST          | LSTM+GCNN (K=7)     | 7              | 3,792,400   | 1.61        | 3446 (Cross-entropy)          |
| Penn Treebank         | GCRN-M1 (One-hot)   | -              | 42,011,602  | -           | 98.67 (Perplexity)            |

The GCRN is evaluated on the moving-MNIST dataset and the Penn Treebank dataset, demonstrating that combining graph convolutions with LSTMs improves spatio-temporal predictions on structured but non-Euclidean data. 


**Dataset**: Chickenpox

GConvLSTM combines graph convolutional layers with LSTM to improve the prediction of time-series data on graph structures.

## Results on Chickenpox Dataset:

- **MSE**: 0.9646
- **MAE**: 0.6288
- **MAPE**: 554.9057%
- **RMSE**: 0.9821
- **R-squared**: 0.0420

## Version History:

- v1.0
