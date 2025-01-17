# Attention-based Graph Convolutional Recurrent Network (AGCRN)

**Dataset**: Chickenpox

AGCRN combines the strengths of graph convolutional networks and recurrent neural networks to capture both spatial and temporal dependencies for time-series forecasting.

## Framework Overview
Node Adaptive Parameter Learning (NAPL):

Learns node-specific patterns by generating unique parameters for each traffic node.
Uses a matrix factorization approach to avoid the overfitting issue of a fully independent parameter space.
Data Adaptive Graph Generation (DAGG):

Automatically infers inter-dependencies among traffic nodes without requiring a pre-defined graph.
Dynamically generates graph adjacency matrices during training based on node embeddings.


## Results on Chickenpox Dataset:

- **MSE**: 1.0422
- **MAE**: 0.6881
- **MAPE**: 822.2222%
- **RMSE**: 1.0209
- **R-squared**: -0.0364

## Version History:

- v1.0
