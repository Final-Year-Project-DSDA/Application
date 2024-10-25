# AGCRN: Adaptive Graph Convolutional Recurrent Network

**Year**: 2020  
**Publication**: arXiv preprint  
**Paper Link**: [arXiv:2007.02842](https://arxiv.org/abs/2007.02842)

#### Task:
- Traffic Flow Prediction

#### Architecture:
- Adaptive Graph Convolutional Recurrent Network (AGCRN)

#### Spatial Module:
- Node Adaptive Parameter Learning (NAPL) for node-specific patterns
- Data Adaptive Graph Generation (DAGG) for learning inter-dependencies

#### Temporal Module:
- Gated Recurrent Unit (GRU)

#### Missing Values:
- Managed with Linear Interpolation

#### Input Graph:
- Not required; DAGG automatically learns dependencies

#### Learned Graph Relations:
- Dynamic (learned data-driven relations)

#### Graph Heuristics:
- Node embeddings inferred through DAGG, with node-specific patterns through NAPL

# AGCRN: Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting

## 1. Introduction
**AGCRN** addresses spatial-temporal dependencies in traffic forecasting by enabling adaptive learning of node-specific parameters and dynamically inferring inter-dependencies among traffic series. The architecture focuses on capturing both spatial and temporal nuances without relying on pre-defined graphs, making it highly adaptable.

## 2. Methodology
The model is designed to:
- **Spatial Dependency**: Captured through NAPL, which learns unique parameters for each node, and DAGG, which dynamically generates the graph structure.
- **Temporal Dependency**: Modeled with GRU, which handles short-term sequence patterns.
- **Dynamic Traffic Events**: Learns from both consistent and sudden shifts in traffic patterns, using adaptive modules.

## 3. Model Components
- **Spatial Aggregation**: NAPL provides node-specific parameter learning, while DAGG adapts the graph structure based on data-driven dependencies.
- **Temporal Aggregation**: GRU handles temporal sequence patterns in traffic.
- **Attention Mechanism**: NAPL and DAGG combine to refine spatial-temporal relations within traffic data.

## 4. Hyperparameters
- **Learning Rate**: 0.003 (Adam optimizer)
- **Embedding Dimension**: 10 for PeMSD4, 2 for PeMSD8
- **Hidden Dimensions**: 64
- **Batch Size**: 64
- **Evaluation Metrics**: MAE, RMSE, MAPE

## 5. Performance Metrics

| Dataset     | 15-min Forecast       | 30-min Forecast       | 60-min Forecast       |
|-------------|------------------------|------------------------|------------------------|
| PeMSD4      | MAE: 19.15, RMSE: 30.65, MAPE: 13.15% | MAE: 21.98, RMSE: 34.91, MAPE: 14.82% | MAE: 26.54, RMSE: 41.07, MAPE: 17.91% |
| PeMSD8      | MAE: 15.95, RMSE: 25.22, MAPE: 10.09% | N/A                   | N/A                   |

The datasets (PeMSD4 and PeMSD8) consist of traffic flow data from urban road sensors. Missing values are handled through linear interpolation, and traffic data is aggregated into 5-minute intervals.


**Dataset**: Chickenpox

AGCRN combines the strengths of graph convolutional networks and recurrent neural networks to capture both spatial and temporal dependencies for time-series forecasting.

## Results on Chickenpox Dataset:

- **MSE**: 1.0422
- **MAE**: 0.6881
- **MAPE**: 822.2222%
- **RMSE**: 1.0209
- **R-squared**: -0.0364

## Version History:

- v1.0
