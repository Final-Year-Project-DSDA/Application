# STGNN: Spatial-Temporal Graph Neural Network for Traffic Flow Prediction

**Year**: 2020  
**Publication**: WWW Conference  
**Paper Link**: https://doi.org/10.1145/3366423.3380186

#### Task:
- Traffic Flow Prediction

#### Architecture:
- Spatial-Temporal Graph Neural Network (STGNN)

#### Spatial Module:
- Graph Neural Network (GNN) with Positional Attention

#### Temporal Module:
- Gated Recurrent Unit (GRU) and Transformer Layers

#### Missing Values:
- Managed with Zero-Imputation and Predicted Filling

#### Input Graph:
- Required

#### Learned Graph Relations:
- Dynamic (learned positional relations)

#### Graph Heuristics:
- Positional Attention Mechanism, Temporal Dependencies through GRU and Transformer

# STGNN: Spatial-Temporal Graph Neural Network for Traffic Flow Prediction

## 1. Introduction
**STGNN** addresses complex spatial-temporal dependencies in traffic forecasting by combining positional attention in GNN with GRU and Transformer layers to enhance both local and global temporal dependencies. This design overcomes limitations in capturing distant spatial dependencies and long-term temporal patterns in urban traffic data.

## 2. Methodology
The model is designed to:
- **Spatial Dependency**: Captured using GNN with positional attention to aggregate traffic information from adjacent nodes, adjusted by learned node positions.
- **Temporal Dependency**: Modeled using GRU for short-term sequence patterns and a Transformer layer for global temporal dependencies.
- **Dynamic Traffic Events**: Supports learning from sudden changes and long-lasting impacts of events (e.g., accidents) using recurrent and attention mechanisms.

## 3. Model Components
- **Spatial Aggregation**: GNN with positional attention to balance influence from nearby and distant nodes.
- **Temporal Aggregation**: GRU captures local dependencies; Transformer layer captures long-term dependencies.
- **Attention Mechanism**: Positional encoding refines the graph structure, considering both geographical and dynamic relations.

## 4. Hyperparameters
- **Learning Rate**: 0.001 (Adam optimizer with learning rate decay)
- **Hidden Dimensions**: 64
- **Attention Heads**: 4
- **Batch Size**: 64
- **Evaluation Metrics**: MAE, RMSE, MAPE

## 5. Performance Metrics

| Dataset     | 15-min Forecast       | 30-min Forecast       | 60-min Forecast       |
|-------------|------------------------|------------------------|------------------------|
| METR-LA     | MAE: 2.62, RMSE: 4.99, MAPE: 6.55% | MAE: 2.98, RMSE: 5.88, MAPE: 7.77% | MAE: 3.49, RMSE: 6.94, MAPE: 9.69% |
| PEMS-BAY    | MAE: 1.17, RMSE: 2.43, MAPE: 2.34% | MAE: 1.46, RMSE: 3.27, MAPE: 3.09% | MAE: 1.83, RMSE: 4.20, MAPE: 4.15% |

The datasets (METR-LA and PEMS-BAY) consist of traffic speed data from sensors on urban road networks, sampled every 5 minutes.
