# Evolving Graph Convolutional Network (Original) (EvolveGCNO)

**Year**: 2020  
**Publication**: AAAI Conference  
**Paper Link**: [arXiv:1902.10191](https://arxiv.org/abs/1902.10191)

#### Task:
- Link Prediction, Edge Classification, Node Classification

#### Architecture:
- Evolving Graph Convolutional Network (EvolveGCNo)

#### Spatial Module:
- Graph Convolutional Network (GCN) layers with weight evolution

#### Temporal Module:
- Recurrent Neural Networks (RNN) for evolving GCN parameters (GRU and LSTM variants)

#### Missing Values:
- Not explicitly addressed; relies on node embeddings and edge data from available information

#### Input Graph:
- Required with dynamic changes over time

#### Learned Graph Relations:
- Dynamic relations through RNN-based weight evolution

#### Graph Heuristics:
- Two versions: EvolveGCN-H (GRU-based weight update with node embeddings) and EvolveGCN-O (LSTM-based weight update without node embeddings)

# Evolving Graph Convolutional Network (Original) (EvolveGCNO)

## 1. Introduction
**EvolveGCN** focuses on adapting GCNs to dynamically evolving graphs by evolving the model parameters instead of fixed node embeddings. Through RNNs, EvolveGCN updates the GCN weights at each time step to capture temporal graph changes, making it suitable for applications where node sets change frequently.

## 2. Methodology
The model is designed to:
- **Spatial Dependency**: Handled through GCN layers, which learn node relations based on input graphs.
- **Temporal Dependency**: Modeled with RNN-based weight updates, evolving the GCN parameters over time.
- **Dynamic Graph Adaptability**: Maintains flexibility with changing node sets and graphs over time, making it robust for evolving graph data.

## 3. Model Components
- **Spatial Aggregation**: Standard GCN layers transform node features based on their neighborhoods.
- **Temporal Aggregation**: Two RNN variants manage GCN parameter updates over time:
  - **EvolveGCN-H**: GRU-based updates using node embeddings.
  - **EvolveGCN-O**: LSTM-based updates without node embeddings.
- **Evolution Mechanism**: Dynamic updates in GCN weights, allowing adaptability to new and changing nodes and edges in the graph sequence.

## 4. Hyperparameters
- **Embedding Size**: Tuned per dataset; consistent for both GCN layers.
- **Time Window**: Set to 10 for most datasets.
- **Evaluation Metrics**: MAP, MRR, F1 Score

## 5. Performance Metrics

| Dataset       | Link Prediction (MAP) | Link Prediction (MRR) | Edge Classification (F1) | Node Classification (F1 - Illicit Class) |
|---------------|------------------------|------------------------|---------------------------|-------------------------------------------|
| SBM           | 0.1989                 | 0.0138                | -                         | -                                         |
| BC-OTC        | 0.0028                 | 0.0968                | 0.68                      | -                                         |
| BC-Alpha      | 0.0036                 | 0.1185                | 0.71                      | -                                         |
| UCI           | 0.0270                 | 0.1379                | -                         | -                                         |
| AS            | 0.1139                 | 0.2746                | -                         | -                                         |
| Elliptic      | -                      | -                     | -                         | 0.95 (Micro Avg.)                         |

The datasets include synthetic and real-world networks, covering social interactions, financial transactions, and communication networks with varying temporal granularity.


**Dataset**: Chickenpox

EvolveGCNO applies a graph convolutional framework to model dynamic graphs while focusing on original homogeneous node representations.

## Results on Chickenpox Dataset:

- **MSE**: 0.9558
- **MAE**: 0.6299
- **MAPE**: 684.6207%
- **RMSE**: 0.9777
- **R-squared**: 0.0507

## Version History:

- v1.0
