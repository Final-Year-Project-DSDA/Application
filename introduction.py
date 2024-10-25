# File: Introduction.py
import streamlit as st
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Application:
    name: str
    description: str
    domains: List[str]
    key_papers: List[Dict[str, str]]

class Introduction:
    def __init__(self):
        # GNN and STGNN Motivational Points
        self.motivation = {
            "Conventional Limitations": "Traditional Neural Networks struggle to capture complex relationships in interconnected data. GNNs address this by enabling relational learning within highly connected structures.",
            "Spatial-Temporal Modeling": "GNN-based methods can effectively model spatial and temporal dependencies, making them suitable for multivariate time series data.",
            "Architecture Complexity": "Despite their effectiveness, GNNs come with challenges in understanding and implementation due to complex architectures and algorithms."
        }

        self.literature_survey_data = {
            "Title": [
                "Graph neural networks: A review of methods and applications",
                "A review of graph neural networks: concepts, architectures, techniques, challenges, datasets, applications, and future directions",
                "Attention Based Spatial-Temporal Graph Convolutional Networks(ASTGCN) for Traffic flow forecasting",
                "A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection",
                "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs",
                "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting",
                "A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting",
                "Network Traffic Prediction based on Diffusion Convolutional Recurrent Neural Networks",
                "Gated Graph Sequence Neural Networks",
                "Inductive Representation Learning on Large Graphs",
                "Invariance-Preserving Localized Activation Functions for Graph Neural Networks",
                "Short-term load forecasting using spatial-temporal embedding graph neural network"
            ],
            "Publication": [
                "AI open- 2020",
                "Springer(Journal of Big Data)-2024",
                "AAAI - 2019",
                "IEEE Transactions on Pattern Analysis and Machine Intelligence-2021",
                "AAAI conference on artificial intelligence-2020",
                "IJCAI-2018",
                "ISPRS-2021",
                "IEEE INFOCOM-2019",
                "ICLR - 2016",
                "ACM-Digital Library-2017",
                "IEEE Transactions on Signal Processing-2020",
                "Electric Power Systems Research-2023"
            ],
            "Methodology": [
                "Provides a Base Design Pipeline for GNN identification and training",
                "Usage of Propagation, Sampling and Pooling modules to process Graph Structures in GCN, GAT and GraphSage",
                "Usage of GCN to capture spatial dependencies as well as Spatial Attention & Temporal Attention for spatial features and temporal patterns",
                "Classify multiple GNN frameworks for different tasks such as classification, imputation, forecasting",
                "Usage of Recurrent Models to evolve the GCN Parameters",
                "Leverages the power of graph convolutions to model spatial dependencies and convolutional sequence learning to capture temporal dynamics",
                "2 GCN layers for Spatial feature extraction And 2 GRU Layers for Temporal Feature extraction",
                "Capturing both topological properties and temporal dependencies through diffusion convolution and recurrent units",
                "Node annotations representing the program's heap state and a propagation module to update the graph structure",
                "Generate node embeddings by sampling and aggregating features from a node's local neighborhood",
                "Adapting activation functions to the graph structure",
                "Construct directed static and dynamic graphs and leveraging EMA GCNs to capture spatial dependencies"
            ],
            "Research Gaps": [
                "Fails to address the interpretability and the effect of multiple hyperparameters",
                "Fails to address the case of lack of ability to work on imputed and noisy data (Imbalanced datasets)",
                "Need to test it out on Multiple domains and datasets, Need to consider external non-trivial features affecting the graph",
                "Pre-training, Transfer Learning, and Large Models, Robustness and Interpretability for Spatio Temporal GNNs",
                "Focused only on GCN to capture spatial dependencies",
                "Comparison only with limited models on limited datasets, The Graph Structures discussed here are homogenous in nature",
                "Comparison only with limited models on limited datasets, Did not consider other attention mechanisms other than GRU",
                "It struggles with accurately predicting sudden traffic spikes or bursts.",
                "Not well-suited for data with inherent temporal dependencies, Primarily focuses on binary relations represented as edges",
                "The discussion and conclusion were done on Undirected Graphs, Lack of explanation on non-uniform neighborhood sampling functions",
                "Limited Exploration of Activation Functions for GNNs, Neglect of Neighborhood Structure in Nonlinearities",
                "Existing methods often rely on undirected or static graphs, which fail to fully capture the dynamic spatial dependencies and periodicity in load data."
            ]
        }
        
        self.df_literature_survey = pd.DataFrame(self.literature_survey_data)
        
        # GNN Concepts
        self.gnn_concepts = {
            "Introduction to GNNs": """
            Graph Neural Networks (GNNs) are a class of neural networks designed to process graph-structured data, where nodes represent entities and edges represent relationships.
            """,
            "Graph Categories": """
            - **Homogeneous Graphs**: All nodes and edges are of the same type.
            - **Heterogeneous Graphs**: Nodes and edges have different types, representing richer structures.
            - **Static Graphs**: Graphs where relationships (edges) remain constant over time.
            - **Dynamic Graphs**: Graphs where both nodes and edges change over time, reflecting evolving relationships.
            """,
            "Dynamic Graph Types": """
            - **Discrete-time dynamic graphs (DTDG)**: Represent sequences of static graph snapshots taken at discrete intervals of time.
            - **Continuous-time dynamic graphs (CTDG)**: Represent events that happen continuously over time, such as edge/node addition or deletion, feature transformation.
            """
        }

        # Key Concepts of GNNs and STGNNs
        self.key_concepts = {
            "GNN Architecture": """
            - **Message Passing**: Each node sends its current feature representation to all neighboring nodes.
            - **Aggregation**: The neighboring nodes' messages are aggregated into new node representations using functions like summation or more complex, learnable methods.
            - **Updating**: Node representations are updated iteratively using aggregation results and nonlinear transformations (e.g., ReLU, Tanh).
            """,
            "STGNN Architecture": """
            **Spatial Module**:
            - **Spectral GNNs**: Use spectral graph theory to capture node relationships in the frequency domain.
            - **Spatial GNNs**: Simplified approach based on neighborhood information, focusing on direct relationships between nodes.
            - **Hybrid Approaches**: Combining spectral and spatial methods to leverage both benefits.

            **Temporal Module**:
            - Capture temporal dependencies using recurrence-based methods (e.g., RNNs), convolutional methods (e.g., TCNs), or attention-based techniques (e.g., Transformers).
            - Frequency domain methods (e.g., Fourier transform) can also be applied for temporal dependencies.

            **Model Architectures**:
            - **Factorized**: Temporal processing occurs before or after spatial processing.
            - **Coupled**: Temporal and spatial modules are interleaved for integrated learning.
            """
        }
        
        # Research Gap Section
        self.research_gap = """
        **Research Gap**:
        - Limited comparisons between models performing similar tasks.
        - Insufficient exploration of layer types and mathematical functions in GNNs.
        - Lack of studies exploring the complexity-performance trade-off in GNNs.
        - Few studies have developed or studied localized activation functions for graph-based algorithms.
        """
        
        # Tools Section
        self.tools = {
            "PyTorch Geometric": {
                "description": "Extensive library for GNNs with PyTorch backend",
                "features": [
                    "Wide range of GNN layers",
                    "Efficient sparse matrix operations",
                    "Built-in datasets and benchmarks",
                    "Easy-to-use data handling"
                ],
                "url": "https://pytorch-geometric.readthedocs.io/"
            },
            "Deep Graph Library (DGL)": {
                "description": "Framework-agnostic library for deep learning on graphs",
                "features": [
                    "Supports multiple deep learning frameworks",
                    "High performance sparse matrix operations",
                    "Comprehensive graph algorithms",
                    "Distributed training support"
                ],
                "url": "https://www.dgl.ai/"
            },
            "Spektral": {
                "description": "GNN library based on Keras and TensorFlow",
                "features": [
                    "Keras-style API",
                    "Ready-to-use GNN layers",
                    "Graph preprocessing tools",
                    "Built-in visualization tools"
                ],
                "url": "https://graphneural.network/"
            }
        }
        
        # Applications Section
        self.applications = [
            Application(
                name="Drug Discovery",
                description="Predicting molecular properties and drug-protein interactions",
                domains=["Healthcare", "Chemistry", "Pharmaceuticals"],
                key_papers=[
                    {"title": "Drug-Target Interaction Prediction", "year": "2020"},
                    {"title": "Molecular Property Prediction", "year": "2019"}
                ]
            ),
            Application(
                name="Social Network Analysis",
                description="Understanding user interactions and community structure",
                domains=["Social Media", "Marketing", "Community Detection"],
                key_papers=[
                    {"title": "Social Influence Prediction", "year": "2021"},
                    {"title": "Community Evolution Analysis", "year": "2020"}
                ]
            ),
            Application(
                name="Traffic Prediction",
                description="Forecasting traffic flow and optimizing transportation networks",
                domains=["Transportation", "Urban Planning", "IoT"],
                key_papers=[
                    {"title": "Traffic Flow Prediction", "year": "2021"},
                    {"title": "Spatial-Temporal Forecasting", "year": "2020"}
                ]
            )
        ]

    def display_motivation(self):
        st.header("Motivation for Using Graph Neural Networks")
        cols = st.columns(len(self.motivation))
        for col, (title, desc) in zip(cols, self.motivation.items()):
            with col:
                st.markdown(f"**{title}**")
                st.write(desc)

    def display_gnn_concepts(self):
        st.header("Introduction to GNNs")
        for title, content in self.gnn_concepts.items():
            st.subheader(title)
            st.markdown(content)

    def display_key_concepts(self):
        st.header("Key Concepts of GNN and STGNN Architecture")
        
        stgnn_image_path = './images/introduction/stgnn/Screenshot 2024-10-25 at 10.59.09 AM.png'
        tabs = st.tabs(list(self.key_concepts.keys()))
        
        for tab, (concept, content) in zip(tabs, self.key_concepts.items()):
            with tab:
                st.markdown(content)
                if concept == "STGNN Architecture":
                    st.image(stgnn_image_path, caption="STGNN Architecture", use_column_width=True)

    def display_tools(self):
        st.header("Popular Tools and Frameworks")
        cols = st.columns(len(self.tools))
        
        for col, (tool_name, tool_info) in zip(cols, self.tools.items()):
            with col:
                st.subheader(tool_name)
                st.write(tool_info["description"])
                st.markdown("**Key Features:**")
                for feature in tool_info["features"]:
                    st.markdown(f"- {feature}")
                st.markdown(f"[Documentation]({tool_info['url']})")

    def display_applications(self):
        st.header("Real-world Applications")
        tabs = st.tabs([app.name for app in self.applications])
        
        for tab, app in zip(tabs, self.applications):
            with tab:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {app.description}")
                    st.markdown("**Domains:**")
                    for domain in app.domains:
                        st.markdown(f"- {domain}")
                
                with col2:
                    st.markdown("**Key Papers:**")
                    for paper in app.key_papers:
                        st.markdown(f"- {paper['title']} ({paper['year']})")

    def display_research_gap(self):
        st.header("Research Gap")
        st.markdown(self.research_gap)

    def display_literature_survey(self):
        st.header("Literature Survey")
        st.dataframe(self.df_literature_survey)  # Display the table in an interactive format


    def display_problem_statement(self):
        st.header("Problem Statement")
        st.markdown("""
        **Goal**: Conduct an empirical study that analyses GNN layers and variants, benchmarks models on similar datasets, 
        examines the complexity-performance trade-offs with additional layers, and explores optimization techniques, 
        aiming to develop a novel, optimized STGNN architecture for forecasting.
        """)

    def display(self):
        st.title("Introduction to Graph Neural Networks")
        
        # Brief overview
        st.markdown("""
        Graph Neural Networks (GNNs) are powerful deep learning models designed to work with graph-structured data.
        They can learn from both the features of nodes and the relationships between them, making them ideal for 
        many real-world applications where data is naturally represented as a graph.
        """)
        
        # Display all sections
        self.display_motivation()
        st.divider()
        
        self.display_gnn_concepts()
        st.divider()
        
        self.display_key_concepts()
        st.divider()
        
        self.display_tools()
        st.divider()
        
        self.display_applications()
        st.divider()

        self.display_problem_statement()
        st.divider()

        self.display_literature_survey()  # Add the literature survey table
        st.divider()

        self.display_research_gap()
        st.divider()


def main():
    st.set_page_config(page_title="GNN Introduction", layout="wide")
    intro = Introduction()
    intro.display()

if __name__ == "__main__":
    main()
