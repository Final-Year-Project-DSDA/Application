
import streamlit as st

def main():
    st.set_page_config(
        page_title="Strengths and Weaknesses of STGNN Models",
        page_icon="📊",
        layout="wide"
    )

    # Title
    st.title("Comparative Analysis of the Discussed ST-GNN Architectures")

    # Introduction
    st.markdown("""
    Through empirical evaluation across three benchmark datasets - **METR-LA**, **PeMS-D7**, and **Chickenpox** - significant variations in model performance and applicability have emerged. 
    This analysis presents a systematic comparison of leading STGNN architectures, emphasizing their operational characteristics and domain-specific applicability.
    """)

    # Table for Architectural Characteristics and Performance Analysis
    st.header("Architectural Characteristics and Performance Analysis of STGNN Models")

    # Data for the table
    table_data = [
        {
            "Architecture": "DCRNN",
            "Strengths": """
            - Demonstrated superior accuracy in high-frequency traffic analysis
            - Robust temporal dependency modeling
            - Enhanced resilience to missing data points
            """,
            "Limitations": """
            - Substantial computational overhead
            - Complex optimization requirements
            - Intensive data preprocessing demands
            """
        },
        {
            "Architecture": "A3T-GCN",
            "Strengths": """
            - Consistent performance across domains
            - Optimized training convergence
            - Linear computational scaling
            """,
            "Limitations": """
            - Diminished performance with sparse datasets
            - Temporal sequence length constraints
            """
        },
        {
            "Architecture": "MPNN-LSTM",
            "Strengths": """
            - Flexible message passing mechanism
            - Strong sequential pattern learning
            - Adaptable to various graph structures
            """,
            "Limitations": """
            - Higher error rates on Chickenpox data
            - Complex message passing optimization
            """
        },
        {
            "Architecture": "EvolveGCN-H",
            "Strengths": """
            - Dynamic graph evolution modeling
            - Moderate performance on Chickenpox data
            - Adaptive weight updates
            """,
            "Limitations": """
            - Higher error rates on Chickenpox data
            - Resource-intensive training
            """
        },
        {
            "Architecture": "EvolveGCN-O",
            "Strengths": """
            - Optimized graph evolution
            - Flexible architecture adaptation
            """,
            "Limitations": """
            - Complex architecture optimization
            - Training stability challenges
            """
        },
        {
            "Architecture": "ASTGCN",
            "Strengths": """
            - Enhanced performance in dense networks
            - Rapid convergence characteristics
            - Effective seasonal pattern recognition
            """,
            "Limitations": """
            - Significant resource requirements
            - Performance degradation in sparse scenarios
            """
        },
        {
            "Architecture": "SSTGCN",
            "Strengths": """
            - Efficient spatial dependency modeling
            - Optimized memory utilization
            """,
            "Limitations": """
            - Sensitivity to data discontinuities
            - Limited temporal pattern capture
            - Higher error rates in irregular sequences
            """
        },
        {
            "Architecture": "ST-GCN",
            "Strengths": """
            - Reliable baseline performance
            - Resource efficiency and Implementation Simplicity
            """,
            "Limitations": """
            - Limited complex pattern modeling
            - Moderate accuracy metrics
            - Basic temporal representation
            """
        },
        {
            "Architecture": "AGRCN",
            "Strengths": """
            - Adaptive graph learning capability
            - Node-specific pattern recognition
            - Efficient parameter sharing mechanism
            """,
            "Limitations": """
            - Limited performance on sparse graphs
            - Higher complexity in graph structure learning
            - Resource-intensive adaptive mechanisms
            """
        }
    ]

    # Display the table
    for row in table_data:
        with st.expander(f"**{row['Architecture']}**"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Strengths**")
                st.markdown(row["Strengths"])
            with col2:
                st.markdown("**Limitations**")
                st.markdown(row["Limitations"])
            st.divider()

    # Footer
    st.markdown("""
    **Note**: This analysis is based on empirical evaluations across benchmark datasets and highlights the key strengths and limitations of each architecture.
    """)

if __name__ == "__main__":
    main()