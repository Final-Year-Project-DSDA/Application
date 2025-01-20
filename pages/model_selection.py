import streamlit as st
import pandas as pd

def get_model_recommendation(temporal_resolution, resources, prediction_horizon, data_volume, architecture_preference):
    # Scores Initialization
    models = {
        'DCRNN': 0,
        'A3TGCN': 0,
        'ASTGCN': 0,
        'SSTGCN': 0,
        'EvolveGCN': 0,
        'ST-GCN': 0
    }

    # Temporal Resolution scoring
    if temporal_resolution == "High (5-minute intervals)":
        models['DCRNN'] += 3
        models['A3TGCN'] += 1
        models['ST-GCN'] += 2
    elif temporal_resolution == "Medium (Hourly)":
        models['A3TGCN'] += 2
        models['ASTGCN'] += 2
        models['ST-GCN'] += 2
    else:
        models['A3TGCN'] += 3
        models['EvolveGCN'] += 3
        models['SSTGCN'] += 2

    # Computational Resources scoring
    if resources == "High (Powerful hardware available)":
        models['DCRNN'] += 2
        models['ASTGCN'] += 3
    elif resources == "Medium":
        models['A3TGCN'] += 2
        models['ST-GCN'] += 3
    else:
        models['SSTGCN'] += 3
        models['ST-GCN'] += 2

    # Prediction Horizon scoring
    if prediction_horizon == "Short-term":
        models['SSTGCN'] += 2
        models['A3TGCN'] += 2
    else:
        models['DCRNN'] += 2
        models['ASTGCN'] += 2
        models['EvolveGCN'] += 2

    # Data Volume scoring
    if data_volume == "Large dataset":
        models['SSTGCN'] += 3
        models['ASTGCN'] += 2
        models['DCRNN'] += 2
    elif data_volume == "Medium dataset":
        models['A3TGCN'] += 2
        models['ST-GCN'] += 2
    else:
        models['A3TGCN'] += 3
        models['EvolveGCN'] += 2

    # Architecture Preference scoring
    if architecture_preference == "Factorized (Faster training)":
        models['ST-GCN'] += 2
        models['SSTGCN'] += 2
    else:
        models['DCRNN'] += 2
        models['ASTGCN'] += 2

    # Get the model with highest score
    recommended_model = max(models.items(), key=lambda x: x[1])[0]

    return recommended_model, models

def main():
    st.title("STGNN Model Selection Advisor")
    st.write("""
    This tool helps you select the most appropriate Spatio-Temporal Graph Neural Network (STGNN)
    model based on your specific requirements and constraints.
    """)


    st.subheader("Please answer the following questions:")

    temporal_resolution = st.selectbox(
        "What is your required temporal resolution?",
        ["High (5-minute intervals)", "Medium (Hourly)", "Low (Weekly or longer)"],
        help="Select the time interval between consecutive data points in your dataset"
    )

    resources = st.selectbox(
        "What computational resources do you have available?",
        ["High (Powerful hardware available)", "Medium", "Limited"],
        help="Consider your available computing power and memory"
    )

    prediction_horizon = st.selectbox(
        "What is your prediction horizon requirement?",
        ["Short-term", "Long-term"],
        help="Short-term is typically hours to days, long-term is weeks to months"
    )

    data_volume = st.selectbox(
        "What is your data volume?",
        ["Large dataset", "Medium dataset", "Sparse dataset"],
        help="Consider both the number of nodes and temporal length of your dataset"
    )

    architecture_preference = st.selectbox(
        "What is your architecture preference?",
        ["Factorized (Faster training)", "Coupled (Better accuracy)"],
        help="Factorized architectures offer faster training, while coupled architectures provide better accuracy"
    )


    if st.button("Get Recommendation"):
        recommended_model, scores = get_model_recommendation(
            temporal_resolution, resources, prediction_horizon, data_volume, architecture_preference
        )

        st.success(f"### Recommended Model: {recommended_model}")

        # Display model characteristics
        st.subheader("Model Characteristics:")
        model_info = {
            'DCRNN': {
                'Computational Requirements': 'High',
                'Best Use-Case': 'High-frequency traffic data',
                'Key Strength': 'Superior performance with high-frequency data',
                'Architecture Type': 'Coupled'
            },
            'A3TGCN': {
                'Computational Requirements': 'Moderate',
                'Best Use-Case': 'Balanced applications',
                'Key Strength': 'Stability with data limitations',
                'Architecture Type': 'Hybrid'
            },
            'ASTGCN': {
                'Computational Requirements': 'High',
                'Best Use-Case': 'Dense network analysis',
                'Key Strength': 'Rapid convergence (20 epochs)',
                'Architecture Type': 'Coupled'
            },
            'SSTGCN': {
                'Computational Requirements': 'Low',
                'Best Use-Case': 'Resource-constrained scenarios',
                'Key Strength': 'Efficient with large datasets',
                'Architecture Type': 'Factorized'
            },
            'EvolveGCN': {
                'Computational Requirements': 'Moderate',
                'Best Use-Case': 'Long-term temporal patterns',
                'Key Strength': 'Good performance with weekly intervals',
                'Architecture Type': 'Hybrid'
            },
            'ST-GCN': {
                'Computational Requirements': 'Moderate',
                'Best Use-Case': 'Baseline performance',
                'Key Strength': 'Reliable baseline with moderate compute',
                'Architecture Type': 'Factorized'
            }
        }

        for key, value in model_info[recommended_model].items():
            st.write(f"**{key}:** {value}")


        st.subheader("Model Scoring Breakdown:")
        scores_df = pd.DataFrame(
            list(scores.items()),
            columns=['Model', 'Score']
        ).sort_values('Score', ascending=False)

        st.bar_chart(scores_df.set_index('Model'))

        st.subheader("Additional Considerations:")
        st.write("""
        - Model performance is highly context-dependent
        - Architecture coupling directly impacts prediction horizon capability
        - Consider running benchmarks with your specific dataset
        - Monitor model training time and resource usage
        - Evaluate the trade-off between accuracy and computational requirements
        - Higher computational requirements generally correlate with improved accuracy
        - Model selection should account for expected data growth patterns
        """)


if __name__ == "__main__":
    main()
