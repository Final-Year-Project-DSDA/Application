import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_sample_data():
    data = {
        "METR-LA": pd.DataFrame({
            "Model": ["A3T-GCN", "DCRNN"],
            "RMSE": [3.8989, 5.38],
            "MAE": [2.6480, 2.77],
            "MAPE": [168.376, 122.95],
        }),
        "PEMS-D7": pd.DataFrame({
            "Model": ["ASTGCN", "SSTGCN", "ST-GCN"],
            "RMSE": [25.27, 59.7008, 6.90],
            "MAE": [16.63, 58.1919, 4.21],
            "MAPE": [192.87, 98.446, 160.32],
        }),
        "CHICKENPOX": pd.DataFrame({
            "Model": ["A3T-GCN", "DCRNN", "EvolveGCN-O", "EvolveGCN-H", "MPNN-LSTM", "AGCRN"],
            "RMSE": [0.9752, 0.9672, 0.9777, 0.9786, 1.0679, 1.0209],
            "MAE": [0.6303, 0.6183, 0.6299, 0.6373, 0.7670, 0.6881],
            "MAPE": [542.53, 538.73, 684.62, 944.008, 2138.42, 822.22],
        }),
    }
    return data


data_dict = load_sample_data()

def main():
    st.title("Model Performance Viewer")

    st.sidebar.header("Options")
    dataset_name = st.sidebar.selectbox("Select Dataset", list(data_dict.keys()))
    metric = st.sidebar.selectbox("Select Performance Metric", ["RMSE", "MAE", "MAPE"])


    selected_data = data_dict[dataset_name]


    st.subheader(f"{metric} of Models on {dataset_name}")
    
    try:
        fig, ax = plt.subplots()
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_data)))
        bars = ax.bar(selected_data["Model"], selected_data[metric], color=colors)
        

        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} of Models on {dataset_name}")
        ax.set_xticks(range(len(selected_data["Model"])))
        ax.set_xticklabels(selected_data["Model"], rotation=45, ha="right")
        

        st.pyplot(fig)
    except KeyError:
        st.error("Selected metric is not available in the dataset.")

if __name__ == "__main__":
    main()