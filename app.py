# File: app.py
import streamlit as st
from introduction import Introduction

def main():
    st.set_page_config(
        page_title="GNN Information Center",
        page_icon="🧠",
        layout="wide"
    )
    
    # Hide the "Made with Streamlit" footer
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Main content
    st.title("GNN Information Center")
    
    # Display introduction content
    intro = Introduction()
    intro.display()

if __name__ == "__main__":
    main()