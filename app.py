import streamlit as st
from src.data_loader import load_dataset
from components.dataset_oveview import dataset_overview



#Title 
st.set_page_config(
    page_title='House Prize Prediction',
    page_icon='🏡',
    layout='wide'

)
st.title("House Prize Prediction🏡")

# Read Dataset 
df = load_dataset()


# Show dataset Overview
dataset_overview(df)

# Now perform EDA
# Eda perform in notebook/eda.ipynb file 



with st.sidebar:
    st.write("House prize prediction parameters")