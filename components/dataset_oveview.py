
import streamlit as st

def dataset_overview(df):
    head_color = '#fff0d4'
    # dataset shape
    st.markdown('<h4 style="font-weight:600; font-size:32px; color:head_color;">Dataset Shape</h4>', unsafe_allow_html=True)
    rows, columns = st.columns(2)
    with rows:
        st.metric("Rows: ", df.shape[0])
    with columns: 
        st.metric('Columns:', df.shape[1])

    # st.write('Missing Value: \n',df.isnull().sum())
    
    # dataset
    st.markdown('<h4 style="font-weight:600; font-size:32px; color:head_color;">Dataset Overview</h4>', unsafe_allow_html=True)
    st.write(df)


    # dataset Information
    st.markdown('<h4 style="font-weight:600; font-size:32px; color:head_color;">Dataset (Numerical Columns) Information</h4>', unsafe_allow_html=True)
    st.write(df.describe())
    
