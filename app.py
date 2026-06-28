import streamlit as st
import numpy as np 
import pandas as pd 

# Import load dataset
from src.data_loader import load_dataset

# Import Files
from components.dataset_oveview import dataset_overview
from notebook.preprocessing import preprocess_data
from notebook.feature_encoding import feature_encoding
from notebook.model_training_and_prediction import model_training_and_prediction

# Import Train test split and standard scaler 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import all metrices
from notebook.metrices import metrices_cal

#Title 
st.set_page_config(
    page_title='House Price Prediction',
    page_icon='🏡',
    layout='wide'

)
st.title("House Price Prediction🏡")

# Read Dataset 
df = load_dataset()


# Show dataset Overview
dataset_overview(df)

# Now perform EDA

# Preprocessing 
df = preprocess_data(df)
# st.write(df.shape)

# Feature Scaling 
df = feature_encoding(df)

head_color = "#8df097"

# dataset shape
# st.markdown('<h4 style="font-weight:600; font-size:32px; color: #f7e5c2;">Preprocessed Dataset</h4>', unsafe_allow_html=True)
# st.write(df)

# Scaling and Spliting dataset
X = df.drop(columns=['price']) # Input Features
Y = df['price'] # Target Features


# Split the dataset into training and testing
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# st.write('before scaling\n',X_test)

# Apply Standard Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


st.markdown('<h4 style="font-weight:600; font-size:28px; color: #8df097;">Model Performance Metrices</h4>', unsafe_allow_html=True)

lr_model, Y_pred = model_training_and_prediction(X_train, X_test, Y_train) # return linear regression model and y_prediction 
# st.write('X test After Scaling: ', X_test)
# st.write('Y pred', np.exp(Y_pred))



# Calculate Metrices
metric_list = metrices_cal(Y_test, Y_pred)

r2_score, mae, rmse = metric_list
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('R2 score', round(r2_score,3))
with col2:
    st.metric('MAE (₹)', f"{mae:,.0f}")
with col3:
    st.metric('RMSE (₹)', f"{rmse:,.0f}")


with st.sidebar:
    st.markdown('<h5 style="font-weight:600; font-size:20px; color: #f7e5c2;">House price prediction parameters</h5>', unsafe_allow_html=True)

    # Area
    area = st.number_input('Enter Area in Square feet', step=1, value=0)
    # st.write('area', area)

    # Bedroom
    bedrooms = st.number_input('Enter Number of bedroom', step=1, min_value=1, max_value=4, value=1)

    # Bathroom
    bathrooms = st.number_input('Enter Number of bathroom', step=1, min_value=1, max_value=4, value=1)

    # stories
    stories = st.number_input('Enter number of stories in house', step=1, min_value=1, max_value=4, value=1)

    # Main Road
    mainroad = st.selectbox('Main Road', 
                            ['yes', 'no'])
    
    # Guestroom
    guestroom = st.selectbox('Guestroom', 
                            ['yes', 'no'])
    
    # Basement
    basement = st.selectbox('Basement', 
                            ['yes', 'no'])
    # Hotwaterheating
    hotwaterheating = st.selectbox('Hot water heating', 
                            ['yes', 'no'])
    
    # Air Conditioning
    airconditioning = st.selectbox('Air Conditioning', 
                            ['yes', 'no'])
    
    # Parking
    parking = st.number_input('Parking', step=1, min_value=0, max_value=3, value=1)

    # Prefarea
    prefarea = st.selectbox('Prefarea', 
                            ['yes', 'no'])
    
    # furninshingstatus
    furnishingstatus = st.selectbox('Furnishing status', 
                            ['furnished', 'semi-furnished', 'unfurnished'])
    
    userdata = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'mainroad': [mainroad],
            'guestroom': [guestroom],
            'basement': [basement],
            'hotwaterheating': [hotwaterheating],
            'airconditioning': [airconditioning],
            'parking': [parking],
            'prefarea': [prefarea],
            'furnishingstatus': [furnishingstatus]
        })

st.markdown('<h5 style="font-weight:600; font-size:20px; color: #8df097;">House detail</h5>', unsafe_allow_html=True)
st.write(userdata)

predicted_price = 0
with st.sidebar:
    if st.button('Predict HousePrice'):
        
        userdata = feature_encoding(userdata)
        # st.write('User data: ',userdata)

        userdata_scaled = scaler.transform(userdata.iloc[:,:])
        # print('scaled user data: ',  userdata_scaled)

        predicted_price = np.round(np.expm1(lr_model.predict(userdata_scaled)), 2)
        # st.write('Predicted Price: ',predicted_price)

# st.markdown('<h5 style="font-weight:600; font-size:20px; color: #f7e5c2;">Predicted House Price</h5>', unsafe_allow_html=True)
st.metric('Predicted Price (₹)',predicted_price)