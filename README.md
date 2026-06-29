# 🏠 House Price Prediction System

A Machine Learning based web application that predicts house prices based on various property features such as area, number of bedrooms, bathrooms, stories, furnishing status, parking availability, and more.

## 🚀 Project Overview

This project uses a Multiple Linear Regression model to estimate house prices from user-provided inputs. The model was trained on a housing dataset and deployed using Streamlit to provide an interactive user interface.

## ✨ Features

- Predict house prices instantly
- Interactive Streamlit web interface
- Data preprocessing pipeline included
- Feature scaling for improved model performance
- Supports categorical and numerical inputs
- Real-time prediction results

## 📊 Input Features

The model uses the following features for prediction:

- Area (sq ft)
- Number of Bedrooms
- Number of Bathrooms
- Number of Stories
- Main Road Access
- Guest Room Availability
- Basement Availability
- Hot Water Heating
- Air Conditioning
- Parking Spaces
- Preferred Area
- Furnishing Status

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| R² Score | 0.68 |

The model explains approximately **68%** of the variance in house prices.

## ▶️ How to Run
1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application

```bash
streamlit run app.py
```

## Demo
Link: https://drive.google.com/file/d/1Kajn2jUhTsCkLkd_PYrfBbw0Kj0lJIvn/view?usp=sharing

## 🎯 Future Improvements

- Improve prediction accuracy using advanced regression models.
- Hyperparameter tuning.
- Feature engineering.
- Model comparison with Random Forest and XGBoost.
- Deploy the application on Streamlit Cloud.

## 👨‍💻 Author

**Ritik Kumar**

Assistant System Engineer at TCS | GATE CSE Qualified | Aspiring AI/ML Engineer
