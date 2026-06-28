from sklearn.linear_model import LinearRegression

def model_training_and_prediction(X_train, X_test, Y_train):

    # Linear Regression instance
    model = LinearRegression()

    # Model training 
    model.fit(X_train, Y_train)
    
    # Prediction
    Y_pred = model.predict(X_test)
    return (model, Y_pred)
