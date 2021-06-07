import numpy as np

# Metric - RMSLE

# Reference: https://www.kaggle.com/marknagelberg/rmsle-function

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))