from joblib import load, dump
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import numpy as np

def train(train_data:pd.DataFrame, train_labels: pd.DataFrame, 
          test_data: pd.DataFrame, test_labels: pd.DataFrame, 
          model_type, cv: int = None):

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'rfr':
        model = RandomForestRegressor()
    elif model_type == 'gbr':
        model = GradientBoostingRegressor()
    elif model_type == 'hgb':
        model = HistGradientBoostingRegressor()
    elif model_type == 'svr':
        model = SVR(kernel= 'rbf')
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if cv is not None and cv > 1:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, train_data, train_labels,
                                 scoring="neg_root_mean_squared_error", cv=kf)
        rmse_scores = -scores  # convert to positive RMSE
        print(f"Cross-validated RMSEs: {rmse_scores}")
        print(f"Mean RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
        return rmse_scores.mean()  # or return all scores if you like
    
    model.fit(train_data, train_labels)

    preds = model.predict(test_data)
    rmse = np.sqrt(mse(preds, test_labels))

    return rmse

