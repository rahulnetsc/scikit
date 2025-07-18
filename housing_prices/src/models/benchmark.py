from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm 
import time 

def benchmark(X_train, y_train, X_test, y_test):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(), 
        'Lasso': Lasso(),
        'RFG': RandomForestRegressor(), 
        'GBR': GradientBoostingRegressor(), 
        'HistGBR': HistGradientBoostingRegressor(), 
        'KNeighbors': KNeighborsRegressor(),
        'DecisionTreeRegr': DecisionTreeRegressor(),
        'MLPRegressor' : MLPRegressor(max_iter=1500)
        }
    results = {}
    pbar = tqdm(models.items(), desc="Benchmarking models")

    for name, model in pbar:
        pbar.set_description(f"Training {name}")  
        try:
            start_time = time.time()
            model.fit(X_train,y_train)
            preds = model.predict(X_test)
            model_time = round(time.time()-start_time, 2) 
            rmse = np.sqrt(mean_squared_error(y_test,preds))
            results[name] = [rmse, model_time]
        except Exception as e:
            results[name] = f"Error: {e}"
    return results
