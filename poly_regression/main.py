from pathlib import Path
import yaml
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from src.data import data_gen

def main():

    config_path = Path("configs/config.yaml")

    with open(config_path,'r') as f:
        env_configs: dict = yaml.safe_load(f)

    X, y, coeff = data_gen(poly_degree=4)

    # if not Path(env_configs["data_path"]).is_file():
    #     X, y, coeff = data_gen()
        
    #     with open(Path(env_configs["data_path"]),'wb') as f:
    #         pickle.dump((X, y, coeff),f)
    # else:
    #     with open(Path(env_configs["data_path"]),'rb') as f:
    #         X, y, coeff = pickle.load(f)
    import time
    start_time = time.time()
    poly = PolynomialFeatures(degree=3, include_bias=True)
    features = poly.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(features,y)
    recon_coeff  = np.array(lin_reg.coef_)
    recon_y = lin_reg.predict(features)

    print(f"lin_reg.intercept_,lin_reg.coef_: {lin_reg.intercept_,lin_reg.coef_}")
    print(f"original coeff: {coeff}")
    print(f"exec_time: {(time.time()-start_time)/60:.2f} mins")

    # plt.scatter(X,y,c='green', linestyle = '--', label= 'target')
    # plt.scatter(X,recon_y,c='red', linestyle = '--', label = 'prediction')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.title('Polynomial Regression')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    from sklearn.metrics import mean_squared_error, r2_score
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]
    recon_y_sorted = recon_y[sort_idx]
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='green', s=15, alpha=0.6, label='Target')
    plt.plot(X_sorted, recon_y_sorted, color='red', linewidth=2.5, label='Model Prediction')

    # Labels and title
    plt.xlabel("Input Feature $x$", fontsize=12)
    plt.ylabel("Target Value $y$", fontsize=12)
    plt.title("Polynomial Regression Fit", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Performance metrics
    mse = mean_squared_error(y, recon_y)
    r2 = r2_score(y, recon_y)
    # plt.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nMSE = {mse:.2f}",
    #         transform=plt.gca().transAxes,
    #         fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    from matplotlib.offsetbox import AnchoredText

    stats_box = AnchoredText(
        f"$R^2$ = {r2:.3f}\nMSE = {mse:.2f}",
        loc="upper left", frameon=True, prop={'size': 10}
    )
    plt.gca().add_artist(stats_box)
    
    ridge = Ridge(alpha=0.1,solver='cholesky')
    ridge.fit(features,y)
    ridge_y = ridge.predict(features)
    plt.plot(X_sorted, ridge_y[sort_idx], color='blue', linewidth=2.5, label='Ridge Prediction')
    
    from sklearn.linear_model import SGDRegressor
    sgd = SGDRegressor(penalty= 'l2', alpha= 0.01/len(X),random_state=42, max_iter=1000)
    sgd.fit(features,y)
    sgd_y = sgd.predict(features)
    plt.plot(X_sorted, sgd_y[sort_idx], color='orange', linewidth=2.5, label='SGD Prediction')
    
    lasso = Lasso(alpha= 0.01)
    lasso.fit(features,y)
    lasso_y = lasso.predict(features)

    plt.plot(X_sorted, lasso_y[sort_idx], color='violet', linewidth=2.5, label='Lasso Prediction')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/polynomial_fit.png", dpi=300, bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    main()