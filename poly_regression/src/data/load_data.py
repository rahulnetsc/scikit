import random
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple

def data_gen(num_samples:int = 1000, dim: int = 1, poly_degree: int|None = None, noise_std: float = 1)-> Tuple[
                                np.ndarray, np.ndarray, np.ndarray]:

    if poly_degree is None:
        poly_degree = random.randint(0,10)
    
    X= np.random.randn(num_samples, dim)
    poly = PolynomialFeatures(degree=poly_degree,include_bias=True)
    features = poly.fit_transform(X)
    coeff = np.random.randn(features.shape[1])
    data_label = np.dot(features,coeff) + np.random.randn(num_samples) * noise_std

    return X, data_label, coeff
