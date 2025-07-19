from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from pathlib import Path
import yaml

from src.data import data_loader
from src.utils import plt_digit

def main():
    config_path = Path('config/config.yaml')
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)

    X,y = data_loader()

    print(X.shape)
    print(y.shape)
    # print(X[0])

    plt_digit(X[0])
    print(y[0])

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    X_train, y_train = shuffle(X_train,y_train, random_state= config["state"])

if __name__ == '__main__':
    main()