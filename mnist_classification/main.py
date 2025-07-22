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

    # plt_digit(X[0])
    # print(y[0])

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    # X_train, y_train = shuffle(X_train,y_train, random_state= config["state"])
    # y_train_5 = (y_train==5)
    # y_test_5 = (y_test==5)

    from sklearn.linear_model import SGDClassifier
    import time 
    start_time = time.time()

    sgd_clf = SGDClassifier(random_state= config["state"])
    sgd_clf.fit(X_train,y_train)

    preds = sgd_clf.predict(X_test)

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    acc = accuracy_score(y_true=y_test,y_pred=preds)
    prec = precision_score(y_true=y_test, y_pred=preds, average='weighted')
    recall = recall_score(y_true=y_test, y_pred=preds, average='weighted')
    print(f"acc: {acc:.2f}, prec: {prec:.2f}, recall: {recall:.2f} time: {(time.time()-start_time)/60:.2f} mins")

if __name__ == '__main__':
    main()