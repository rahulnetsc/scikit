from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data: pd.DataFrame,  strat_col: str ,test_size: float= 0.2, valid_size: float| None = None, random_state: int =42):

    if not valid_size:
        data_train, data_test  = train_test_split(data,test_size=test_size,stratify=strat_col,random_state=random_state)
        return data_train, data_test, None
    else:
        data_temp, data_test = train_test_split(data,test_size=test_size,stratify= strat_col,random_state=random_state)
        data_train, data_valid = train_test_split(data_temp, test_size=valid_size/(1-test_size),stratify= strat_col, random_state=random_state)
        return data_train, data_test, data_valid