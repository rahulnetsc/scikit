# feature engineering, missing values
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,MinMaxScaler
import numpy as np
from sklearn.base import TransformerMixin

def fit_imputer(df: pd.DataFrame):
    imputer = SimpleImputer(strategy='median')
    min_max = MinMaxScaler(feature_range=(-1,1))
    df_num = df.select_dtypes(include=[np.number])
    imputer.fit(df_num)
    min_max.fit(df_num)
    return imputer, min_max

def fit_encoder(df_cat:pd.DataFrame):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(df_cat)
    return encoder 
    

def transform_data(df: pd.DataFrame, transform_num: list[TransformerMixin], transform_cat: list[TransformerMixin])->pd.DataFrame:

    df_num = df.select_dtypes(include=[np.number])
    df_cat = df.select_dtypes(exclude=[np.number])

    for transform in transform_num:
        assert df_num.columns.equals(transform.feature_names_in_), f"Mismatch in columns used for {transform}"
        df_num = pd.DataFrame(transform.transform(df_num),columns=df_num.columns, index=df.index)
    
    for transform in transform_cat:
        assert df_cat.columns.equals(transform.feature_names_in_), f"Mismatch in columns used for {transform}"
        try:
            columns = transform.get_feature_names_out(df_cat.columns)
        except:
            columns = df_cat.columns
        df_cat = pd.DataFrame(transform.transform(df_cat), columns = columns, index= df.index)

    return pd.concat([df_num,df_cat],axis=1)


