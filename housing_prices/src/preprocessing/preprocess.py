# feature engineering, missing values
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np

def fit_imputer(df: pd.DataFrame):
    imputer = SimpleImputer(strategy='median')
    df_num = df.select_dtypes(include=[np.number])
    imputer.fit(df_num)
    return imputer

def cat_transform(df_cat:pd.DataFrame):
    transform = OneHotEncoder()
    return transform.fit_transform(df_cat,sparse = False), transform
    

def transform_data(df: pd.DataFrame, imputer: SimpleImputer)->pd.DataFrame:

    df_num = df.select_dtypes(include=[np.number])
    assert df_num.columns.equals(imputer.feature_names_in_), "Mismatch in columns used for imputation"

    df_cat = df.select_dtypes(exclude=[np.number])
    cat_array = cat_transform(df_cat)
    cat_pd = pd.DataFrame(cat_array,columns=df_cat.columns, index= df.index)
    imputed_array = imputer.transform(df_num)
    impute_df= pd.DataFrame(imputed_array,columns=df_num.columns,index=df.index)
    return pd.concat([impute_df,df_cat,cat_pd],axis=1)


