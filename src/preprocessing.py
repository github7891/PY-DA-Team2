from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class PreprocessConfig:
    drop_cols: list | None = None
    expect_numeric: bool = True # False if categorical exist

def build_preprocessor(df:pd.DataFrame, cfg:PreprocessConfig):

    X = df.copy()
    if cfg.drop_cols:
        X = X.drop(columns = [c for c in cfg.drop_cols if c in X.columns], errors="ignore")
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [] if cfg.expect_numeric else X.select_dtypes(include=['object','category','bool']).columns.tolist()
    
    # Pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
    ])

    transformers = []
    if num_cols:
        transformers.append(('num', num_pipe, num_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipe, cat_cols))
    ct = ColumnTransformer(
        transformers=transformers, 
        remainder='drop',
        verbose_feature_names_out=False
    )

    steps = [('ct', ct)]

    # Full pipeline
    preprocessor = Pipeline(steps)

    # Post transform - feature list
    def get_feature_names() -> list[str]:
        return preprocessor.named_steps['ct'].get_feature_names_out().tolist()

    return preprocessor, get_feature_names