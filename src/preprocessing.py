from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    # Self-define features
    def __init__(self):
        self.new_cols_ = []

    def fit(self, X, y=None):
        Xc = X.copy()
        cols = set(Xc.columns)

        # Only create features if column exist
        if {"MonthlyCharges", "tenure"}.issubset(cols):
            self.new_cols_.append(("MCxTenure", "MonthlyCharges", "tenure"))
        if {"Dependents", "Contract_Two year"}.issubset(cols):
            self.new_cols_.append(("Dependents_x_TwoYear", "Dependents", "Contract_Two year"))
        if {"Dependents", "Contract_One year"}.issubset(cols):
            self.new_cols_.append(("Dependents_x_OneYear", "Dependents", "Contract_One year"))
        if {"PhoneService", "MultipleLines"}.issubset(cols):
            self.new_cols_.append(("Phone_x_Multi", "PhoneService", "MultipleLines"))
        if {"InternetService_Fiber optic", "MonthlyCharges"}.issubset(cols):
            self.new_cols_.append(("Fiber_x_Charges", "InternetService_Fiber optic", "MonthlyCharges"))
        if {"SeniorCitizen", "Contract_One year"}.issubset(cols):
            self.new_cols_.append(("Senior_x_OneYear", "SeniorCitizen", "Contract_One year"))
            
        return self

    def transform(self, X):
        Xc = X.copy()

        for new_name, a, b in self.new_cols_:
            Xc[new_name] = Xc[a]*Xc[b]

        return Xc

@dataclass
class PreprocessConfig:
    drop_cols: list | None = None
    corr_threshold: float | None = None # None if no correlation dropping
    expect_numeric: bool = True # False if categorical exist

def build_preprocessor(df:pd.DataFrame, cfg:PreprocessConfig, include_interactions: bool=True) -> tuple[Pipeline, callable]:
    X = df.copy()
    if cfg.drop_cols:
        X.drop(columns=[c for c in cfg.drop_cols if c in X.columns], errors='ignore', inplace=True)
    
    # Pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
    ])

    # Selectors
    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_include=['object','category','bool']) if not cfg.expect_numeric else []

    ct = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_sel), 
            ('cat', cat_pipe, cat_sel)], 
        remainder='drop',
        verbose_feature_names_out=False
    )

    steps = []
    if include_interactions:
        steps.append(('fe', FeatureEngineer()))
    steps.append(('ct', ct))

    # Full pipeline
    preprocessor = Pipeline(steps)

    # Post transform - drop highly correlated cols
    def get_feature_names() -> list[str]:
        return preprocessor.named_steps['ct'].get_feature_names_out().tolist()

    return preprocessor, get_feature_names