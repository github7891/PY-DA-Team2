from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# This is a scikit-learn model preprocessing pipeline
class churn_preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None, corr_threshold=0.999, num_cols=None, cat_cols=None, remove_redundant=True):
        '''Remove redundant features'''
        self.drop_cols = drop_cols or []
        self.corr_threshold = corr_threshold
        self.num_cols = None if num_cols is None else list(num_cols)
        self.cat_cols = None if cat_cols is None else list(cat_cols)
        self.remove_redundant = remove_redundant

        # Set in fit
        self.auto_drop_ = []
        self.kept_columns_ = []
        self.prep_ = None
        self.feature_names_ = []
        self.corr_matrix_ = None

    def num_for_corr(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Transform to numeric data to check correlation'''
        df_num = df.copy()

        # Convert object/categorical
        for c in df_num.select_dtypes(include=["object","category"]).columns:
            df_num[c] = pd.Categorical(df_num[c]).codes
        return df_num.apply(pd.to_numeric, errors="coerce")

    def _infer_column_types(self, Xdf: pd.DataFrame):
        if self.num_cols is None or self.cat_cols is None:
            num = Xdf.select_dtypes(include=[np.number]).columns.tolist()
            cat = Xdf.columns.difference(num).tolist()
            if self.num_cols is None: self.num_cols = num
            if self.cat_cols is None: self.cat_cols = cat
            
    def _find_redundant(self, df_num: pd.DataFrame):
        if df_num.shape[1] <= 1:
            return []
        corr = df_num.corr(numeric_only=True).abs()
        self.corr_matrix_ = corr
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] >= self.corr_threshold)]
        return to_drop
        
    def fit(self, X, y=None):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Drop useless columns
        Xdf = Xdf.drop(columns=self.drop_cols, errors="ignore")
        self._infer_column_types(Xdf)

        # Remove highly correlated numeric columns
        self.auto_drop_ = []
        if self.remove_redundant and len(self.num_cols) > 1:
            Xnum_for_corr = self.num_for_corr(Xdf[self.num_cols]).select_dtypes(include=[np.number])
            redundant = self._find_redundant(Xnum_for_corr)
            self.auto_drop_ = redundant

        # Kept columns
        drop_set = set(self.drop_cols) | set(self.auto_drop_)
        self.kept_columns_ = [c for c in X.columns if c in Xdf.columns and c not in drop_set]

        # Build column transformer
        used_num = [c for c in self.num_cols if c in self.kept_columns_ and c not in self.auto_drop_]
        used_cat = [c for c in self.cat_cols if c in self.kept_columns_]

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False))
        ])

        self.prep_ = ColumnTransformer(
            transformers=[
                ("num", num_pipe, used_num),
                ("cat", cat_pipe, used_cat)
            ], remainder="drop"
        ).fit(Xdf)

        # Feature names
        names_num = used_num
        cat_ohe = self.prep_.named_transformers_["cat"]["ohe"] if used_cat else None
        names_cat = cat_ohe.get_features_names_out(used_cat).tolist() if catohe is not None else []
        self.feature_names_ = names_num + names_cat # For viz later
        return self
        
    def transform(self, X):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Xdf.drop(columns=self.drop_cols+self.auto_drop_, errors="ignore")

        # Align to kept column order for new data
        Xdf = Xdf.reindex(columns=[c for c in self.kept_columns_ if c in Xdf.columns])
        return self.prep_transform(Xdf)
