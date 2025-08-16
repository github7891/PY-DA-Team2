import numpy as np
import pandas as pd
import time
import tensorflow as tf
import json, joblib
from pathlib import Path
from tensorflow.keras import mixed_precision as mp
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

import sys, os
sys.path.append(os.path.abspath(".."))
import json, joblib

from preprocessing import build_preprocessor, PreprocessConfig
from models.model import build_model, make_tuner
from config import Config

class ChurnPredictor:
    def __init__(self, drop_cols=None, corr_threshold=None, expect_numeric=True):
        self.cfg = PreprocessConfig(drop_cols=drop_cols, corr_threshold=corr_threshold, expect_numeric=expect_numeric)
        self.preprocessor = None
        self.feature_names_ = None
        self.tuner = None
        self.best_hp = None
        self.model = None

        # Model artefacts
        self.model_path = Path(Config.MODEL_PATH)
        self.scaler_path = Path(Config.SCALER_PATH)
        self.params_path = Path(Config.PARAMS_PATH)

    def split(self, df:pd.DataFrame, y_col: str="Churn", test_size: float=0.30, val_size: float=0.15, seed: int=42):
        X_df = df.drop(columns=[y_col])
        y = df[y_col].values

        # Split train test val sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_df, y, test_size=test_size, stratify=y, random_state=seed)
        rel_val = val_size / (1.0 - test_size)
        X_train_full, X_valid, y_train_full, y_valid = train_test_split(
            X_temp, y_temp, test_size=rel_val, stratify=y_temp, random_state=seed)
        self._splits = (X_train_full, X_valid, X_test, y_train_full, y_valid, y_test)
        return self._splits

    def preprocess_fit(self, X_train_full:pd.DataFrame):
        self.preprocessor, get_names = build_preprocessor(X_train_full, self.cfg)
        self.preprocessor.fit(X_train_full)
        self.feature_names_ = get_names()
        return self

    def transform_all(self, X_train_full, X_valid, X_test):
        X_tr = self.preprocessor.transform(X_train_full)
        X_va = self.preprocessor.transform(X_valid)
        X_te = self.preprocessor.transform(X_test)
        return X_tr, X_va, X_te

    def tune(self, X_tr, y_tr, X_va, y_va, project_name='krs_hyperband'):
        self.tuner = make_tuner(input_dim=X_tr.shape[1], project_name=project_name)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auprc', mode='max', patience=8, restore_best_weights=True)
        self.tuner.search(X_tr, y_tr, validation_data=(X_va, y_va), callbacks=[early_stop], verbose=1)
        self.best_hp = self.tuner.get_best_hyperparameters(1)[0]
        return self.best_hp
        
    def pick_best_params(self, min_val_acc=0.80, keys=('units1', 'units2', 'lr', 'l2', 'drop')):
        trials = [t for t in self.tuner.oracle.get_best_trials() if t.status == 'COMPLETED']
        trials = [t for t in trials
                 if (t.metrics.get_last_value('val_accuracy') is not None 
                 and t.metrics.get_last_value('val_accuracy') >= min_val_acc
                 and t.metrics.get_last_value('val_auprc') is not None)]
        
        if not trials:
                hp = self.best_hp
                return {k:hp.get(k) for k in keys}
            
        best = max(trials, key=lambda t: t.metrics.get_last_value('val_auprc'))
        hpv = best.hyperparameters.values
        return {k:hpv[k] for k in keys}

    def compute_class_weight(self, y):
        classes = np.unique(y)
        w = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=y)
        return dict(enumerate(w))

    def fit_final(self, X_tr, y_tr, X_va, y_va, 
                  best_params:dict, epochs=50, batch_size=32, class_weights=None, 
                  save_path:str | None=None, use_gpu: bool=True, use_mixed_precision: bool=False):
        # Speed up tuning and training time
        policy_ctx = None
        if use_mixed_precision:
            mp.set_global_policy('mixed_float16')
            
        tf.keras.backend.clear_session()
        self.model = build_model(X_tr.shape[1], **best_params)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_auprc', mode='max', patience=8, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_auprc', mode='max', factor=0.5, patience=4, verbose=0),
                    # Keep max auprc
                    ModelCheckpoint("models/best_churn_model.keras", monitor='val_auprc', mode='max', save_best_only=True)]

        device_name = '/GPU:0' if (use_gpu and tf.config.list_physical_devices('GPU')) else 'CPU:0' # Choose device if available
        start = time.perf_counter()
        with tf.device(device_name):
            history = self.model.fit(X_tr, y_tr, 
                    validation_data=(X_va, y_va), 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks, 
                    class_weight=class_weights,
                    verbose=0)
        train_time_sec = time.perf_counter() - start

        # Save best model
        if save_path:
            self.model.save(save_path)
            
        return history, train_time_sec

    def evaluate(self, X_te, y_te):
        results = self.model.evaluate(X_te, y_te, verbose=0)

        # This model calculates accuracy and auprc, change other metrics here
        if isinstance(results, (list, tuple)) and len(results) >= 3:
            loss, auprc, acc = results[:3]
            return {"loss": loss, "auprc": auprc, "accuracy": acc} # For readibility
        return results

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0).ravel()

    def save_artifacts(self, model_path: Path, scaler_path: Path, params_path: Path):
        # Save model
        if self.model is None:
            raise RuntimeError("Model not trained")
        self.model.save(str(model_path))

        # Save preprocessor
        joblib.dump(self.preprocessor, scaler_path)

        # Save best hp
        if self.best_hp is not None:
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump({k:self.best_hp.get(k) for k in ['units1', 'units2', 'units3', 'lr'] if self.best_hp.get(k) is not None}, f, indent=2)

    # Fit with tensorboard
    def fit_with_tensorboard(self, X_train, y_train, X_val, y_val, 
                             best_params:dict, 
                             epochs:int, batch_size:int, class_weights=None, 
                             tb_cb:Callback | None=None):
        tf.keras.backend.clear_session()
        self.model = build_model(X_train.shape[1], **best_params)
        cbs = [tf.keras.callbacks.EarlyStopping(monitor='val_auprc', mode='max', patience=8, restore_best_weights=True)]
        if tb_cb is not None:
            cbs.append(tb_cb)
        
        history = self.model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs, 
            batch_size=batch_size, 
            class_weight=class_weights, 
            callbacks=cbs, 
            verbose=0
        )
        return history