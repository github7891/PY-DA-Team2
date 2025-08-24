#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load libraries
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.abspath('src'))
from src.churn_predictor import ChurnPredictor


# In[3]:


import os; print(os.getcwd())
import sys; print(sys.path[:3])


# In[ ]:


# Load dataframe

cp = ChurnPredictor(drop_cols=['Unnamed: 0', 'customer_id'], corr_threshold=None, expect_numeric=True)

X_train_full, X_valid, X_test, y_train_full, y_valid, y_test = cp.split(preprocessed_df, y_col='Churn')
cp.preprocess_fit(X_train_full)
X_tr, X_va, X_te = cp.transform_all(X_train_full, X_valid, X_test)

# Tuning
best_hp = cp.tune(X_tr, y_train_full, X_va, y_valid, project_name='krs_hyperband')
best_params = cp.pick_best_params(min_val_acc=0.78)
print("Best parameters: ", best_params)

# Final fit
class_w = cp.compute_class_weight(y_train_full)
hist = cp.fit_final(X_tr, y_train_full, X_va, y_valid, best_params, epochs=50, batch_size=32, class_weights=class_w)

# Evaluate
loss, auprc, acc = cp.evaluate(X_te, y_test)
print(f"Test AUPRC: {auprc:.4f} | Test accuracy: {acc:.4f}")

