# Load libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import random, os
import json, joblib
import sklearn.metrics as metrics
from pathlib import Path
from sklearn.metrics import average_precision_score, accuracy_score, precision_recall_curve, f1_score

import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # For py file
sys.path.append(os.path.abspath('..'))

from config import Config
from churn_predictor import ChurnPredictor
from preprocessing import build_preprocessor, PreprocessConfig
from models.model import build_gb, calibrate_prefit

os.environ["PYTHONHASHSEED"] = str(Config.SEED)
random.seed(Config.SEED)
np.random.seed(Config.SEED)
tf.keras.utils.set_random_seed(Config.SEED)
tf.config.experimental.enable_op_determinism()

# Load dataframe+split
df = pd.read_csv(Config.DATA_URL)
cp = ChurnPredictor(drop_cols=['Unnamed: 0', 'customer_id'], corr_threshold=None, expect_numeric=True) # False if str data still exist

X_train_full, X_valid, X_test, y_train_full, y_valid, y_test = cp.split(
    df, y_col='Churn', test_size=Config.TEST_SIZE, val_size=Config.VAL_SIZE, seed=Config.SEED)

# Training without new features (based on test file)
cfg = PreprocessConfig(drop_cols=['Unnamed: 0', 'customer_id'], corr_threshold=None, expect_numeric=True)
preproc, get_names = build_preprocessor(X_train_full, cfg, include_interactions=False)
preproc.fit(X_train_full)

# Transform all splits
X_tr = preproc.transform(X_train_full)
X_va = preproc.transform(X_valid)
X_te = preproc.transform(X_test)

# Tuning
best_hp = cp.tune(X_tr, y_train_full, X_va, y_valid, project_name='krs_hyperband')
best_params = cp.pick_best_params(min_val_acc=Config.MIN_VAL_ACCURACY)
print("Best parameters: ", best_params)

# Final fit
class_w = cp.compute_class_weight(y_train_full)
hist = cp.fit_final(X_tr, y_train_full, X_va, y_valid, 
                    best_params, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, class_weights=class_w)

# Evaluate
metrics_nn = cp.evaluate(X_te, y_test)
print(f"(Keras NN)Test AUPRC: {metrics_nn['auprc']:.4f} | Test accuracy: {metrics_nn['accuracy']:.4f}")

# GB baseline
gb = build_gb(random_state=Config.SEED)
gb.fit(X_tr, y_train_full)

# Calibrate on validation
gb_cal = calibrate_prefit(gb, X_va, y_valid, method='isotonic')

# Pick threshold on validation
proba_va = gb_cal.predict_proba(X_va)[:,1]
prec, rec, thr = precision_recall_curve(y_valid, proba_va)
preds_va = (proba_va[:, None] >= thr[None, :]).astype(int)

# Vectorised metrics
auprc_curve = np.array([
    average_precision_score(y_valid, preds_va[:, i]) for i in range(preds_va.shape[1])
])
acc_curve = np.mean(preds_va==y_valid[:, None], axis=0)
f1_curve = np.array([f1_score(y_valid, preds_va[:, i]) for i in range(preds_va.shape[1])])

# Choose threshold
target_auprc = 0.6
feasible = auprc_curve >= target_auprc
if feasible.any():
    best_idx_acc = np.argmax(acc_curve*feasible)
else:
    best_idx_acc = np.argmax(acc_curve)
    
best_thr_acc = float(thr[best_idx_acc])
best_idx_f1 = np.argmax(f1_curve)
best_thr_f1 = float(thr[best_idx_f1])

print(f"[VAL] Best-ACC threshold:{best_thr_acc:.3f} | " f"ACC={acc_curve[best_idx_acc]:.4f} | " f"AUPRC={auprc_curve[best_idx_acc]:.4f}")
print(f"[VAL] Best-F1 threshold:{best_thr_f1:.3f} | " f"F1={f1_curve[best_idx_f1]:.4f} | " 
      f"AUPRC={auprc_curve[best_idx_f1]:.4f} | " f"ACC={acc_curve[best_idx_f1]:.4f}")

# Evaluate on test with best acc threshold
proba_te_acc = gb_cal.predict_proba(X_te)[:,1]
pred_te_acc = (proba_te_acc>=best_thr_acc).astype(int)

auprc_acc = average_precision_score(y_test, proba_te_acc)
acc_acc = accuracy_score(y_test, pred_te_acc)
f1_acc = f1_score(y_test, pred_te_acc)
print(f"[TEST] Best threshold: {best_thr_acc:.3f} | Accuracy: {acc_acc:.4f} | F1: {f1_acc:.4f} | AUPRC: {auprc_acc:.4f}")

# Evaluate on test with best f1 threshold
proba_te_f1 = gb_cal.predict_proba(X_te)[:,1]
pred_te_f1 = (proba_te_f1>=best_thr_f1).astype(int)

auprc_f1 = average_precision_score(y_test, proba_te_f1)
acc_f1 = accuracy_score(y_test, pred_te_f1)
f1_f1 = f1_score(y_test, pred_te_f1)
print(f"[TEST] Best threshold: {best_thr_f1:.3f} | Accuracy: {acc_f1:.4f} | F1: {f1_f1:.4f} | AUPRC: {auprc_f1:.4f}")

# Save keras model, preprocessor, params
cp.save_artifacts(Config.MODEL_PATH, Config.SCALER_PATH, Config.PARAMS_PATH)

final_model = gb_cal
final_name = "keras_nn + gb_calibrated_isotonic"

# Save calibrated GB
joblib.dump(final_model, Path(Config.MODEL_PATH).with_stem('gb_calibrated').with_suffix('.joblib'))

# Save path
def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Final results
final_results = {
    "model_name": final_name,
    "keras_nn_metrics": {
        "loss": float(metrics_nn['loss']),
        "auprc": float(metrics_nn['auprc']),
        "accuracy": float(metrics_nn['accuracy']),
    },
    "gb_cal_metrics_f1": { 
        "auprc": float(auprc_f1),
        "accuracy": float(acc_f1),
        "f1": float(f1_f1),
        "threshold": float(best_thr_f1),
    },
    "gb_cal_metrics_acc": { 
        "auprc": float(auprc_acc),
        "accuracy": float(acc_acc),
        "f1": float(f1_acc),
        "threshold": float(best_thr_acc),
    },
    "feature_count": int(X_tr.shape[1]),
    "split": {"train":0.70, "val":0.15, "test":0.15}
}

# Save artifacts
save_json(Path(Config.PARAMS_PATH).with_stem("final_results.json"), final_results)