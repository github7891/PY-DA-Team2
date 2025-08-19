import datetime, time
import pandas as pd
import numpy as np
import tensorflow as tf
import random, os
import sys, json, joblib
import sklearn.metrics as metrics
from pathlib import Path
from sklearn.metrics import average_precision_score, accuracy_score, precision_recall_curve, f1_score

ROOT = Path.cwd()
SRC_DIR = ROOT/"src"
MODELS_DIR = ROOT/"models"

for p in (SRC_DIR, MODELS_DIR, ROOT):
    p_str = str(p.resolve())
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from src.churn_predictor import ChurnPredictor
from src.preprocessing import build_preprocessor, PreprocessConfig
from config import Config
from models.model import build_gb, calibrate_prefit

from tensorflow.keras.callbacks import TensorBoard

os.environ["PYTHONHASHSEED"] = str(Config.SEED)
random.seed(Config.SEED)
np.random.seed(Config.SEED)
tf.keras.utils.set_random_seed(Config.SEED)
#tf.config.experimental.enable_op_determinism()

%load_ext autoreload
%autoreload 2

# Tensorboard callback
log_dir = Path('logs')/("integrate_run"+datetime.datetime.now().strftime('%d%m%Y-%H%M%S'))
tb_cb = TensorBoard(log_dir=str(log_dir), histogram_freq=1, write_graph=True, write_images=True)

# Load dataframe+split
df = pd.read_csv(Config.DATA_URL)
df.head()

cp = ChurnPredictor(drop_cols=['Unnamed: 0', 'customer_id'], corr_threshold=None, expect_numeric=True) # False if str data still exist

X_train_full, X_valid, X_test, y_train_full, y_valid, y_test = cp.split(
    df, y_col='Churn', test_size=Config.TEST_SIZE, val_size=Config.VAL_SIZE, seed=Config.SEED)

# Training without new features (based on test file)
cfg = PreprocessConfig(drop_cols=['Unnamed: 0', 'customer_id'], corr_threshold=None, expect_numeric=True)
preproc, get_names = build_preprocessor(X_train_full, cfg, include_interactions=False)
preproc.fit(X_train_full)
feature_names = get_names()
print(feature_names)

# Transform all splits
X_tr = preproc.transform(X_train_full)
X_va = preproc.transform(X_valid)
X_te = preproc.transform(X_test)

# Tuning
best_hp = cp.tune(X_tr, y_train_full, X_va, y_valid, project_name='krs_hyperband')
best_params = cp.pick_best_params(min_val_acc=Config.MIN_VAL_ACCURACY)
print("Best parameters: ", best_params)

# Checkpoint: Save for fit
payload = {
    "best_params": best_params,
    "seed": Config.SEED,
    "feature_count": int(X_tr.shape[1]),
    "saved_at": time.strftime('%d-%m-%Y-%H:%M:%S')
}

Path(Config.PARAMS_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(Config.PARAMS_PATH, 'w', encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"Saved_params -> {Config.PARAMS_PATH}")

with open(Path(Config.PARAMS_PATH).with_name("best_hp.json"), 'w', encoding="utf-8") as f:
    json.dump(best_hp.values, f, indent=2)

# Final fit (without tensorboard)
class_w = cp.compute_class_weight(y_train_full)
hist = cp.fit_final(X_tr, y_train_full, X_va, y_valid, 
                    best_params, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, class_weights=class_w)

# Evaluate
metrics_nn = cp.evaluate(X_te, y_test)
print(f"(Keras NN)Test AUPRC: {metrics_nn['auprc']:.4f} | Test accuracy: {metrics_nn['accuracy']:.4f}")

# Reload best parameters for fitting (fresh session)
'''
Redo preprocessing and transforms, records are saved after tuning
'''
with open(Config.PARAMS_PATH, 'r', encoding="utf-8") as f:
    saved = json.load(f)
best_params = saved["best_params"]
log_dir = Path('logs')/("integrate_run"+datetime.datetime.now().strftime('%d%m%Y-%H%M%S'))
tb_cb = TensorBoard(log_dir=str(log_dir), histogram_freq=1)

# Final fit (with tensorboard)
class_w = cp.compute_class_weight(y_train_full)
hist = cp.fit_with_tensorboard(X_tr, y_train_full, X_va, y_valid, 
                               best_params, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, class_weights=class_w, tb_cb=tb_cb)

nn_tr = cp.predict_proba(X_tr).ravel()
nn_va = cp.predict_proba(X_va).ravel()
nn_te = cp.predict_proba(X_te).ravel()

X_tr_st = np.column_stack([X_tr, nn_tr])
X_va_st = np.column_stack([X_va, nn_va])
X_te_st = np.column_stack([X_te, nn_te])

# Fit on GB
gb = build_gb(random_state=Config.SEED)
gb.fit(X_tr_st, y_train_full)

# Calibrate on val
gb_cal = calibrate_prefit(gb, X_va_st, y_valid, method='isotonic')
gb_cal.fit(X_va_st, y_valid)

# Pick threshold on validation
proba_va = gb_cal.predict_proba(X_va_st)[:,1]
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

# Evaluate on test with best threshold
proba_te = gb_cal.predict_proba(X_te_st)[:,1]
pred_te_f1 = (proba_te>=best_thr_f1).astype(int)
pred_te_acc = (proba_te>=best_thr_acc).astype(int)

auprc_te = average_precision_score(y_test, proba_te)
acc_te_f1 = accuracy_score(y_test, pred_te_f1)
f1_te_f1 = f1_score(y_test, pred_te_f1)
acc_te_acc = accuracy_score(y_test, pred_te_acc)
f1_te_acc = f1_score(y_test, pred_te_acc)
print(f"AUPRC: {auprc_te:.4f}")
print(f"[TEST-acc_thr] Accuracy: {acc_te_acc:.4f} | F1: {f1_te_acc:.4f}")
print(f"[TEST-f1_thr] Accuracy: {acc_te_f1:.4f} | F1: {f1_te_f1:.4f}")

model = cp.model
model.summary()

# Load Tensorboard in Jupyter
%reload_ext tensorboard
%tensorboard --logdir logs --port 6008

# Save model
joblib.dump(gb_cal, "gb_cal_model.pkl")
cp.model.save("nn_model.keras")

# Save predictions
gb_train = gb_cal.predict_proba(X_tr_st)[:,1]
gb_valid = gb_cal.predict_proba(X_va_st)[:,1]
gb_test = gb_cal.predict_proba(X_te_st)[:,1]
train_df = pd.DataFrame({
    "y_train": y_train_full,
    "nn_train": nn_tr,
    "gb_train": gb_train
})
valid_df = pd.DataFrame({
    "y_valid": y_valid,
    "nn_valid": nn_va,
    "gb_valid": gb_valid
})
test_df = pd.DataFrame({
    "y_test": y_test,
    "nn_test": nn_te,
    "gb_test": gb_test
})

train_df.to_csv("preds_train.csv", index=False)
valid_df.to_csv("preds_valid.csv", index=False)
test_df.to_csv("preds_test.csv", index=False)