from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

class Config:
    SEED = 42

    # Load data
    DATA_URL = ("https://raw.githubusercontent.com/github7891/PY-DA-Team2/"
       "feature-engineer/data-preprocessing/data/processed/"
       "Dataset_ATS_v2_processed.csv")
    
    # Model artefact
    ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = ROOT/"models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    PARAMS_PATH = MODELS_DIR/"best_param.json"
    
    # Split (default)
    VAL_SIZE = 0.15
    TEST_SIZE = 0.3
    
    # Train
    BATCH_SIZE = 32
    EPOCHS = 50
    EARLY_STOP_PATIENCE = 10
    MIN_VAL_ACCURACY = 0.8

