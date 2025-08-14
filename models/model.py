import tensorflow as tf
import keras_tuner as kt
from keras_tuner import Objective
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Build keras model
def build_model(input_dim:int, units1=64, units2=32, units3=16, lr=0.001):
    # Initialise ANN
    model = Sequential()
    
    # Input layer and 1st hidden layer
    model.add(Dense(units1, kernel_initializer='he_uniform', activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # 2nd layer
    model.add(Dense(units2, kernel_initializer='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # 3rd layer
    model.add(Dense(units3, kernel_initializer='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    
    # Output layer
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    
    # Optimise
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc'), 'accuracy'])
    
    return model 

def hypermodel(hp:kt.HyperParameters, input_dim:int):
    u1 = hp.Int('units1', min_value=32, max_value=128, step=32)
    u2 = hp.Int('units2', min_value=16, max_value=64, step=16)
    u3 = hp.Int('units3', min_value=8, max_value=32, step=8)
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
    return build_model(input_dim, u1, u2, u3, lr)

def make_tuner(input_dim:int, project_name='krs_hyperband', directory='hyperband'):
    tuner = kt.Hyperband(
    hypermodel=lambda hp:hypermodel(hp, input_dim),
    max_epochs=30,
    objective=Objective('val_auprc', direction='max'),
    executions_per_trial=3,
    directory=directory,
    project_name=project_name,
    overwrite=True)
    return tuner

def build_gb(random_state:int=42, **kwargs) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=random_state, **kwargs)

def calibrate_prefit(estimator, X_valid, y_valid, method:str='isotonic') -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(estimator=estimator, method=method, cv='prefit')
    cal.fit(X_valid, y_valid)
    return cal