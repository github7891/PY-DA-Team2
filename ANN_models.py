#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# In[2]:


# For ANN tensorflow keras
class ANN_tf:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        
    # Define the ANN architecture
    def model_keras(self, hp):
    
        # Initialise ANN
        model = Sequential()
    
        # Input layer and 1st hidden layer
        model.add(Dense(units=hp.Int('units1', min_value=32, max_value=128, step=32), kernel_initializer = 'normal', activation = 'relu', input_shape = (self.input_dim, )))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    
        # 2nd layer
        model.add(Dense(units=hp.Int('units2', min_value=16, max_value=64, step=16), kernel_initializer = 'normal', activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    
        # 3rd layer
        model.add(Dense(units=hp.Int('units3', min_value=8, max_value=32, step=8), kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
    
        # Output layer
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'sigmoid'))
    
        # Optimise
        model.compile(optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')), loss = 'binary_crossentropy', metrics = [tf.keras.metrics.AUC(curve='PR', name='auprc'), 'accuracy'])
    
        return model 
    
    def model_rebuild(self, units1, units2, units3, lr):
    
        # Initialise ANN
        model = Sequential()
    
        # Input layer and 1st hidden layer
        model.add(Dense(units=units1, kernel_initializer = 'normal', activation = 'relu', input_shape = (self.input_dim, )))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    
        # 2nd layer
        model.add(Dense(units=units2, kernel_initializer = 'normal', activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    
        # 3rd layer
        model.add(Dense(units=units3, kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization())
    
        # Output layer
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'sigmoid'))
    
        # Optimise
        model.compile(optimizer=Adam(learning_rate=lr), loss = 'binary_crossentropy', metrics = [tf.keras.metrics.AUC(curve='PR', name='auprc'), 'accuracy'])
    
        return model

