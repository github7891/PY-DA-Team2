# Churn Prediction Model with Sklearn and Tensorflow

The telecommunications industry is facing a significant transformation driven by the rapid advancement of digital technologies. 
However, high customer churn rates have long been a pressing issue for leading companies. This project aims to analyse the provided telecommunication dataset and identify the key drivers that influence customer churn.
In this stage of project, the objectives for this stage are:

* Define ANN model architecture
* Train and optimise the model
* Predict customer churn based on critical attributes
* Evaluate model performance and analyse predictions

It is recommended that this project should be worked under an isolated environment to prevent libraries installation conflicts. To set up a virtual environment, 
```
$ python3 -m pip install --user -U virtualenv
Successfully installed virtualenv
$ cd $PROJECT_PATH
$ python3 -m virtualenv my_env
Installing...done.
```
To activate the environment,
```
$ cd $PROJECT_PATH
$ source my_env/bin/activate # Linux/macOS
$.\my_env\Scripts\activate # Windows
```
Then you may simply install the required libraries using pip commands. If you are working on Jupyter with a virtualenv, you need to register it to Jupyter.
If you are not working in a virtual environment, the library versions for this project are:
```
Pandas version: 2.2.3
Numpy version: 2.1.3
TensorFlow version: 2.20.0
Scikit-learn version: 1.6.1
Keras Tuner version: 1.0.5
Matplot version: 3.10.0
Seaborn version: 0.13.2
```

The structure of this project includes a main folder named **project** and the subfolders as follows
```  
 project/
├─ config.py
├─ datasets
├─ main.py/ipynb
├─ models/
│  └─model.py
├─ src/
│  ├─ preprocessing.py
│  └─churn_predictor.py                   
└─ tests/
   └─ test.py    
```

## src
This folder is home for the `preprocessing.py` and `churn_predictor.py` script that calls the transformers and run through the churn prediction process.

### preprocessing.py
**`FeatureEngineer`**: This class controls the feature interactions, the new features one wish to add to the model training can be defined from here.

**`PreprocessConfig`**: This class is for dropping redundant columns.

`build_preprocessor()`: This function creates a Scikit-learn preprocessing pipeline, based on the settings in `PreprocessConfig`, helps selecting and dealing with the numeric and categorical columns. 
Additionally, it provides a switch for feature combination decision; at last, gather those columns `get_feature_names()` for future visualisations.

### churn_predictor.py
**`ChurnPredictor`**: This class is the full NN workflow manager for predicting customer churn, covering data spitting, preprocessing, fine tuning, model training, and evaluation.

`Hyperparameter Tuning(tune(), pick_best_params())`: Uses Keras tuner with Hyperband search to find the best neural network architecture and parameters based on validation metrics (accuracy and AUPRC).

`fit_final()`: Trains the final nn model with optional GPU usage, mixed precision for speed, class weighting, and early stopping. It can also save the trained model.

`fit_with_tensorboard()`: This function trains the model while integrating TensorBoard logging and early stopping, using the best hyperparameters discovered during tuning.

## datasets
This is a folder that saves the optional splits and predictions for future visualisation.

## models
### model.py
This script defines and configures the keras models used in churn prediction pipeline. 

`make_tuner()`: Uses Hyperband search to find the best hyperparameters for the neural network.

`build_gb()`: Builds and returns a Gradient Boosting classifier with optional parameters.

`calibrate_prefit()`: Wraps a fitted classifier with calibration to adjust predicted probabilities to reach a target accuracy.

## viz.py
Some visualisations for business analysis.
