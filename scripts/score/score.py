import json
import numpy as np
import os
import pickle
import joblib
import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback
from keras.models import load_model
import flask
import cv2
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.applications.densenet import preprocess_input
import glob
import pandas as pd
from flask import send_file

 

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'weights.best.dense_generator_callback.hdf5')
    model = load_model(model_path,compile=False)


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model.predict(data)
    return y_hat.tolist()