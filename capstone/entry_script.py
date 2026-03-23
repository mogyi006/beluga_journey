import json

import numpy as np
import joblib

from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('Credit_Default_Prediction')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        result = model.predict(data)
        # Any data type can be returned if json can be serialized
        return result.tolist()
    except Exception as e:
       error = str(e)
       return error
