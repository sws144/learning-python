from sklearn.externals import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import gunicorn

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hola!"

@app.route('/predict',methods=['POST'])
def api_call():
    try:
        test_json = request.get_json()
        val = []
        for dic in test_json:
            row = []
            row.append(dic['sepal_length'])
            row.append(dic['sepal_width'])
            row.append(dic['petal_length'])
            row.append(dic['petal_width'])
            val.append(row)
        #load model
        loaded_model = joblib.load('model/iris_svm_model.pkl')
        y_pred = loaded_model.predict(np.array(val))
        pred_dict = {}
        for i, pred in enumerate(y_pred):
            pred_dict['prediction_' + str(i)] = int(pred)
        responses = jsonify(predictions=pred_dict)
        responses.status_code = 200
    except Exception as e:
        responses = jsonify(predictions={'error':'some error occured, please try again later'})
        responses.status_code = 404
    return (responses)        