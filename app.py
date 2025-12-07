from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

#Load trained model and scale

model = pickle.load(open("svc_model.pkl", "rb"))
scale = pickle.load(open("scale.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]          #Expect list of 30 values
    data = np.array(data).reshape(1, -1)
    data = scale.transform(data)

    prediction = model.predict(data)

    result = "benign  (Safe)" if prediction[0] == 1 else "Malignant  (UnSafe)"
    return jsonify({"prediction": result})


