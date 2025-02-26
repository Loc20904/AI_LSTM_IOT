from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load model khi khởi động app
model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    input_data = np.array(data)
    prediction = model.predict(input_data)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
