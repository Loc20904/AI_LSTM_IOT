from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load mô hình đã huấn luyện (mô hình được huấn luyện trên dữ liệu đã được chuẩn hóa về [0,1])
model = tf.keras.models.load_model("model.h5")

# Giả sử dữ liệu gốc (raw) được chuẩn hóa từ khoảng [20, 30]
RAW_MIN = 20.0
RAW_MAX = 30.0

@app.route("/predict", methods=["POST"])
def predict():
    # Kiểm tra request có JSON hay không
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400

    # Kiểm tra key "data" có tồn tại không
    if "data" not in request.json:
        return jsonify({"error": "Missing 'data' key in JSON payload"}), 400

    data = request.json["data"]

    # Chuyển dữ liệu sang numpy array (giá trị nhiệt độ thô)
    try:
        input_data = np.array(data, dtype=np.float32)
    except Exception as e:
        return jsonify({"error": "Invalid data format", "message": str(e)}), 400

    # Đảm bảo input_data có shape (batch_size, sequence_length, 1)
    if input_data.ndim == 1:
        # Nếu chỉ có 1 chuỗi với shape (sequence_length,)
        input_data = input_data.reshape((1, input_data.shape[0], 1))
    elif input_data.ndim == 2:
        # Nếu gửi nhiều chuỗi, mỗi chuỗi có shape (sequence_length,)
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        return jsonify({"error": "Invalid input dimensions"}), 400

    # Chuẩn hóa dữ liệu từ raw về [0,1]
    input_normalized = (input_data - RAW_MIN) / (RAW_MAX - RAW_MIN)

    # Dự đoán với mô hình (mô hình dự đoán giá trị đã chuẩn hóa)
    try:
        prediction_normalized = model.predict(input_normalized)
    except Exception as e:
        return jsonify({"error": "Prediction error", "message": str(e)}), 500

    # Ngược chuẩn hóa kết quả dự đoán về giá trị nhiệt độ gốc
    prediction_raw = prediction_normalized * (RAW_MAX - RAW_MIN) + RAW_MIN

    return jsonify({"prediction": prediction_raw.tolist()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
