from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json

app = Flask(__name__)

# Load mô hình đã huấn luyện
model = load_model("temperature_model.h5")

# Khởi tạo MinMaxScaler như đã dùng khi huấn luyện
# (Lưu ý: Trong thực tế, bạn nên lưu lại đối tượng scaler hoặc tham số để scale dữ liệu mới)
scaler = MinMaxScaler(feature_range=(0, 1))
# Ví dụ, giả sử giá trị nhiệt độ nằm trong khoảng [20, 30]:
scaler.fit(np.array([[20], [30]]))

sequence_length = 10

# Hàm tạo chuỗi dữ liệu cho mô hình
def create_sequence(input_data, seq_length=10):
    # Giả sử input_data là một danh sách các giá trị nhiệt độ (đã được scale)
    if len(input_data) < seq_length:
        return None
    return np.array(input_data[-seq_length:]).reshape(1, seq_length, 1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Giả sử client gửi dữ liệu nhiệt độ dưới dạng danh sách: {"temperature_data": [..]}
    temp_data = data.get("temperature_data", [])
    if len(temp_data) < sequence_length:
        return jsonify({"error": "Not enough data, need at least {} data points.".format(sequence_length)}), 400
    
    # Chuẩn hóa dữ liệu theo scaler
    temp_array = np.array(temp_data).reshape(-1, 1)
    temp_scaled = scaler.transform(temp_array)
    sequence = create_sequence(temp_scaled.flatten().tolist(), sequence_length)
    if sequence is None:
        return jsonify({"error": "Error creating sequence."}), 500
    
    # Dự đoán
    prediction_scaled = model.predict(sequence)
    # Chuyển đổi dự đoán về giá trị thực (giả sử scaler ngược lại)
    # Vì scaler của chúng ta được fit trên [20, 30], ngược lại:
    prediction_real = scaler.inverse_transform(prediction_scaled)[0][0]
    
    return jsonify({"predicted_temperature": prediction_real})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
