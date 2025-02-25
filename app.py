import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify

# Giả lập dữ liệu
time = np.arange(0, 500)
temperature = 25 + 5 * np.sin(0.02 * time) + np.random.normal(0, 0.5, len(time))
df = pd.DataFrame({'temperature': temperature})

# Tạo chuỗi dữ liệu (ví dụ dự đoán giá trị tiếp theo dựa vào 10 giá trị trước)
sequence_length = 10
X, y = [], []
for i in range(len(df) - sequence_length):
    X.append(df['temperature'].values[i:i+sequence_length])
    y.append(df['temperature'].values[i+sequence_length])
X = np.array(X)
y = np.array(y)

# Chuẩn hóa (nếu cần)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_scaled, y_scaled)

# Tạo API Flask đơn giản
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    temp_data = data.get("temperature_data", [])
    if len(temp_data) < sequence_length:
        return jsonify({"error": "Not enough data, need at least {} data points.".format(sequence_length)}), 400
    # Chuẩn hóa dữ liệu mới
    temp_data = np.array(temp_data).reshape(1, -1)
    temp_data_scaled = scaler.transform(temp_data)
    prediction_scaled = model.predict(temp_data_scaled)
    # Chuyển về giá trị thực (ở đây scaler đơn giản có thể đảo ngược qua hàm inverse_transform)
    prediction_real = scaler.inverse_transform(prediction_scaled.reshape(-1,1))[0][0]
    return jsonify({"predicted_temperature": prediction_real})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
