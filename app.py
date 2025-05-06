from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# 🔁 Load models và scalers
model_gen = tf.keras.models.load_model("model_generator.keras")
model_fore = tf.keras.models.load_model("model_forecaster.keras")
scaler_gen = joblib.load("scaler_generator.save")
scaler_fore = joblib.load("scaler_forecaster.save")
scaler_fs = joblib.load("scaler_fs_output.save")

# ✅ Khởi tạo Flask app
app = Flask(__name__)

# 🚨 Hàm phân loại cảnh báo dựa vào FS
def classify_fs(fs):
    if fs >= 1.5:
        return "An toàn"
    elif fs >= 1.0:
        return "Có dấu hiệu"
    else:
        return "Nguy cơ cao"

# ✅ Route gốc (GET)
@app.route('/')
def index():
    return (
        "<h2>✅ Landslide FS Prediction API is running!</h2>"
        "<p>Use <code>POST /predict</code> with JSON: <br>"
        "<code>{ \"features\": [c, L, gamma, h, u, phi, beta, elevation, slope_type] }</code></p>"
    )

# ✅ Route dự đoán (POST)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([data["features"]])  # đảm bảo shape (1, 9)

        # 🔁 Bước 1: Chuẩn hóa + dự đoán 3 ngày đặc trưng
        input_scaled = scaler_gen.transform(features)
        sequence = model_gen.predict(input_scaled)

        # 🔁 Bước 2: reshape + dự đoán FS
        sequence_lstm = sequence.reshape((1, 1, sequence.shape[1]))
        fs_scaled = model_fore.predict(sequence_lstm)
        fs = scaler_fs.inverse_transform(fs_scaled)[0]

        result = []
        for i, val in enumerate(fs, start=1):
            result.append({
                "day": i,
                "fs": round(float(val), 2),
                "label": classify_fs(val)
            })

        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "❌ Kiểm tra lại định dạng JSON. Bạn cần gửi {\"features\": [9 số đầu vào]}"
        }), 400

# ✅ Chạy khi debug cục bộ (Render sẽ dùng gunicorn nên không cần dòng này)
if __name__ == '__main__':
    app.run(debug=True)
