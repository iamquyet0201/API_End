from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import os

# ✅ Kiểm tra các file model & scaler
required_files = [
    "model_generator.keras",
    "model_forecaster.keras",
    "scaler_generator.save",
    "scaler_forecaster.save",
    "scaler_fs_output.save"
]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Thiếu file: {missing}")

# ✅ Khởi tạo Flask app
app = Flask(__name__)

# ✅ Kích hoạt CORS toàn cục
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model và scaler
model_gen = tf.keras.models.load_model("model_generator.keras")
model_fore = tf.keras.models.load_model("model_forecaster.keras")
scaler_gen = joblib.load("scaler_generator.save")
scaler_fore = joblib.load("scaler_forecaster.save")
scaler_fs = joblib.load("scaler_fs_output.save")

# ✅ Phân loại FS
def classify_fs(fs):
    if fs >= 1.5: return "An toàn"
    elif fs >= 1.0: return "Có dấu hiệu"
    else: return "Nguy cơ cao"

# ✅ Trang chủ kiểm tra
@app.route('/')
def home():
    return (
        "<h2>✅ Landslide FS Prediction API is running!</h2>"
        "<p>Use <code>POST /predict</code> or try GET with query string.</p>"
    )

# ✅ API chính
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            json_data = request.get_json(force=True)
            if not json_data or "features" not in json_data:
                return jsonify({"success": False, "error": "Thiếu trường 'features'"}), 400
            features = np.array([json_data["features"]])

        elif request.method == 'GET':
            keys = ["c", "L", "gamma", "h", "u", "phi", "beta", "elevation", "slope_type"]
            features = [float(request.args.get(k, 0)) for k in keys]
            features = np.array([features])

        else:
            return jsonify({"success": False, "message": "Chỉ hỗ trợ GET và POST"}), 405

        # ✅ Dự đoán
        scaled = scaler_gen.transform(features)
        sequence = model_gen.predict(scaled)
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
            "message": "⚠️ Kiểm tra định dạng dữ liệu. Cần gửi 9 giá trị đặc trưng."
        }), 400

# Không cần chạy trực tiếp khi deploy trên Render
if __name__ == '__main__':
    app.run(debug=True)
