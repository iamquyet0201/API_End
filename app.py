from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import os

# 🔁 Kiểm tra mô hình có tồn tại không
required_files = [
    "model_generator.keras",
    "model_forecaster.keras",
    "scaler_generator.save",
    "scaler_forecaster.save",
    "scaler_fs_output.save"
]

missing_files = [f for f in required_files if not os.path.isfile(f)]
if missing_files:
    raise FileNotFoundError(f"❌ Thiếu file mô hình hoặc scaler: {missing_files}")

# ✅ Load models và scaler
model_gen = tf.keras.models.load_model("model_generator.keras")
model_fore = tf.keras.models.load_model("model_forecaster.keras")
scaler_gen = joblib.load("scaler_generator.save")
scaler_fore = joblib.load("scaler_forecaster.save")
scaler_fs = joblib.load("scaler_fs_output.save")

# ✅ Flask setup
app = Flask(__name__)
CORS(app)

# ✅ Gán nhãn theo FS
def classify_fs(fs):
    if fs >= 1.5: return "An toàn"
    elif fs >= 1.0: return "Có dấu hiệu"
    else: return "Nguy cơ cao"

# ✅ Trang chủ kiểm tra nhanh
@app.route('/')
def index():
    return (
        "<h2>✅ Landslide FS Prediction API is running!</h2>"
        "<p>Use <code>POST /predict</code> with JSON:<br>"
        "<code>{ \"features\": [c, L, gamma, h, u, phi, beta, elevation, slope_type] }</code></p>"
        "<p>Or use GET with query string: "
        "<code>?c=...&L=...&gamma=...&h=...&u=...&phi=...&beta=...&elevation=...&slope_type=...</code></p>"
    )

# ✅ Dự đoán FS
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # GET hoặc POST
        if request.method == 'POST':
            json_data = request.get_json(force=True)
            if not json_data or "features" not in json_data:
                raise ValueError("Dữ liệu POST phải chứa trường 'features'")
            features = np.array([json_data["features"]])
        elif request.method == 'GET':
            params = ["c", "L", "gamma", "h", "u", "phi", "beta", "elevation", "slope_type"]
            features = [float(request.args.get(p)) for p in params]
            features = np.array([features])
        else:
            return jsonify({"success": False, "message": "Chỉ hỗ trợ POST hoặc GET"}), 405

        if features.shape[1] != 9:
            raise ValueError(f"Số lượng đầu vào không hợp lệ. Cần 9 giá trị, nhận {features.shape[1]}")

        # 🔁 Dự đoán
        input_scaled = scaler_gen.transform(features)
        sequence = model_gen.predict(input_scaled)
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
            "message": "⚠️ Kiểm tra đầu vào JSON hoặc query. Định dạng phải giống {\"features\": [9 số]}"
        }), 400

# ✅ Không dùng dòng này trên Render vì đã dùng Gunicorn
if __name__ == '__main__':
    app.run(debug=True)
