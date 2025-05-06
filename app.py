from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import os

# ✅ Kiểm tra đầy đủ file model trước khi chạy
required_files = [
    "model_generator.keras",
    "model_forecaster.keras",
    "scaler_generator.save",
    "scaler_forecaster.save",
    "scaler_fs_output.save"
]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"❌ Thiếu file: {missing}")

# ✅ Tải model và scaler
model_gen = tf.keras.models.load_model("model_generator.keras")
model_fore = tf.keras.models.load_model("model_forecaster.keras")
scaler_gen = joblib.load("scaler_generator.save")
scaler_fore = joblib.load("scaler_forecaster.save")
scaler_fs = joblib.load("scaler_fs_output.save")

# ✅ Khởi tạo Flask app và bật CORS toàn cục
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Cho phép mọi web gọi API  # Cho phép tất cả nguồn gọi API

# ✅ Hàm phân loại FS
def classify_fs(fs):
    if fs >= 1.5:
        return "An toàn"
    elif fs >= 1.0:
        return "Có dấu hiệu"
    else:
        return "Nguy cơ cao"

# ✅ Route kiểm tra API đang chạy
@app.route('/')
def home():
    return (
        "<h2>✅ Landslide FS Prediction API is running!</h2>"
        "<p>Use <b>POST</b> at <code>/predict</code> with JSON:<br>"
        "<code>{ \"features\": [c, L, gamma, h, u, phi, beta, elevation, slope_type] }</code></p>"
        "<p>Or try <b>GET</b>:<br>"
        "<code>/predict?c=...&L=...&gamma=...&h=...&u=...&phi=...&beta=...&elevation=...&slope_type=...</code></p>"
    )

# ✅ Route dự đoán chính
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Xử lý POST
        if request.method == 'POST':
            data = request.get_json(force=True)
            if not data or "features" not in data:
                return jsonify({"success": False, "error": "Thiếu trường 'features'"}), 400
            features = np.array([data["features"]])

        # Xử lý GET
        elif request.method == 'GET':
            keys = ["c", "L", "gamma", "h", "u", "phi", "beta", "elevation", "slope_type"]
            try:
                features = [float(request.args.get(k, "0")) for k in keys]
            except:
                return jsonify({"success": False, "error": "Tham số GET không hợp lệ"}), 400
            features = np.array([features])
        else:
            return jsonify({"success": False, "message": "Chỉ hỗ trợ GET và POST"}), 405

        # ✅ Kiểm tra số lượng đặc trưng
        if features.shape[1] != 9:
            return jsonify({"success": False, "error": "Bạn cần nhập đúng 9 đặc trưng đầu vào."}), 400

        # ✅ Chuỗi xử lý dự đoán
        input_scaled = scaler_gen.transform(features)
        generated_seq = model_gen.predict(input_scaled)
        sequence_lstm = generated_seq.reshape((1, 1, generated_seq.shape[1]))
        fs_scaled = model_fore.predict(sequence_lstm)
        fs = scaler_fs.inverse_transform(fs_scaled)[0]

        # ✅ Trả kết quả
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
            "message": "❌ Dữ liệu đầu vào sai định dạng. Hãy gửi {\"features\": [9 số]} hoặc query đầy đủ."
        }), 400

# ✅ Không cần khi chạy trên Render (sử dụng Gunicorn)
if __name__ == '__main__':
    app.run(debug=True)
