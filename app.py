from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import os

# ✅ Tạo Flask app & bật CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Cho phép mọi web sử dụng API

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


# ✅ Load model và scaler
model_gen = tf.keras.models.load_model("model_generator.keras")
model_fore = tf.keras.models.load_model("model_forecaster.keras")
scaler_gen = joblib.load("scaler_generator.save")
scaler_fore = joblib.load("scaler_forecaster.save")
scaler_fs = joblib.load("scaler_fs_output.save")

# ✅ Hàm phân loại cảnh báo
def classify_fs(fs):
    if fs >= 1.5:
        return "An toàn"
    elif fs >= 1.0:
        return "Có dấu hiệu"
    else:
        return "Nguy cơ cao"

# ✅ Trang gốc để kiểm tra API hoạt động
@app.route('/')
def home():
    return (
        "<h2>✅ Landslide FS Prediction API is running!</h2>"
        "<p>POST JSON to <code>/predict</code> or test via GET:</p>"
        "<code>?c=..., L=..., gamma=..., h=..., u=..., phi=..., beta=..., elevation=..., slope_type=...</code>"
    )

# ✅ API POST & GET
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
            if not data or "features" not in data:
                return jsonify({"success": False, "error": "Thiếu trường 'features'"}), 400
            features = np.array([data["features"]])

        elif request.method == 'GET':
            keys = ["c", "L", "gamma", "h", "u", "phi", "beta", "elevation", "slope_type"]
            features = [float(request.args.get(k, "0")) for k in keys]
            features = np.array([features])
        else:
            return jsonify({"success": False, "message": "Chỉ hỗ trợ GET và POST"}), 405

        if features.shape[1] != 9:
            return jsonify({"success": False, "error": "Bạn cần nhập đủ 9 đặc trưng."}), 400

        # ✅ Dự đoán
        sample_scaled = scaler_gen.transform(features)
        generated_seq = model_gen.predict(sample_scaled)
        seq_input = generated_seq.reshape((1, 1, generated_seq.shape[1]))
        fs_scaled = model_fore.predict(seq_input)
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
        return jsonify({"success": False, "error": str(e)}), 400

# ✅ Render sẽ sử dụng Gunicorn, không cần run trực tiếp
if __name__ == "__main__":
    app.run(debug=True)
