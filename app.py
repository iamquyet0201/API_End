from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kiểm tra các file cần thiết
required_files = [
    "model_generator.keras",
    "model_forecaster.keras",
    "scaler_generator.save",
    "scaler_forecaster.save",
    "scaler_fs_output.save"
]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    logger.error(f"Thiếu file: {missing}")
    raise FileNotFoundError(f"Thiếu file: {missing}")

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Bật CORS cho endpoint /predict

# Load model và scaler
try:
    logger.info("Đang tải model và scaler...")
    model_gen = tf.keras.models.load_model("model_generator.keras")
    model_fore = tf.keras.models.load_model("model_forecaster.keras")
    scaler_gen = joblib.load("scaler_generator.save")
    scaler_fore = joblib.load("scaler_forecaster.save")
    scaler_fs = joblib.load("scaler_fs_output.save")
    logger.info("Tải model và scaler thành công")
except Exception as e:
    logger.error(f"Lỗi khi tải model/scaler: {str(e)}")
    raise

# Phân loại FS
def classify_fs(fs):
    if fs >= 1.5:
        return "An toàn"
    elif fs >= 1.0:
        return "Có dấu hiệu"
    else:
        return "Nguy cơ cao"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Nhận yêu cầu POST")
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"success": False, "error": "Thiếu trường 'features'"}), 400
        features = np.array([data["features"]])
        if features.shape[1] != 9:
            return jsonify({"success": False, "error": "Cần gửi đúng 9 giá trị đặc trưng"}), 400

        logger.info(f"Dữ liệu nhận được: {features}")
        sample_scaled = scaler_gen.transform(features)
        sequence_generated = model_gen.predict(sample_scaled, verbose=0)
        sequence_input = sequence_generated.reshape((1, 1, sequence_generated.shape[1]))
        fs_scaled = model_fore.predict(sequence_input, verbose=0)
        fs = scaler_fs.inverse_transform(fs_scaled)[0]

        result = []
        for i, val in enumerate(fs, start=1):
            result.append({
                "day": i,
                "fs": round(float(val), 2),
                "label": classify_fs(val)
            })

        logger.info(f"Kết quả dự đoán: {result}")
        return jsonify({"success": True, "result": result})

    except Exception as e:
        logger.error(f"Lỗi trong dự đoán: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
