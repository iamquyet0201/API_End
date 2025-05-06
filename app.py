from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib

# Load model và scaler
model_gen = tf.keras.models.load_model("model_generator.keras")
model_fore = tf.keras.models.load_model("model_forecaster.keras")
scaler_gen = joblib.load("scaler_generator.save")
scaler_fore = joblib.load("scaler_forecaster.save")
scaler_fs = joblib.load("scaler_fs_output.save")

app = Flask(__name__)
CORS(app)

def classify_fs(fs):
    if fs >= 1.5: return "An toàn"
    elif fs >= 1.0: return "Có dấu hiệu"
    else: return "Nguy cơ cao"

@app.route('/')
def index():
    return (
        "<h2>✅ Landslide FS Prediction API is running!</h2>"
        "<p>Use <code>POST /predict</code> with JSON body:<br>"
        "<code>{ \"features\": [c, L, gamma, h, u, phi, beta, elevation, slope_type] }</code></p>"
        "<p>Or try GET:<br>"
        "<code>/predict?c=...&L=...&gamma=...&h=...&u=...&phi=...&beta=...&elevation=...&slope_type=...</code></p>"
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json()
            features = np.array([data["features"]])
        elif request.method == 'GET':
            params = ["c", "L", "gamma", "h", "u", "phi", "beta", "elevation", "slope_type"]
            features = [float(request.args.get(p)) for p in params]
            features = np.array([features])
        else:
            return jsonify({"success": False, "message": "❌ Chỉ hỗ trợ GET hoặc POST"}), 405

        # Chuỗi dự đoán
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
            "message": "❌ Vui lòng kiểm tra dữ liệu đầu vào (POST JSON hoặc GET query)"
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
