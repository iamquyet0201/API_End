from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# Load model và scaler
model_gen = tf.keras.models.load_model("model_generator.keras")
model_fore = tf.keras.models.load_model("model_forecaster.keras")
scaler_gen = joblib.load("scaler_generator.save")
scaler_fore = joblib.load("scaler_forecaster.save")
scaler_fs = joblib.load("scaler_fs_output.save")

# Flask app
app = Flask(__name__)

# Phân loại FS
def classify_fs(fs):
    if fs >= 1.5: return "An toàn"
    elif fs >= 1.0: return "Có dấu hiệu"
    else: return "Nguy cơ cao"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([data["features"]])  # 9 đặc trưng
        sample_scaled = scaler_gen.transform(features)
        
        # Model 1
        sequence_generated = model_gen.predict(sample_scaled)
        
        # Model 2
        sequence_input = sequence_generated.reshape((1, 1, sequence_generated.shape[1]))
        fs_scaled = model_fore.predict(sequence_input)
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
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
