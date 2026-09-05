"""Flask server for the diabetes prediction web application."""

import pickle
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

with (MODELS_DIR / 'model.pkl').open('rb') as file:
  loaded_model = pickle.load(file)

with (MODELS_DIR / 'scaler.pkl').open('rb') as file:
  loaded_scaler = pickle.load(file)

with (MODELS_DIR / 'le_gender.pkl').open('rb') as file:
  loaded_le_gender = pickle.load(file)

with (MODELS_DIR / 'le_smoke.pkl').open('rb') as file:
  loaded_le_smoke = pickle.load(file)


def safe_encode(label_encoder, value):
  """Encode a value; use the first trained class for unknown values."""
  try:
    return int(label_encoder.transform([str(value)])[0])
  except ValueError:
    return int(label_encoder.transform([label_encoder.classes_[0]])[0])


app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
  try:
    data = request.get_json()
    if not data:
      raise ValueError('Dữ liệu đầu vào không hợp lệ.')

    gender_enc = safe_encode(loaded_le_gender, data.get('gender', 'Female'))
    smoking_enc = safe_encode(
        loaded_le_smoke, data.get('smoking_history', 'never')
    )

    features = [
        float(gender_enc),
        float(data['age']),
        float(data['hypertension']),
        float(data['heart_disease']),
        float(smoking_enc),
        float(data['bmi']),
        float(data['HbA1c_level']),
        float(data['blood_glucose_level']),
    ]

    x_input = np.array([features])
    x_scaled = loaded_scaler.transform(x_input)
    prediction = loaded_model.predict(x_scaled)[0]
    probability = loaded_model.predict_proba(x_scaled)[0]

    return jsonify(
        {
            'prediction': int(prediction),
            'label': 'Có tiểu đường' if prediction == 1 else 'Không tiểu đường',
            'probability_no': round(float(probability[0]) * 100, 2),
            'probability_yes': round(float(probability[1]) * 100, 2),
            'risk_level': (
                'Cao'
                if probability[1] >= 0.7
                else 'Trung bình'
                if probability[1] >= 0.4
                else 'Thấp'
            ),
        }
    )
  except Exception as error:
    return jsonify({'error': str(error)}), 400


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001, debug=False)
