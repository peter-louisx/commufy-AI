from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)
CORS(app)

model = joblib.load('model_Random_Forest.pkl')

@app.route('/')
def home():
    return jsonify({"message": "Welcome to Commufy AI"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    required_fields = ['age', 'sex', 'weight', 'height', 'speed']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    X = pd.DataFrame({
        'age': [data['age']],
        'sex': [data['sex']],
        'weight': [data['weight']],
        'height': [data['height']],
        'speed': [data['speed']]
    })
    
    prediction = model.predict(X)
    
    return jsonify({"prediction": prediction.tolist()[0]}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
