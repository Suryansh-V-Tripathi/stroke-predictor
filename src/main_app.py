import pickle

from flask import Flask
from flask import request
from flask import jsonify

input_file = 'model.pkl'


with open(input_file, 'rb') as f_in: 
    model = pickle.load(f_in)
    dv = None  # We aren't using a separate vectorizer anymore



app = Flask('stroke')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    stroke = y_pred >= 0.5

    result = {
        'stroke_probability': float(y_pred),
        'stroke': bool(stroke)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    
def clinical_expert_check(age, hyper, heart, glucose):
    # Rule-based logic for edge cases (Module 2 syllabus alignment)
    if age > 75 and hyper == 1 and heart == 1:
        return "CRITICAL: Patient fits High-Age Multi-Risk profile."
    if glucose > 200:
        return "WARNING: Glucose levels exceed clinical safety thresholds."
    return "Patient parameters within standard range."
