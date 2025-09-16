from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity = int(form_data['parity'])
    weight = float(form_data['weight'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    smoke = float(form_data['smoke'])

    cleaned_data = {
        "gestation": gestation,
        "parity": parity,
        "weight": weight,
        "age": age,
        "height": height,
        "smoke": smoke
    }
    return cleaned_data

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Support both JSON (API) and form (HTML) submissions
    if request.is_json:
        baby_data = request.get_json()
    else:
        baby_data = request.form

    baby_data_cleaned = get_cleaned_data(baby_data)
    # Ensure the feature order matches the order used during model training
    feature_order = ["gestation", "parity", "age", "height", "weight", "smoke"]  # Must match training order
    # Reorder the data accordingly
    baby_df = pd.DataFrame([[baby_data_cleaned[feat] for feat in feature_order]], columns=feature_order)

    # Load the trained model
    with open('model.pkl', 'rb') as obj:
        model = pickle.load(obj)

    # Predict using the model
    prediction = model.predict(baby_df)
    prediction = round(float(prediction[0]), 2)

    response = {"Prediction": prediction}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)



   
