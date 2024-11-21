import pickle
import flask
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from analysis.scripts.data_load import preprocess_data, sample_default, SERVER_DATA_PATH, LOCAL_DATA_PATH

feature_transformer, _, _, _ = preprocess_data(data_path=SERVER_DATA_PATH)
default_audio_track = sample_default(data_path=SERVER_DATA_PATH).to_dict()

output_file = f'./analysis/model.pkl'
with open(output_file, 'rb') as inF:
    model = pickle.load(inF)

print(f"Loaded the SVC model from {output_file}")

def predict_proba(audio):
    audio = np.array(list(audio.values())).astype(float)
    x_audio = feature_transformer.transform([audio])
    fake_prediction = model.predict(x_audio)[0]
    return bool(fake_prediction)

app = Flask("deepvoice")
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if flask.request.method == 'POST':
        audio = request.get_json()
        decision = predict_proba(audio)
        result = {"decision": decision, "is_valid": True}
        return jsonify(result)
    
    if flask.request.method == 'GET':
        print("Warning!! You are getting results for random audio clip")
        audio = default_audio_track
        decision = predict_proba(audio)
        result = {"decision": decision, "is_valid": False}
        return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

# # # run the files with  -> python -m file_name