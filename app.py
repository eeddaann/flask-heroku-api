import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# read our pickle file and label our logisticmodel as model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'GET': # handle QR
        int_features = [float(x) for x in list(request.args.values())]
    else: # handle manual form
        int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    proba = dict(zip([
        0,30,60,90,120,150,180,210
    ],list(model.predict_proba(final_features))[0]))
    return render_template('index.html',
                            prediction_text="prediction: %s"%(prediction[0]),
                            proba=proba,
                            )

if __name__ == "__main__":
    app.run(debug=True)