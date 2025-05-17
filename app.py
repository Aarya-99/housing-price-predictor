import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])


def predict():
    features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    data = [float(request.form[feature]) for feature in features]

    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("index.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True,port=5050)
   
     