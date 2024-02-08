from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

Forest_Fire_Dataset=pd.read_csv("Forest_Fire_Dataset.csv")
# forest_fire_d = Forest_Fire_Dataset["City","Wildlife"]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/forest_fire')
def forest_fire():
    return render_template("forest_fire.html")

@app.route('/wildlife')
def wildlife():
    return render_template("wildlife.html")

@app.route('/safety')
def safety():
    return render_template("safety.html")

@app.route('/preprocessing')
def preprocessing():
    return render_template("preprocessing.html")




@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    print(prediction)

    if output>str(0.5):
        return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output))
    else:
        return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
