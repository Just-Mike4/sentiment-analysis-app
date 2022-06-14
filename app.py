from flask import Flask, request, jsonify, render_template 
import numpy as np
import pickle
from model import text_cleaner
app = Flask(__name__)

model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
          return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
          values=[str(x) for x in request.form.values()]
          prediction=model.predict(values)
          if prediction== 0:
                    prediction="Negative review"
          else:
                    prediction="Positive review"
          return render_template("index.html",prediction_text = "The sentiment is {}".format(prediction))

if __name__=="__main__":
          app.run(debug=True)
