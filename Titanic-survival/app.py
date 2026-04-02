from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Pclass = int(request.form['Pclass'])
    Sex = int(request.form['Sex'])
    Age = float(request.form['Age'])
    SibSp = int(request.form['SibSp'])
    Parch = int(request.form['Parch'])
    Fare = float(request.form['Fare'])
    Embarked = int(request.form['Embarked'])

    features = np.array([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Passenger Survived"
    else:
        result = "Passenger Did Not Survive"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
