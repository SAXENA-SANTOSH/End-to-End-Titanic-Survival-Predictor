from flask import Flask, render_template, request , jsonify, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)


@app.route("/" , methods =["GET", "POST"])
def home_page():
    return render_template("Homepage.html")

@app.route("/analysis", methods = ["GET","POST"])
def analysis():
    return render_template("Analysis.html")


@app.route("/values", methods = ["POST"])
def fetch():

    if(request.method == "POST"):

        name = request.form["name"]

        if(name == ""):
            name = "NAN"

        sex = request.form["sex"]
        
        if(sex == ""):
            sex = "NAN"

        age = request.form["age"]

        if(age == ""):
            age = "NAN"

        sibsp = request.form["family_size"]

        if(sibsp == ""):
            sibsp = "NAN"

        parch = request.form["parent_size"]

        if(parch == ""):
            parch = "NAN"

        pclass = request.form["pclass"]

        if(pclass == ""):
            pclass = "NAN"

        ticket = request.form["ticket"]

        if(ticket == ""):
            ticket = "NAN"

        fare = request.form["fare"]

        if(fare == ""):
            fare = "NAN"

        deck = request.form["cabin"]

        if(deck == ""):
            deck = "NAN"
    
        embarked = request.form["embarked"]

        if(embarked == ""):
            embarked = "NAN"


        if(sex == "male"):
            sex_ = 1
        else:
            sex_ = 0

        if(sex == "NAN" or sibsp == "NAN" or parch == "NAN" or age == "NAN" or fare == "NAN"):
            return render_template("error.html")

        model = pickle.load(open("Titanic.pickle", "rb"))
        sc = pickle.load(open("age_scaler.pickle","rb"))
        sc1 = pickle.load(open("fare_scaler.pickle","rb"))

        age_ = age
        age_ = np.array(age_) 
        age_ = np.reshape(age_, (1,1))
        age_ = sc.transform(age_)
        age_ = age_[0][0]

        fare_ = fare
        fare_ = np.array(fare_) 
        fare_ = np.reshape(fare_, (1,1))
        fare_ = sc1.transform(fare_)
        fare_ = fare_[0][0]
        
        x = [sex_, sibsp, parch, age_, fare_]
        x = np.array(x)
        x = x.reshape(1, len(x))


        if(pclass == 1):
            pclass = "Upper"
        elif(pclass == 2):
            pclass = "Middle"
        else:
            pclass = "Lower"

        y_pred = model.predict(x)
        if(y_pred[0] == 0):
            result = "Dead"
        elif(y_pred[0] == 1):
            result = "Survived"
        else:
            print("Invalid")
        return render_template("Prediction.html", result = [result, name, sex, age, sibsp, parch, pclass, ticket, fare, deck , embarked])

if __name__ == '__main__':
    app.run(debug=True)
