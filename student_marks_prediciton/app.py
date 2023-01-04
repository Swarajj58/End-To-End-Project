# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#Flask framework 
#flask will take the input from front end and then returen it
#request values will be requested
# render will render the value which we take
from flask import Flask, request, render_template
import joblib

app = Flask(__name__) # this is name
#here we are importing our pickel file in which we have our model we created
model = joblib.load(r"C:\Users\swaraj jaiswal\OneDrive\Desktop\prakesh senapati\notes\27 april\Student Mark Predictor Project Deployment\student_mark_predictor.pkl")

# create a empty dataframe
df = pd.DataFrame()
#whenever we open our server route our srever to our main page index.html
@app.route('/') 
def home():
    return render_template('index.html')
# route to predict our input 
@app.route('/predict',methods=['POST'])#post is used to send html data to server
def predict():
    #to store our input in empty df we give empty df as global
    global df
    #user enter input 
    input_features = [int(x) for x in request.form.values()] 
    #aims to provide an array object that is up to 50x faster than traditional Python lists
    features_value = np.array(input_features)
    
    #validate input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
        
    #here we get our output
    output = model.predict([features_value])[0][0].round(2)

    # input and predicted value store in df then save in csv file
    df= pd.concat([df,pd.DataFrame({'Study Hours':input_features,'Predicted Output':[output]})],ignore_index=True)
    print(df)   
    df.to_csv('smp_data_from_app.csv')
    
    return render_template('index.html', prediction_text='You will get [{}%] marks, when you do study [{}] hours per day '.format(output, int(features_value[0])))


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
    