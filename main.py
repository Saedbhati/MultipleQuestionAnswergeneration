from flask import Flask,request,render_template
from Model import Model

app= Flask(__name__)

model= Model()

@app.route('/genrate',methods=['POST'])
def Genrate():
 
    data=model.gen(request.form['context'])   

    return data, 200

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')



app.run('0.0.0.0',8080)