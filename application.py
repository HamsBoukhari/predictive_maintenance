from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app=Flask(__name__)
model=pickle.load(open('pred_maintenance.pkl','rb'))
df=pd.read_csv('cleaned_pred_maintenance_data.csv')
@app.route('/')
def home():
    type=df['Type'].unique()
    return render_template('index.html',type=type)
@app.route('/predict',methods=['POST'])
def predict():
    Type=request.form.get('type')
    air_temp=request.form.get('air_temp')
    pr_temp=request.form.get('pr_temp')
    rot=request.form.get('rot')
    torque=request.form.get('torque')
    tool=request.form.get('tool')
    pred=model.predict(pd.DataFrame(columns=['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]'],data=np.array([Type,air_temp,pr_temp,rot,torque,tool]).reshape(1,-1)))
    if pred==0:
        return 'No Failure'
    elif pred==1:
        return 'Heat Dissipation Failure'
    elif pred==2:
         return 'Power Failure'
    elif pred==3:
         return 'Overstrain Failure'
    else:
         return'Tool Wear Failure'
if __name__=="__main__":
    app.run(debug=True)