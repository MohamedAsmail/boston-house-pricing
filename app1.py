import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
#define name of the app
st.write("""
# House priceing regression
""")
st.write('----')

# load data to know boundries boston is data set from sklearn
#data columns
boston=datasets.load_boston()
X=pd.DataFrame(boston.data,columns=boston.feature_names)
#target outputs
Y=pd.DataFrame(boston.target,columns=['MEDV'])

# crate sidebar
# declare the input section user
# build function 
#sidebar slider ___________
st.sidebar.header('please Mohamed inter input')
#function to take input from user
def user_input_features():
    # declaration for variables to hold the data
    CRIM=st.sidebar.slider('CRIM',float(X.CRIM.min()),float(X.CRIM.max()),float(X.CRIM.mean()))
    ZN=st.sidebar.slider('ZN',float(X.ZN.min()),float(X.ZN.max()),float(X.ZN.mean()))
    INDUS=st.sidebar.slider('INDUS',float(X.INDUS.min()),float(X.INDUS.max()),float(X.INDUS.mean()))
    CHAS=st.sidebar.slider('CHAS',(X.CHAS.min()),float(X.CHAS.max()),float(X.CHAS.mean()))
    NOX=st.sidebar.slider('NOX',float(X.NOX.min()),float(X.NOX.max()),float(X.NOX.mean()))
    RM=st.sidebar.slider('RM',float(X.RM.min()),(X.RM.max()),float(X.RM.mean()))
    AGE=st.sidebar.slider('AGE',float(X.AGE.min()),float(X.AGE.max()),float(X.AGE.mean()))
    DIS=st.sidebar.slider('DIS',float(X.DIS.min()),float(X.DIS.max()),float(X.DIS.mean()))
    RAD=st.sidebar.slider('RAD',float(X.RAD.min()),float(X.RAD.max()),float(X.RAD.mean()))
    TAX=st.sidebar.slider('TAX',float(X.TAX.min()),float(X.TAX.max()),float(X.TAX.mean()))
    PTRATIO=st.sidebar.slider('PTRATIO',float(X.PTRATIO.min()),float(X.PTRATIO.max()),float(X.PTRATIO.mean()))
    B=st.sidebar.slider('B',float(X.B.min()),float(X.B.max()),float(X.B.mean()))
    LSTAT=st.sidebar.slider('LSTAT',float(X.LSTAT.min()),float(X.LSTAT.max(),float(X.LSTAT.mean()))
    #preparing the output of the function
#bas data set EX: 'CRIM':CRIM    
    data={
        'CRIM':CRIM,
        'ZN':ZN,
        'INDUS':INDUS,
        'CHAS':CHAS,
        'NOX':NOX,
        'RM':RM,
        'AGE':AGE,
        'DIS':DIS,
        'RAD':RAD,
        'TAX':TAX,
        'PTRATIO':PTRATIO,
        'B':B,
        'LSTAT':LSTAT}
    # prepare to model
    features=pd.DataFrame(data,index=[0])
    return features
    
df=user_input_features()
# print the input to user
st.write('this is your input')
st.write(df)
# bulid line _____---------
st.write('----')

# load model train model-
model=RandomForestRegressor()
model.fit(X,Y)
prediction=model.predict(df)

#output section
st.header('predicton of MEDV')
st.write(prediction)
