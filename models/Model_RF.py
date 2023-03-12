# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# EDA Pkgs
import pandas as pd
import numpy as np

# Plot Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

def randomforest_predict(dataset):
    st.write("this is th test demo of RF")

    ## Check Data
    data = pd.read_csv(dataset)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'],inplace=True,axis=1)
    st.dataframe(data=data)
    for i in range(len(data.dtypes)):
        if pd.api.types.is_integer_dtype(data.dtypes[i]) or pd.api.types.is_float_dtype(data.dtypes[i]): 
            if i == len(data.dtypes) - 1:
                st.success("the dataset is qualified.")
            continue
        else:
            return st.warning("the dataset is unqualified.")
        
    ## Split the Features and Target
    x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
    y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
    for i in [Xtrain,Xtest,Ytrain,Ytest]:
        i.index = range(i.shape[0])

    ## Run the DecisionTree Model
    rfc = RandomForestClassifier(n_estimators=200,random_state=30)
    rfc = rfc.fit(Xtrain,Ytrain)
    score = rfc.score(Xtest,Ytest)
    with st.container():
        st.caption("accuracy rating of RF: {}".format(score))
    with st.sidebar:
        st.write("accuracy rating of RF: {}".format(score))

    return rfc


def randomforest_parameter_add():
    st.write("more parameter is waiting to be added.")