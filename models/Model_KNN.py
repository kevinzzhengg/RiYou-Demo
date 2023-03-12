# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# EDA Pkgs
import pandas as pd
import numpy as np

# Plot Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

def knn_predict(dataset):
    st.write("this is the test demo of KNN")

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
    Ytrain = pd.DataFrame(Ytrain).values.ravel()
    Ytest = pd.DataFrame(Ytest).values.ravel()
    
    ## Run the DecisionTree Model
    knn = KNeighborsClassifier()
    params = {"n_neighbors":[i for i in range(1,30)],"weights":["distance","uniform"],"p":[1,2]}
    gcv = GridSearchCV(knn,params,scoring = "accuracy",cv = 6)
    gcv.fit(Xtrain,Ytrain)
    y_ = gcv.predict(Xtest)
    score = gcv.score(Xtest,Ytest)
    ct = pd.crosstab(index = Ytest,columns = y_,rownames=["True"],colnames=["Predict"])
    with st.container():
        st.caption("accuracy rating of KNN: {}".format(score))
        st.write(ct)
    with st.sidebar:
        st.write("accuracy rating of KNN: {}".format(score))
    
    return gcv

def knn_parameter_add():
    st.write("more parameter is waiting to be added.")