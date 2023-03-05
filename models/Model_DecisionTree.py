# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# EDA Pkgs
import pandas as pd
import numpy as np

# Plot Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

def DecisionTreeCLF_PRE(dataset):
    st.write("this is the test demo of DecisionTree")

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
    with st.container():
        x_col, y_col = st.columns(2)
        with x_col:
            st.subheader("Features:")
            st.dataframe(x,use_container_width=True)
        with y_col:
            st.subheader("Target")
            st.dataframe(y,use_container_width=True)

    ## Split the Train Samples and the Test Samples
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
    for i in [Xtrain,Xtest,Ytrain,Ytest]:
        i.index = range(i.shape[0])
    
    ## Run the DecisionTree Model
    clf = DecisionTreeClassifier(random_state=25,max_depth=3)
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    with st.container():
        st.caption("accuracy rating: {}".format(score))
    with st.sidebar:
        st.write("accuracy rating: {}".format(score))
       

def DecisionTreeCLF_Parameter_Add():
    st.write("more parameter is waiting to be added.")