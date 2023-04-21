# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,BernoulliNB,ComplementNB,MultinomialNB
import joblib

# EDA Pkgs
import pandas as pd
import numpy as np

# Sys Pkgs
import os
import re

# Plot Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

# exfuction Pkgs
from exfuction.main_fxn import *

def naive_bayes_predict(dataset):
    #st.write("this is the test demo of Naive Bayes")
    dataset_name = re.search(r"[/|\\](.*?)[/|\\](.*)\.csv",dataset).groups()[1]

    ## Check Data
    data = pd.read_csv(dataset)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'],inplace=True,axis=1)
    # st.dataframe(data=data)
    for i in range(len(data.dtypes)):
        if pd.api.types.is_integer_dtype(data.dtypes[i]) or pd.api.types.is_float_dtype(data.dtypes[i]): 
            # if i == len(data.dtypes) - 1:
            #     st.success("the dataset is qualified.")
            continue
        else:
            return st.warning("the dataset is unqualified.")
    
    if check_save_models_file(dataset,'nb'):
        bayes_model = joblib.load('./userdata/{}/nb.pkl'.format(dataset_name))
        return bayes_model
    else:
        ## Split the Features and Target
        x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
        y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
        for i in [Xtrain,Xtest,Ytrain,Ytest]:
            i.index = range(i.shape[0])

        bayes_model = GaussianNB()
        bayes_model.fit(Xtrain,Ytrain)
        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(bayes_model,'./userdata/{}/nb.pkl'.format(dataset_name)) 
        return bayes_model

        # result = pd.DataFrame(bayes_model.predict(Xtest))
        # # st.write(result)

        # score = bayes_model.score(Xtest,Ytest)
        # with st.container():
        #     st.caption("accuracy rating of NB: {}".format(score))
        # with st.sidebar:
        #     st.write("accuracy rating of NB: {}".format(score))
        
        # return bayes_model


def naive_bayes_parameter_add():
    st.write("more parameter is waiting to be added.")