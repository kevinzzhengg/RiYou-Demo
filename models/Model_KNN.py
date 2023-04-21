# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
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

def knn_train(dataset,parameter_dict={}):
    # st.write("this is the test demo of KNN")
    dataset_name = re.search(r"[/|\\](.*?)[/|\\](.*)\.csv",dataset).groups()[1]

    ## Check Data
    data = pd.read_csv(dataset,index_col=False)
    # if 'Unnamed: 0' in data.columns:
    #     data.drop(['Unnamed: 0'],inplace=True,axis=1)
    # st.dataframe(data=data)

    # 检查输入数据是否符合数据
    for i in range(len(data.dtypes)):
        if pd.api.types.is_integer_dtype(data.dtypes[i]) or pd.api.types.is_float_dtype(data.dtypes[i]): 
            # if i == len(data.dtypes) - 1:
            #     st.success("the dataset is qualified.")
            continue
        else:
            return st.warning("the dataset is unqualified.")
        
    if check_save_models_file(dataset,'knn') and parameter_dict == {}:
        gcv = joblib.load('./userdata/{}/knn.pkl'.format(dataset_name))
        return gcv
    else:
        
        ## Split the Features and Target
        x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
        y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
        for i in [Xtrain,Xtest,Ytrain,Ytest]:
            i.index = range(i.shape[0])
        Ytrain = pd.DataFrame(Ytrain).values.ravel()
        Ytest = pd.DataFrame(Ytest).values.ravel()
        
        ## Train the DecisionTree Model
        knn = KNeighborsClassifier(n_neighbors=parameter_dict['n_neighbors'],
                                   weights=parameter_dict['weights'],
                                   algorithm=parameter_dict['algorithm'],
                                   leaf_size=parameter_dict['leaf_size'],
                                   metric=parameter_dict['metric'])
        knn.fit(Xtrain,Ytrain)

        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(knn,'./userdata/{}/knn.pkl'.format(dataset_name))
        st.success('trainning completed!')
        return knn

        # # result display
        # y_ = gcv.predict(Xtest)
        # score = gcv.score(Xtest,Ytest)
        # ct = pd.crosstab(index = Ytest,columns = y_,rownames=["True"],colnames=["Predict"])
        # with st.container():
        #     st.caption("accuracy rating of KNN: {}".format(score))
        #     st.write(ct)
        # with st.sidebar:
        #     st.write("accuracy rating of KNN: {}".format(score))
        
        # return gcv

def knn_parameter_add():
    # 提供超参数选择
    col_1,col_2 = st.columns([3,1])
    with col_1:
        with st.container():
            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # n_neighbors参数
                    n_neighbors_mode = st.selectbox('n_neighbors',['default','user defined'],key='n_neighbors')
                    if n_neighbors_mode == 'default':
                        n_neighbors = 5
                    elif n_neighbors_mode == 'user defined':
                        n_neighbors = st.slider('',3,20,label_visibility='collapsed')
                with col_right:
                    # weight参数
                    weights = st.selectbox('weights',['uniform','distance'])

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # algorithm参数
                    algorithm = st.selectbox('algorithm',['auto','kd_tree','ball_tree','brute'])
                with col_right:
                    # leaf_size参数
                    leaf_size_mode = st.selectbox('leaf_size',['default','user defined'])
                    if leaf_size_mode == 'default':
                        leaf_size = 30
                    elif leaf_size_mode == 'user defined':
                        leaf_size = st.slider('',10,50,label_visibility='collapsed')

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # metric参数
                    metric = st.selectbox('metric',['euclidean','manhattan','chebyshev','minkowski','wminkowski','seuclidean','mahalanobis'])
                with col_right:
                    pass
            
                
        parameter_dict = {      'n_neighbors':n_neighbors,
                                'weights':weights,
                                'algorithm':algorithm,
                                'leaf_size':leaf_size,
                                'metric':metric}
    
    # 展示最终选择的参数清单-便于后续检验输入参数是否符合需求
    with col_2:
        st.write('参数统计')
        for each in parameter_dict:
            st.write(each,':',parameter_dict[each])

    return parameter_dict