# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.svm import SVC
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

# exfuction Pkgs
from exfuction.main_fxn import *


def svm_predict(dataset,parameter_dict={}):
    #st.write("this is the test demo")
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

    if check_save_models_file(dataset,'svm') and parameter_dict == {}:
        svm = joblib.load('./userdata/{}/svm.pkl'.format(dataset_name))
        return svm
    else:

        ## Split the Features and Target
        x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
        y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
        for i in [Xtrain,Xtest,Ytrain,Ytest]:
            i.index = range(i.shape[0])

        ## Run the DecisionTree Model
        svm = SVC(C=parameter_dict['C'],
                  kernel=parameter_dict['kernel'],
                  probability=parameter_dict['probability'],
                  shrinking=parameter_dict['shrinking'],
                  tol=parameter_dict['tol'],
                  random_state=parameter_dict['random_state'])
        svm = svm.fit(Xtrain,Ytrain)

        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(svm,'./userdata/{}/svm.pkl'.format(dataset_name))
        st.success('trainning completed!')  
        return svm
    
    

def svm_parameter_add():
    # 提供超参数选择
    # C - [default=1.0,user defined] #误差项的惩罚参数
	# kernel - ['linear','poly','rbf','sigmoid','precomputed'] # SVC核函数

	# degree - [default=3,user defined] # 仅当kernel为poly时有效
	# gamma - [default='auto'] # 当kernel为rbf,ploy,sigmoid时的kernel系数
	# coef0 - [default=0.0,user defined] # ploy或sigmoid时有效，kernel函数的常数项

	# probability - [False,True] # 是否采用概率估计，该方法的使用会降低运算速度，默认为False
	# shrinking - [True,False] # 是否使用shrinking技术
	# tol - [default=1e-3,user defined] # 误差项达到指定值时停止训练
	# random_state - text input # 伪随机数使用数据
    col_1,col_2 = st.columns([3,1])
    with col_1:
        with st.container():
            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # C参数
                    c_mode = st.selectbox('C',['default','user defined'],key='c')
                    if c_mode == 'default':
                        c = 1.0
                    elif c_mode == 'user defined':
                        c = st.slider('',1.0,5.0,label_visibility='collapsed')
                with col_right:
                    # kernel参数
                    kernel = st.selectbox('kernel',['linear','poly','rbf','sigmoid','precomputed'])

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # probability参数
                    probability = st.selectbox('probability',[True,False])
                with col_right:
                    # shrinking参数
                    shrinking = st.selectbox('shrinking',[True,False])

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # tol参数
                    tol_mode = st.selectbox('tol',['default','user defined'])
                    if tol_mode == 'default':
                        tol = 1e-7
                    elif tol_mode == 'user defined':
                        tol = float(st.text_input('',value='1e-7',label_visibility='collapsed'))
                with col_right:
                    # random_state 参数
                    random_state = int(st.text_input('random state',value=2023))
            
                
        parameter_dict = {      'C':c,
                                'kernel':kernel,
                                'probability':probability,
                                'shrinking':shrinking,
                                'tol':tol,
                                'random_state':random_state}
    
    # 展示最终选择的参数清单-便于后续检验输入参数是否符合需求
    with col_2:
        st.write('参数统计')
        for each in parameter_dict:
            st.write(each,':',parameter_dict[each])

    return parameter_dict