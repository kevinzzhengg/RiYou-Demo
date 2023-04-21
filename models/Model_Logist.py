# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib

# EDA Pkgs
import pandas as pd
import numpy as np

# Sys Pkgs
import os
import re

# exfuction Pkgs
from exfuction.main_fxn import *

def logistic_train(dataset,parameter_dict={}):
    # st.write("this is th test demo of Logistic")
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
    
    if check_save_models_file(dataset,'lc') and parameter_dict == {}:
        lc = joblib.load('./userdata/{}/lc.pkl'.format(dataset_name))
        return lc
    else:

        ## Split the Features and Target
        x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
        y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
        for i in [Xtrain,Xtest,Ytrain,Ytest]:
            i.index = range(i.shape[0])

        ## Run the DecisionTree Model
        lc = LogisticRegression(penalty=parameter_dict['penalty'],
                                dual=parameter_dict['dual'],
                                tol=parameter_dict['tol'],
                                fit_intercept=parameter_dict['fit_intercept'],
                                solver=parameter_dict['solver'],
                                max_iter=parameter_dict['max_iter'],
                                verbose=parameter_dict['verbose'],
                                random_state=parameter_dict['random_state'])
        lc = lc.fit(Xtrain,Ytrain)

        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(lc,'./userdata/{}/lc.pkl'.format(dataset_name))
        st.success('trainning completed!')
        return lc


def logistic_parameter_add():
    # 提供超参数选择
    # penalty - ['l2','l1'] # 正则化类型
	# dual - [False,True] # 样本数>特征数时，令dual=False
	# tol - [default=1e-4,user defined] #迭代终止判断的误差范围
	# C - [default=1.0,user defined]
	# fit_intercept - [True,False] # 指定是否应该向决策函数添加常量
	# solver - ['newton-cg','lbfgs',default='liblinear','sag','saga'] # 用于优化问题的算法
	# max_iter - [default=100,user defined] #
	# verbose - [default=0, user defined] # 
	# random state
    col_1,col_2 = st.columns([3,1])
    with col_1:
        with st.container():
            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # penalty参数
                    penalty = st.selectbox('penalty',['l2','l1'],key='penalty')
                with col_right:
                    # dual参数
                    dual = st.selectbox('dual',[False,True])

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # tol参数
                    tol_mode = st.selectbox('tol',['default','user defined'])
                    if tol_mode == 'default':
                        tol = 1e-7
                    elif tol_mode == 'user defined':
                        tol = st.text_input('',value=1e-7,label_visibility='collapsed')
                with col_right:
                    # C参数
                    c_mode = st.selectbox('C',['default','user defined'],key='c')
                    if c_mode == 'default':
                        c = 1.0
                    elif c_mode == 'user defined':
                        c = st.slider('',1.0,5.0,label_visibility='collapsed')

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # fit_intercept参数
                    fit_intercept = st.selectbox('fit_intercept',[True,False])
                with col_right:
                    # solver参数
                    solver = st.selectbox('solver',['newton-cg','lbfgs','liblinear','sag','saga'])

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # max_iter参数
                    max_iter_mode = st.selectbox('max_iter',['default','user defined'])
                    if max_iter_mode == 'default':
                        max_iter = 100
                    elif max_iter_mode == 'user defined':
                        max_iter = st.slider('',100,500,label_visibility='collapsed')

                with col_right:
                    # verbose参数
                    verbose_mode = st.selectbox('verbose',['default','user defined'])
                    if verbose_mode == 'default':
                        verbose = 0
                    elif verbose_mode == 'user defined':
                        verbose = st.slider('',0,5,label_visibility='collapsed')
            
            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    random_state = int(st.text_input('random state',value=2023))
                with col_right:
                    pass
                
        parameter_dict = {      'penalty':penalty,
                                'dual':dual,
                                'tol':tol,
                                'fit_intercept':fit_intercept,
                                'solver':solver,
                                'max_iter':max_iter,
                                'verbose':verbose,
                                'random_state':random_state}
    
    # 展示最终选择的参数清单-便于后续检验输入参数是否符合需求
    with col_2:
        st.write('参数统计')
        for each in parameter_dict:
            st.write(each,':',parameter_dict[each])

    return parameter_dict