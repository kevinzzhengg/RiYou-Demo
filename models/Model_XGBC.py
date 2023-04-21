# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
import joblib

# EDA Pkgs
import pandas as pd
import numpy as np

# Sys Pkgs
import os
import re

# exfuction Pkgs
from exfuction.main_fxn import *

def xgbc_train(dataset,parameter_dict={}):
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
    
    if check_save_models_file(dataset,'xgbc') and parameter_dict == {}:
        xgbc = joblib.load('./userdata/{}/xgbc.pkl'.format(dataset_name))
        return xgbc
    else:

        ## Split the Features and Target
        x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
        y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
        for i in [Xtrain,Xtest,Ytrain,Ytest]:
            i.index = range(i.shape[0])

        ## Run the DecisionTree Model
        xgbc = XGBClassifier(booster=parameter_dict['booster'],
                             silent=parameter_dict['silent'],
                             n_estimators=parameter_dict['n_estimators'],
                             max_depth=parameter_dict['max_depth'],
                             min_child_weight=parameter_dict['min_child_weight'],
                             subsample=parameter_dict['subsample'],
                             verbose=parameter_dict['verbose'],
                             colsample_bytree=parameter_dict['colsample_bytree'],
                             learning_rate=parameter_dict['learning_rate'],
                             objective=parameter_dict['objective'],
                             gamma=parameter_dict['gamma'],
                             reg_alpha=parameter_dict['alpha'])
        xgbc = xgbc.fit(Xtrain,Ytrain)

        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(xgbc,'./userdata/{}/xgbc.pkl'.format(dataset_name))
        st.success('trainning completed!') 
        return xgbc


def xgbc_parameter_add():
    # 提供超参数选择
    # # 常规参数
	# booster - ['gbtree','gbliner'] # 基分类器
	# silent - [0,1]
	
	# # 模型参数
	# n_estimatores - [default=100,user defined 100~500]
	# max_depth - [default=6,uer defined 3~10]
	# min_child_weight - [default=1,user defiend 1~10]
	# subsample - [default=1,user defined 0.5~1]
	# colsample_bytree - [default=1,user defined 0.5~1]
	
	# #学习任务参数
	# learning_rate - [default=0.3,user defined 0.01~0.3]
	# objective - ['binary:logistic','binary:logitraw'] # 暂时二分类
	# gamma - 0.0~10.0
	# alpha - default=1,user defined 0.0~10.0
	# lambda - default=1,user defined 0.0~10.0
    col_1,col_2 = st.columns([3,1])
    with col_1:
        with st.container():
            st.subheader('常规参数')
            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # booster参数
                    booster = st.selectbox('booster',['gbtree','gblinear'],key='booster')
                with col_right:
                    # silent参数
                    silent = st.selectbox('silent',[0,1])

            st.subheader('模型参数')
            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # n_estimators参数
                    n_estimators = st.slider('n_estimators',100,500,key='n_estimators')
                with col_right:
                    # max_depth参数
                    max_depth_mode = st.selectbox('max depth',['default','user defined'],key='max_depth')
                    if max_depth_mode == 'default':
                        max_depth = 6
                    elif max_depth_mode == 'user defined':
                        max_depth = st.slider('',3,8,label_visibility='collapsed',key='max_depth_slider')

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # min_child_weight参数
                    min_child_weight_mode = st.selectbox('min_child_weight',['default','user defined'],key='min_child_weight')
                    if min_child_weight_mode == 'default':
                        min_child_weight = 1
                    elif min_child_weight_mode == 'user defined':
                        min_child_weight = st.slider('',1,10,label_visibility='collapsed',key='min_child_weight_slider')
                with col_right:
                    # subsample参数
                    subsample_mode = st.selectbox('subsample',['default','user defined'],key='subsample')
                    if subsample_mode == 'default':
                        subsample = 1.0
                    elif subsample_mode == 'user defined':
                        subsample = st.slider('',0.5,1.0,label_visibility='collapsed',key='subsample_slider')

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # colsample_bytree参数
                    colsample_bytree_mode = st.selectbox('colsample_bytree',['default','user defined'],key='colsample_bytree')
                    if colsample_bytree_mode == 'default':
                        colsample_bytree = 1.0
                    elif colsample_bytree_mode == 'user defined':
                        colsample_bytree = st.slider('',0.5,1.0,label_visibility='collapsed',key='colsample_bytree_slider')

                with col_right:
                    # verbose参数
                    verbose_mode = st.selectbox('verbose',['default','user defined'])
                    if verbose_mode == 'default':
                        verbose = 0
                    elif verbose_mode == 'user defined':
                        verbose = st.slider('',0,5,label_visibility='collapsed')
            
            # 学习任务参数
            st.subheader('学习任务参数')
            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # learning_rate参数
                    learning_rate_mode = st.selectbox('learning_rate',['default','user defined'],key='learning_rate')
                    if learning_rate_mode == 'default':
                        learning_rate = 0.3
                    elif learning_rate_mode == 'user defined':
                        learning_rate = st.slider('',0.01,0.30,label_visibility='collapsed',key='learning_rate_slider')
                with col_right:
                    # objective参数
                    objective = st.selectbox('objective',['binary:logistic','binary:logitraw'])
            
            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # gamma参数
                    gamma_mode = st.selectbox('gamma',['default','user defined'],key='gamma')
                    if gamma_mode == 'default':
                        gamma = 0.1
                    elif gamma_mode == 'user defined':
                        gamma = st.slider('',0.0,2.0,label_visibility='collapsed',key='gamma_slider')
                with col_right:
                    # alpha参数
                    alpha_mode = st.selectbox('alpha',['default','user defined'],key='alpha')
                    if alpha_mode == 'default':
                        alpha = 1
                    elif gamma_mode == 'user defined':
                        alpha = st.slider('',0.0,10.0,label_visibility='collapsed',key='alpha_slider')

            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # lambda参数
                    lambda_mode = st.selectbox('lambda',['default','user defined'],key='lambda')
                    if lambda_mode == 'default':
                        lambda_ = 1
                    elif lambda_mode == 'user defined':
                        lambda_ = st.slider('',0.0,10.0,label_visibility='collapsed',key='lambda_slider')
                with col_right:
                    pass
                
        parameter_dict = {      'booster':booster,
                                'silent':silent,
                                'n_estimators':n_estimators,
                                'max_depth':max_depth,
                                'min_child_weight':min_child_weight,
                                'subsample':subsample,
                                'verbose':verbose,
                                'colsample_bytree':colsample_bytree,
                                'learning_rate':learning_rate,
                                'objective':objective,
                                'gamma':gamma,
                                'alpha':alpha,
                                'lambda':lambda_}
    
    # 展示最终选择的参数清单-便于后续检验输入参数是否符合需求
    with col_2:
        st.write('参数统计')
        for each in parameter_dict:
            st.write(each,':',parameter_dict[each])

    return parameter_dict