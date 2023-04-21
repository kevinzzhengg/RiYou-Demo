# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import joblib

# EDA Pkgs
import pandas as pd
import numpy as np

# Sys Pkgs
import os
import re

# exfuction Pkgs
from exfuction.main_fxn import *

def lgbm_train(dataset,parameter_dict={}):
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
    
    if check_save_models_file(dataset,'lgbm') and parameter_dict == {}:
        lgbm = joblib.load('./userdata/{}/lgbm.pkl'.format(dataset_name))
        return lgbm
    else:

        ## Split the Features and Target
        x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
        y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
        for i in [Xtrain,Xtest,Ytrain,Ytest]:
            i.index = range(i.shape[0])

        ## Run the DecisionTree Model
        lgbm = LGBMClassifier(boosting_type=parameter_dict['boosting_type'],
                              num_leaves=parameter_dict['num_leaves'],
                              n_estimators=parameter_dict['n_estimators'],
                              max_depth=parameter_dict['max_depth'],
                              min_child_weight=parameter_dict['min_child_weight'],
                              subsample=parameter_dict['subsample'],
                              colsample_bytree=parameter_dict['colsample_bytree'],
                              learning_rate=parameter_dict['learning_rate'],
                              reg_alpha=parameter_dict['reg_alpha'],
                              reg_lambda=parameter_dict['reg_lambda'])
        lgbm = lgbm.fit(Xtrain,Ytrain)

        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(lgbm,'./userdata/{}/lgbm.pkl'.format(dataset_name))
        st.success('trainning completed!')
        return lgbm


def lgbm_parameter_add():
    # 提供超参数选择
    col_1,col_2 = st.columns([3,1])
    with col_1:
        with st.container():
            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # booster参数
                    boosting_type = st.selectbox('boosting_type',['gbdt','rf','dart','goss'],key='boosting_type')
                with col_right:
                    # num_leaves参数
                    num_leaves_mode = st.selectbox('num_leaves',['default','user defined'],key='num_leaves')
                    if num_leaves_mode == 'default':
                        num_leaves = 31
                    elif num_leaves_mode == 'user defined':
                        num_leaves = st.slider('',20,50,label_visibility='collapsed',key='num_leaves_slider')

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
                    # learning_rate参数
                    learning_rate_mode = st.selectbox('learning_rate',['default','user defined'],key='learning_rate')
                    if learning_rate_mode == 'default':
                        learning_rate = 0.3
                    elif learning_rate_mode == 'user defined':
                        learning_rate = st.slider('',0.01,0.30,label_visibility='collapsed',key='learning_rate_slider')
                with col_right:
                    # colsample_bytree参数
                    colsample_bytree_mode = st.selectbox('colsample_bytree',['default','user defined'],key='colsample_bytree')
                    if colsample_bytree_mode == 'default':
                        colsample_bytree = 1.0
                    elif colsample_bytree_mode == 'user defined':
                        colsample_bytree = st.slider('',0.5,1.0,label_visibility='collapsed',key='colsample_bytree_slider')

            
            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # reg_alpha参数
                    reg_alpha_mode = st.selectbox('reg_alpha',['default','user defined'],key='reg_alpha')
                    if reg_alpha_mode == 'default':
                        reg_alpha = 0.0
                    elif reg_alpha_mode == 'user defined':
                        reg_alpha = st.slider('',0.0,10.0,label_visibility='collapsed',key='reg_alpha_slider')
                with col_right:
                    # reg_lambda参数
                    reg_lambda_mode = st.selectbox('reg_lambda',['default','user defined'],key='reg_lambda')
                    if reg_lambda_mode == 'default':
                        reg_lambda = 0.0
                    elif reg_lambda_mode == 'user defined':
                        reg_lambda = st.slider('',0.0,10.0,label_visibility='collapsed',key='reg_lambda_slider')
                
        parameter_dict = {      'boosting_type':boosting_type,
                                'num_leaves':num_leaves,
                                'n_estimators':n_estimators,
                                'max_depth':max_depth,
                                'min_child_weight':min_child_weight,
                                'subsample':subsample,
                                'colsample_bytree':colsample_bytree,
                                'learning_rate':learning_rate,
                                'reg_alpha':reg_alpha,
                                'reg_lambda':reg_lambda}
    
    # 展示最终选择的参数清单-便于后续检验输入参数是否符合需求
    with col_2:
        st.write('参数统计')
        for each in parameter_dict:
            st.write(each,':',parameter_dict[each])

    return parameter_dict