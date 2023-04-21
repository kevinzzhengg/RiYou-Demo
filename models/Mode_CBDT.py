# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# EDA Pkgs
import pandas as pd
import numpy as np

# Sys Pkgs
import os
import re

# exfuction Pkgs
from exfuction.main_fxn import *

def cbdt_train(dataset,parameter_dict={}):
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
    
    if check_save_models_file(dataset,'cbdt') and parameter_dict == {}:
        gbdt = joblib.load('./userdata/{}/cbdt.pkl'.format(dataset_name))
        return gbdt
    else:

        ## Split the Features and Target
        x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
        y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
        for i in [Xtrain,Xtest,Ytrain,Ytest]:
            i.index = range(i.shape[0])

        ## Run the DecisionTree Model
        gbdt = GradientBoostingClassifier(n_estimators=parameter_dict['n_estimators'],
                                          loss=parameter_dict['loss'],
                                          subsample=parameter_dict['subsample'],
                                          max_features=parameter_dict['max_features'],
                                          max_depth=parameter_dict['max_depth'],
                                          random_state=parameter_dict['random_state'],
                                          min_samples_split=parameter_dict['min_samples_split'],
                                          min_weight_fraction_leaf=parameter_dict['min_weight_fraction_leaf'],
                                          max_leaf_nodes=parameter_dict['max_leaf_nodes'],
                                          min_impurity_decrease=parameter_dict['min_impurity_decrease'])
        gbdt = gbdt.fit(Xtrain,Ytrain)

        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(gbdt,'./userdata/{}/cbdt.pkl'.format(dataset_name))
        st.success('trainning completed!')
        return gbdt


def cbdt_parameter_add():
    # 提供超参数选择
    # # boosting框架参数
	# n_estimators - [default=100,user defined 100~500]
	# learning_rate - slider 0.0~1.0
	# subsample - slider 0.0~1.0
	# loss - ['deviance','exponential'] #损失函数
	
	# # 弱学期器参数
	# max_features
	# max_depth
	# min_samples_split
	# min_samples_leaf
	# min_weight_fraction_leaf
	# max_leaf_nodes
	# min_impurity_split
    col_1,col_2 = st.columns([3,1])
    with col_1:
        with st.container():
            st.subheader('boosting框架参数')
            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # n_estimators参数
                    n_estimators = st.slider('n_estimators',100,500,key='n_estimators')
                with col_right:
                    # loss参数
                    loss = st.selectbox('loss',['deviance','exponential'])

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # subsample参数
                    subsample_mode = st.selectbox('subsample',['default','user defined'],key='subsample')
                    if subsample_mode == 'default':
                        subsample = 1.0
                    elif subsample_mode == 'user defined':
                        subsample = st.slider('',0.5,1.0,label_visibility='collapsed',key='subsample_slider')
                with col_right:
                    # learning_rate参数
                    learning_rate_mode = st.selectbox('learning_rate',['default','user defined'],key='learning_rate')
                    if learning_rate_mode == 'default':
                        learning_rate = 0.3
                    elif learning_rate_mode == 'user defined':
                        learning_rate = st.slider('',0.01,0.30,label_visibility='collapsed',key='learning_rate_slider')

            st.subheader('弱学习器参数')
            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # max features参数
                    max_features = st.selectbox('max_featrues',[None,'auto','sqrt','log2'])
                with col_right:
                    # max_depth参数
                    max_depth_mode = st.selectbox('max depth',['None','user defined'],key='max_depth')
                    if max_depth_mode == 'None':
                        max_depth = None
                    elif max_depth_mode == 'user defined':
                        max_depth = st.slider('',10,100,label_visibility='collapsed',key='max_depth_slider')

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    # min_sample_split参数
                    min_samples_split_mode = st.selectbox('min samples split mode',['default','user defined'],key='min_samples_split')
                    if min_samples_split_mode == 'default':
                        min_samples_split = 2
                    elif min_samples_split_mode == 'user defined':
                        min_samples_split = st.slider('',1,10,label_visibility='collapsed',key='min_samples_split slider')

                with col_right:
                    # min_samples_leaf参数
                    min_samples_leaf_mode = st.selectbox('min samples leaf mode',['default','user defined'])
                    if min_samples_leaf_mode == 'default':
                        min_samples_leaf = 1
                    elif min_samples_leaf_mode == 'user defined':
                        min_samples_leaf = st.slider('',1,5,label_visibility='collapsed')
            
            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # min_weight_fraction_leaf参数
                    min_weight_fraction_leaf_mode = st.selectbox('min weight fraction leaf',['default','user defined'])
                    if min_weight_fraction_leaf_mode == 'default':
                        min_weight_fraction_leaf = 0.0
                    elif min_weight_fraction_leaf_mode == 'user defined':
                        min_weight_fraction_leaf = st.slider('',0.0,1.0,label_visibility='collapsed')
                with col_right:
                    # max_leaf_nodes参数
                    max_leaf_nodes_mode = st.selectbox('max_leaf_nodes',['default','user defined'])
                    if max_leaf_nodes_mode == 'default':
                        max_leaf_nodes = None
                    elif max_leaf_nodes_mode == 'user defined':
                        max_leaf_nodes = st.slider('',1,10,label_visibility='collapsed')
            
            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # min impurity split(decrease)
                    min_impurity_decrease_mode = st.selectbox('min impurity decrease',['default','user defined'])
                    if min_impurity_decrease_mode == 'default':
                        min_impurity_decrease = 1e-7
                    elif min_impurity_decrease_mode == 'user defined':
                        min_impurity_decrease = st.text_input('',label_visibility='collapsed')
                with col_right:
                    random_state = int(st.text_input('random state',value=2023))
                
        parameter_dict = {      'n_estimators':n_estimators,
                                'loss':loss,
                                'subsample':subsample,
                                'max_features':max_features,
                                'max_depth':max_depth,
                                'random_state':random_state,
                                'min_samples_split':min_samples_split,
                                'learning_rate':learning_rate,
                                'min_samples_leaf':min_samples_leaf,
                                'min_weight_fraction_leaf':min_weight_fraction_leaf,
                                'max_leaf_nodes':max_leaf_nodes,
                                'min_impurity_decrease':min_impurity_decrease}
    
    # 展示最终选择的参数清单-便于后续检验输入参数是否符合需求
    with col_2:
        st.write('参数统计')
        for each in parameter_dict:
            st.write(each,':',parameter_dict[each])

    return parameter_dict