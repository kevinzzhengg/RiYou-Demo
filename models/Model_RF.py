# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

def randomforest_predict(dataset,parameter_dict={}):
    # st.write("this is th test demo of RF")
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
    
    if check_save_models_file(dataset,'rf') and parameter_dict == {}:
        rfc = joblib.load('./userdata/{}/rf.pkl'.format(dataset_name))
        return rfc
    else:

        ## Split the Features and Target
        x = data.iloc[:,data.columns != "Recidivism_Within_3years"]
        y = data.iloc[:,data.columns == "Recidivism_Within_3years"]

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)
        for i in [Xtrain,Xtest,Ytrain,Ytest]:
            i.index = range(i.shape[0])

        ## Run the DecisionTree Model
        rfc = RandomForestClassifier(n_estimators=parameter_dict['n_estimators'],
                                     oob_score=parameter_dict['oob_score'],
                                     criterion=parameter_dict['criterion'],
                                     max_features=parameter_dict['max_features'],
                                     max_depth=parameter_dict['max_depth'],
                                     random_state=parameter_dict['random_state'],
                                     min_samples_split=parameter_dict['min_samples_split'],
                                     min_samples_leaf=parameter_dict['min_samples_leaf'],
                                     min_weight_fraction_leaf=parameter_dict['min_weight_fraction_leaf'],
                                     max_leaf_nodes=parameter_dict['max_leaf_nodes'],
                                     class_weight=parameter_dict['class_weight'],
                                     min_impurity_decrease=parameter_dict['min_impurity_decrease'])
        rfc = rfc.fit(Xtrain,Ytrain)

        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(rfc,'./userdata/{}/rf.pkl'.format(dataset_name))
        st.success('trainning completed!')

        n_features = x.shape[1]
        plt.figure(figsize=(3,8))
        plt.barh(range(n_features), rfc.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), x.columns.tolist())
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        st.pyplot(plt)
        plt.close()

        return rfc
        # score = rfc.score(Xtest,Ytest)
        # with st.container():
        #     st.caption("accuracy rating of RF: {}".format(score))
        # with st.sidebar:
        #     st.write("accuracy rating of RF: {}".format(score))

        # return rfc


def randomforest_parameter_add():
    # 提供超参数选择
    col_1,col_2 = st.columns([3,1])
    with col_1:
        with st.container():

            st.subheader('Bagging框架参数')

            with st.container():                
                col_left,col_right = st.columns(2)
                with col_left:
                    n_estimators = st.slider('n_estimators',100,500,key='n_estimators')
                    oob_score = st.selectbox('oob score',[False,True],key='oob_score')
                with col_right:
                    criterion = st.selectbox('criterion',['gini','entropy'],key='criterion')
            
            st.subheader('决策树参数')

            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # max_depth参数
                    max_depth_mode = st.selectbox('max depth',[None,'user defined'],key='max_depth')
                    if max_depth_mode == None:
                        max_depth = None
                    elif max_depth_mode == 'user defined':
                        max_depth = st.slider('',10,100,label_visibility='collapsed',key='max_depth_slider')
                        
                with col_right:
                    # min_sample_split_mode参数
                    min_samples_split_mode = st.selectbox('min samples split mode',['default','user defined'],key='min_samples_split')
                    if min_samples_split_mode == 'default':
                        min_samples_split = 2
                    elif min_samples_split_mode == 'user defined':
                        min_samples_split = st.slider('',1,10,label_visibility='collapsed',key='min_samples_split slider')
            
            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # min_samples_leaf参数
                    min_samples_leaf_mode = st.selectbox('min samples leaf mode',['default','user defined'])
                    if min_samples_leaf_mode == 'default':
                        min_samples_leaf = 1
                    elif min_samples_leaf_mode == 'user defined':
                        min_samples_leaf = st.slider('',1,5,label_visibility='collapsed')
                with col_right:
                    # min_weight_fraction_leaf参数
                    min_weight_fraction_leaf_mode = st.selectbox('min weight fraction leaf',['default','user defined'])
                    if min_weight_fraction_leaf_mode == 'default':
                        min_weight_fraction_leaf = 0.0
                    elif min_weight_fraction_leaf_mode == 'user defined':
                        min_weight_fraction_leaf = st.slider('',0.0,1.0,label_visibility='collapsed')

            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    # max features参数
                    max_features = st.selectbox('max_featrues',[None,'auto','sqrt','log2'])
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
                    # min_impurity_decrease参数
                    min_impurity_decrease_mode = st.selectbox('min_impurity_decrease',['default','user defined'])
                    if min_impurity_decrease_mode == 'default':
                        min_impurity_decrease = 1e-7
                    elif min_impurity_decrease_mode == 'user defined':
                        min_impurity_decrease = st.text_input('',label_visibility='collapsed')
                with col_right:
                    # class_weight参数
                    class_weight = st.selectbox('class weight',[None,'balanced'])

            with st.container():
                col_left,col_right = st.columns(2)
                with col_left:
                    random_state = int(st.text_input('random state',value=2023))
                with col_right:
                    pass
                
        parameter_dict = {      'n_estimators':n_estimators,
                                'oob_score':oob_score,
                                'criterion':criterion,
                                'max_features':max_features,
                                'max_depth':max_depth,
                                'random_state':random_state,
                                'min_samples_split':min_samples_split,
                                'min_samples_leaf':min_samples_leaf,
                                'min_weight_fraction_leaf':min_weight_fraction_leaf,
                                'max_leaf_nodes':max_leaf_nodes,
                                'class_weight':class_weight,
                                'min_impurity_decrease':min_impurity_decrease}
    
    # 展示最终选择的参数清单-便于后续检验输入参数是否符合需求
    with col_2:
        st.write('参数统计')
        for each in parameter_dict:
            st.write(each,':',parameter_dict[each])

    return parameter_dict