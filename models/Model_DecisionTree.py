# Render Pkgs
import streamlit as st

# ML Pkgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
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

def decisiontree_clf_predict(dataset,parameter_dict={}):
    # st.write("this is the test demo of DecisionTree")
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
            return st.warning("the dataset is unqualified.\nyou need do some preprocessing first")
        
    # 首先检查是否有已保存的模型
    if check_save_models_file(dataset,'dt') and parameter_dict == {}:
        clf = joblib.load('./userdata/{}/dt.pkl'.format(dataset_name))
        return clf
    else:
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
        clf = DecisionTreeClassifier(criterion=parameter_dict['criterion'],
                                    splitter=parameter_dict['splitter'],
                                    max_features=parameter_dict['max_features'],
                                    max_depth=parameter_dict['max_depth'],
                                    random_state=parameter_dict['random_state'],
                                    min_samples_split=parameter_dict['min_samples_split'],
                                    min_weight_fraction_leaf=parameter_dict['min_weight_fraction_leaf'],
                                    max_leaf_nodes=parameter_dict['max_leaf_nodes'],
                                    class_weight=parameter_dict['class_weight'],
                                    min_impurity_decrease=parameter_dict['min_impurity_decrease'])
        clf = clf.fit(Xtrain,Ytrain)

        # 训练之后保存模型文件
        if not os.path.exists('./userdata/{}'.format(dataset_name)):
            os.mkdir('./userdata/{}'.format(dataset_name))
        joblib.dump(clf,'./userdata/{}/dt.pkl'.format(dataset_name))
        st.success('trainning completed!')   
        return clf
       

def decisiontree_clf_parameter_add():
    # 提供超参数选择
    col_1,col_2 = st.columns([3,1])
    with col_1:
        with st.container():
            col_left,col_right = st.columns(2)
            with col_left:
                # criterion
                criterion = st.selectbox('criterion',['gini','entropy'])
                # splitter
                splitter = st.selectbox('splitter',['best','random'])
                # max features
                max_features = st.selectbox('max features',['auto','sqrt','log2','None'])\
                # max depth
                max_depth = st.selectbox('max depth',[None,'user definded'])
                if max_depth == 'user definded':
                    max_depth = st.slider('',10,100,label_visibility='collapsed')
                # random state
                random_state = int(st.text_input('random state',value=2023))

            with col_right:
                # min samples split
                min_samples_split_mode = st.selectbox('min samples split',['default','user defined'])
                if min_samples_split_mode == 'default':
                    min_samples_split = 2
                elif min_samples_split_mode == 'user defined':
                    min_samples_split = st.text_input('请输入自定义的值',label_visibility='collapsed')
                # min weight fraction leaf
                min_weight_fraction_leaf_mode = st.selectbox('min weight fraction leaf',['default','user defined'])
                if min_weight_fraction_leaf_mode == 'default':
                    min_weight_fraction_leaf = 0
                elif min_weight_fraction_leaf_mode == 'user defined':
                    min_weight_fraction_leaf = st.slider('',0.0,1.0,0.1,label_visibility='collapsed')

                # max leaf nodes mode
                max_leaf_nodes_mode = st.selectbox('max leaf nodes',['default','user defined'])
                if max_leaf_nodes_mode == 'default':
                    max_leaf_nodes = None
                elif max_leaf_nodes_mode == 'user defined':
                    max_leaf_nodes = st.text_input('max leaf nodes')
                # class weight
                class_weight = st.selectbox('class weight',[None,'balanced'])
                # min_impurity_decrease
                min_impurity_decrease_mode = st.selectbox('min_impurity_decrease',['default','user defined'])
                if min_impurity_decrease_mode == 'default':
                    min_impurity_decrease = 1e-7
                elif min_impurity_decrease_mode == 'user defined':
                    min_impurity_decrease = st.text_input('',label_visibility='collapsed')
        parameter_dict = {'criterion':criterion,
                                'splitter':splitter,
                                'max_features':max_features,
                                'max_depth':max_depth,
                                'random_state':random_state,
                                'min_samples_split':min_samples_split,
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