# WEB Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# ML Pkgs

# custom Pkgs
from exfuction.main_fxn import *

with st.sidebar:
    data_source = yaml_read()['data_source']
    # 数据库逻辑
    if data_source == "database":
        st.write("you choose the {} as the data source".format(data_source))
        ## 选择所要使用的数据表
        selected_tables = st.multiselect("选择需要使用的表",db_tables(),key='tables')
        ## 读取数据
        ## 选择选用的特征
        temp_list4features = db_get_all_columns(selectde_tables=selected_tables)
        selected_features = st.multiselect("选取需要使用的特征",temp_list4features,key='features')

    # 本地文件逻辑
    elif data_source == "local":
        st.write("you choose the {} as the data source".format(data_source))

    ## 选择所要使用的数据表
    selected_tables = st.multiselect("选择需要使用的表",db_tables())

    ## 读取数据

    ## 选择选用的特征

    ## 选择编码方案
    selected_encode_type = st.selectbox("选择编码方式",["OneHot","digital"]) 
    
    ## 选择特征过滤方案


## 根据筛选出的特征组织DataFrame
    if data_source == 'database':
        st.write("")

    elif data_source == 'local':
        st.write("")
## 统计缺失项
## 对于缺失项进行填补（反预测） --分两部分，名义离散用分类评估预测；

