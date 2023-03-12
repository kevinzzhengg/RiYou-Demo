# WEB Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# ML Pkgs

# Sys Pkgs
import time

# custom Pkgs
from exfuction.main_fxn import *

st.set_page_config(page_title="Preprocessing",layout="wide")



# sidebar
with st.sidebar:

    filename = file_selector(".\dataset")
    config_list = yaml_read()
    config_list['dataset_path'] = filename
    yaml_write(config_list=config_list)

    data_source = yaml_read()['data_source']
    # 数据库逻辑
    if data_source == "database":
        st.info("{}".format(data_source))

    # 本地文件逻辑
    elif data_source == "local":
        st.info("{}".format(data_source))
        dataset_path = yaml_read()['dataset_path']
        data = pd.read_csv(dataset_path,index_col=False)

    # st.info("You Selected {}".format(filename))

# 选项模块
with st.container():
    st.subheader("数据预处理")
    single_tab,entirety_tab = st.tabs(['单列操作','整表操作'])
    with single_tab:
        col1,col2,col3 = st.columns([2,2,1])
        with col1:
            # 获取表的所有列
            features = data.columns.tolist()
            selected_column = st.selectbox("选择操作列",features)
        with col2:
            #选择所需要的操作
            opt_list = ['删除','无量纲化','数据映射','编码','填补缺失值']
            selected_opt = st.selectbox("选择对于该列数据的操作",opt_list)
        with col3:
            # 对齐
            st.write('')
            st.write('')
            submit = st.button("do it",use_container_width=False)


    with entirety_tab:
        st.caption("模型预测填补缺失值")
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.write("预测离散数据的分类模型")
        with col2:
            selected_clf_fill = st.selectbox('',[1,2],label_visibility="collapsed",key='clf')
        with col3:
            st.write("预测连续数据的回归模型")
        with col4:
            selected_reg_fill = st.selectbox('',[1,2],label_visibility="collapsed",key='reg')

# 数据表的一些基本数据展示
with st.container():
    rows_col, columns_col, missing_rate_col = st.columns([1,1,2])
    with rows_col:
        st.write("⬇️rows:",data.shape[0])
    with columns_col:
        st.write("➡️columns:",len(data.columns.tolist()))
    with missing_rate_col:
        missing_rate = (sum(data.isnull().sum().to_dict().values())/(data.shape[0]*len(data.columns.tolist())))
        st.write("❌missing rate:",missing_rate)

with st.container():
    main_tab,sub_tab = st.columns([3,1])
    with main_tab:
        dataframe_tab,datainfo_tab = st.tabs(["展示","概览"])
        with dataframe_tab:
            df_empty = st.empty()
            with df_empty.container():
                st.dataframe(data.head(20))
            #time.sleep(1)
            #df_empty.empty()
        with datainfo_tab:
            st.write("wait")
                
    with sub_tab:
        column_tab,info_tab = st.tabs([selected_column,'info'])
        with column_tab:
            st.dataframe(data.loc[1:20,selected_column],use_container_width=True)

with st.container():
    if st.checkbox("保存文件"):
        col1,col2,col3 = st.columns(3)
        with col1:
            save_filename = st.text_input("保存的文件名",value="保存的文件名.csv",label_visibility="collapsed")
        with col2:
            save_type = st.radio('',['local','plantform'],label_visibility="collapsed")
        with col3:
            if save_type == 'local':
                st.download_button(
                    label='Download data as CSV',
                    data=download_local(data),
                    file_name=save_filename,
                    mime='text/csv'
                )
            elif save_type == 'plantform':
                if st.button("save on the platform"):
                    download_plantform(filename=save_filename)

        

    
        

# 特征过滤
# 数据标准化
# 数据归一化
# 数据缺失值处理


