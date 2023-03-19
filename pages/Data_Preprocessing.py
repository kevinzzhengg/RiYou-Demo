# WEB Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# ML Pkgs
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer

# Sys Pkgs
import time
import os
import re

# custom Pkgs
from exfuction.main_fxn import *

st.set_page_config(page_title="Preprocessing",layout="wide")

def read_cache_pt():
    if 'temp.csv' not in os.listdir('./cache'):
        st.error('please choose dataset first.')
    else:
        data = pd.read_csv('./cache/temp.csv',index_col=False)
        return data
    
def write_cache_pt(data):
    data.to_csv('./cache/temp.csv',index=False)


# sidebar
with st.sidebar:

    # 选择后更改配置文件
    st.caption('选择数据集(平台)')
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
        # 在平台本地创建暂存副本
        if 'temp.csv' in os.listdir('./cache'):
            data = pd.read_csv('./cache/temp.csv',index_col=False)
        else:
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
            opt_list = ['删除','无量纲化','编码','填补缺失值']
            selected_opt = st.selectbox("选择对于该列数据的操作",opt_list)
            # 无量纲化
            if selected_opt == '无量纲化':
                st.caption('建议先填补缺失值再进行无量纲化')
                nond_text_col, nond_input_col = st.columns([1,1])
                with nond_text_col:
                    st.write("选择无量纲化的方式")
                with nond_input_col:
                    selected_nond_type = st.radio('',['MinMaxScaler','StandardScaler'],label_visibility='collapsed')
            
            # 编码
            elif selected_opt == '编码':
                if data.loc[:,selected_column].nunique() < 10:
                    st.caption('所选的列值为离散值')
                    flag_apply_in_all = st.checkbox('应用于全表')
                    st.caption('选择编码方式')
                    selected_encode_type = st.radio('',['Onehot','Number Replace'],label_visibility='collapsed')
                else:
                    st.caption('所选的列值为连续值')
                    flag_apply_in_all = st.checkbox('应用于全表')
                    st.caption('选择编码方式')
                    selected_encode_type = st.radio('',['分箱','Other(not add yet)'],label_visibility='collapsed')

            # 填补缺失值
            elif selected_opt == '填补缺失值':
                miss_count = data.loc[:,selected_column].isnull().sum()
                st.caption("the selected column has {} NaN values".format(miss_count))
                select_fill_type = st.selectbox('选择填补方式',['众数','平均数','中位数','0'])

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
            clf_mod_fill = ['DT','RF']
            selected_clf_fill = st.selectbox('',clf_mod_fill,label_visibility="collapsed",key='clf')
        with col3:
            st.write("预测连续数据的回归模型")
        with col4:
            reg_mod_fill = ['DT','RF']
            selected_reg_fill = st.selectbox('',reg_mod_fill,label_visibility="collapsed",key='reg')

# 提交逻辑判定,并操作数据后写入暂存文件
if submit:
    # 删除逻辑
    if selected_opt == '删除':
        data.drop([selected_column],axis=1,inplace=True)
        write_cache_pt(data)

    # 无量纲化操作
    elif selected_opt == '无量纲化':
        if selected_nond_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
            data.loc[:,selected_column] = scaler.fit_transform(data.loc[:,selected_column].values.reshape(-1,1))
            write_cache_pt(data)
        elif selected_nond_type == 'StandardScaler':
            scaler = StandardScaler()
            data.loc[:,selected_column] = scaler.fit_transform(data.loc[:,selected_column].values.reshape(-1,1))
            write_cache_pt(data)

    # 离散数据编码,连续数据进行分箱
    elif selected_opt == '编码':
        if flag_apply_in_all:
            time.sleep(0.5)
            write_cache_pt(data)
        else:
            if selected_encode_type == 'Onehot':
                new_col_name = []
                ohe = OneHotEncoder(categories='auto')
                result = ohe.fit_transform(data.loc[:,selected_column].values.reshape(-1,1)).toarray()
                encoding_col = ohe.get_feature_names_out()
                for each in encoding_col:
                    res_match = re.search(r"_(.*)",each)
                    tmp = selected_column + '_' + res_match.groups()[0]
                    new_col_name += [tmp]
                new_col = pd.DataFrame(result)
                new_col.columns = new_col_name
                data.drop([selected_column],axis=1,inplace=True)
                data = pd.concat([data,new_col],axis=1)
                write_cache_pt(data)

            elif selected_encode_type == 'Number Replace':
                le = LabelEncoder()
                data.loc[:,selected_column] = le.fit_transform(data.loc[:,selected_column])
                write_cache_pt(data)

            # elif selected_encode_type == '分箱':

    # 一般数据进行填补缺失值
    elif selected_opt == '填补缺失值':
        if select_fill_type == '众数':
            imp = SimpleImputer(strategy='most frequent')
            data.loc[:,selected_column] = imp.fit_transform(data.loc[:,selected_column])
            write_cache_pt(data)
        elif select_fill_type == '平均数':
            imp = SimpleImputer(strategy='mean')
            data.loc[:,selected_column] = imp.fit_transform(data.loc[:,selected_column])
            write_cache_pt(data)
        elif select_fill_type == '中位数':
            imp = SimpleImputer(strategy='median')
            data.loc[:,selected_column] = imp.fit_transform(data.loc[:,selected_column])
            write_cache_pt(data)
        elif select_fill_type == '0':
            imp = SimpleImputer(strategy='constant',fill_value=0)
            data.loc[:,selected_column] = imp.fit_transform(data.loc[:,selected_column])
            write_cache_pt(data)

else:
    write_cache_pt(data)

# 数据表的一些基本数据展示

data = read_cache_pt()

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
            try:
                st.dataframe(data.loc[1:20,selected_column],use_container_width=True)
            except KeyError:
                st.write('')

# 暂存的文件的下载与处理
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
