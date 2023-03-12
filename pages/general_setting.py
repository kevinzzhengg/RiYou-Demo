import streamlit as st
import pandas as pd 
import numpy as np

from exfuction.main_fxn import yaml_read,yaml_write,yaml_change,file_selector

st.markdown("# General Setting\n------")

## 数据源选择
st.header("data source choose")
ds_selected = st.selectbox("",['local','database'],label_visibility="collapsed")

# 选择本地文件
if ds_selected == "local":
    st.write("choose local csv file")
    file_selected = file_selector('.\dataset')
    try:
        data = pd.read_csv(file_selected,index_col=False)
        if 'Unnamed: 0' in data.columns:
            data.drop(['Unnamed: 0'],inplace=True,axis=1)
        with st.expander("data overview"):
            tab1, tab2 = st.tabs(["Data Frame Overview(top ten)","Count"])
            with tab1:
                st.dataframe(data.head(10))
            with tab2:
                st.write("rows count:",len(data))
                st.write("columns count:",len(data.columns.tolist()))
    except FileNotFoundError:
        st.error("file path error")


# 数据库表单
elif ds_selected == "database":
    in_left, in_right = st.columns(2)
    with in_left:
        db_IP = st.text_input("数据库IP地址",key="ip")
    with in_right:
        db_port = st.text_input("端口号",key="port")
    db_username = st.text_input("用户名")
    db_passwd = st.text_input("密码",type="password")
    db_name = st.text_input("数据库名称")
    button_conn = st.button("连接",use_container_width=True)


