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
    uploaded_file = st.file_uploader("请选择一个CSV文件", type='csv', key="csvfile")
    data = pd.DataFrame()
    try:
        if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                data.to_csv('./dataset/{}'.format(uploaded_file.name),index=False)
                st.success('upload success!')
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
    with st.expander('log info'):
        in_left, in_middle,in_right = st.columns([1,3,1])
        with in_middle:
            with st.container():
                col1,col2 = st.columns(2)
                with col1:
                    db_IP = st.text_input("数据库IP地址",key="ip")
                with col2:
                    db_port = st.text_input("端口号",key="port")
                db_username = st.text_input("用户名")
                db_passwd = st.text_input("密码",type="password")
                db_name = st.text_input("数据库名称")
                button_conn = st.button("连接",use_container_width=True)
            
            # with out_right:
            #     if button_conn:
            #         yaml_change(['db','IPaddress'],db_IP)
            #         yaml_change(['db','port'],int(db_port))
            #         yaml_change(['db','username'],db_username)
            #         yaml_change(['db','passwd'],db_passwd)
            #         yaml_change(['db','dbname'],db_name)
            #         db_connection()


