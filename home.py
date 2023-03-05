# -- coding : utf-8 --
import streamlit as st
import pandas as pd
from exfuction.main_fxn import yaml_read,yaml_write,yaml_change
from exfuction.main_fxn import db_connection

def main():
    st.set_page_config(page_title="HomePage"
                    , page_icon="bar_chart"
                    #, layout="wide"
                    , initial_sidebar_state="expanded"
                )
    
    st.title("Community Corrector Recidivism Predicted System")
    st.caption("use different AI models to predict the possibility of the community corrector to recidivism.")
    
    with st.container():
        selected_ds = st.selectbox("数据来源",["local","database"])
        if selected_ds == "local":
            st.write("  ")
            yaml_change(['data_source'],'local')
        elif selected_ds == "database":
            yaml_change(['data_source'],'database')
            out_left, out_right = st.columns(2)
            with out_left:
                in_left, in_right = st.columns(2)
                with in_left:
                    db_IP = st.text_input("数据库IP地址",key="ip")
                with in_right:
                    db_port = st.text_input("端口号",key="port")
                db_username = st.text_input("用户名")
                db_passwd = st.text_input("密码",type="password")
                db_name = st.text_input("数据库名称")
                button_conn = st.button("连接",use_container_width=True)
                
            with out_right:
                if button_conn:
                    yaml_change(['db','IPaddress'],db_IP)
                    yaml_change(['db','port'],int(db_port))
                    yaml_change(['db','username'],db_username)
                    yaml_change(['db','passwd'],db_passwd)
                    yaml_change(['db','dbname'],db_name)
                    db_connection()

if __name__ == '__main__':
    main()