import yaml
import streamlit as st
import pymysql
import pandas as pd

# 对于yaml配置文件的读写操作
def yaml_read():
    """读取yaml配置文件"""
    with open("./config.yaml",'r',encoding='utf-8') as f:
        res = yaml.load(f,Loader=yaml.FullLoader)
    return res

def yaml_write(config_list={}):
    """覆写yaml配置文件"""
    with open('./config.yaml','w',encoding='utf-8') as f:
        yaml.dump(data=config_list, stream=f, allow_unicode=True)

def yaml_change(key:list,value):
    """修改对应键值"""
    config_list = yaml_read()
    if len(key) == 1:
        config_list[key[0]] = value
    elif len(key) == 2:
        config_list[key[0]][key[1]] = value
    else:
        return "key value error"
    yaml_write(config_list=config_list)

## 对于数据库的相关操作
# 数据库连接
@st.cache_resource
def db_connection():
    try:
        conn = pymysql.connect(host=yaml_read()['db']['IPaddress'],
                               user=yaml_read()['db']['username'],
                               password=yaml_read()['db']['passwd'],
                               port=yaml_read()['db']['port'],
                               database=yaml_read()['db']['dbname'])
        st.success("连接成功")
        return conn
    except Exception as err:
        st.error("连接失败\n{}".format(err))

# 检查数据库状态
def db_ping():
    c = db_connection()
    try:
        c.ping()
        with st.sidebar:
            st.success("the db is already connected")
    except:
        c = db_connection()

# 查询一共有多少表,并返回列表
def db_tables():
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute("show tables;")
    data = cursor.fetchall()
    res_list = []
    for i in range(len(data)):
        res_list.append(data[i][0])
    cursor.close()
    conn.close()
    return res_list

# 查询选中的表中一共有多少列，并返回列表
def db_get_all_columns(selectde_tables):
    conn = db_connection()
    cursor = conn.cursor()
    sql = """SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = '{0}' AND TABLE_NAME = '{1}';"""
    res = []
    for i in range(len(selectde_tables)):
        cursor.execute(sql.format(yaml_read()['db']['dbname'],selectde_tables[i]))
        data = cursor.fetchall()
        for i in range(len(data)):
            res.append(data[i][0])     
    # conn.close() 
    return res

# 选择需要的表项集合成一张DataFrame(important)
def sql_2_dataframe():
    sql_view1 = """ CREATE VIEW dataset AS SELECT {} from {} """
    sql_view2 = """SELECT * FROM dataset"""

# 选定需要的features和target(at end)-->data preprocessing part

