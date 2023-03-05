# 这个文件只是暂时用来将csv写入数据库，以及涉及数据库操作的函数的集合

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from exfuction.main_fxn import yaml_read

## 读取csv为DataFrame格式，将pd.DataFrame类写入数据库
def csv2db(filepath='./dataset/NIJ_s_Recidivism_Datasets.csv'):
    # 读取文件
    data = pd.read_csv(filepath)

    db_username = yaml_read()['db']['username']
    db_passwd = yaml_read()['db']['passwd']
    db_port = yaml_read()['db']['port']
    db_IPaddress = yaml_read()['db']['IPaddress']
    db_name = yaml_read()['db']['dbname']

    # 创建表
    # SQL = "CREATE TABLE IF NOT EXISTS t_nij"

    # 连接数据库
    engine = create_engine(
        f'mysql+pymysql://{db_username}:{quote_plus(db_passwd)}@{db_IPaddress}:{db_port}/{db_name}?charset=utf8'
    )

    # 写入数据库
    data.to_sql('t_nij',engine,if_exists='replace',chunksize=10000,index=True,index_label='Index')
    
csv2db()

## 从数据库中查询数据，并返回DataFrame格式