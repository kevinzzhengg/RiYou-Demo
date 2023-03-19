import streamlit as st
import os
import time
import yaml

# EDA Pkgs
import pandas as pd

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

st.set_page_config(page_title="Dataset Config Page",page_icon='🎲',layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Dataset Configure Page")
st.caption("Choose your dataset to show the analyse results")

## FXN
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select A file",filenames)
    return os.path.join(folder_path,selected_filename)

def yaml_read():
    with open("./config.yaml",'r',encoding='utf-8') as f:
        res = yaml.load(f,Loader=yaml.FullLoader)
    return res

def yaml_write(config_list={}):
    with open('./config.yaml','w',encoding='utf-8') as f:
        yaml.dump(data=config_list, stream=f, allow_unicode=True)

### SideBar and the parameters
with st.sidebar:
    filename = file_selector(".\dataset")
    DATASET_PATH = filename

### Main Part
st.info("You Selected {}".format(filename))
config_list = yaml_read()
config_list['dataset_path'] = filename
yaml_write(config_list=config_list)


## read datasets files
data = pd.read_csv(filename)

## tab1 is for DataFrame,and 2 is for the Dtype
tab1, tab2,tab3 = st.tabs(["DataFrame", "Dtype", "Counts"])
with tab1:
    st.write("overview")
    st.dataframe(data=data.head(10))

with tab2:
    st.dataframe(data.dtypes,use_container_width=True)

with tab3:
    # rows count
    rows_count = data.shape[0]
    st.write("Data Rows Count:",rows_count)

    # feature count
    feature_count = len(data.columns.tolist())
    st.write("feature counts:",feature_count)

    # full records feature count
    # full records rows count

## Basic Info Part
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Plot[Seaborn]:")
        st.write(sns.heatmap(data.corr(),annot=False))
        st.pyplot()
    with col2:
        st.subheader("Plot of Value Counts")

## Plot Part