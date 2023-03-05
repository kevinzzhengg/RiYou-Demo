# WEB Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd

# Plot Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

# Model Pkgs
from models.Model_DecisionTree import DecisionTreeCLF_PRE,DecisionTreeCLF_Parameter_Add

# Configure Read
import yaml

st.set_page_config(page_title="Model",layout="wide")

model_list = ["Comprehensive","DecisionTreeClassifier","KNN"]
CONFIG_LIST = {}

# FXN
def model_render(model_name="Comprehensive"):
    if model_name not in model_list: return "the model error"
    st.title(model_name)

def model_parameter_add(model_name="Comprehensive"):
    st.write(model_name)
    if model_name == "DecisionTreeClassifier":
        DecisionTreeCLF_Parameter_Add()

def yaml_read():
    with open("./config.yaml",'r',encoding='utf-8') as f:
        res = yaml.load(f,Loader=yaml.FullLoader)
    return res

def yaml_write(config_list=CONFIG_LIST):
    with open('./config.yaml','w',encoding='utf-8') as f:
        yaml.dump(data=config_list, stream=f, allow_unicode=True)

## Sidebar Parameters
with st.sidebar:
    selected_model = st.selectbox("Select A Model",model_list)
    model_parameter_add(selected_model)

## Main Part
## Model Choose
model_render(selected_model)
dataset_name = yaml_read()['dataset_path']
st.info("The Model is using dataset {}".format(dataset_name))

if selected_model == "DecisionTreeClassifier":
    DecisionTreeCLF_PRE(dataset_name)

