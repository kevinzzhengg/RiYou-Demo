# WEB Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd

# Plot Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

# Model Pkgs
from models.Model_DecisionTree import decisiontree_clf_predict,decisiontree_clf_parameter_add
from models.Model_RF import randomforest_predict,randomforest_parameter_add
from models.Model_KNN import knn_predict,knn_parameter_add
from models.Model_SVM import svm_predict,svm_parameter_add
from models.Naive_Bayes import naive_bayes_predict,naive_bayes_parameter_add

# Configure Read
from exfuction.main_fxn import yaml_read,yaml_change,yaml_write

st.set_page_config(page_title="Model",layout="wide")

model_list = ["OverView","DecisionTreeClassifier","RandomForest","KNN","SVM","Naive Bayes"]
CONFIG_LIST = {}

# FXN
def model_render(model_name="OverView"):
    if model_name not in model_list: return "the model error"
    st.title(model_name)

def model_parameter_add(model_name="OverView"):
    st.write(model_name)
    if model_name == "DecisionTreeClassifier":
        decisiontree_clf_parameter_add()
    elif model_name == "RandomForest":
        randomforest_parameter_add()
    elif model_name == "KNN":
        knn_parameter_add()
    elif model_name == "SVM":
        svm_parameter_add()
    elif model_name == "Naive Bayes":
        naive_bayes_parameter_add()
        


## Sidebar Parameters
with st.sidebar:
    selected_model = st.selectbox("Select A Model or Algorithm",model_list)
    model_parameter_add(selected_model)

## Main Part
## Model Choose
model_render(selected_model)
dataset_name = yaml_read()['dataset_path']
st.info("Dataset {} is selected.".format(dataset_name))

if selected_model == "OverView":
    st.write("the model predict results are roughly as follows")
    data = pd.read_csv(dataset_name)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'],inplace=True,axis=1)
    temp_df = data.head(10)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'],inplace=True,axis=1)
    samples = temp_df.iloc[:,data.columns == "ID"]
    x = temp_df.iloc[:,data.columns != "Recidivism_Within_3years"]
    # DTClf
    clf = decisiontree_clf_predict(dataset_name)
    res = pd.DataFrame(clf.predict(x))
    res.columns = ['DT']
    res_dt = res

    # RF
    rfc = randomforest_predict(dataset_name)
    res = pd.DataFrame(rfc.predict(x))
    res.columns = ['RF']
    res_rf = res

    # KNN
    knn = knn_predict(dataset_name)
    res = pd.DataFrame(knn.predict(x))
    res.columns = ['KNN']
    res_knn = res

    # SVM
    
    # NB
    nb = naive_bayes_predict(dataset_name)
    res = pd.DataFrame(knn.predict(x))
    res.columns = ['NB']
    res_nb = res

    frames = [samples,res_dt,res_rf,res_knn,res_nb]
    final_res = pd.concat(frames,axis=1)
    st.dataframe(final_res,use_container_width=True)


elif selected_model == "DecisionTreeClassifier":
    decisiontree_clf_predict(dataset_name)

elif selected_model == "RandomForest":
    randomforest_predict(dataset_name)

elif selected_model == "KNN":
    knn_predict(dataset_name)

elif selected_model == "SVM":
    svm_predict(dataset_name)

elif selected_model == "Naive Bayes":
    naive_bayes_predict(dataset_name)

