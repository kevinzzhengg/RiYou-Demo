# WEB Pkgs
import streamlit as st

# Sys Pkgs
import os
import re

# EDA Pkgs
import pandas as pd

# Plot Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

# Model Pkgs
from models.Model_DecisionTree import decisiontree_clf_predict,decisiontree_clf_parameter_add
from models.Model_RF import randomforest_predict,randomforest_parameter_add
from models.Model_KNN import knn_train,knn_parameter_add
from models.Model_SVM import svm_predict,svm_parameter_add
from models.Naive_Bayes import naive_bayes_predict,naive_bayes_parameter_add
from models.Model_Logist import logistic_train,logistic_parameter_add
from models.Model_XGBC import xgbc_train,xgbc_parameter_add
from models.Model_LGBM import lgbm_train,lgbm_parameter_add
from models.Mode_CBDT import cbdt_train,cbdt_parameter_add

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve

# Configure Read
from exfuction.main_fxn import yaml_read,yaml_change,yaml_write

st.set_page_config(page_title="Model",layout="wide")

model_list = ["请选择训练模型","DecisionTreeClassifier","RandomForest","KNN","SVM","Naive Bayes","LogisticRegression","XGBoost","LGBM","GBDT"]
CONFIG_LIST = {}

# FXN
def model_render(model_name="OverView"):
    if model_name not in model_list: return "the model error"
    st.title(model_name)

def model_parameter_add(model_name="OverView"):
    # st.write(model_name)
    if model_name == "DecisionTreeClassifier":
        parameter_dict = decisiontree_clf_parameter_add()
    elif model_name == "RandomForest":
        parameter_dict = randomforest_parameter_add()
    elif model_name == "KNN":
        parameter_dict = knn_parameter_add()
    elif model_name == "SVM":
        parameter_dict = svm_parameter_add()
    elif model_name == "Naive Bayes":
        parameter_dict = naive_bayes_parameter_add()
    elif model_name == "LogisticRegression":
        parameter_dict = logistic_parameter_add()
    elif model_name == "XGBoost":
        parameter_dict = xgbc_parameter_add()
    elif model_name == "LGBM":
        parameter_dict = lgbm_parameter_add()
    elif model_name == "GBDT":
        parameter_dict = cbdt_parameter_add()
    
    return parameter_dict
        


## Sidebar Parameters
with st.sidebar:
    selected_module = st.selectbox("选择你的模块",["监测","训练"])

## Main Part
## Model Choose
# model_render(selected_model)
# dataset_path = yaml_read()['dataset_path']
# st.info("Dataset {} is selected.".format(dataset_path))

# 判断对于目标数据集是否已有训练好的模型
if selected_module == '监测':
    with st.sidebar:
        dataset_path = "./dataset/" + st.selectbox('选择测试集',os.listdir('./dataset/'))
    st.info("Dataset {} is selected.".format(dataset_path))
    dataset_name = re.search(r"[/|\\](.*?)[/|\\](.*)\.csv",dataset_path).groups()[1]

    # 划分测试集
    data = pd.read_csv(dataset_path,index_col=False)
    X_test = data.iloc[:,data.columns != "Recidivism_Within_3years"]
    Y_test = data.iloc[:,data.columns == "Recidivism_Within_3years"]

    if os.path.exists('./userdata/{}'.format(dataset_name)):
        st.write('use existed model')
        with st.container():
            col1,col2,col3,col4 = st.columns(4)
            # 决策树模型评估
            with col1:
                st.subheader('DT')
                if os.path.exists('./userdata/{}/dt.pkl'.format(dataset_name)):
                    clf = decisiontree_clf_predict(dataset_path)
                    tr_y_pre = clf.predict(X_test) # 模型预测
                    tr_y_proba = clf.predict_proba(X_test)
                    tr_score = clf.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    tr_accuracy_score = accuracy_score(Y_test,tr_y_pre)
                    tr_preci_score = precision_score(Y_test,tr_y_pre)
                    tr_recall_score = recall_score(Y_test,tr_y_pre)
                    tr_f1_score = f1_score(Y_test,tr_y_pre)
                    tr_auc = roc_auc_score(Y_test,tr_y_proba[:,1])

                    st.write('评分:\n',tr_score)
                    st.write('精确率:\n',tr_accuracy_score)
                    st.write('命中率:\n',tr_preci_score)
                    st.write('召回率:\n',tr_recall_score)
                    st.write('f1指数:\n',tr_f1_score)
                    st.write('auc:\n',tr_auc)
                else:
                    st.write('need trainning')

           
            # 随机森林模型评估
            with col2:
                st.subheader('RF')
                if os.path.exists('./userdata/{}/rf.pkl'.format(dataset_name)):
                    rfc = randomforest_predict(dataset_path)
                    rf_y_pre = rfc.predict(X_test) # 模型预测
                    rf_y_proba = rfc.predict_proba(X_test)
                    rf_score = rfc.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    rf_accuracy_score = accuracy_score(Y_test,rf_y_pre)
                    rf_preci_score=precision_score(Y_test,rf_y_pre)
                    rf_recall_score=recall_score(Y_test,rf_y_pre)
                    rf_f1_score=f1_score(Y_test,rf_y_pre)
                    rf_auc=roc_auc_score(Y_test,rf_y_proba[:,1])

                    st.write('评分:\n',rf_score)
                    st.write('精确率:\n',rf_accuracy_score)
                    st.write('命中率:\n',rf_preci_score)
                    st.write('召回率:\n',rf_recall_score)
                    st.write('f1指数:\n',rf_f1_score)
                    st.write('auc:\n',rf_auc)
                else:
                    st.write('need trainning')

            # KNN模型评估
            with col3:
                st.subheader('KNN')
                if os.path.exists('./userdata/{}/knn.pkl'.format(dataset_name)):
                    knn = knn_train(dataset_path)
                    knn_y_pre = knn.predict(X_test) # 模型预测
                    knn_y_proba = knn.predict_proba(X_test)
                    knn_score = knn.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    knn_accuracy_score = accuracy_score(Y_test,knn_y_pre)
                    knn_preci_score=precision_score(Y_test,knn_y_pre)
                    knn_recall_score=recall_score(Y_test,knn_y_pre)
                    knn_f1_score=f1_score(Y_test,knn_y_pre)
                    knn_auc=roc_auc_score(Y_test,knn_y_proba[:,1])

                    st.write('评分:\n',knn_score)
                    st.write('精确率:\n',knn_accuracy_score)
                    st.write('命中率:\n',knn_preci_score)
                    st.write('召回率:\n',knn_recall_score)
                    st.write('f1指数:\n',knn_f1_score)
                    st.write('auc:\n',knn_auc)
                else:
                    st.write("need trainning")

            # SVM模型评估
            with col4:
                st.subheader('SVM')
                if os.path.exists('./userdata/{}/svm.pkl'.format(dataset_name)):
                    svm = svm_predict(dataset_path)
                    svm_y_pre = svm.predict(X_test) # 模型预测
                    svm_y_proba = svm.predict_proba(X_test)
                    svm_score = svm.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    svm_accuracy_score = accuracy_score(Y_test,svm_y_pre)
                    svm_preci_score=precision_score(Y_test,svm_y_pre)
                    svm_recall_score=recall_score(Y_test,svm_y_pre)
                    svm_f1_score=f1_score(Y_test,svm_y_pre)
                    svm_auc=roc_auc_score(Y_test,svm_y_proba[:,1])

                    st.write('评分:\n',svm_score)
                    st.write('精确率:\n',svm_accuracy_score)
                    st.write('命中率:\n',svm_preci_score)
                    st.write('召回率:\n',svm_recall_score)
                    st.write('f1指数:\n',svm_f1_score)
                    st.write('auc:\n',svm_auc)
                else:
                    st.write("need trainning")

        st.markdown('------')
        with st.container():
            col1,col2,col3,col4 = st.columns(4)
            
            # 朴素贝叶斯模型评估
            with col1:
                st.subheader('NB')
                if os.path.exists('./userdata/{}/nb.pkl'.format(dataset_name)):
                    bayes_model = naive_bayes_predict(dataset_path)
                    nb_y_pre = bayes_model.predict(X_test) # 模型预测
                    nb_y_proba = bayes_model.predict_proba(X_test)
                    nb_score = bayes_model.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    nb_accuracy_score = accuracy_score(Y_test,nb_y_pre)
                    nb_preci_score=precision_score(Y_test,nb_y_pre)
                    nb_recall_score=recall_score(Y_test,nb_y_pre)
                    nb_f1_score=f1_score(Y_test,nb_y_pre)
                    nb_auc=roc_auc_score(Y_test,nb_y_proba[:,1])

                    st.write('评分:\n',nb_score)
                    st.write('精确率:\n',nb_accuracy_score)
                    st.write('命中率:\n',nb_preci_score)
                    st.write('召回率:\n',nb_recall_score)
                    st.write('f1指数:\n',nb_f1_score)
                    st.write('auc:\n',nb_auc)
                else:
                    st.write("need trainning")

            # 逻辑回归模型评估
            with col2:
                st.subheader('Logist')
                if os.path.exists('./userdata/{}/lc.pkl'.format(dataset_name)):
                    lc = logistic_train(dataset_path)
                    lc_y_pre = lc.predict(X_test) # 模型预测
                    lc_y_proba = lc.predict_proba(X_test)
                    lc_score = lc.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    lc_accuracy_score = accuracy_score(Y_test,lc_y_pre)
                    lc_preci_score=precision_score(Y_test,lc_y_pre)
                    lc_recall_score=recall_score(Y_test,lc_y_pre)
                    lc_f1_score=f1_score(Y_test,lc_y_pre)
                    lc_auc=roc_auc_score(Y_test,lc_y_proba[:,1])

                    st.write('评分:\n',lc_score)
                    st.write('精确率:\n',lc_accuracy_score)
                    st.write('命中率:\n',lc_preci_score)
                    st.write('召回率:\n',lc_recall_score)
                    st.write('f1指数:\n',lc_f1_score)
                    st.write('auc:\n',lc_auc)
                else:
                    st.write("need trainning")

            # XGBoost分类器模型评估
            with col3:
                st.subheader('XGBC')
                if os.path.exists('./userdata/{}/xgbc.pkl'.format(dataset_name)):
                    xgbc = xgbc_train(dataset_path)
                    xgbc_y_pre = xgbc.predict(X_test) # 模型预测
                    xgbc_y_proba = xgbc.predict_proba(X_test)
                    xgbc_score = xgbc.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    xgbc_accuracy_score = accuracy_score(Y_test,xgbc_y_pre)
                    xgbc_preci_score=precision_score(Y_test,xgbc_y_pre)
                    xgbc_recall_score=recall_score(Y_test,xgbc_y_pre)
                    xgbc_f1_score=f1_score(Y_test,xgbc_y_pre)
                    xgbc_auc=roc_auc_score(Y_test,xgbc_y_proba[:,1])

                    st.write('评分:\n',xgbc_score)
                    st.write('精确率:\n',xgbc_accuracy_score)
                    st.write('命中率:\n',xgbc_preci_score)
                    st.write('召回率:\n',xgbc_recall_score)
                    st.write('f1指数:\n',xgbc_f1_score)
                    st.write('auc:\n',xgbc_auc)
                else:
                    st.write("need trainning")

            # LGBM模型评估    
            with col4:
                st.subheader('LGBM')
                if os.path.exists('./userdata/{}/lgbm.pkl'.format(dataset_name)):
                    lgbm = lgbm_train(dataset_path)
                    lgbm_y_pre = lgbm.predict(X_test) # 模型预测
                    lgbm_y_proba = lgbm.predict_proba(X_test)
                    lgbm_score = lgbm.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    lgbm_accuracy_score = accuracy_score(Y_test,lgbm_y_pre)
                    lgbm_preci_score=precision_score(Y_test,lgbm_y_pre)
                    lgbm_recall_score=recall_score(Y_test,lgbm_y_pre)
                    lgbm_f1_score=f1_score(Y_test,lgbm_y_pre)
                    lgbm_auc=roc_auc_score(Y_test,lgbm_y_proba[:,1])

                    st.write('评分:\n',lgbm_score)
                    st.write('精确率:\n',lgbm_accuracy_score)
                    st.write('命中率:\n',lgbm_preci_score)
                    st.write('召回率:\n',lgbm_recall_score)
                    st.write('f1指数:\n',lgbm_f1_score)
                    st.write('auc:\n',lgbm_auc)
                else:
                    st.write("need trainning")

        st.markdown('------')
        with st.container():
            col1,col2,col3,col4 = st.columns(4)
            # CBDT模型评估
            with col1:
                st.subheader('GBDT')
                if os.path.exists('./userdata/{}/cbdt.pkl'.format(dataset_name)):
                    cbdt = cbdt_train(dataset_path)
                    cbdt_y_pre = cbdt.predict(X_test) # 模型预测
                    cbdt_y_proba = cbdt.predict_proba(X_test)
                    cbdt_score = cbdt.score(X_test,Y_test.values.reshape(-1,1)) # 模型评分
                    cbdt_accuracy_score = accuracy_score(Y_test,cbdt_y_pre)
                    cbdt_preci_score=precision_score(Y_test,cbdt_y_pre)
                    cbdt_recall_score=recall_score(Y_test,cbdt_y_pre)
                    cbdt_f1_score=f1_score(Y_test,cbdt_y_pre)
                    cbdt_auc=roc_auc_score(Y_test,cbdt_y_proba[:,1])

                    st.write('评分:\n',cbdt_score)
                    st.write('精确率:\n',cbdt_accuracy_score)
                    st.write('命中率:\n',cbdt_preci_score)
                    st.write('召回率:\n',cbdt_recall_score)
                    st.write('f1指数:\n',cbdt_f1_score)
                    st.write('auc:\n',cbdt_auc)
                else:
                    st.write("need trainning")

        st.markdown('------')

        st.write("the model predict results are roughly as follows")
        data = pd.read_csv(dataset_path,index_col=False)
        if 'Unnamed: 0' in data.columns:
            data.drop(['Unnamed: 0'],inplace=True,axis=1)

        
        # 取前十条数据进行预测
        temp_df = data.head(10)
        if 'Unnamed: 0' in data.columns:
            data.drop(['Unnamed: 0'],inplace=True,axis=1)
        samples = temp_df.iloc[:,data.columns == "ID"]
        x_test = temp_df.iloc[:,data.columns != "Recidivism_Within_3years"]
        # DTClf
        res = pd.DataFrame(clf.predict(x_test))
        res.columns = ['DT']
        res_dt = res
        # RF
        res = pd.DataFrame(rfc.predict(x_test))
        res.columns = ['RF']
        res_rf = res
        # KNN
        res = pd.DataFrame(knn.predict(x_test))
        res.columns = ['KNN']
        res_knn = res
        # SVM
        res = pd.DataFrame(svm.predict(x_test))
        res.columns = ['SVM']
        res_svm = res
        # NB
        res = pd.DataFrame(bayes_model.predict(x_test))
        res.columns = ['NB']
        res_nb = res
        # Logist
        res = pd.DataFrame(lc.predict(x_test))
        res.columns = ['Logist']
        res_lc = res
        # XGBC
        res = pd.DataFrame(xgbc.predict(x_test))
        res.columns = ['XGBC']
        res_xgbc = res
        # LGBM
        res = pd.DataFrame(lgbm.predict(x_test))
        res.columns = ['LGBM']
        res_lgbm = res
        # CBDT
        res = pd.DataFrame(cbdt.predict(x_test))
        res.columns = ['GBDT']
        res_cbdt = res
        # 组合最终的结果并输出
        frames = [samples,res_dt,res_rf,res_knn,res_svm,res_nb,res_lc,res_xgbc,res_lgbm,res_cbdt]
        final_res = pd.concat(frames,axis=1)
        st.dataframe(final_res,use_container_width=True)


        ## 模型评估图表
        with st.container():
            col_1,col_2 = st.columns(2)
            with col_1:
                st.write("混淆矩阵")
                cm_dt,cm_rf,cm_knn,cm_nb,cm_svm,cm_logist,cm_xgbc,cm_lgbm,cm_cbdt = st.tabs(['DT','RF','KNN','NB','SVM','Logist','XGBC','LGBM','CBDT'])
                with cm_dt:
                    y_pred = clf.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()
                with cm_rf:
                    y_pred = rfc.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()
                with cm_knn:
                    y_pred = knn.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()
                with cm_nb:
                    y_pred = bayes_model.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()
                with cm_svm:
                    y_pred = svm.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()
                with cm_logist:
                    y_pred = lc.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()
                with cm_xgbc:
                    y_pred = xgbc.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()
                with cm_lgbm:
                    y_pred = lgbm.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()
                with cm_cbdt:
                    y_pred = cbdt.predict(X_test)
                    cm = confusion_matrix(Y_test,y_pred)
                    cm_display = ConfusionMatrixDisplay(cm).plot()
                    st.pyplot(cm_display.figure_)
                    plt.close()

            with col_2:
                st.write("ROC")
                roc_dt,roc_rf,roc_knn,roc_nb,roc_svm,roc_logist,roc_xgbc,roc_lgbm,roc_cbdt = st.tabs(['DT','RF','KNN','NB','SVM','Logist','XGBC','LGBM','CBDT'])
                with roc_dt:
                    tr_fpr,tr_tpr,tr_threasholds = roc_curve(Y_test,tr_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('decisiontree',tr_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(tr_fpr,tr_tpr)
                    st.pyplot(plt)
                    plt.close()

                with roc_rf:
                    rf_fpr,rf_tpr,rf_threasholds = roc_curve(Y_test,rf_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('RandomForest',rf_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(rf_fpr,rf_tpr)
                    st.pyplot(plt)
                    plt.close()

                with roc_knn:
                    knn_fpr,knn_tpr,knn_threasholds = roc_curve(Y_test,knn_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('KNN',knn_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(knn_fpr,knn_tpr)
                    st.pyplot(plt)
                    plt.close()

                with roc_nb:
                    nb_fpr,nb_tpr,nb_threasholds = roc_curve(Y_test,nb_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('Naive Bayes',nb_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(nb_fpr,nb_tpr)
                    st.pyplot(plt)
                    plt.close()
                
                with roc_svm:
                    svm_fpr,svm_tpr,svm_threasholds = roc_curve(Y_test,svm_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('SVM',svm_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(svm_fpr,svm_tpr)
                    st.pyplot(plt)
                    plt.close()

                with roc_logist:
                    lc_fpr,lc_tpr,lc_threasholds = roc_curve(Y_test,lc_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('Logist',lc_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(lc_fpr,lc_tpr)
                    st.pyplot(plt)
                    plt.close()

                with roc_xgbc:
                    xgbc_fpr,xgbc_tpr,xgbc_threasholds = roc_curve(Y_test,xgbc_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('XGBC',xgbc_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(xgbc_fpr,xgbc_tpr)
                    st.pyplot(plt)
                    plt.close()

                with roc_lgbm:
                    lgbm_fpr,lgbm_tpr,lgbm_threasholds = roc_curve(Y_test,lgbm_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('LGBM',lgbm_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(lgbm_fpr,lgbm_tpr)
                    st.pyplot(plt)
                    plt.close()

                with roc_cbdt:
                    cbdt_fpr,cbdt_tpr,cbdt_threasholds = roc_curve(Y_test,cbdt_y_proba[:,1]) # 计算ROC的值,lr_threasholds为阈值
                    plt.title("roc_curve of %s(AUC=%.4f)" %('CBDT',cbdt_auc))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(cbdt_fpr,cbdt_tpr)
                    st.pyplot(plt)
                    plt.close()

        st.markdown('------')
        # st.write("P-R")
        # pr_dt,pr_rf,pr_knn,pr_nb = st.tabs(['DT','RF','KNN','NB'])
        # with roc_dt:
        # with roc_rf:
        # with roc_knn:
        # with roc_nb:

    else:
        st.write('need trainning')
    



# selected_model参数说明用户选择了训练模块
elif selected_module == '训练':
    with st.sidebar:
        selected_model = st.selectbox("Select A Model or Algorithm",model_list)
        dataset_path = "./dataset/" + st.selectbox("请选择训练集",os.listdir('./dataset/'))
    
    if selected_model == "DecisionTreeClassifier":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            decisiontree_clf_predict(dataset_path,parameter_dict)

    elif selected_model == "RandomForest":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            randomforest_predict(dataset_path,parameter_dict)

    elif selected_model == "KNN":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            knn_train(dataset_path,parameter_dict)

    elif selected_model == "SVM":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            svm_predict(dataset_path,parameter_dict)

    elif selected_model == "Naive Bayes":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            naive_bayes_predict(dataset_path)

    elif selected_model == "LogisticRegression":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            logistic_train(dataset_path,parameter_dict)

    elif selected_model == "XGBoost":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            xgbc_train(dataset_path,parameter_dict)

    elif selected_model == "LGBM":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            lgbm_train(dataset_path,parameter_dict)
    
    elif selected_model == "GBDT":
        model_render(selected_model)
        parameter_dict = model_parameter_add(selected_model)
        if st.button('训练'):
            cbdt_train(dataset_path,parameter_dict)
