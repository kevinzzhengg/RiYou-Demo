# -- coding : utf-8 --
import streamlit as st
import pandas as pd
from exfuction.main_fxn import yaml_read,yaml_write,yaml_change
from exfuction.main_fxn import db_connection
import os
import re

def main():
    st.set_page_config(page_title="HomePage"
                    , page_icon="bar_chart"
                    , layout="wide"
                    , initial_sidebar_state="expanded"
                )
    
    st.title("Community Corrector Recidivism Predicted System")
    st.caption("use different AI models to predict the possibility of the community corrector to recidivism.")
    
    # 展示平台资源清单
    with st.container():
        df = pd.DataFrame(columns=['DATASET','DT','RF','KNN','SVM','NB','Logist','XGBC','LGBM','CBDT'])
        for each in os.listdir('./dataset/'):
            dataset_name = re.search(r"(.*?)\.csv",each).groups()[0]
            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'dt')):
                dt_model_file = '✅'
            else:
                dt_model_file = ''
            
            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'rf')):
                rf_model_file = '✅'
            else:
                rf_model_file = ''

            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'knn')):
                knn_model_file = '✅'
            else:
                knn_model_file = ''

            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'svm')):
                svm_model_file = '✅'
            else:
                svm_model_file = ''

            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'nb')):
                nb_model_file = '✅'
            else:
                nb_model_file = ''

            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'lc')):
                lc_model_file = '✅'
            else:
                lc_model_file = ''

            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'xgbc')):
                xgbc_model_file = '✅'
            else:
                xgbc_model_file = ''

            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'lgbm')):
                lgbm_model_file = '✅'
            else:
                lgbm_model_file = ''

            if os.path.exists('./userdata/{0}/{1}.pkl'.format(dataset_name,'cbdt')):
                cbdt_model_file = '✅'
            else:
                cbdt_model_file = ''
            
            df_temp = pd.DataFrame({'DATASET':each,'DT':dt_model_file,'RF':rf_model_file,'KNN':knn_model_file,'SVM':svm_model_file,'NB':nb_model_file,'Logist':lc_model_file,'XGBC':xgbc_model_file,'LGBM':lgbm_model_file,'CBDT':cbdt_model_file},index=[''])
            df = pd.concat([df,df_temp],axis=0)
        st.subheader('平台资源清单:')
        st.dataframe(df.iloc[:,0:10],use_container_width=True)

if __name__ == '__main__':
    main()