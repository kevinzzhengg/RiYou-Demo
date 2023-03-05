import os 
import streamlit as st

# EDA Pkgs
import pandas as pd

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

def main():
    """ Common ML Dataset Explorer"""

    st.set_page_config(page_title="RiYou", layout="wide")

    st.title("Common ML Dataset Explorer")
    st.subheader("Simple Data Science Explorer with Streamlit")

    html_temp = """
    <div style="background-color:tomato;"><p style="color:white;font-size:50px">Stream is Awesome</p></div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    def file_selector(folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Select A file",filenames)
        return os.path.join(folder_path,selected_filename)

    filename = file_selector(".\dataset")
    st.info("You Selected {}".format(filename))

    # Read Data 
    df = pd.read_csv(filename)

    # Show Datasets
    if st.checkbox("Show Datasete"):
        number = st.number_input("Number of Rows to View")
        st.dataframe(df.head(int(number)))
    
    # Show Columns
    if st.button("Columns Names"):
        st.write(df.columns)

    # Show Shape
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Show Dimension By",("Rows","Columns"))
        if data_dim == 'Rows':
            st.text("Number of Rows")
            st.write(df.shape[0])
        if data_dim == 'Columns':
            st.text("Number of Columns")
            st.write(df.shape[1])
        else:
            st.write(df.shape)

    # Select Columns
    if st.checkbox("Select Columns To Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select",all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # Show Values
    if st.button("Value Counts"):
        st.text("Value Counts By Target/Class")
        st.write(df.iloc[:,-1].value_counts())

    # Show Datatypes
    if st.button("Data Types"):
        st.write(df.dtypes)

    # Show Sunmmary
    if st.checkbox("Summary"):
        st.write(df.describe().T)

    ## PLot and Visualization

    st.subheader("Data Visualization")
    
    # Correlation
    # Seaborn Plot
    if st.checkbox("Correlation Plot[Seaborn]"):
        st.write(sns.heatmap(df.corr(),annot=True))
        st.pyplot()

    # Count Plot
    if st.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target")
        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox("Primary Column to GroupBy",all_columns_names)
        selected_columns_names = st.multiselect("Select Columns",all_columns_names)
        if st.button("Plot"):
            st.text("Generate Plot")
            if selected_columns_names:
                vc_plot = df.groupby(primary_col)[selected_columns_names].count()
            else:
                vc_plot = df.iloc[:,-1].value_counts()
            st.write(vc_plot.plot(kind="bar"))
            st.pyplot()

    # Pie Chart
    if st.checkbox("Pie Plot"):
        all_columns_names = df.columns.tolist()
        if st.button("Generate Pie Plot"):
            st.success("Generatin A Pie Plot")
            st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    # Customizable Plot

    st.subheader("Customizable Plot")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of PLot",["area","bar","line","hist","box","kde"])
    selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generatin Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

        # Plot By Streamlit
        if type_of_plot == 'area':
            cust_data =df[selected_columns_names]
            st.area_chart(cust_data)

        elif type_of_plot == 'bar':
            cust_data =df[selected_columns_names]
            st.bar_chart(cust_data)
        
        elif type_of_plot == 'line':
            cust_data =df[selected_columns_names]
            st.line_chart(cust_data)
        
        # Custom Plot 
        elif type_of_plot:
            cust_plot =df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()


if __name__ == '__main__':
    main()