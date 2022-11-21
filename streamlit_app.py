# ! pip install streamlit
# ! pip install pandas_profiling streamlit_pandas_profiling
# ! pip install pycaret
# example usage:
# > streamlit run app.py --server.maxUploadSize 1000

import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# import profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


with st.sidebar:
    st.image('https://robohash.org/corvidautoml')
    st.title('VautoML')
    choice = st.radio('Navigation', ['Upload', 'Profile Data', 'Create Model'])
    st.info('This application allows you to create ML models automagically. Built with Python \
             using Streamlit, Pandas, and PyCaret.')

data_exists = False
if os.path.exists('source_data.csv'):
    df = pd.read_csv('source_data.csv', index_col=None)
    data_exists = True

if choice == 'Upload':
    st.title('Upload your Data for modeling!')

    placeholder = st.empty()

    if data_exists:
        with placeholder.container():
            st.dataframe(df)
            st.info('Note: Data already exists in the working directory. If you would like to upload new data, use the uploader below.')

    file = st.file_uploader('Upload your dataset here')
    st.info('The input file is expected to be a comma-delimited text file')

    if file:
        placeholder.empty()
        df = pd.read_csv(file, index_col=None)
        df.to_csv('source_data.csv', index=None)
        st.dataframe(df)

elif choice == 'Profile Data':
    st.title('Automated exploratory data analysis')
    st.info('using the pandas_profiling library')
    profile_report = df.profile_report()
    st_profile_report(profile_report)

elif choice == 'Create Model':
    st.title('Create ML Model')

    type = st.selectbox('Select model type', ['', 'classification', 'regression', 'clustering'])

    # classification
    if type == 'classification':
        import VautoClassification

        # select input features and target
        options = list(df.columns)
        cols = st.multiselect("Select features", options, default=options)
        target = st.selectbox('Select the target variable', df.columns, index=len(df.columns)-1)
        if target not in cols:
            cols = cols + [target]
        st.dataframe(df[cols])

        if st.button('Train model'):
            VautoClassification.make_model(df[cols], target=target)

    
    # regression
    if type == 'regression':
        import VautoRegression
        
        # select input features and target
        options = list(df.columns)
        cols = st.multiselect("Select features", options, default=options)
        target = st.selectbox('Select the target variable', df.columns, index=len(df.columns)-1)
        if target not in cols:
            cols = cols + [target]
        st.dataframe(df[cols])

        if st.button('Train model'):
            VautoRegression.make_model(df[cols], target=target)

    # clustering
    elif type == 'clustering':
        import VautoClustering
        options = list(df.columns)
        cols = st.multiselect("Select features", options, default=options)
        n_clusters = st.slider('Specify number of clusters', min_value=1, max_value=12, value=4)
        if st.button('Train model'):
            VautoClustering.make_model(df[cols], num_clusters=n_clusters)

# import pickle
# import pandas as pd
# from pycaret.classification import load_model

# model = load_model('trained_model.pkl')
# model
