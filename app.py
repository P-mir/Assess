# streamlit run app.py to run in local
import os
import base64
import json
import pickle
import uuid
import re
from subprocess import Popen
import shutil

import streamlit as st
import requests
import pandas as pd
import io
import src.train
from src.train import fit, predict


host = "sumit-up-api.herokuapp.com"
# host = '127.0.0.1:8000'

st.beta_set_page_config(
    initial_sidebar_state="expanded",
)

# st.image("logo3.png")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)

def main(): 

    st.info("ASSESS allows you to evaluate the signal of a dataset in one click")
    
    # remove eventual traces from previous runs
    path = "outputs/plots/mlflow_artifacts/shap"
    if os.path.exists(path):
        shutil.rmtree(path)

    data = pd.read_csv('data\breast_cancer.csv')
    
    uploaded_file = st.file_uploader("Play with the demo or upload your own data (CSV format)", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

    if data is not None: 
       
        st.dataframe(data,height=200)
        features = st.sidebar.multiselect('Select features',options = list(data.columns), default=list(data.columns))
        target = st.sidebar.selectbox('Target variable:',(data.columns))
        task = st.sidebar.radio("Task",
        ('classification', 'regression'))
        explanations = st.sidebar.checkbox("Explain model's decision (might take longer)")
        if st.button('Train'):
            model, preprocessing, x = fit(data,features,target, task, explainmodel=explanations) 
            preds = predict(model, preprocessing, x) 
            data['Predictions'] =  preds      
            st.write('Model trained, launching the MLFlow tracking server...')
            Popen("mlflow ui")
            st.write("[Click here to browse your results](http://localhost:5000)")


def get_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


if __name__ == '__main__':
    main()




