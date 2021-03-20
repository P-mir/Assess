.. role:: raw-html-m2r(raw)
   :format: html


ASSESS (Automated SignalS Evaluation Service System)
====================================================

 ASSESS allows you to assess the signal in your dataset, understand the important variables and deploy your model in seconds.


Command line interface
----------------------

Training a model
^^^^^^^^^^^^^^^^


.. code-block::

        python train.py 
            --label "label"# the name of the target variable.
            --path ../data/iris2.csv # path to the dataset csv file.
            --explainmodel # optional argument, to store plots explaining the model' decisions during the predictive process.

example:


.. code-block::

        python train.py --target "label" --path "../data/iris2.csv" --explainmodel


Monitoring the model via the MLflow UI
--------------------------------------

Place yourself in the ASSESS/src directory then:


.. code-block::

        mlflow ui

Generate predictions
^^^^^^^^^^^^^^^^^^^^

.. code-block::

        python predict.py --path ../data/iris2.csv


Deploy the model as a local REST API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from the ASSESS directory:

.. code-block::
        
		mlflow models serve -m "src\mlruns\0\\ :raw-html-m2r:`<model id>`\ \artifacts\sk_model" -p 1234 

HTTPs query from a Python script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block::

       # ... define X as the pandas dataframe containing the observations
       http_data = X.to_json(orient='split')
       host = '127.0.0.1'
       port = '1234'
       url = f'http://{host}:{port}/invocations'
       headers = {'Content-Type': 'application/json'}
       r = requests.post(url=url, headers=headers, data=http_data)
       return r.text


