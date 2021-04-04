# ASSESS (Automated SignalS Evaluation Service System)

 ASSESS allows you to assess the signal in your dataset, understand the important variables and deploy your model in seconds.

![backend_schema.PNG]()



## Web application

```python
streamlit run src/app.py
```

## Sphinx documentation


## Command line interface


### Training a model 

```python
python train.py 
	--label "label"# the name of the target variable.
	--path ../data/iris2.csv # path to the dataset csv file.
	--explainmodel # optional argument, to store plots explaining the model' decisions during the predictive process.
```
example:

```
python src/train.py --target "label" --path "../data/iris2.csv" --explainmodel
```

### Monitoring the model via the MLflow UI

Place yourself in the ASSESS/src directory then:

```
mlflow ui
```

### Generate predictions

```
python src/predict.py --path ../data/iris2.csv
```

### Deploy the model as a local REST API

from the ASSESS directory:

```
mlflow models serve -m "src\mlruns\0\<model id>\artifacts\sk_model" -p 1234 
```
	
### HTTPs query from a Python script

```python
# ... define X as the pandas dataframe containing the observations
http_data = X.to_json(orient='split')
host = '127.0.0.1'
port = '1234'
url = f'http://{host}:{port}/invocations'
headers = {'Content-Type': 'application/json'}
r = requests.post(url=url, headers=headers, data=http_data)
return r.text
```
