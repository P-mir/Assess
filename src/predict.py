import argparse
import requests
import pandas as pd

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(prog='predict', description='Predict the label given supplied observation(s)')
    my_parser.add_argument('--path', type=str, help='path to the csv file for which prediction is needed')
    args = my_parser.parse_args()
    path = args.path

    df = pd.read_csv(path, index_col=False, header=0)
    if 'label' in df.columns:
        X = df.drop('label', axis=1)
    else:
        X = df
    http_data = X.to_json(orient='split')

    host = '127.0.0.1'
    port = '1234'
    url = f'http://{host}:{port}/invocations'
    headers = {'Content-Type': 'application/json'}
    r = requests.post(url=url, headers=headers, data=http_data)
    r.text
    print(r.text)
