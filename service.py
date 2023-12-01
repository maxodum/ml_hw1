from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from io import BytesIO
import pandas as pd
import numpy as np
import random
import re
import pickle

random.seed(42)
np.random.seed(42)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


def floater(my_str):
    if pd.isna(my_str):
        return np.nan
    result = re.sub(r'[^0-9.]', '', my_str)
    if result == '':
        return np.nan
    return float(result)


def preprocessing(item: Item):
    df = pd.DataFrame(dict(item), index=[0])
    df['mileage'] = df['mileage'].apply(floater)
    df['engine'] = df['engine'].apply(floater)
    df['max_power'] = df['max_power'].apply(floater)
    df_train = pd.read_pickle('median_df.pkl')
    df['mileage'] = df['mileage'].fillna(df_train['mileage'].median())
    df['engine'] = df['engine'].fillna(df_train['engine'].median())
    df['max_power'] = df['max_power'].fillna(df_train['max_power'].median())
    df['seats'] = df['seats'].fillna(df_train['seats'].median())
    df['bhp/engine'] = df['max_power'] / df['engine']
    df['year^2'] = df['year'] ** 2
    df = df.drop(['year', 'torque'], axis=1)
    X = df.drop(['selling_price', 'name'], axis=1)
    labels = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    categorical_data = X[labels]
    ohe = pickle.load(open('ohe.pkl', 'rb'))
    feature_arr = ohe.transform(categorical_data).toarray()
    ohe_labels = ohe.get_feature_names_out(labels)
    features = pd.DataFrame(feature_arr, columns=ohe_labels)
    X = X.drop(['fuel', 'seller_type', 'transmission', 'owner', 'seats'], axis=1).join(features)
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    X_num = pd.DataFrame(scaler.transform(X[['year^2', 'km_driven', 'engine', 'max_power', 'mileage', 'bhp/engine']]), columns=X[['year^2', 'km_driven', 'engine', 'max_power', 'mileage', 'bhp/engine']].columns)
    X = X.drop(['year^2', 'km_driven', 'engine', 'max_power', 'mileage', 'bhp/engine'], axis=1).join(X_num)
    return X


def predict(df):
    model = pickle.load(open('model.pkl', 'rb'))
    return model.predict(df)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = preprocessing(item)
    return predict(df)


@app.post("/predict_items")
def predict_items(file: UploadFile):
    content = file.file.read()
    buffer = BytesIO(content)
    items = pd.read_csv(buffer)
    predictions = []
    for i in items.index:
        df = preprocessing(items.iloc[i])
        predictions.append(predict(df))
    items['predictions'] = predictions
    items.to_csv('predictions.csv', index=False)
    response = FileResponse(path='predictions.csv',media_type='text/csv')
    return response