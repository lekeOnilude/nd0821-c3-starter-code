import pickle
import pytest

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference
import pandas as pd

from sklearn.model_selection import train_test_split



@pytest.fixture
def data():
    """
    Load the dataset so it can be used for testing
    """
    data_path = "data/census_clean.csv"
    df = pd.read_csv(data_path)
    return df

def test_process_data(data):
    """
    Test the process data fuction
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        ]

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

    X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

    assert X_train.shape[0] == train.shape[0]
    assert y_train.shape[0] == train.shape[0]

    assert X_test.shape[0] == test.shape[0]
    assert y_test.shape[0] == test.shape[0] 

def test_inference(data):
    """
    Test to make sure the model can perform inference
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        ]

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

    X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

    with open("model/log_reg_model.sav", 'rb') as file:
        model = pickle.load(file)

    pred = inference(model, X_test)

    assert pred.shape == y_test.shape

def test_compute_model_metrics(data):
    """
    Test the compute_model_metrics funtion
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        ]

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

    X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

    with open("model/log_reg_model.sav", 'rb') as file:
        model = pickle.load(file)
    
    pred = inference(model, X_test)
    metrics = compute_model_metrics(y_test, pred)

    assert len(metrics) == 3
    assert pd.Series(metrics).between(0, 1).all()



