import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data


# Synthetic data fixture
def get_sample_data():
    data = pd.DataFrame({
        "age": [39, 50],
        "workclass": ["State-gov", "Self-emp-not-inc"],
        "education": ["Bachelors", "Bachelors"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    return process_data(data, categorical_features=cat_features, label="salary", training=True)

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_random_forest():
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    # Your code here
    X, y, _, _ = get_sample_data()
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference_output_shape_matches_input(data=get_sample_data):
    """
    Test that inference returns predictions with the same number of rows as input.
    """
    # Your code here
    X, y, _, _ = get_sample_data()
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_output_range():
    """
    Test that precision, recall, and F1 score are between 0 and 1.
    """
    # Your code here
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 0])
    p, r, f = compute_model_metrics(y_true, y_pred)
    for metric in [p, r, f]:
        assert 0.0 <= metric <= 1.0