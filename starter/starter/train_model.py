# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pickle

# Add code to load in the data.
data_path = "starter/data/census.csv"
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

# Train and save a model.
Trained_model = train_model(X_train, y_train)

filename = 'starter/model/log_reg_model.sav'
pickle.dump(Trained_model, open(filename, 'wb'))
