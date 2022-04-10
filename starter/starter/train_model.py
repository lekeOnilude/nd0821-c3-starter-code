# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import joblib

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, slice_dataset_inference, get_inference_pipeline, compute_model_metrics
import pickle

# Add code to load in the data.
data_path = "starter/data/census_clean.csv"
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


# Test with slice data
result = {}
slice_cat_features = [
    "workclass",
]
for cat_feature in slice_cat_features:
    result.update(
        slice_dataset_inference(cat_feature, test,
        Trained_model, X_test, y_test)
        )

#print(result)

# Build pipe line
sk_pipe = get_inference_pipeline()

y_train = train['salary']
x_train = train.drop(["salary"], axis=1)

sk_pipe.fit(x_train, y_train)

y_test = test['salary']
x_test = test.drop(["salary"], axis=1)
y_pred = sk_pipe.predict(x_test)

lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)
y_pred = lb.transform(y_pred)
print(compute_model_metrics(y_test, y_pred))


# ###########################
# request_output = {
#     "age": 27,
#     "workclass": "Private",
#     "fnlgt": 160178,
#     "education": "Some-college",
#     "education-num": 10,
#     "marital-status": "Divorced",
#     "occupation": "Adm-clerical",
#     "relationship": "Not-in-family",
#     "race": "White",
#     "sex": "Female",
#     "capital-gain": 0,
#     "capital-loss": 0,
#     "hours-per-week": 38,
#     "native-country": "United-States"}

# x_t = pd.DataFrame(request_output, index=[0])
# print(x_t.shape)
# print(sk_pipe.predict(x_t))


joblib.dump(sk_pipe, 'starter/model/lr_model.pkl')