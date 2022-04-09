from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline, make_pipeline

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def slice_dataset_inference(categorical_feature, data, model, processed_X, processed_y):
    """ Run Inference and output results base on
    hold a particular categorical feature constants

    Inputs
    -----------
    categorical_features: list()
        List of categorical feature
    model : ???
        Trained machine learning model.
    data : pd.Dataframe
        Data used for prediction.
    
    processed_X: 

    Output
    --------
    result: dict()
        {
            categorical_features: {
                class: [precision, recall, fbeta]
            }
        }
    """
    result = {categorical_feature: {}}

    for cls in data[categorical_feature].unique():
        x_temp = processed_X[data[categorical_feature] == cls]
        y_temp = processed_y[data[categorical_feature] == cls]
    
        preds = inference(model, x_temp)
        metrics = compute_model_metrics(y_temp, preds)
        result[categorical_feature][cls] = metrics
        
    return result


def get_inference_pipeline():

    """
    Build inference Pipeline
    """

    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",]

    categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", categorical_preproc, cat_features),
        ],
        remainder="passthrough", 
    )

    liner_reg = LogisticRegression()

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("Linear_Regression", liner_reg),]
    )

    return sk_pipe
