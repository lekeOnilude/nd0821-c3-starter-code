from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200

def test_api_locally_post_root():
    request_boby = {
        "age": 27,
        "workclass": "Private",
        "fnlgt": 160178,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 38,
        "native-country": "United-States"}

    r = client.post("/get-salary", json=request_boby)
    assert r.json() == {"result": "<=50K"}

def test_api_locally_post_50K():
    request_boby = {
        "age": 58,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 93664,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"}

    r = client.post("/get-salary", json=request_boby)
    assert r.json() == {"result": ">50K"}