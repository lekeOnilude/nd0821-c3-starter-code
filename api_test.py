import requests
import json


def live_api_test(data):

    response = requests.post('https://census-udacity.herokuapp.com/get-salary/',
    data=json.dumps(data))

    return response.status_code, response.json()


if __name__=="__main__":
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

    print(live_api_test(request_boby))
