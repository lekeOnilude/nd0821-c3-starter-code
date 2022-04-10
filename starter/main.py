# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field

import pandas as pd
import joblib

app = FastAPI()


class Value(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True


@app.get("/")
async def greetings():
    return{"Welcome"}

@app.post("/get-salary")
async def exercise_function(body: Value):
    sk_pipe = joblib.load('starter/model/lr_model.pkl')
    query = body.dict(by_alias=True)
    x = pd.DataFrame(query, index=[0])
    y_pred = sk_pipe.predict(x)
    return {"result": y_pred[0]}
