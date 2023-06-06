
    # A route at / that accept:
    #     GET request and return "alive" if the server is alive.
    # A route at /predict that accept:
    #     POST request that receives the data of a house in JSON format.
    #     GET request returning a string to explain what the POST expect (data and format).

# run the server with command line : uvicorn app:app --reload

from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib

lrm = joblib.load('model.pkl')

app = FastAPI()

class Textin(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message" : "alive"}

@app.post("/predict")
def predict(text: Textin):

    if(not(text.text)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid text message")
    prediction = lrm.predict([text.text])

    return {"prediction" : prediction}

@app.get("/predict")
def predict():
  return {
      "message": "Required : data of a house in JSON format",
      "data_format": {
            "area": "int",
            "property-type": "str",
            "rooms-number": "int",
            "zip-code": "int",
            "land-area": "optional[int]",
            "garden": "optional[bool]",
            "garden-area": "optional[int]",
            "equipped-kitchen": "optional[bool]",
            "full-address": "optional[str]",
            "swimming-pool": "optional[bool]",
            "furnished": "optional[bool]",
            "open-fire": "optional[bool]",
            "terrace": "optional[bool]",
            "terrace-area": "optional[int]",
            "facades-number": "optional[int]",
            "building-state": 'optional["NEW" | "GOOD" | "TO RENOVATE" | "JUST RENOVATED" | "TO REBUILD"]'
        }
    }