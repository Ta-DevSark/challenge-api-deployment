
    # A route at / that accept:
    #     GET request and return "alive" if the server is alive.
    # A route at /predict that accept:
    #     POST request that receives the data of a house in JSON format.
    #     GET request returning a string to explain what the POST expect (data and format).

# run the server with command line : uvicorn app:app --reload

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.externals import joblib

app = FastAPI()

@app.get("/")
def read_root():
    return {"alive"}

@app.post("/predict")
def predict(data):# not sure what to write here
    return {""}# same here

@app.get("/predict")
def predict():
    return ("Required : data of a house in JSON format")

class Property_variables :
    Optional = ""
    {
  "data": {
    "area": int,
    "property-type": "APARTMENT" | "HOUSE" | "OTHERS",
    "rooms-number": int,
    "zip-code": int,
    "land-area": Optional[int],
    "garden": Optional[bool],
    "garden-area": Optional[int],
    "equipped-kitchen": Optional[bool],
    "full-address": Optional[str],
    "swimming-pool": Optional[bool],
    "furnished": Optional[bool],
    "open-fire": Optional[bool],
    "terrace": Optional[bool],
    "terrace-area": Optional[int],
    "facades-number": Optional[int],
    "building-state": Optional[
      "NEW" | "GOOD" | "TO RENOVATE" | "JUST RENOVATED" | "TO REBUILD"
    ]
  }
}