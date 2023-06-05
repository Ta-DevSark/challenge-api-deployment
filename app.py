
    # A route at / that accept:
    #     GET request and return "alive" if the server is alive.
    # A route at /predict that accept:
    #     POST request that receives the data of a house in JSON format.
    #     GET request returning a string to explain what the POST expect (data and format).

# run the server with command line : uvicorn app:app --reload


from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

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