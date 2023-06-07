
    # A route at / that accept:
    #     GET request and return "alive" if the server is alive.
    # A route at /predict that accept:
    #     POST request that receives the data of a house in JSON format.
    #     GET request returning a string to explain what the POST expect (data and format).

# run the server with command line : uvicorn app:app --reload

from typing import Literal, Union
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib
from preprocessing.cleaning_data import preprocess

lrm = joblib.load('model.pkl')

app = FastAPI()

@app.get("/")
def read_root():
    return {"message" : "alive"}

@app.post("/predict")
def predict(
    Locality: str = "bruxelles",
    Type_of_sale: Literal["sale", "rent"] = "sale", 
    Type_of_property: Literal["house", "apartment"] = "house",
    Subtype_of_property: str = "house",
    Number_of_facade: int = 4,
    Number_of_rooms: int = 4,
    Fully_equipped_kitchen: bool = True,
    Open_fire: bool = True,
    Surface_of_the_land: float = 124.4
):
    house_details = {
        "Locality" : Locality,
        "Type_of_sale": Type_of_sale,
        "Type_of_property": Type_of_property,
        "Subtype_of_property": Subtype_of_property,
        "Number_of_facades": Number_of_facade,
        "Number_of_rooms": Number_of_rooms,
        "Fully_equipped_kitchen": Fully_equipped_kitchen,
        "Open_fire": Open_fire,
        "Surface_of_the_land": Surface_of_the_land,
    }
    df = pd.DataFrame(house_details)
    cleaned_data = preprocess(df)
    prediction = lrm.predict(cleaned_data)

    return {"prediction" : prediction}

Property_variables = {
    "Locality" : "float",
    "Type_of_sale": "float",
    "Type_of_property": "float",
    "Subtype_of_property": "float",
    "Number_of_facades": "float",
    "Number_of_rooms": "float",
    "Fully_equipped_kitchen": "float",
    "Open_fire": "float",
    "Surface_of_the_land": "float",
    }

@app.get("/predict")
def predict():
  return {
      "message": Property_variables,
    }