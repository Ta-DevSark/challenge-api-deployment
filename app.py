
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

class Property_variables:
    {
    "area": "float",
    "property-type": "float",
    "rooms-number": "float",
    "zip": "flot",
    "Locality" : "float",
    "Type of sale": "float",
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
      "message": "Required : data of a house in JSON format",
    }