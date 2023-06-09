# run the server with command line : uvicorn app:app --reload

from typing import Literal, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
from preprocessing.cleaning_data import preprocess

lrm = joblib.load('model.pkl')

app = FastAPI()

@app.get("/")
def read_root():
    return {"message" : "alive"}

# Index(['Number_of_rooms', 'Living_area', 'Zip', 'Primary_energy_consumption',
#        'Construction_year'],
#       dtype='object')

@app.post("/predict")
def predict(
   Number_of_rooms: float = 3,
   Living_area: float = 154, 
   Zip: float = 1000, 
   Primary_energy_consumption: float = 122,
   Construction_year: float = 1991
):
    data = np.array([[Number_of_rooms, Living_area, Zip, Primary_energy_consumption, Construction_year]])
    data_prediction = lrm.predict(data)

    converted_prediction = float(data_prediction[0])

    # house_details = {
    #     "Locality" : Locality,
    #     "Type_of_sale": Type_of_sale,
    #     "Type_of_property": Type_of_property,
    #     "Subtype_of_property": Subtype_of_property,
    #     "Number_of_facades": Number_of_facade,
    #     "Number_of_rooms": Number_of_rooms,
    #     "Fully_equipped_kitchen": Fully_equipped_kitchen,
    #     "Open_fire": Open_fire,
    #     "Surface_of_the_land": Surface_of_the_land,
    # }  

    # df = pd.DataFrame(data)
    # cleaned_data = preprocess(df)
    # prediction = lrm.predict(cleaned_data)

    return {"converted_prediction" : converted_prediction}

Property_variables = {
   "Number_of_rooms": "float",
   "Living_area": "float", 
   "Zip": "float", 
   "Primary_energy_consumption": "float",
   "Construction_year": "float",
    }

@app.get("/predict")
def predict():
  return {
      "message": Property_variables,
    }