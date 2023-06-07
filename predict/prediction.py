# This file will contain all the code used to predict a new house's price.

# def predict() will take your preprocessed data as an input and return a price as output.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
from sklearn.metrics import r2_score
from typing import List
import joblib

house_df = pd.read_csv("final_house.csv")

X = house_df

y = house_df['Price']

X =  X.drop(['Heating type','Surroundings type','Unnamed: 0', 'Price' , 'Locality','id', 'Primary energy consumption','Type of property',
                                         'Subtype of property', 'Energy class', 'Province'], axis=1)

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lrm = LinearRegression() 

lrm.fit(X_train,y_train)

y_prediction = lrm.predict(X_test) 

score = r2_score(y_test, y_prediction)

print(score)
joblib.dump(lrm, 'model.pkl')