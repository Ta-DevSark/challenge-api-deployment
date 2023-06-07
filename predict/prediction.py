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
from sklearn.feature_selection import SelectPercentile, mutual_info_regression

df = pd.read_csv("final_house.csv")

df = df[['Price', 'Number_of_rooms', 'Living area', 'Zip', 
          "Primary energy consumption", 'Construction year']]

df.isna().sum()

df.columns = df.columns.str.replace(' ', '_')

df = df.fillna(0)

X = df.drop('Price', axis = 1)

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lrm = LinearRegression() 

lrm.fit(X_train,y_train)

y_prediction = lrm.predict(X_test) 

score = r2_score(y_test, y_prediction)

print(score)

joblib.dump(lrm, 'model.pkl')

print(X_train.columns)

d = np.array([[4, 150, 1000, 100, 1991]])

print(lrm.predict(d))