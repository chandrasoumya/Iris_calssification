import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("iris.csv")
print(df.head(3))

df.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']
print(df.head(3))

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
print(df.head(3))

x = df[['sepal_length','sepal_width','petal_width','petal_length']]
y = df['species']

x = np.array(x)
print(df.isna().sum(axis=0))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

log = LogisticRegression()
log.fit(x_train,y_train)

with open("iris.pkl","wb") as f:
    pickle.dump(log,f)