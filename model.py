import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('Zomato_df1.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())
X=df.drop('rate',axis=1)
y=df['rate']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=10)


#Preparing Extra Tree Regression
from sklearn.ensemble import  RandomForestRegressor
RF_Model=RandomForestRegressor(n_estimators = 120)
RF_Model.fit(X_train,y_train)


y_predict=RF_Model.predict(X_test)


import pickle
# # Saving model to disk
pickle.dump(RF_Model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_predict)