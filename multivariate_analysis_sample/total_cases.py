# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 06:56:00 2021

@author: kalya
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('owid-covid-data.csv')


from urllib.request import urlopen
  
# import json
import json
# store the URL in url as 
# parameter for urlopen
url = "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson"
  
# store the response of URL
response = urlopen(url)
  
# storing the JSON response 
# from url in data
data_json = json.loads(response.read())
  
# print the json response



required_countries = []
required_countries_hash = {}
for i in range(len(data_json['features'])):
    country_name = data_json['features'][i]['properties']['name']
    required_countries.append(data_json['features'][i]['properties']['name'])
    required_countries_hash[country_name.upper()] = True


final_countries_list = []
for item in df['location'].unique():
    if item.upper() in required_countries_hash:
        final_countries_list.append(item)


df_filtered = df[df['location'].isin(final_countries_list)]


df_filtered = df_filtered[['location', 'date', 'new_vaccinations', 'new_deaths', 'new_cases']]
df_filtered['new_vaccinations'].fillna(0, inplace=True)
df_filtered['new_deaths'].fillna(0, inplace=True)
df_filtered['new_cases'].fillna(0, inplace=True)


gp = df_filtered.groupby('date').sum()

test = df_filtered[ df_filtered['date'] == '2020-01-22']

plt.plot(gp['new_cases'])
plt.title('new_cases')
plt.show()

plt.plot(gp['new_vaccinations'])
plt.title('new_vaccinations')
plt.show()

plt.plot(gp['new_deaths'])
plt.title('new_deaths')
plt.show()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

cases_scaler = StandardScaler()

cases_scaler.fit(np.array(gp['new_cases']).reshape(-1, 1))


multi_dt = np.array(gp)

multi_dt = sc.fit_transform(multi_dt)

X_train = multi_dt[:500]
X_test = multi_dt[500:]

train_series_X = []
train_series_y = []

test_series_X = []
test_series_y = []

for i in range(4, 500):
    train_series_X.append(X_train[i-4: i, :])
    train_series_y.append(X_train[i, 2])

train_series_X = np.array(train_series_X)
train_series_y = np.array(train_series_y)

train_series_X = np.reshape(train_series_X, (train_series_X.shape[0], train_series_X.shape[1], 3))


for i in range(4, len(X_test)):
    test_series_X.append(X_test[i-4: i, :])
    test_series_y.append(X_test[i, 2])

test_series_X = np.array(test_series_X)
test_series_y = np.array(test_series_y)

test_series_X = np.reshape(test_series_X, (test_series_X.shape[0], test_series_X.shape[1], 3))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

regressor.add(LSTM(units= 100, return_sequences=True, input_shape = (train_series_X.shape[1], 3)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 100, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 100, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 100, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 100, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 100))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(train_series_X, train_series_y, epochs=100, batch_size=32)

predict_train = regressor.predict(test_series_X)

plt.plot(test_series_y, color='blue')
plt.plot(predict_train, color='green')
plt.show()

total_predict = []

for i in range(4, len(multi_dt)):
    total_predict.append(multi_dt[i-4:i, :])

total_predict = np.array(total_predict)
total_predict = np.reshape(total_predict, (671, 4, 3))

total_predict_plot = regressor.predict(total_predict)

test = cases_scaler.inverse_transform(total_predict_plot)

print(multi_dt[2])
plt.plot(gp['new_cases'], color='blue')
plt.plot(test, color='green')
plt.plot()