# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:48:39 2022

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

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

cases_scaler = StandardScaler()

cases_scaler.fit(np.array(gp['new_cases']).reshape(-1, 1))


multi_dt = np.array(gp)

multi_dt = sc.fit_transform(multi_dt)

X_train = multi_dt[:738]
X_test = multi_dt[600:]

train_series_X = []
train_series_y = []

test_series_X = []
test_series_y = []

for i in range(30, 738):
    train_series_X.append(X_train[i-30: i, :])
    train_series_y.append(X_train[i, 2])

train_series_X = np.array(train_series_X)
train_series_y = np.array(train_series_y)

train_series_X = np.reshape(train_series_X, (train_series_X.shape[0], train_series_X.shape[1], 3))


for i in range(30, len(X_test)):
    test_series_X.append(X_test[i-30: i, :])
    test_series_y.append(X_test[i, 2])

test_series_X = np.array(test_series_X)
test_series_y = np.array(test_series_y)

test_series_X = np.reshape(test_series_X, (test_series_X.shape[0], test_series_X.shape[1], 3))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

regressor.add(LSTM(units= 1250, return_sequences=True, input_shape = (train_series_X.shape[1], 3)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=1250, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 1250, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 1250, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 1250, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 1250, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 1250, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 1250))
regressor.add(Dropout(0.2))


regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(train_series_X, train_series_y, epochs=100, batch_size=100)


new_vaccination_data = np.array(gp['new_vaccinations']).reshape(-1,1)

vaccination_scaler = StandardScaler()

vaccination_scaler.fit(new_vaccination_data)

train_data = vaccination_scaler.transform(new_vaccination_data[:738])
test_data = vaccination_scaler.transform(new_vaccination_data[600:])

train_X, train_y = [], []

for i in range(30, 738):
    train_X.append(train_data[i-30: i])
    train_y.append(train_data[i])
    
train_X, train_y = np.array(train_X), np.array(train_y)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))


train_test_X = []
train_test_y = []

for i in range(30, len(test_data)):
    train_test_X.append(test_data[i-30: i])
    train_test_y.append(test_data[i])
    
train_test_X = np.array(train_test_X)
train_test_y = np.array(train_test_y)

train_test_X = np.reshape(train_test_X, (train_test_X.shape[0], train_test_X.shape[1], 1))


vaccination_regressor = Sequential()

vaccination_regressor.add(LSTM(units=380, return_sequences=True, input_shape = (train_X.shape[1], 1)))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(LSTM(units= 384, return_sequences=True))
vaccination_regressor.add(Dropout(0.2))
vaccination_regressor.add(LSTM(units= 384, return_sequences=True))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(LSTM(units= 384, return_sequences=True))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(LSTM(units= 384, return_sequences=True))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(LSTM(units= 384))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(Dense(units = 1))


vaccination_regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

vaccination_regressor.fit(train_X, train_y, epochs = 100, batch_size = 100)


new_X_data = np.array(gp['new_deaths']).reshape(-1,1)

new_X_scaler = StandardScaler()

new_X_scaler.fit(new_X_data)


train_data = new_X_scaler.transform(new_X_data[:738])
test_data = new_X_scaler.transform(new_X_data[600:])

train_X, train_y = [], []

for i in range(30, 738):
    train_X.append(train_data[i-30: i])
    train_y.append(train_data[i])
    
train_X, train_y = np.array(train_X), np.array(train_y)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))


train_test_X = []
train_test_y = []

for i in range(30, len(test_data)):
    train_test_X.append(test_data[i-30: i])
    train_test_y.append(test_data[i])
    
train_test_X = np.array(train_test_X)
train_test_y = np.array(train_test_y)

train_test_X = np.reshape(train_test_X, (train_test_X.shape[0], train_test_X.shape[1], 1))


new_X_regressor = Sequential()

new_X_regressor.add(LSTM(units=100, return_sequences=True, input_shape = (train_X.shape[1], 1)))
new_X_regressor.add(Dropout(0.2))

new_X_regressor.add(LSTM(units= 100, return_sequences=True))
new_X_regressor.add(Dropout(0.2))
new_X_regressor.add(LSTM(units= 100, return_sequences=True))
new_X_regressor.add(Dropout(0.2))

new_X_regressor.add(LSTM(units= 100, return_sequences=True))
new_X_regressor.add(Dropout(0.2))

new_X_regressor.add(LSTM(units= 100, return_sequences=True))
new_X_regressor.add(Dropout(0.2))

new_X_regressor.add(LSTM(units= 100))
new_X_regressor.add(Dropout(0.2))

new_X_regressor.add(Dense(units = 1))


new_X_regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

new_X_regressor.fit(train_X, train_y, epochs = 100, batch_size = 100)




n_multi = multi_dt

vac_data = n_multi[:, 0]
nw_data = n_multi[:, 1]

for i in range(300):
    
    vac_data = n_multi[:, 0]
    nw_data = n_multi[:, 1]
    
    start_multi = np.reshape(n_multi[len(n_multi)-30:], (1, 30, 3))
    start_vacci = np.reshape(vac_data[len(vac_data)-30:], (1, 30, 1))
    start_nw = np.reshape(nw_data[len(nw_data)-30:], (1,30,1))
    
    start_multi_predict = regressor.predict(start_multi)[0][0]
    start_vacci_predict = vaccination_regressor.predict(start_vacci)[0][0]
    start_nw_predict = new_X_regressor.predict(start_nw)[0][0]
        
    n_multi = np.append(n_multi, [[start_vacci_predict, start_nw_predict, start_multi_predict]], axis=0)

#check_with_latest = np.array(gp_2)

n_multi = sc.inverse_transform(n_multi)

#plt.plot(gp['new_vaccinations'], color='blue')
plt.plot(n_multi[738:, 0], color='blue')
plt.title('new_vaccinations')
plt.show()

#plt.plot(gp['new_deaths'], color='blue')
plt.plot(n_multi[738:, 1], color='blue')
plt.title('new_deaths')
plt.show()

#plt.plot(gp_2['new_cases'], color='blue')
plt.plot(n_multi[738:, 2], color='blue')
plt.title('new_cases')
plt.show()

import joblib

joblib.dump(regressor, 'cases_model.pkl')

joblib.dump(vaccination_regressor, 'vaccination_regressor.pkl')

joblib.dump(new_X_regressor, 'new_X_regressor.pkl')

regressor = joblib.load('cases_model.pkl')
vaccination_regressor = joblib.load('vaccination_regressor.pkl')
new_X_regressor = joblib.load('new_X_regressor.pkl')