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

df_2 = pd.read_csv('owid-covid-data_new.csv')


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


df_filtered_2 = df_2[df_2['location'].isin(final_countries_list)]


df_filtered_2 = df_filtered_2[['location', 'date', 'new_vaccinations', 'new_deaths', 'new_cases']]
df_filtered_2['new_vaccinations'].fillna(0, inplace=True)
df_filtered_2['new_deaths'].fillna(0, inplace=True)
df_filtered_2['new_cases'].fillna(0, inplace=True)

gp = df_filtered.groupby('date').sum()
gp_2 = df_filtered_2.groupby('date').sum()


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

new_test = np.array(gp_2)
new_test = new_test[676:]
new_test = sc.transform(new_test)

new_test_data = []
new_test_data_y = []
for i in range(4, len(new_test)):
    new_test_data.append(new_test[i-4:i, :])
    new_test_data_y.append(new_test[i, 2])

new_test_data = np.array(new_test_data)
new_test_data = np.reshape(new_test_data, (37, 4, 3))

predict_new = regressor.predict(new_test_data)

plt.plot(predict_new, color='blue')
plt.plot(new_test_data_y, color='green')
plt.show()

rnd = [[0.787749,	-0.571368,	-0.0519976],
[1.19736,	-0.229999	,0.22633],
[1.51699,	0.060427,	0.372232],
[1.6208,	0.0255875,	0.713595]
]
rnd = np.reshape(rnd, (1, 4, 3))

print(regressor.predict(rnd))

print(new_test_data)



# Need to maintain an original table, and for every 4 previous series need to take from this table and np.reshape and append again,




#new_vaccinations predictor single

new_vaccination_data = np.array(gp['new_vaccinations']).reshape(-1,1)

vaccination_scaler = StandardScaler()

vaccination_scaler.fit(new_vaccination_data)

train_data = vaccination_scaler.transform(new_vaccination_data[:500])
test_data = vaccination_scaler.transform(new_vaccination_data[500:])

train_X, train_y = [], []

for i in range(4, 500):
    train_X.append(train_data[i-4: i])
    train_y.append(train_data[i])
    
train_X, train_y = np.array(train_X), np.array(train_y)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    

vaccination_regressor = Sequential()

vaccination_regressor(LSTM(units=100, return_sequences=True, input_shape = (train_X.shape[1], 1)))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(LSTM(units= 100, return_sequences=True))
vaccination_regressor.add(Dropout(0.2))
vaccination_regressor.add(LSTM(units= 100, return_sequences=True))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(LSTM(units= 100, return_sequences=True))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(LSTM(units= 100, return_sequences=True))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(LSTM(units= 100))
vaccination_regressor.add(Dropout(0.2))

vaccination_regressor.add(Dense(units = 1))


vaccination_regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

vaccination_regressor.fit(train_X, train_y, epochs = 100, batch_size = 32)

train_test_X = []
train_test_y = []

for i in range(4, len(test_data)):
    train_test_X.append(test_data[i-4: i])
    train_test_y.append(test_data[i])
    
train_test_X = np.array(train_test_X)
train_test_y = np.array(train_test_y)

train_test_X = np.reshape(train_test_X, (train_test_X.shape[0], train_test_X.shape[1], 1))


vac_predicted = vaccination_regressor.predict(train_test_X)

vac_predicted = vaccination_scaler.inverse_transform(vac_predicted)

train_test_y = vaccination_scaler.inverse_transform(train_test_y)

plt.title('vaccinations')
plt.plot(train_test_y, color='blue')
plt.plot(vac_predicted, color='green')
plt.show()


new_test_data = []
new_test_data_y = []
for i in range(4, len(new_test)):
    new_test_data.append(new_test[i-4:i, 0])
    new_test_data_y.append(new_test[i, 0])

new_test_data = np.array(new_test_data)
new_test_data = np.reshape(new_test_data, (37, 4, 1))

new_predict = vaccination_regressor.predict(new_test_data)

new_predict = vaccination_scaler.inverse_transform(new_predict)
new_test_data_y = vaccination_scaler.inverse_transform(new_test_data_y)

plt.title('vaccinations_new')
plt.plot(new_test_data_y, color='blue')
plt.plot(new_predict, color='green')
plt.show()


#new_x predictor single

new_X_data = np.array(gp['new_deaths']).reshape(-1,1)

new_X_scaler = StandardScaler()

new_X_scaler.fit(new_X_data)


train_data = new_X_scaler.transform(new_X_data[:500])
test_data = new_X_scaler.transform(new_X_data[500:])

train_X, train_y = [], []

for i in range(4, 500):
    train_X.append(train_data[i-4: i])
    train_y.append(train_data[i])
    
train_X, train_y = np.array(train_X), np.array(train_y)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))


new_X_regressor = Sequential()

new_X_regressor(LSTM(units=100, return_sequences=True, input_shape = (train_X.shape[1], 1)))
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

new_X_regressor.fit(train_X, train_y, epochs = 100, batch_size = 32)


train_test_X = []
train_test_y = []

for i in range(4, len(test_data)):
    train_test_X.append(test_data[i-4: i])
    train_test_y.append(test_data[i])
    
train_test_X = np.array(train_test_X)
train_test_y = np.array(train_test_y)

train_test_X = np.reshape(train_test_X, (train_test_X.shape[0], train_test_X.shape[1], 1))

new_X_predicted = new_X_regressor.predict(train_test_X)

new_X_predicted = new_X_scaler.inverse_transform(new_X_predicted)

train_test_y = new_X_scaler.inverse_transform(train_test_y)

plt.title('new_X')
plt.plot(train_test_y, color='blue')
plt.plot(new_X_predicted, color='green')
plt.show()


#To append to the end of np array
tt = multi_dt

tt = np.append(tt, [[1,1,1]], axis=0)


#test prediction into 1 day in future

start_multi = multi_dt[len(multi_dt)-4:]

vac_data = vaccination_scaler.fit_transform(new_vaccination_data)

start_vaccination = vac_data[len(vac_data)-4:]

x_data = new_X_scaler.fit_transform(new_X_data)

start_x = x_data[len(x_data)-4:]

start_multi = np.reshape(start_multi, (1,  4, 3))
start_vaccination = np.reshape(start_vaccination, (1,4,1))
start_x = np.reshape(start_x, (1,4,1))



start_vac_predict = vaccination_regressor.predict(start_vaccination)[0][0]
start_x_predict = new_X_regressor.predict(start_x)[0][0]
start_multi_predict = regressor.predict(start_multi)[0][0]

print(sc.inverse_transform( [[start_vac_predict, start_x_predict, start_multi_predict]]))
print(vaccination_scaler.inverse_transform([start_vac_predict]))
print(new_X_scaler.inverse_transform([start_x_predict]))
multi_dt = np.append(multi_dt, [[start_vac_predict, start_x_predict, start_multi_predict]], axis=0)

print(multi_dt[:, 0])
#for loop to predict N days into future

n_multi = multi_dt

vac_data = n_multi[:, 0]
nw_data = n_multi[:, 1]

for i in range(50):
    
    vac_data = n_multi[:, 0]
    nw_data = n_multi[:, 1]
    
    start_multi = np.reshape(n_multi[len(n_multi)-4:], (1, 4, 3))
    start_vacci = np.reshape(vac_data[len(vac_data)-4:], (1, 4, 1))
    start_nw = np.reshape(nw_data[len(nw_data)-4:], (1,4,1))
    
    start_multi_predict = regressor.predict(start_multi)[0][0]
    start_vacci_predict = vaccination_regressor.predict(start_vacci)[0][0]
    start_nw_predict = new_X_regressor.predict(start_nw)[0][0]
        
    n_multi = np.append(n_multi, [[start_vacci_predict, start_nw_predict, start_multi_predict]], axis=0)

check_with_latest = np.array(gp_2)

n_multi = sc.inverse_transform(n_multi)

plt.plot(gp['new_vaccinations'], color='blue')
plt.plot(n_multi[:, 0], color='red')
plt.title('new_vaccinations')
plt.show()

plt.plot(gp['new_deaths'], color='blue')
plt.plot(n_multi[:, 1], color='red')
plt.title('new_deaths')
plt.show()

plt.plot(gp_2['new_cases'], color='blue')
plt.plot(n_multi[:, 2], color='red')
plt.title('new_cases')
plt.show()

