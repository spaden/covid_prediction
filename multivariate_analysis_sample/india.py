# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 07:27:14 2021

@author: kalya

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('owid-covid-data.csv')

df.head()

india = df[df['location'] == 'India'][['total_cases']]


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

india = np.array(india)


india.reshape(-1, 1)
india = sc.fit_transform(india)

X_train = india[:600]
X_test = india[600:]



print(46%8)

train_X = []
train_y = []

for i in range(4, 600):
    train_X.append(X_train[i - 4: i])
    train_y.append(X_train[i])

train_X, train_y = np.array(train_X), np.array(train_y)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()


regressor.add(LSTM(units= 100, return_sequences=True, input_shape = (train_X.shape[1], 1)))
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

regressor.fit(train_X, train_y, epochs = 100, batch_size = 32)

train_test_X = []
train_test_y = []


for i in range(4, 46):
    train_test_X.append(X_test[i-4: i])
    train_test_y.append(X_test[i])
    
train_test_X = np.array(train_test_X)
train_test_y = np.array(train_test_y)

train_test_X = np.reshape(train_test_X, (train_test_X.shape[0], train_test_X.shape[1], 1))

predicted = regressor.predict(train_test_X)

predicted = sc.inverse_transform(predicted)

train_test_y = sc.inverse_transform(train_test_y)


plt.plot(train_test_y, color='red')
plt.plot(predicted, color='blue')
plt.show()

original = df[df['location'] == 'India'][['total_cases']].values

total_data =  df[df['location'] == 'India'][['total_cases']].values


total_data = sc.transform(total_data)

total_predict = []

for i in range(4, len(total_data)):
    total_predict.append(total_data[i-4: i])


total_predict = np.array(total_predict)

total_predict = np.reshape(total_predict, (total_predict.shape[0], total_predict.shape[1], 1))

total_predicted_plot = regressor.predict(total_predict)

total_predicted_plot = sc.inverse_transform(total_predicted_plot)

plt.plot(original, color='red')
plt.plot(total_predicted_plot, color = 'green')
plt.show()


#multivariate analysis

df_filtered = df[['location', 'date', 'total_vaccinations', 'total_deaths', 'total_cases']]

df_filtered['total_vaccinations'].fillna(0, inplace=True)
df_filtered['total_deaths'].fillna(0, inplace=True)
df_filtered['total_cases'].fillna(0, inplace=True)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

cases_scaler = StandardScaler()

total_vaccinations = df_filtered[df['location'] == 'India'][['total_vaccinations', 'total_deaths', 'total_cases']]


cases_scaler.fit(np.array(total_vaccinations['total_cases']).reshape(-1, 1))



total_vaccinations = np.array(total_vaccinations)

total_vaccinations = sc.fit_transform(total_vaccinations)

total_vaccinations_train = total_vaccinations[:500]
total_vaccinations_test = total_vaccinations[500:]

total_vaccinations_x = []
total_vaccinations_y = []




for i in range(4, 500):
    total_vaccinations_x.append(total_vaccinations_train[i-4: i, :])
    total_vaccinations_y.append(total_vaccinations_train[i, 2])


total_vaccinations_x = np.array(total_vaccinations_x)
total_vaccinations_y = np.array(total_vaccinations_y)

total_vaccinations_x = np.reshape(total_vaccinations_x, (total_vaccinations_x.shape[0], total_vaccinations_x.shape[1], 3))
   

regressor = Sequential()


regressor.add(LSTM(units= 100, return_sequences=True, input_shape = (total_vaccinations_x.shape[1], 3)))
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

regressor.fit(total_vaccinations_x, total_vaccinations_y, epochs = 100, batch_size = 32)

vac_test_x = []
vac_test_y = []

for i in range(4, len(total_vaccinations_test)):
    vac_test_x.append(total_vaccinations_test[i-4:i, :])
    vac_test_y.append(total_vaccinations_test[i, 2])

vac_test_x = np.array(vac_test_x)
vac_test_y = np.array(vac_test_y)

vac_test_x = np.reshape(vac_test_x, (vac_test_x.shape[0], vac_test_x.shape[1], vac_test_x.shape[2]))

vac_pred = regressor.predict(vac_test_x)

plt.plot(vac_test_y, color='red')
plt.plot(vac_pred, color='blue')
plt.show()

total_vac_pred = []
for i in range(4, len(total_vaccinations)):
    total_vac_pred.append(total_vaccinations[i-4:i, :])

total_vac_pred = np.array(total_vac_pred)
total_vac_pred = np.reshape(total_vac_pred, (642, 4, 3))

total_vac_plot = regressor.predict(total_vac_pred)

total_vac_plot = cases_scaler.inverse_transform(total_vac_plot)

plt.plot(original, color='red')
plt.plot(total_vac_plot, color='green')
plt.show()