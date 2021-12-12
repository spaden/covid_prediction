# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 07:23:04 2021

@author: kalya
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('owid-covid-data.csv')

df.head()

gp = df.groupby('location')

test =  gp.count()['total_cases']

rand = df[df['location'] == 'Tajikistan']





india = df[df['location'] == 'India']

hd = india.head()


india_mod = india.drop(['population', 'population_density', 'median_age', 'aged_65_older',
       'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
       'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
       'life_expectancy', 'human_development_index',
       'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
       'excess_mortality', 'excess_mortality_cumulative_per_million', 'icu_patients',
       'icu_patients_per_million', 'hosp_patients',
       'hosp_patients_per_million', 'weekly_icu_admissions',
       'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
       'weekly_hosp_admissions_per_million'], axis = 1)


plt.plot(india_mod['date'] ,india_mod['total_cases'])
plt.show()

plt.hist(india_mod['total_cases'])
plt.show()

sns.heatmap(india_mod.corr(), cmap='coolwarm')

cr = india_mod.corr()

for i in range(2, 21):
    if 646 % i == 0:
        print(i)
print(600// i)    



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


test2 = []
final_countries_list = []
for item in df['location'].unique():
    if test[item] > 0 and item.upper() in required_countries_hash:
        test2.append([item, test[item]])
        final_countries_list.append(item)


df_filtered = df[df['location'].isin(final_countries_list)]

df_filtered = df[['location', 'date', 'total_vaccinations', 'total_deaths', 'total_cases']]

df_filtered['total_vaccinations'].fillna(0, inplace=True)
df_filtered['total_deaths'].fillna(0, inplace=True)
df_filtered['total_cases'].fillna(0, inplace=True)


gp2 = df_filtered.groupby('location')

test3 =  gp2.count()['total_cases']

lookback = []
for item in final_countries_list:
    mx = 1
    for i in range(2, 10):
        if test3[item] % i == 0:
            mx = i
    
    lookback.append([item, test3[item], mx])
    

df2 = df_filtered[ df['location'] == 'Afghanistan']



for item in lookback[1:]:
    if item[-1] <= 2:
        df2 = pd.concat([df2, df_filtered[ df['location'] == item[0]].iloc[item[-1]:]], ignore_index = True)
    else:
        df2 = pd.concat([df2, df_filtered[ df['location'] == item[0]]])

test_df2 = df2.groupby('location').count()['total_cases']

lookback2 = []
for item in final_countries_list:
    mx = 1
    for i in range(3, 21):
        if test_df2[item] % i == 0:
            mx = i
        
    lookback2.append([item, test_df2[item], mx])
    
        
df2.info()

rand = df2[ df2['location'] == 'India'].iloc[:, 1:].values

for item in lookback2:
    X_train = {'rn1': [], 'rn2': [], 'rn3':[], 'rn4':[]}
    y_train = {'rn1':[], 'rn2': [], 'rn3':[], 'rn4':[]}
    
    current_country = df2[df2['location'] == item[0]].iloc[:, 2:].values
    
    for i in range(item[2], item[1]):
        X_train['rn1'].append(current_country[ i - item[2]: i, 0 ])
        y_train['rn1'].append(current_country[ i, 0 ])
        
        
        X_train['rn2'].append(current_country[ i - item[2]: i, 1 ])
        y_train['rn2'].append(current_country[ i, 1 ])
        
        
        X_train['rn3'].append(current_country[ i - item[2]: i, 2 ])
        y_train['rn3'].append(current_country[ i, 2 ])
        
        X_train['rn4'].append(current_country[i - item[2]: i, :])
        y_train['rn4'].append(current_country[i, :])
    
    item.append([X_train, y_train])
    

india = lookback2[61][3] 

india_X = np.array(india[0]['rn3'])
india_Y = np.array(india[1]['rn3'])


test_X = india_X[600:]
test_y = india_Y[600:]    

india_X = india_X[:600]
india_Y = india_Y[:600]


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

india_X = sc.fit_transform(india_X)
india_Y = india_Y.reshape(-1, 1)
india_Y = sc.transform(india_Y)

india_X = np.reshape(india_X, (india_X.shape[0], india_X.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM( units = 50, return_sequences = True, input_shape = (india_X.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))
regressor.compile(optimizer = 'adam', loss='mean_squared_error')

regressor.fit(india_X, india_Y, epochs = 50, batch_size= 32)

test_X = np.array(test_X)

test_X = sc.transform(test_X)
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

predicted_value = regressor.predict(test_X)

print(predicted_value)



