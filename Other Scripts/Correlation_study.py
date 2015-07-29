# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:45:56 2015

@author: dantalian
"""

from Classes import *

p=pd.read_csv('Stationary_data_with_weather.csv')
colnames=['Latitude', 'Longitude', 'Value', 'ID', 'Height',
       'Loader ID','Sensor', 'Distance',
      'TemperatureF', 'Dew PointF', 'Humidity', 'Sea Level PressureIn',
       'VisibilityMPH', 'Wind SpeedMPH', 'PrecipitationIn', 'Conditions',
       'WindDirDegrees', 'Captured Time']
p.columns=colnames


start=time.time()
test={}
for n in p.Sensor.unique():
    #print(n)
    model=None
    r=p[p.Sensor==n]
    if len(r)<500:
        '''The p-values are not entirely
        reliable but are probably reasonable for datasets larger than 500 or so.'''
        continue
    model=Model(r)
    model.findCorrelations()
    df=model.CorrelationTable
    if df.empty:
        continue
    test[n]=df.loc['Results']
elapsed_time = time.time() - start
print 'TestModel time:' + str(elapsed_time)

ee=pd.DataFrame(test)

plt.subplot(321)
plt.title('Significance test for correlation between background radiation and weather data ')
plt.ylabel('Value-PrecipitationIn')
ee.loc['PrecipitationIn'].value_counts().plot(kind='barh')
plt.subplot(322)
plt.title('HO: there is no correlation between the two variables')
plt.ylabel('Value-Humidity')
ee.loc['Humidity'].value_counts().plot(kind='barh')
plt.subplot(323)
plt.ylabel('Value-Dew PointF')
ee.loc['Dew PointF'].value_counts().plot(kind='barh')
plt.subplot(324)
plt.ylabel('Value-Temperature')
ee.loc['Temperature'].value_counts().plot(kind='barh')
plt.xlabel('')
plt.subplot(325)
plt.ylabel('Value-Sea Level PressureIn')
ee.loc['Sea Level PressureIn'].value_counts().plot(kind='barh')
plt.show()