# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:32:46 2015

@author: dantalian
"""

from Classes import *


colnames=['Captured Time','Latitude','Longitude',  \
          'Value','Unit','Location','ID','Height', \
          'Surface','Radiation','Upload Time','Loader ID','Sensor','Distance']
#US_cleanedResult contains all the measurements clustered, it was obtained using hadoop
data = pd.read_csv('US_results.csv',header=None,names=colnames)
data.columns=colnames
data=data.sort('Captured Time')
# I am looking just for the stationary detectors
grouped = data.groupby('Sensor')
filtered=grouped.filter(lambda x: len(x)>10) #it drops the group that have less then 10 elements
regrouped=filtered.groupby('Sensor')
stationary=regrouped.filter(Sensor.isStationary) #it returns just the meassurements that belong to stationary detectors
stationarySensor=stationary.Sensor.unique()


d={}
start_time = time.time()
for n in stationarySensor:
    print n
    if len(data[data['Sensor']==n])<1000:
        continue
    sens=data[data['Sensor']==n]
    sens=sens.iloc[:10]
    d[n]=Model.createDataset(sens,printing=True)
elapsed_time = time.time() - start_time
print 'Datasets creation:' + str(elapsed_time)
