# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from Classes import *
from Classes import Model
#the dataset is loaded

def EvaluateResults(filename,plot=False,printing=False):
    columns=['Technique','10','15','20','25','30','35','40','45','50','55','60','Sensor','Measurements','Outliers']
    results=pd.read_csv(filename,names=columns)
    results.index=results.Technique
    results=results.drop('Technique',1)
    grouped= results.groupby('Sensor')
    a={}
    b={}
    lowestError={}
    for group in grouped:
        if group[1].Measurements.iloc[0]>500:
             #it finds the technique and the best percentage for every sensor
            a[group[0]]=group[1].idxmin().value_counts().index[0] #it finds the technique that results better most of the time
            b[group[0]]=group[1].iloc[:,:9].min().idxmin() #it find the best percentage for training set
            lowestError[group[0]]=min(group[1].iloc[:,:9].min()) #it find the lowest error for every person
    metrics=pd.Series(a.values()).value_counts() #it counts the occurences of every metric
    percentage=pd.Series(b.values()).value_counts() #it counts the occurrences of every percentage
    errors=lowestError.values() 
    df=pd.DataFrame(lowestError.values()) #it creates a dataframe with the lowest error for every sensor
    df.index=lowestError.keys()
    if plot:
        plt.figure(1)
        metrics.plot(kind='barh',colormap='winter')
        plt.title('Dimensionality techniques that gave the best metric')
        plt.figure(2)
        percentage.plot(kind='barh',color='r')
        plt.title('Percentage of training set that gave the best metric')
        plt.figure(3)
        plt.plot(errors,'go')
        plt.plot(errors,'r')
        plt.title('Error between predicted and real output,median value:' + str(round(np.median(errors),2)))
        plt.show()
    if printing:
        print 'The dimensionality reduction technique that gives best result in terms of metric achived is ' + metrics.index[0] 
        print 'The best percentage for training set is ' + percentage.index[0] + '%'
        print 'The median of the lowest error achieved by every sensor is ' +str(np.median(errors))
        print 'These results are achieved analyzing ' + str(len(a)) + ' sensors, with more than 500 measurements, that corresponds to more than 5 consecutive days'
    return metrics,percentage,df


p=pd.read_csv('Stationary_data_with_weather.csv')
colnames=['Latitude', 'Longitude', 'Value', 'ID', 'Height',
       'Loader ID','Sensor', 'Distance',
      'TemperatureF', 'Dew PointF', 'Humidity', 'Sea Level PressureIn',
       'VisibilityMPH', 'Wind SpeedMPH', 'PrecipitationIn', 'Conditions',
       'WindDirDegrees', 'Captured Time']
p.columns=colnames

start_time = time.time()          


def printResults(dic,n,filename,printing=True):
    dataframe=pd.DataFrame.from_dict(dic)
    dataframe['Sensor']=n
    dataframe['Total samples']=len(model.dataset)
    dataframe['Outliers removed']=len(model.outliers)

    if printing:
        with open(filename, 'a') as f:
                       dataframe.to_csv(f,header=False)#,index=False,header=False)
    return dataframe
li=list(p.Sensor.unique())
li.sort()

for dd in p.groupby('Sensor'):
    dic1={}
    dic2={}
    dic3={}
    dic4={}
    sensor=dd[0]
    print sensor
    df=dd[1]
    if l>3:
        continue
    if len(df)<500:
        continue
    model=None
    model=Model(df)
    model=model.remove_outliers()
    model=model.prepareDataset(n=4,w=0,l=1)
    model=model.getInput()
    model=model.getOutput()
    for percentage in xrange(10,65,5): #different percentage of training set
        model=model.applyOnOutput(method='movingaverage',window=4)
        model=model.applyOnOutput(method='standardize',percentage=percentage)
        model=model.reduceDataset(method='All',nr=10)
        keys=model.getDatasetsAvailable()
        li=[model.applyOnInputs(inp=e,method='standardize',percentage=percentage) for e in keys]
        model=li[-1]
        values1=[model.SVregression(percentage,inp=a,kern='rbf')[1] for a in keys]
        values4=[model.KNregression(percentage,inp=b,neighbors=5)[1] for b in keys]
        values2=[model.SVregression(percentage,inp=c,kern='sigmoid')[1] for c in keys]
        values3=[model.SVregression(percentage,inp=d,kern='linear')[1] for d in keys]

        dic1[percentage]=dict(zip(keys,values1))
        dic2[percentage]=dict(zip(keys,values2))
        dic3[percentage]=dict(zip(keys,values3))
        dic4[percentage]=dict(zip(keys,values4))
        
    filenames=['SVRrbf.csv','SVRsigmoid.csv','SVRlinear.csv','KN.csv']
    dictionaries=[dic1,dic2,dic3,dic4]
    df=[printResults(dic=d,filename=f,n=sensor) for d,f in zip(dictionaries,filenames)]

elapsed_time = time.time() - start_time

print 'TestModel time:' + str(elapsed_time)


a=EvaluateResults('SVRrbf.csv',plot=True,printing=True)
b=EvaluateResults('SVRlinear.csv',plot=True,printing=True)
c=EvaluateResults('SVRsigmoid.csv',plot=True,printing=True)
d=EvaluateResults('KN.csv',plot=True,printing=True)