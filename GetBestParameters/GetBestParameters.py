
# coding: utf-8

# In[2]:

import numpy as np
from sklearn import preprocessing 
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA as sklearnPCA ,KernelPCA,FactorAnalysis,IncrementalPCA,FastICA
from sklearn.manifold import Isomap,LocallyLinearEmbedding
from scipy.stats.mstats import normaltest 
from scipy.stats import spearmanr
from math import *
import matplotlib.pyplot as plt
import time
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import urllib2
import json


# In[ ]:

class Sensor(object):
    ID=None
    owner=None
    days={} #days is a dictionary containing a dataframe with the safecast data for that specific day
    daysList=[] #it contains the days of the measurement, it is a list of the keys of days dictionary
    dataset=None
    latitude=None
    longitude=None
    stationary=None
    def __init__(self,a,date='Captured Time'):
        import pandas as pd
        #given a series of measurement it creates a dataframe for every day
        df=pd.DataFrame(a)
        df=df.sort('Captured Time')
        self.latitude,self.longitude,self.ID=df[['Latitude','Longitude','Sensor']].iloc[0].values
        i=lambda x: str(x.year) + '-' + str(x.month) + '-' +str(x.day) #I take just year,month and day 
        try:
             dates= df[date].apply(i) 
        except AttributeError:
            df=df.convert_objects(convert_dates='coerce')
            dates= df[date].apply(i)
        df['Date']=dates
        daysList=dates.unique()
        self.stationary=Sensor.isStationary(df)
        self.days=dict([(day,df[df['Date']==day]) for day in daysList])
        self.daysList=daysList
    
    def apply(self,f):
        '''Apply a generic function on historical data'''
        self.days.update((x, f(y)) for x, y in self.days.items())
        return self
    
    def addDay(self,a,date='Captured Time'): 
        ''' It adds another day to the days dictionary
        
        '''
        df=pd.DataFrame(a)
        i=lambda x: str(x.year) + '-' + str(x.month) + '-' +str(x.day) #I take just year,month and day 
        try:
             dates= df[date].apply(i) 
        except AttributeError:
            df=df.convert_objects(convert_dates='coerce')
            dates= df[date].apply(i)
        df['Day']=dates
        daysList=dates.unique()
        [self.days.update({day:df[df['Day']==day]}) for day in daysList] 
        [self.daysList.append(day) for day in daysList]
        return self
    
    def cleanAll(self):
        '''It cleans all the measurements applying the static method clean to every day
        '''
        self.days.update((x, Sensor.clean(y)) for x, y in self.days.items())
        return self
    
    @staticmethod
    def clean(df):
        '''It cleans a single day
        '''
        from string import strip
        columns=['Captured Time','Latitude','Longitude','Value','Unit','ID','Height','Loader ID','Sensor','Distance']
        df=df[columns]
        #df=df.dropna(1) #empty rows are deleted 
        df=df.drop_duplicates('Captured Time') #sometimes there are some duplicates
        df.index=xrange(0,len(df))
        today=dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        df=df.convert_objects(convert_dates='coerce')
        df=df[df['Captured Time']<=today] #every row with date field incorrect is deleted
        df['Unit']=df['Unit'].apply(strip)
        df=df[df.Unit=='cpm'] #all the units that are not expressed in cpm are deleted
        #I should add some lines to remove special character like \n and \t
        return df
    
    @staticmethod
    def convertDate(df,date='Captured Time'):
        df[date]=0
        try:
            f = lambda x: str(int(x.Year)) + '-'+ str(int(x.Month)) + '-' + str(int(x.Day)) + ' ' + str(int(x.Hour)) + ':' + str(int(x.Minute)) + ':' + '00'
            df[date]=df.apply(f,1)      
        except AttributeError:  
            diz={0:'00',0.25:'15',0.5:'30',0.75:'45'}
            g = lambda x: str(int(x.Year)) + '-'+ str(int(x.Month)) + '-' + str(int(x.Day)) + ' ' + str(int(x.Hour)) + ':' + diz[x.Hour - int(x.Hour)] + ':' + '00'
            df[date]=df.apply(g,1)                                                                                                                     
        df=df.drop(['Year','Month','Day','Hour'],axis=1)
        fmt="%Y-%m-%d %H:%M:%S"
        try:
            df[date]=df[date].apply(dt.datetime.strptime(date,fmt))
        except ValueError:
            pass
        return df                                                                                                                
    
    def createDataset(self):
        '''It merge all the dataframe in the days dictionary in a single dataframe
        '''
        tmp=self.days.values()
        df = pd.concat(tmp)
        self.dataset=df#.sort('Captured Time')
        return self.dataset
   
    def delDay(self,day):
        try:
            self.days.pop(day)
            self.daysList.remove(day)
        except KeyError:
            print 'The day ' + str(day) + ' is not present'
            return self
        return self
    
    @staticmethod
    def distance(a1,b1,a2,b2):
        '''Evaluates the distance in m between two points with coordinates expressed in
        Latitude and Longitude 
        '''
        a1=a1*np.pi/180
        a2=a2*np.pi/180
        b1=b1*np.pi/180
        b2=b2*np.pi/180
        return np.arccos(np.cos(a1-a2)*np.cos(b1)*np.cos(b2)+np.sin(b1)*np.sin(b2))*6378*1000    
    
    def extractDates(self,date='Captured Time',delta=0.25):
        '''It applies the extracDate static method on every day 
        '''
        self.days.update((x, Sensor.extractDate(y,date,delta)) for x, y in self.days.items())
        return self
    
    @staticmethod
    def extractDate(df,date='Captured Time',delta=0.25):
        '''Add two different fields useful to couple with weather data.
        
        The field 'DAY': year-month-day and the field 'Hour': hour.minutes
        
        '''
        import datetime as dt
        fmt="%Y-%m-%d"
        i=lambda x: str(x.year) + '-' + str(x.month) + '-' +str(x.day) #I take just year,month and day 
        try:
             dates= df[date].apply(i) 
        except AttributeError:
            df=df.convert_objects(convert_dates='coerce')
            dates= df[date].apply(i)
        g = lambda x: dt.datetime.strptime(x,fmt)
        dates= dates.apply(g)
        h=lambda x : str(x).split(' ')[0]#the conversion adds hour,minutes and seconds 
        dates= dates.apply(h) #I drop it and return a list of string
        df['Year']=df[date].apply(lambda x : x.year)
        df['Month']=df[date].apply(lambda x: x.month)
        df['Day']=df[date].apply(lambda x: x.day)  
        tmp=df[date].apply(lambda x: x.to_datetime())
        df['Hour']=tmp.apply(lambda x: x.hour)
        tmp=df[date].apply(lambda x: x.minute)
        f=lambda x: round(round(x/(60*delta))*delta,3)
        
        df['Hour']=df['Hour']+tmp.apply(f)
        df['Hour']=df['Hour'].replace(24,0.00)
        
        return df
    
    def getDays(self):
        print self.daysList
        
    @staticmethod
    def isStationary(df):
        '''It returns True if the measurement in df belong to a stationary detector
        '''
        l1=df.Latitude.iloc[0]
        l2=df.Longitude.iloc[0]
        m1=df.Latitude.iloc[len(df)-1]
        m2=df.Longitude.iloc[len(df)-1]
        if df.Distance.max()>15: #it checks if the distance between two consevutive measurements is more than
            #the maximum value of gps spatial inaccuracy
            return False
        if Sensor.distance(l1,l2,m1,m2)>100: #it checks if the distance between the first and the last point 
                                                #is too much
            return False
        if df.Distance.sum()>2*len(df):
            return False
        return True

    def timeSampling(self,day):
        '''It returns the time sampling of the measurement in the day indicated
        '''
        from numpy import median
        df=self.days[day]
        df=df.clean()
        return median([(df['Captured Time'].loc[n]-df['Captured Time'].loc[m]).total_seconds() for n,m in zip(xrange(1,len(df)),xrange(0,(len(df)-1)))])
    def to_csv(self,filename):
        with open(filename, 'a') as f:
               self.dataset.to_csv(f,index=False,float_format = '%.4f',header=False)

class Weather(object):
    '''The weather info for every day requested are saved in the dictionary historical {'year-month-day:weather df}
    '''
    lat=None
    lon=None
    historical={}
    stations=None
    state=None
    icao=None
    dataset=pd.DataFrame()
    daysUnavailable=[]
    daysList=[]
    closestStation=None
    key='3187b62a57755d52'
    #key=0
    def __init__(self,lat,lon):
        '''Given latitude and longitude it find the closest weather station
        
        it will be used after to find weather  informations'''
        self.parser=ParseWeather()
        self.city,self.country,self.state=self.parser.getLocation(lat,lon)
                                                                                                                                                                                                                                                                                    
    def addDay(self,a,date='DateUTC'): 
        '''Add another day to the historical dictionary'''
        df=pd.DataFrame(a)
        i=lambda x: str(x.year) + '-' + str(x.month) + '-' +str(x.day) #I take just year,month and day 
        try:
             dates= df[date].apply(i) 
        except AttributeError:
            df=df.convert_objects(convert_dates='coerce')
            dates= df[date].apply(i)
        df['Day']=dates
        daysList=dates.unique()
        [self.historical.update({day:df[df['Day']==day]}) for day in daysList] 
        [self.daysList.append(day) for day in daysList]
        return self
    
    def apply(self,f):
        '''Apply a function on historical data'''
        self.historical.update((x, f(y)) for x, y in self.historical.items())
        return self
    
    @staticmethod
    def clean(df):
        '''Clean a specific dataframe containing weather informations'''
        info=df.copy()
        
        pre={'Light Rain':1,'Heavy Rain':1,'Rain':1,'Light Rain Mist':1,           'Heavy Rain Mist':1,'Rain Mist':1,'Light Rain Showers':1,'Heavy Rain Showers':1,           'Rain Showers':1,'Light Thunderstorms and Rain':1,'Heavy Thunderstorms and Rain':1,           'Thunderstorms and Rain':1,'Light Freezing Drizzle':1,'Heavy Freezing Drizzle':1,               'Freezing Drizzle':1,'Light Freezing Rain':1,'Heavy Freezing Rain':1,'Freezing Rain':1,         'Light Snow':1,'Heavy Snow':1,'Snow':1,'Light Snow Grains':1,'Heavy Snow Grains':1,         'Snow Grains':1,'LightSnow Showers':1,'Heavy Snow Showers':1,'Snow Showers':1,
        'Light Ice Crystals':1,'Heavy Ice Crystals':1,'Ice Crystals':1,'Light Ice Pellets':1,  \
        'Heavy Ice Pellets':1,'Ice Pellets':1,'LightIce Pellet Showers':1,'HeavyIce Pellet Showers':1,   \
        'Ice Pellet Showers':1,'LightHail Showers':1,'Heavy Hail Showers':1, \
        'Hail Showers':1,'Light Small Hail Showers':1,'Heavy Small Hail Showers':1, \
        'Small Hail Showers':1}
        f=lambda x: pre.get(x , 0)        
        info['Conditions']=info['Conditions'].apply(f)
        
        #cleaning of NaN and other unexpected values
        info.PrecipitationIn=info.PrecipitationIn.fillna(value=0)
        info['Wind SpeedMPH']=info['Wind SpeedMPH'].fillna(value=0)
        info['Wind Direction']=info['Wind Direction'].replace('Calm',0)
        info['Wind SpeedMPH']=info['Wind SpeedMPH'].replace('Calm',0)
        #windspeedmph contains strings so it is considered as a generic object type, I convert it in float type
        info['Wind SpeedMPH']=info['Wind SpeedMPH'].apply(float)
        t=info.TemperatureF.copy()    
        h=info.Humidity.copy()
        s=info['Sea Level PressureIn'].copy()    
        d=info['Dew PointF'].copy()
        p=info['PrecipitationIn'].copy()
        #sometimes the weather informations show unexpected values (as -9999)
        t[t < -100] = np.NaN
        h[h<0]=np.NaN
        s[s<0]=np.NaN
        d[d<0]=np.NaN
        p[p<0]=np.NaN
        info['TemperatureF']=t
        info['Humidity']=h
        info['Sea Level PressureIn']=s
        info['Dew PointF']=d
        info['PrecipitationIn']=p
        return info
    
    def conditionsOccurred(self,graph=False):
        '''It returns the weather conditions occurred in the dataset'''
        conditions=self.dataset.Conditions.value_counts()
        print conditions
        self.conditions=self.dataset.Conditions.value_counts()
        if graph:
            conditions.plot(kind='barh')
        return self
    
    def createDataset(self):
        '''It merges all the dataframe in the historical dictionary in a single dataframe
        '''
        tmp=self.historical.values()
        df = pd.concat(tmp)
        self.dataset=df#.sort('DateUTC')
        return self.dataset
    
    @staticmethod
    def extractHour(df,date='DateUTC',delta=0.25): 
        '''It creates a new field hour
        
        The field contains the hour in the format Hour.quarters (i.e 13.25 are 13 hours and 15 mins)'''
        f=lambda x: round(round(x/(60*delta))*delta,3)
        try:
             hour=df[date].apply(lambda x: x.hour)
        except AttributeError:
            df[date]=df[date].convert_objects(convert_dates='coerce')
            hour=df[date].apply(lambda x: x.hour)    
        minute=df[date].dt.minute.apply(f)
        df['Hour']=hour+minute
        df['Hour']=df['Hour'].replace(24,0.00)
        return df
    
    def extractHours(self,date='DateUTC',delta=0.25):
        '''It applies the extractHour static method on every day 
        '''
        self.historical.update((x, Weather.extractHour(y,date,delta)) for x, y in self.historical.items() )
        return self
    
    def getDays(self):
        '''It simply prints the days with weather information available in the instance'''
        print self.weather.keys()
    
    def getHistorical(self, date):
        '''Given a specific day it extract the weather information from wunderground.com
        '''
        s=self.state
        c=self.city
        date=date[:10]
        fmt="%Y-%m-%d"
        date=dt.datetime.strptime(date,fmt)
        day=date.day
        date1=date-dt.timedelta(days=1)
        date=str(date)
        date1=str(date1)
        df1=self.parser.getWeather(date,self.city,self.state)
        df2=self.parser.getWeather(date1,self.city,self.state)
        df1['Day']=df1['DateUTC'].apply(lambda x: x.day)
        df2['Day']=df2['DateUTC'].apply(lambda x: x.day)
        df1=df1[df1['Day']==day]
        df2=df2[df2['Day']==day]
        df=df1.append(df2)
        self.historical[date]=df
        self.daysList.append(date)
        df=Weather.clean(df)
        return df
    
    def timeSampling(self,date='DateUTC'):
        from numpy import median
        df=self
        df=df.clean()
        return median([(df[date].loc[n]-df[date].loc[m]).total_seconds() for n,m in zip(xrange(1,len(df)),xrange(0,(len(df)-1)))])
    

    
class Model(object):
    '''This class contains method to prediction the background radiation using a dataframe with background
     and weather informations
    '''
    debug={}
    outliers=None
    reducedDatasets=None
    weather_columns=['Humidity','TemperatureF','Sea Level PressureIn','PrecipitationIn','Dew PointF','Conditions','Wind SpeedMPH']
    out_columns=['Value']
    #model_columns=['Value','PrecipitationIn','Humidity','Dew PointF','Sea Level PressureIn','TemperatureF']
    columns=['Captured Time','Humidity','TemperatureF','Sea Level PressureIn','Conditions','PrecipitationIn','Dew PointF','Value','Wind SpeedMPH']
        
    def __init__(self,df):
        from sklearn import preprocessing 
        self.ModelInputs={}
        self.ModelOutput=None
        self.prediction=None
        self.metrics={}
        self.Threats=[]
        self.OutputTest={}
        self.CorrelationTable=pd.DataFrame()
        self.datasetsAvailable=['Dataset']
        self.Sensor=df.Sensor.iloc[0]
        self.model_columns=['PrecipitationIn','Humidity','Dew PointF','Sea Level PressureIn','TemperatureF']
        '''Define a model object '''
        df=df[Model.columns]
        df=df.convert_objects(convert_dates='coerce')
        df=self.clean(df)
        t=df['Captured Time'].iloc[0]
        f=lambda x: (x-t).total_seconds()
        index=df['Captured Time'].apply(f)
        #df=df.drop('Captured Time',1)
        self.time=index
        df.index=index
        self.dataset=df
        
        
    def applyOnInputs(self,method,inp,f=None,window=0,percentage=60):
        '''It applies a built-in methods or a custom function f to the input variables
        
        Methods available:  
                            'standardize' , it applies the standardization method of sklearn.preprocessing.scale
        '''
        if not(self.ModelInputs):
            self.getInput()
        index=int(percentage*len(self.dataset)/100)
        d={'Train':self.ModelInputs[inp][:index,:],'Test':self.ModelInputs[inp][index:,:]}
        if method=='standardize':
            d.update((x, preprocessing.scale(y)) for x, y in d.items())
        else:
            d.update((x, f(y)) for x, y in d.items())
            
        #debug
        #dataset=pd.DataFrame(self.ModelInputs['Dataset'])
        #dataset['Output']=self.ModelOutput
        #self.debug['ApplyOnInputs']=dataset
        ###
        self.ModelInputs[inp]=np.append(d['Train'],d['Test'],axis=0)
        return self
    
    def applyOnOutput(self,method,f=None,window=0,percentage=60):
        '''It applies a built-in methods or a custom function f to the output variable
        
        Methods available:  'movingaverage', it requires the variable window
                            'standardize' , it applies the standardization method of sklearn.preprocessing.scale
        '''
        
        if self.ModelOutput==None:
            self.getOutput()
        index=int(percentage*len(self.dataset)/100)
        self.OutputTest['Original']=self.ModelOutput[index:]
        #this function it's used to apply some filtering to the output
        #for this reason the data are splitted , in this way every filtering technique won't be anticasual
        #i.e. a moving average filtering on the train part will consider also some samples from the test part
        #that belong ideally to the "future"
        d={'Train':self.ModelOutput[:index],'Test':self.ModelOutput[index:]}
        if method=='movingaverage':
            if not(window):
                raise ValueError('A value for the window is required')
            d.update((x, Model.moving_average(y,n=window)) for x, y in d.items())
            
        elif method=='standardize':
            self.OutputTest['mean']=np.mean(d['Train'])
            self.OutputTest['std']=np.std(d['Train'])
            d.update((x, preprocessing.scale(y)) for x, y in d.items())
        else:
            d.update((x, f(y)) for x, y in d.items())
        newOutput=np.append(d['Train'],d['Test'])
        
        #the moving_average could drop some values at the end of the time series, so if this happens the last
        #values is repeated to restore the original dimension
        check=len(self.ModelOutput)-len(newOutput)
        if check>0:
            newOutput=np.append(newOutput,newOutput[-check:])
        self.ModelOutput=newOutput
        '''        #debug
        dataset=pd.DataFrame(self.ModelInputs['Dataset'])
        dataset['Output']=self.ModelOutput
        self.debug['ApplyOnOutputs']=dataset
        ###'''
        
        return self
    
    def clean(self,dataset):
        dataset.Value=dataset.Value.replace(0,np.nan)
        #a weighted interpolation is applied on a windows that correspond to a period of 3 hours
        #just for the weather conditions
        colnames=['Humidity','TemperatureF','Sea Level PressureIn','Conditions','PrecipitationIn','Dew PointF']
        dataset[colnames]=dataset[colnames].replace(np.nan,999) 
        #the rolling apply function require that there are no nan values, so I use a dummy number
        dataset[colnames]=pd.rolling_apply(dataset[colnames],13,Model.weightedInterp)
        #at the end a linear interpolation it is used on value field and to fulfill the weather conditions in
        #the case that some period had no value to interpolate
        dataset=dataset.interpolate(method='linear')
        dataset=dataset.dropna() #it drops the NaT captured Time
        return dataset
    
    @staticmethod
    def clustering(var1,var2):
        '''Given two variables it find the clusters according the Meanshift algorithm
        The current function is used by the remove_outliers method
        '''

        X=[var1,var2]
        X=np.array(X)
        X=X.T
        bandwidth = estimate_bandwidth(X, quantile=0.9, n_samples=500) #estimation of bandwidth parameter needed for the 
                                                                    #clustering
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        tmp=pd.DataFrame(X)
        tmp['Label']=labels
        return tmp
    
    def conditionsOccurred(self,graph=False):
        '''It returns the weather conditions occurred in the dataset, if the Condition field has not transformed in 
        a numerical field yet
        '''
        conditions=self.dataset.Conditions.value_counts()
        print conditions
        self.conditions=self.dataset.Conditions.value_counts()
        if graph:
            conditions.plot(kind='barh')
        return self
    
    @staticmethod
    def createDataset(sens,printing=False,filename='Stationary_data_with_weather.csv'):
        '''This function instantiates the objects Weather and Sensor, use their method to clean and collect informations 

        Then merge them in a dataset containing weather and radiation information
        '''
        w=None
        s=Sensor(sens)
        s=s.cleanAll()
        sensor=s.extractDates() 
        #value of lat and lon needed to instantiate the weather class
        lat,lon=sensor.latitude,sensor.longitude
        w= Weather(lat,lon)
        
        for day in sensor.daysList: 
            w.getHistorical(day)
        #the historical weather has a sampling time of 1 hour, so I resample my sensor data every (15 min default)
        #taking the median of the value in that period
        wea=w.extractHours()
        f= lambda x: x.groupby(x.Hour).median() 
        wea=wea.apply(f)
        sensor=sensor.apply(f)
        #pieces contains a list of dataframe corresponding to a single day of measurements coupled with the weater
        #dataframe with all the measurements coupled
        try:
            pieces=[sensor.days[date].join(wea.historical[date]) for date in wea.daysList  if not(wea.historical[date].empty) ]
        except ValueError:
            return pd.DataFrame()
        #to make the single days well sampled the holes are filled with a linear interpolation method
        #the first and the last are skipped because the first piece probably doesn't start at midnight so it would be filled
        #with NaN
        #for the last is the same, it probably doesn't finish at midnight
        filled=[p.reindex(np.arange(0,24,0.25)).interpolate(method='linear') for num,p in enumerate(pieces) if (num!=0 and num!=len(pieces)-1) ]
        try:
            filled.insert(0,pieces[0])
        except IndexError:
            return pd.DataFrame()
        filled.append(pieces[-1])
        try:
            
            dataset=pd.concat(filled)
        except ValueError:
            return pd.DataFrame()
        #after the median on every hour all the field that were string become NaN or are dropped
        dataset.dropna(1,how='all')
        
        dataset = dataset[np.isfinite(dataset['Sensor'])] 
        dataset['Hour']=dataset.index
        dataset.drop
        #in the line below the field Captured Time is recreated
        dataset=Sensor.convertDate(dataset)
        if printing:
            with open(filename, 'a') as f:
                   dataset.to_csv(f,index=False,float_format = '%.4f',header=False)
        return dataset
    
    def dimensionalityReduction(self,nr=5):
        '''It applies all the dimensionality reduction techniques available in this class:
        Techniques available:
                            'PCA'
                            'FactorAnalysis'
                            'KPCArbf','KPCApoly'
                            'KPCAcosine','KPCAsigmoid'
                            'IPCA'
                            'FastICADeflation'
                            'FastICAParallel'
                            'Isomap'
                            'LLE'
                            'LLEmodified'
                            'LLEltsa'
        '''
        dataset=self.ModelInputs['Dataset']
        sklearn_pca = sklearnPCA(n_components=nr)
        p_components = sklearn_pca.fit_transform(dataset)
        fa=FactorAnalysis(n_components=nr)
        factors=fa.fit_transform(dataset)
        kpca=KernelPCA(nr,kernel='rbf')
        rbf=kpca.fit_transform(dataset)
        kpca=KernelPCA(nr,kernel='poly')
        poly=kpca.fit_transform(dataset)
        kpca=KernelPCA(nr,kernel='cosine')
        cosine=kpca.fit_transform(dataset)
        kpca=KernelPCA(nr,kernel='sigmoid')
        sigmoid=kpca.fit_transform(dataset)
        ipca=IncrementalPCA(nr)
        i_components=ipca.fit_transform(dataset)
        fip=FastICA(nr,algorithm='parallel')
        fid=FastICA(nr,algorithm='deflation')
        ficaD=fip.fit_transform(dataset)
        ficaP=fid.fit_transform(dataset)
        '''isomap=Isomap(n_components=nr).fit_transform(dataset)
        try:
            lle1=LocallyLinearEmbedding(n_components=nr).fit_transform(dataset)
        except ValueError:
            lle1=LocallyLinearEmbedding(n_components=nr,eigen_solver='dense').fit_transform(dataset)
        try:
            
            lle2=LocallyLinearEmbedding(n_components=nr,method='modified').fit_transform(dataset)
        except ValueError:
            lle2=LocallyLinearEmbedding(n_components=nr,method='modified',eigen_solver='dense').fit_transform(dataset) 
        try:
            lle3=LocallyLinearEmbedding(n_components=nr,method='ltsa').fit_transform(dataset)
        except ValueError:
            lle3=LocallyLinearEmbedding(n_components=nr,method='ltsa',eigen_solver='dense').fit_transform(dataset)'''
        values=[p_components,factors,rbf,poly,cosine,sigmoid,i_components,ficaD,ficaP]#,isomap,lle1,lle2,lle3]
        keys=['PCA','FactorAnalysis','KPCArbf','KPCApoly','KPCAcosine','KPCAsigmoid','IPCA','FastICADeflation','FastICAParallel']#,'Isomap','LLE','LLEmodified','LLEltsa']
        self.ModelInputs.update(dict(zip(keys, values)))
        [self.datasetsAvailable.append(key) for key in keys ]
        
        #debug
        #dataset=pd.DataFrame(self.ModelInputs['Dataset'])
        #dataset['Output']=self.ModelOutput
        #self.debug['Dimensionalityreduction']=dataset
        ###
        return self
    
    @staticmethod
    def extractMetrics(pred,test_y):
        '''It extracts three different metrics: mean absolute error,median absolute error,mean square error

        '''
        try:
            meanae=mean_absolute_error(test_y,pred)
        except ValueError:
            #sometimes the moving average filter on the output reduce the dimensionality of it
            #so some value of the predition is dropped
            pred=pred[:len(test_y)-len(pred)]
            meanae=mean_absolute_error(test_y,pred)
        mae=median_absolute_error(test_y,pred)
        mse=mean_squared_error(test_y,pred)
        return meanae,mae,mse
    
    def findCorrelations(self,alfa=5,duringRain=False,minimumLength=500):
        '''It discovers if the input variables are correlated with the output making use of Spearman correlation technique
        
        The alfa parameter define the level of significance of the test,it is expressed in percentage
        If the p-value evaluated is less than alfa/100 the Null Hypotesis (there is no correlation between the variables) is refused'''
        e=self.dataset
        if duringRain:
            e=e[e['Conditions']==1]
        e=e[Model.weather_columns]
        e['Value']=self.dataset.Value
        e=e.apply(preprocessing.scale)
        if len(e)<minimumLength:
            self.CorrelationTable=pd.DataFrame()
            return self
        pthresh=alfa/100.0
        val=e.Value.values
        temp=spearmanr(e.TemperatureF.values,val)
        hum=spearmanr(e.Humidity.values,val)
        sea=spearmanr(e['Sea Level PressureIn'].values,val)
        prec=spearmanr(e.PrecipitationIn.values,val)
        dew=spearmanr(e['Dew PointF'].values,val)
        df=pd.DataFrame({'Temperature':temp,'Sea Level PressureIn':sea,'PrecipitationIn':prec,'Humidity':hum,'Dew PointF':dew},index=['Pearson coefficients','p-values'])
        def test(p,threshold):
            if p<threshold:
                return 'Reject H0'
            else:
                return 'Accept H0'
        df.loc['Results']=[test(p,pthresh) for p in df.loc['p-values']]
        self.CorrelationTable=df
        return self
        
    
    def GBregression(self,percentage=60,inp='Dataset',n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls'):
        '''It applies the ensamble method of gradient boosting trees'''
        X=y=prediction=metrics=None
        X=self.ModelInputs[inp] #input dataset
        samples=int(percentage*len(X)/100) #evaluating the samples number given the percentage
        x=X[:samples,:] #training input set
        try:
            y = self.ModelOutput[:samples] #training output set
        except KeyError:
            self.getOutput()
            y = self.ModelOutput[:samples]
        test_x=X[samples:,:] #testing input set
        test_y=self.ModelOutput[samples:] # testing output set
        gb=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')
        model=gb.fit(x,y)
        prediction=model.predict(test_x)
        self.prediction=prediction
        self.OutputTest['Standardized']=test_y
        metrics=Model.extractMetrics(prediction,test_y)
        return prediction,np.median(metrics)
    
    def getDatasetsAvailable(self):
        self.datasetsAvailable=self.ModelInputs.keys()
        return self.ModelInputs.keys()
    
    def getInput(self):
        X=self.dataset[self.model_columns].copy()
        self.ModelInputs['Dataset']=X.as_matrix()
        return self
        
    def getOutput(self):
        Y=self.dataset.copy()
        try:
            self.ModelOutput=Y[Model.out_columns].as_matrix()
        except KeyError:
            self.ModelOutput=self.dataset['Output'].as_matrix() #if the preparare dataset has been called 
            #the output is 'Output' instead of 'Values
        return self
    
    def insertThreat(self,testPercentage=40,wLength=4,meanP=1.1):
        '''Method to simulate and insert a threat in the part of the output series that will be used as test
        wLenght: the lenght of the window in which the threat will be inserted
        testPercentage: indicates the percentage of the test dataset
        meanP: is the mean value of the Poisson distribution from which the "threat" is extracted
        '''
        t=None
        testPercentage=testPercentage/100.0
        t=pd.DataFrame()
        t['Value']=self.dataset.Value.copy()#create a copy of the output
        startTest=int((1-testPercentage)*len(t)) #define the first index of the output that will be used as test
        s=np.random.random_integers(startTest,len(t)) #find a random index in the test part of the output
        values=np.random.poisson(t['Value'].mean()*meanP,wLength) #find random values from poisson distribution with E[x]=m
        window=np.arange(s,s+4)*(self.dataset.index[1]-self.dataset.index[0]) #define the window
        #the window is cleaned, the values are added and the other values are interpolated to maintain the continuity
        t['Value'].loc[window]=values
        #t.loc[window[1:-1]]=values
        self.ThreatsIndex=t.copy()
        self.ThreatsIndex['Value']=0
        self.ThreatsIndex.loc[window]=1
        
        d={'Train':t['Value'].iloc[:startTest],'Test':t['Value'].iloc[startTest:]}
        d.update((x, preprocessing.scale(y)) for x, y in d.items())      
        self.Threats=np.append(d['Train'],d['Test'])#append the window in which there is the threat 
        self.dataset.Value=t['Value'].values #the threat is inserted in the dataset
        return self

    def KNregression(self,percentage,inp='Dataset',neighbors=5,weights='distance',algorithm='auto',leaf=30):
        '''It evaluates a prediction using k-nearest neighbors regression approach
        
        It returns a tuple: (prediction, median of three different metrics) '''
        X=y=prediction=metrics=None
        X=self.ModelInputs[inp] #input matrix
        samples=int(percentage*len(X)/100) #evaulating the number of samples given the percentage
        x=X[:samples,0:] #training input set
        y = self.ModelOutput[:samples] # training output set
        test_x=X[samples:,:] #testing input set
        test_y=self.ModelOutput[samples:] #testing output set
        knn=KNeighborsRegressor(n_neighbors=neighbors,weights=weights,algorithm=algorithm, leaf_size=leaf)
        try:
            model=knn.fit(x,y) #evaluating the model
        except ValueError:
            return np.nan,9999
        prediction=model.predict(test_x) #evaluating of the prediction
        self.prediction=prediction
        self.OutputTest['Standardized']=test_y
        metrics=Model.extractMetrics(prediction,test_y)
        return prediction,np.median(metrics)
    
    @staticmethod   
    def moving_average(a, n=3) :
        ''' Function that implements a moving average filter
            [source]:http://stackoverflow.com/questions/14313510/moving-average-function-on-numpy-scipy    
        '''
        first=np.array([a[0]])
        last=np.array([a[-1]])
        a=np.concatenate((first,a,last))
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    def plotRadiationWeather(self):
        '''It plots the Value field with each weather field separately

        The function returns a plot object
        '''

        df=self.dataset
        plt.figure()

        stand=df.apply(preprocessing.scale,axis=0) #the data are normalized because they have different units
        val=stand['Value'].as_matrix()
        prec=stand['PrecipitationIn'].as_matrix()
        dew=stand['Dew PointF'].as_matrix()
        hum=stand['Humidity'].as_matrix()
        press=stand['Sea Level PressureIn'].as_matrix()
        temp=stand['TemperatureF'].as_matrix()
        plt.subplot(3,3,1)
        plt.plot(val,prec,'bo')
        plt.ylabel('Precipitation')
        plt.xlabel('Background Radiation')
        plt.subplot(3,2,2)
        plt.plot(val,dew,'ro')
        plt.ylabel('Dew Point')
        plt.xlabel('Background Radiation')
        plt.subplot(3,2,3)
        plt.plot(val,hum,'yo')
        plt.ylabel('Humidity')
        plt.xlabel('Background Radiation')
        plt.subplot(3,2,4)
        plt.plot(val,press,'go')
        plt.ylabel('Sea Level Pressure')
        plt.xlabel('Background Radiation')
        plt.subplot(3,2,5)
        plt.plot(val,temp,'mo')
        plt.ylabel('Temperature')
        plt.xlabel('Background Radiation')
        plt.subplot(3,2,6)
        plt.plot(val,prec,'bo')
        plt.plot(val,dew,'ro')
        plt.plot(val,hum,'yo')
        plt.plot(val,press,'go')
        plt.plot(val,temp,'mo')
        #plt.legend(['Precipitation','DewPoint','Humidity','Sea Level Pressure','Temperature'])
        plt.xlabel('Background Radiation')
        plt.show()
        
    def plotDataset(self):
        self.dataset.plot(subplots=True)
        plt.xlabel('Time')
        plt.show()
        
    def  plotPrediction(self):
        '''It creates a figure with two graphs: the real and the predicted output
                                                the absolute error between them
        '''
        predicted=self.prediction
        real=self.OutputTest['Standardized']#[abs(len(self.OutputTest['Standardized'])-len(self.prediction)):]

        rmse=np.sqrt(mean_squared_error(predicted,real))
        plt.figure()
        plt.subplot(211)
        plt.xlabel('Time')
        plt.ylabel('Radiation ')
        plt.title('Comparison between real and predicted output, RMSE=' + str(rmse))
        plt.plot(predicted,'r')
        plt.plot(real,'b')
        plt.legend(['Predicted output','Real output'])
        plt.subplot(212)
        plt.xlabel('Time')
        plt.ylabel('Absolute error')
        plt.plot(abs(real-predicted),'m')
        plt.show()
        
    def prepareDataset(self,n=1,l=1,w=0):
        X=self.dataset[Model.weather_columns].copy()
        self.model_columns=Model.weather_columns[:] #this fake slicing provide a copy of the list 
        values=model.dataset.Value.copy()
        output=values.shift(-l).copy()
        vfield=[]
        for m in xrange(0,n+1): #the n parameter sets how much new fields should be created 
                #if the present value of the output is at the time t there will be created n columns with
                #output from 0,1,2,...t-1 , 0,1,2,...t-2, ....... 0,1,2,...t-n
            field='Values-' + str(m)
            vfield.append(field)
            self.model_columns.append(field)
            X[field]=values.shift(m) #the shift function creates the new fields 
        for k in xrange(1,w+1):
            a=X[Model.weather_columns].shift(k)
            newfields=[col+'-' +str(w) for col in a.columns]
            a.columns=newfields
            #[self.model_columns.append(f) for f in newfields]
            X=pd.concat([X,a], axis=1)
        X['Output']=output
        X=X.dropna()
        ##debug    
        #dataset=X.copy()
        #dataset['Output']=output.copy()
        #self.debug['getInput']=dataset
        ##
        self.dataset=X.copy()
        return self
    
    def reduceDataset(self,nr=3,method='PCA'):
        '''It reduces the dimensionality of a given dataset using different techniques provided by Sklearn library
         Methods available:
                            'PCA'
                            'FactorAnalysis'
                            'KPCArbf','KPCApoly'
                            'KPCAcosine','KPCAsigmoid'
                            'IPCA'
                            'FastICADeflation'
                            'FastICAParallel'
                            'Isomap'
                            'LLE'
                            'LLEmodified'
                            'LLEltsa'
        '''
        dataset=self.ModelInputs['Dataset']
        #dataset=self.dataset[Model.in_columns]
        #dataset=self.dataset[['Humidity','TemperatureF','Sea Level PressureIn','PrecipitationIn','Dew PointF','Value']]
        #PCA
        if method=='PCA':
            sklearn_pca = sklearnPCA(n_components=nr)
            reduced = sklearn_pca.fit_transform(dataset)
        #Factor Analysis
        elif method=='FactorAnalysis':
            fa=FactorAnalysis(n_components=nr)
            reduced=fa.fit_transform(dataset)
        #kernel pca with rbf kernel
        elif method=='KPCArbf':
            kpca=KernelPCA(nr,kernel='rbf')
            reduced=kpca.fit_transform(dataset)
        #kernel pca with poly kernel
        elif method=='KPCApoly':
            kpca=KernelPCA(nr,kernel='poly')
            reduced=kpca.fit_transform(dataset)
        #kernel pca with cosine kernel
        elif method=='KPCAcosine':
            kpca=KernelPCA(nr,kernel='cosine')
            reduced=kpca.fit_transform(dataset)
        #kernel pca with sigmoid kernel
        elif method=='KPCAsigmoid':
            kpca=KernelPCA(nr,kernel='sigmoid')
            reduced=kpca.fit_transform(dataset)
        #ICA
        elif method=='IPCA':
            ipca=IncrementalPCA(nr)
            reduced=ipca.fit_transform(dataset)
        #Fast ICA
        elif method=='FastICAParallel':
            fip=FastICA(nr,algorithm='parallel')
            reduced=fip.fit_transform(dataset)
        elif method=='FastICADeflation':
            fid=FastICA(nr,algorithm='deflation')
            reduced=fid.fit_transform(dataset)
        elif method == 'All':
            self.dimensionalityReduction(nr=nr)
            return self
        
        self.ModelInputs.update({method:reduced})
        self.datasetsAvailable.append(method)
        return self
    
    def remove_outliers(self):
        '''It removes the outliers using the MeanShift clustering techniques
        '''
        dataset=self.dataset[self.model_columns].copy()
        dataset['Value']=self.dataset.Value.copy()
        stand=dataset.apply(preprocessing.scale,axis=0) #the data are standardized because they have different units
        val=stand['Value'].as_matrix()
        prec=stand['PrecipitationIn'].as_matrix()
        dew=stand['Dew PointF'].as_matrix()
        hum=stand['Humidity'].as_matrix()
        press=stand['Sea Level PressureIn'].as_matrix()
        temp=stand['TemperatureF'].as_matrix()
        l=[Model.clustering(val,b) for b in [prec,dew,hum,press,temp] ]
        l1=[a.groupby('Label').count().index[0] for a in l ] #it finds the cluster with most of the data
        l2=[a[a['Label']!=lab] for a,lab in zip(l,l1)] #the biggest cluster is removed in every dataframe
        outliers=pd.concat(l2,join='inner',axis=1).index #the concat with join='inner' option find the intersection between                                              
        #the dataframes, the resulting indexes indicate the outliers
        #the indexes in outliers are not expressed in seconds
        #so I create a fake index
        index=list(xrange(0,len(stand)))
        #and I remove the indexes that corresponds to the outliers
        [index.remove(a) for a in outliers ] 
        #using iloc I remove them from the original dataset
        self.dataset.Value.iloc[outliers]=np.nan
        #the dropped value are replaced using a linear interpolation
        self.dataset.Value=self.dataset.Value.interpolate(method='linear')
        self.dataset=self.dataset.dropna()
        index=self.dataset.index-self.dataset.index[0]
        self.dataset.index=index
        self.outliers=outliers #the outliers are saved
        
        #DEBUG
        self.debug['Removeoutliers']=dataset
        ###
        
        return self

    def SVregression(self,percentage,inp='Dataset',kern='rbf',method='standard',c=2048,eps=0,gamma=0.01,tau=3):
        '''Given the dataset of the input X and the dataset of the output Y it find a regression model using
        Support vector regression algorithm of sklearn library
        
        It returns a tuple: (prediction, median of three different metrics)
        '''       
        
        
        X=y=prediction=metrics=None
        X=self.ModelInputs[inp].copy() #input dataset
        samples=int(percentage*len(X)/100) #evaluating the samples number given the percentage
        x=X[:samples,:] #training input set
        try:
            y = self.ModelOutput[:samples] #training output set
        except KeyError:
            self.getOutput()
        y = self.ModelOutput[:samples]
        test_x=X[samples:,:] #testing input set
        test_y=self.ModelOutput[samples:] # testing output set

        #Parameters settings based on "Selection of Meta-Parameters for support vector regression" 
        # Vladimir Cherkassky and Yunqian Ma
        if method=='standard':
            n=len(y)
            std=y.std()
            c=tau*std
            eps=tau*np.sqrt(log(n)/n)
        #regression
        svr =SVR(kernel=kern,C=c,epsilon=eps,gamma=gamma)
        m=None
        try:
            m=svr.fit(x,y)
        except ValueError:
            return np.nan,9999
            
        #debug
        #self.debug['SVR']=self.ModelOutput
            
            
        prediction=m.predict(test_x)
        self.prediction=prediction
        self.OutputTest['Standardized']=test_y
        metrics=Model.extractMetrics(prediction,test_y)
        return prediction,np.median(metrics)

    
    @staticmethod
    def weightedInterp(array):
        l=int(len(array)/2)
        if array[l]!=999:
            return array[6]

        #other weight function could be inserted using scipy.signal module
        a=list(np.arange(1,l+1))
        l1=[(n*m,m)  for n,m in zip(array[0:6],a) if n!=999]
        a.reverse()
        l2=[(n*m,m)  for n,m in zip(array[7:13],a) if n!=999]
        try:
            num=reduce(lambda x,y: x+y, [x[0] for x in l1+l2])
        except TypeError:
            return np.nan
        den= reduce(lambda x,y: x+y, [x[1] for x in l1+l2])
        return num/den
    

class ParseMap(object):
    way={}
    node={}
    coord={}
    way_limit={}
    way_City={}
    way_Street={}
    way_coor={}
    '''
    
    #notes:
    #the use of the tag_filter seems slower than a simple if-then
    #not used at the moment
    whitelist = set(('name', 'highway'))
    
    #unused
    def tag_filter(tags):
        for key in tags.keys():
            if key not in whitelist:
                del tags[key]
        if 'name' in tags and len(tags) == 1:
            # tags with only a name have no information
            # how to handle this element
            del tags['name']
    '''
    def ways_stationary(self,ways):
        for osmid, tags, refs in ways:
            if tags.has_key('building'): 
                self.way[osmid]=refs
                if tags.has_key('addr:city'):  #sometimes the ways have also the city name in tags
                    self.way_City[osmid]=tags['addr:city']
                else:
                    self.way_City[osmid]=None
                if tags.has_key('name'): 
                    self.way_Street[osmid]=tags['name']
                else:
                    self.way_Street[osmid]=None

    def ways(self,ways):
        for osmid, tags, refs in ways:
            if tags.has_key('highway'): #just the streets are needed 
                self.way[osmid]=refs
                if tags.has_key('addr:city'):  #sometimes the ways have also the city name in tags
                    self.way_City[osmid]=tags['addr:city']
                else:
                    self.way_City[osmid]=None
                if tags.has_key('name'): 
                    self.way_Street[osmid]=tags['name']
                else:
                    self.way_Street[osmid]=None
                    
    def nodes(self,nodes):
        for idnode,tag,coor in nodes:
            lat=coor[1] #it's necessary because the coordinates in the nodes 
            lon=coor[0] #are (lon,lat) while in the coords are (lat,lon)
            self.node[idnode]=((lat,lon), tag)
            
    def coords(self,coords):
          for osm_id, lon, lat in coords:
            self.coord[osm_id]=(lat,lon)

    def fill_way_coords(self): #return a dictionary: {osmid:[list of nodes coordinates]}
        for osmid in self.way.keys():
            l=[]
            for ref in self.way[osmid]:
                try:
                    val=self.node[ref][0]
                except KeyError:
                    val=self.coord[ref]
                l.append(val)
            self.way_coor[osmid]=l
                  
    def getRange(self):
        for osmid in self.way.keys():
            a=self.way_coor[osmid]
            c=map(list, zip(*a)) #to unzip a list of tuples [(lat1,lon1),(lat2,lon2)] in [ [lat1,lat2),(lon1,lon2)]
            lat=c[0]
            lon=c[1]
            self.way_limit[osmid]=[min(lat),min(lon),max(lat),max(lon)]
            
            
class ParseWeather(object):
    '''Class that implement methods to get the weather informations from wunderground.com
    '''
    key='3187b62a57755d52'
    def __init__(self):
        if not(ParseWeather.key):
            raise Exception('Key is not present, register at http://www.wunderground.com/weather/api/ to get one')
    def getLocation(self,lat,lon):
        '''Given latitude and longitude it returns the city,country and state corresponding to the coordinates '''
        key=ParseWeather.key
        url_template='http://api.wunderground.com/api/{key}/geolookup/q/{latitude},{longitude}.json'
        url=url_template.format(key=key,latitude=lat,longitude=lon)
        g = urllib2.urlopen(url)
        json_string = g.read()
        location = json.loads(json_string)
        g.close()
        diz=location['location']['nearby_weather_stations']['airport']['station'][0]
        return diz['city'].replace(' ','_'),diz['country'],diz['state']
    
    def getWeather(self,date,c,s):
        '''Given a date a city and a state it returns a DataFrame '''
        k=ParseWeather.key
        d=date[:10].replace('-','')
        url_template='http://api.wunderground.com/api/{key}/history_{date}/q/{state}/{city}.json'
        url=url_template.format(key=k,date=d,state=s,city=c)  
        f = urllib2.urlopen(url)
        json_string = f.read()
        weather = json.loads(json_string) #parsing the json
        f.close()
        forecast=weather['history']['observations']
        l=[]
        for n in xrange(0,len(forecast)):
            #every cycle define a row containing the weather information for a single hour

            tmp=pd.DataFrame(forecast[n]) #definition of the dataframe
            col=['utcdate','tempi','dewpti','hum','pressurei','visi','wdire','wspdi','precipi','conds','snow','wdird']
            year=tmp.ix['year','utcdate'] #info about the day are extracted
            month=tmp.ix['mon','utcdate']
            day=tmp.ix['mday','utcdate']
            hour=tmp.ix['hour','utcdate']
            minute=tmp.ix['min','utcdate']
            date= year +'-' + month + '-' + day + ' ' + hour + ':' + minute + ':00'
            #the name of the columns are changed
            newcol=['DateUTC', 'TemperatureF', 'Dew PointF', 'Humidity',
                   'Sea Level PressureIn', 'VisibilityMPH', 'Wind Direction',
                   'Wind SpeedMPH',  'PrecipitationIn', 'Conditions','Snow',
                  'WindDirDegrees']
            tmp=tmp[col]
            tmp.columns=newcol
            tmp=tmp.head(1)
            tmp['DateUTC']=date
            tmp.index=[hour]
            l.append(tmp)
            newdate=date[:10]
        df=pd.concat(l) #all the weather info are concatenated in a single dataframe
        df=df.convert_objects(convert_dates='coerce')
        return df
    


# In[ ]:

p=pd.read_csv('Stationary_data_with_weather.csv')
colnames=['Latitude', 'Longitude', 'Value', 'ID', 'Height',
       'Loader ID','Sensor', 'Distance',
      'TemperatureF', 'Dew PointF', 'Humidity', 'Sea Level PressureIn',
       'VisibilityMPH', 'Wind SpeedMPH', 'PrecipitationIn', 'Conditions',
       'WindDirDegrees', 'Captured Time']
p.columns=colnames


# In[ ]:

#r=p[p.Sensor==1721]
params={}
bestinput='Dataset'
percentage=50
start=time.time()
li=p.Sensor.unique()
li.sort()
for n in li:
    print(n)
    model=None
    r=p[p.Sensor==n]
    if len(r)<500:
        continue
        
    model=Model(r)
    model=model.remove_outliers()
    model=model.prepareDataset(n=4,w=0,l=1)
    model=model.getInput()
    model=model.getOutput()
    if bestinput!='Dataset':
        model=model.reduceDataset(method=bestinput,nr=10)
    model=model.applyOnInputs(inp=bestinput,method='standardize',percentage=percentage)
    model=model.applyOnOutput(method='movingaverage',window=4)
    model=model.applyOnOutput(method='standardize',percentage=percentage)

    X=model.ModelInputs[bestinput] #input dataset
    samples=int(percentage*len(X)/100) #evaluating the samples number given the percentage
    x=X[:samples,0:] #training input set
    y = model.ModelOutput[:samples] #training output set
    test_x=X[samples:,:] #testing input set
    test_y=model.ModelOutput[samples:]
    scores = ['precision', 'recall']

    svr = GridSearchCV(SVR(),
                       param_grid={"C": 2**np.arange(1,14),
                                   "gamma": np.logspace(-2, 2, 10),
                                    "epsilon" : [0, 0.01, 0.1, 0.5]})
    svr.fit(x, y)
    params[n]=svr.best_params_

print str(time.time()-start) + 's to complete the evaluation of the best parameters'


# In[4]:

2**np.arange(1,14)


# In[6]:

list(np.logspace(-2, 2, 10))

