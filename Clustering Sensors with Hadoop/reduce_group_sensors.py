#!/usr/bin/env python2

from geopy.distance import vincenty
import sys
import datetime as dt



sens=0
l=[]
di={} #{number of sensor: ((lat,lon),captured time)}
fmt="%Y-%m-%d %H:%M:%S"
dist=None
old=0
spatial_threshold=1000
time_threshold=60*50
#filename='test.csv'
#g=open(filename)
for line in sys.stdin: 
    if not(line):
        continue
    if line.startswith('C'): #the first line with columns names is skipped
        continue
    line=line.split(',')
    if len(line)==1:
        continue
    tID=line[0]
    date=line[1]
    lat=line[2]
    lon=line[3]

    if tID!=old:
        old=tID
        di={}
    p=(lat,lon)
    date=dt.datetime.strptime(date,fmt)
    if len(di)==0:    #first iteration, the dictionary is initialized
        di[0]=(p,date)
        continue

    r=[(x[0],y) for x,y in zip(di.values(),di.keys()) if abs(x[1]-date).total_seconds()<time_threshold]
    tmp=[vincenty(p,c[0]).meters for c in r]
    try:
        m=min(tmp)
    except  ValueError: #there are no measurements in "time_threshold" mins after the previous one
        sens+=1  #another sensor is created and added to the dictionary
        line.append(sens)
        l.append(line)
        di[sens]=(p,date) 
        dist=0
        line.append(dist)
        #print '%s\t,%s' %(sens,','.join(str(p) for p in line[1:]))
        print ','.join(str(p).replace('\n','') for p in line[1:])
        r=tmp=[]
        continue
    index=tmp.index(m)
    index=r[index][1]
    
    if m<spatial_threshold: #if the measurements are not too far
            line.append(index)
            l.append(line)
            di[index]=(p,date)
            dist=m
            line.append(dist)
            print ','.join(str(p).replace('\n','') for p in line[1:])
    else: #else another sensor is defined and is last point and captured time inserted
            #in the dictionary
        sens+=1
        line.append(sens)
        l.append(line)
        di[sens]=(p,date)
        dist=0
        line.append(dist)
        print ','.join(str(p).replace('\n','')  for p in line[1:] ) 
    r=tmp=[]
    m=None
