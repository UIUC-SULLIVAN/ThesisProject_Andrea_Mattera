#!/usr/bin/python2
l=[]
tID=0
fmt="%Y-%m-%d %H:%M:%S"
s='(2015-02-10 07:24:18,42.81229,-70.87251,42.0,cpm,,,,,2015-02-10 07:24:18.769190,2015-02-10 07:24:18,2015)'
import datetime as dt
import sys

def format_string(line):
    '''The function extracts all the field from the string line and fix the formattation
    
    '''
    line=line.split(',')
    lat=round(float(line[1]),5)
    lon=round(float(line[2]),5)
    capt,val,unit=(line[0],line[3],line[4].strip())
    try:
        capt=dt.datetime.strptime(capt,fmt)
    except ValueError:
        #some dates have a different format with microsecond,
        #here they are converted
        capt=capt[:19]
        capt=dt.datetime.strptime(capt,fmt)
    
    return capt,lat,lon,val,unit

for line in sys.stdin:
    if line.startswith('C'): #the first line with columns names is skipped
        continue
    if not line:
        continue

    capt,lat,lon,val,unit=format_string(line)

    #cleaning
    if capt>dt.datetime.now():
        continue 
   # if unit.strip()!='cpm':
       # continue
    if not l: #if the list is empty
        l.append(capt)
        continue
    #print (capt-l[0]).total_seconds()
    tmp=capt-l[0]
    tmp=tmp.seconds
    if tmp>30*60:
        tID+=1
    l[0]=capt
    print '%s\t,%s' %(tID,','.join(line.split(',')))
