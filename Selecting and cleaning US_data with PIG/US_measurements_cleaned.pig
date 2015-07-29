/* Script to load all the measurements and extract us measurements*/
sensors = LOAD 'measurements.csv' USING PigStorage(',') AS (date:chararray,lat:float,lon:float,value:int,unit:chararray,loc:chararray,device:chararray,md5,hei,surf,rad,upl,loader:chararray) ;
us = filter sensors by lat <50 and lat >30 and lon <-60 and lon >-1130 ;
--store us using PigStorage(',') ;
--if a test dataset is required with less measurements 
/*
inpt = load '......' ......;
user_grp = GROUP inpt BY $0;
filtered = FOREACH user_grp {
      top_rec = LIMIT inpt 1;
      GENERATE FLATTEN(top_rec);
};
 it is needed a step to remove duplicates because they create problem during the clustering

EDIT:solved in the next step of the process
*/
converted = foreach us generate lat,lon,value,loc,device,md5,hei,surf,rad,upl,loader,SUBSTRING(date, 0,19) as date,TRIM(unit) as unit ; --trim returns a copy of a string with leading and trailing white space removed. Substring fix incorrect dates with milliseconds
--while todate converts the chararray date in datetime 
cleaned= foreach converted generate date,lat,lon,value,unit,loc,device,md5,hei,surf,rad,upl,loader,ToDate(date, 'YYYY-MM-dd HH:mm:ss') as (date1:datetime);
cleaned1= filter cleaned by date1< CurrentTime() ; --all the dates that are > than the current date are dropped
cleaned2 = filter cleaned1 by unit == 'cpm'; --all the measurements not expressed in cpm are dropped (too much measurements were dropped)
cleaned3 = foreach cleaned generate date,lat,lon,value,unit,loc,device,md5,hei,surf,rad,upl,loader; 
STORE cleaned3 INTO 'usoutput' using PigStorage(',') ;
--hadoop fs -getmerge /user/input/usoutput/ US_cleaned.csv to merge the part in the output directory


