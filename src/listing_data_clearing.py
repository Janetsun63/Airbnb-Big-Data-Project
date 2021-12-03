#!/usr/bin/env python3
#author: Ziyue Cheng
#runuing in terminal as: spark-submit listing_data_clearing.py listings.csv calendar.csv distance out_dir
#distance: is the output file after running osm_04_extract_features.py

#This code working on create new numerical table to do prediction
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import sys,os,uuid,gzip,re

from pyspark.sql import SparkSession, functions, types
from datetime import datetime
from pyspark.sql.functions import lit

# add more functions as necessary
@functions.udf(returnType=types.IntegerType())
def yeartoday(host_since):
	if host_since is not None:
		start_date = datetime.strptime(host_since, "%Y-%m-%d")
		end_date = datetime.strptime('11/6/2021', "%m/%d/%Y")
		return (end_date-start_date).days
	else:
		return 0
	
#this function is returning the number of bathroom. If the bath is shared, the we devided the number by 2
@functions.udf(returnType=types.FloatType())
def bath_count(bathrooms_text):
	if bathrooms_text is not None:
		bathrooms_text =bathrooms_text.strip()
		list = bathrooms_text.split(' ')
		if list[0].replace('.','',1).isdigit():
			num = float(list[0].strip())
		else:
			num = 1.0
		if list[1].find('shared')  == -1:
			num = num
		else:
			num = num * 0.5
		return num
	else:
		return 1.0   

@functions.udf(returnType=types.FloatType())
def get_occupancy(availability_365):
	days = int(availability_365)
	occupancy = (365 - days)/365
	occupancy = round(occupancy,3)
	return occupancy
	
#this function to covert the price from string to float
@functions.udf(returnType=types.FloatType())
def get_price(price):
	price = price.replace(',', '')
	price = price[1:]
	return float(price)

@functions.udf(returnType=types.IntegerType())
def is_avaliable(available):
	if available is not None:
		if available =='f':
			return 1
		else:
			return 0
	else:
		return 0
	

@functions.udf(returnType=types.IntegerType())
def amenities_value(lines):
	ame = ['wifi' ,'alarm'   ,'kitchen'  ,'essentials'  ,'long term stays allowed' ,'hangers' ,'hair dryer'  ,'washer' ,'hot water'  ,'dryer' , 'iron' ,'shampoo' ,'dishes' ,'workspace' ,'refrigerator' ,'fire extinguisher'  ,'microwave' ,'coffee' ,'park' ,'tv' ]	
	lines = lines.lower()
	count = 0
	for i in ame:
		if lines.find(i)>=0:
			count +=1
	return count

@functions.udf(returnType=types.IntegerType())
def is_have(type, word):
	if type.find(word)>= 0:
		return 1
	else:
		return 0
	
@functions.udf(returnType=types.StringType())
def return_month(date):
	date = datetime.strptime(date, "%Y-%m-%d")
	month = str(date.month)
	return month

def main(air_inputs,cal_inputs, osm_inputs, output):
	# main logic starts here
	airbnb = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(air_inputs)
	
	osm = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(osm_inputs)
	osm = osm.withColumnRenamed("listing_id","id")
	
	calendar = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(cal_inputs).cache()
	calendar = calendar.withColumnRenamed("listing_id","id")
		
	d1= airbnb.select('id', 'host_since','neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms_text','beds', 'amenities','price', 'availability_365', )
	d2 = d1.withColumn('amenities', amenities_value(airbnb['amenities'])).withColumn('baths', bath_count(airbnb['bathrooms_text'])).withColumn('occupancy_rate', get_occupancy(airbnb['availability_365'])).withColumn('price',get_price(airbnb['price'])).withColumn('days', yeartoday(airbnb['host_since']))
	df = d2.select('id','neighbourhood_cleansed','room_type','accommodates', 'baths','beds','days','occupancy_rate','price','amenities').cache()
	df = osm.join(df, 'id')
		
	
	c1 = calendar.select(calendar['id'], is_avaliable(calendar['available']).alias('available')).cache()
	feature_occupancy = c1.groupBy('id').agg(functions.avg(c1['available']).alias('future_occupancy')).orderBy('id')
	df = df.join(feature_occupancy,'id')
	df.coalesce(1).write.option("header", "true").csv(output)
	
if __name__ == '__main__':
	air_inputs = sys.argv[1]
	cal_inputs = sys.argv[2]
	osm_inputs = sys.argv[3]
	
	output = sys.argv[4]
	spark = SparkSession.builder.appName('airbnb data clearing').getOrCreate()
	assert spark.version >= '3.0' # make sure we have Spark 3.0+
	spark.sparkContext.setLogLevel('WARN')
	sc = spark.sparkContext
	main(air_inputs,cal_inputs, osm_inputs, output)