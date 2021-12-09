#!/usr/bin/env python3
#author: Ziyue Cheng
#runuing in terminal as: spark-submit listing_data_clearing.py listings.csv distance out_dir
#distance: is the output file after running osm_04_extract_features.py

#This code working on create new numerical table to do prediction
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import sys,os,uuid,gzip,re,math

from pyspark.sql import SparkSession, functions, types
from datetime import datetime
from pyspark.sql.functions import lit

# add more functions as necessary
#cover datetime to how many days from the first day to 11/6/2021
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

#will return next month occupancy 
@functions.udf(returnType=types.FloatType())
def get_occupancy(availability_30):
	days = int(availability_30)
	occupancy = (30 - days)/30
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
	
# from amenities_analysis.py we find the most popular tags in the amenities column, so we will count the number of each listing that how many amenities they have in the top 20 lists.
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

#fixed missing value, if beds column is none, we will return the ceil half of accommodates number.
@functions.udf(returnType=types.IntegerType())
def is_bed_none(beds, acc):
	if beds is not None:
		beds= int(beds)
	else:
		acc = int(acc)
		beds = int(math.ceil(acc/2))
	return beds
	
	
@functions.udf(returnType=types.StringType())
def return_month(date):
	date = datetime.strptime(date, "%Y-%m-%d")
	month = str(date.month)
	return month

def main(air_inputs, osm_inputs, output):
	# main logic starts here
	airbnb = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(air_inputs).repartition(40)
	
	osm = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(osm_inputs)
	osm = osm.withColumnRenamed("listing_id","id").repartition(40)
	
			
	d1= airbnb.select('id', 'latitude','longitude','host_since','neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms_text','beds', 'amenities','price', 'availability_30' )
	d2 = d1.withColumn('amenities', amenities_value(airbnb['amenities'])).withColumn('baths', bath_count(airbnb['bathrooms_text'])).withColumn('beds', is_bed_none(airbnb['beds'],airbnb['accommodates'])).withColumn('price',get_price(airbnb['price'])).withColumn('days', yeartoday(airbnb['host_since'])).withColumn('future_occupancy', get_occupancy(airbnb['availability_30']))
	df = d2.select('id','neighbourhood_cleansed','room_type','accommodates', 'baths','beds','days','amenities','price','future_occupancy').cache()
	df = osm.join(df, 'id')
	area = d2.select('id','latitude','longitude')
	df = area.join(df, 'id')
	
	
	df.coalesce(1).write.option("header", "true").csv(output)
	
if __name__ == '__main__':
	air_inputs = sys.argv[1]
	
	osm_inputs = sys.argv[2]
	
	output = sys.argv[3]
	spark = SparkSession.builder.appName('airbnb data clearing').getOrCreate()
	assert spark.version >= '3.0' # make sure we have Spark 3.0+
	spark.sparkContext.setLogLevel('WARN')
	sc = spark.sparkContext
	main(air_inputs, osm_inputs, output)
