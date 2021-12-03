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
	ls = lines.replace("[","").replace("]","").split(", ")
	return len(ls)

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
	d2 = d1.withColumn('amenities', amenities_value(d1['amenities'])).withColumn('baths', bath_count(airbnb['bathrooms_text'])).withColumn('occupancy_rate', get_occupancy(airbnb['availability_365'])).withColumn('price',get_price(airbnb['price'])).withColumn('days', yeartoday(airbnb['host_since']))
	df = d2.select('id','accommodates', 'baths','beds','days','occupancy_rate','price','amenities').cache()
	df = osm.join(df, 'id')
	room_type = d1.select('id', 'room_type')\
		.withColumn('t1',lit('Entire'))\
		.withColumn('t2',lit('Private'))\
		.withColumn('t3',lit('Shared'))\
		.withColumn('t4',lit('Hotel'))
	room_bional = room_type.select("id", \
		is_have('room_type', 't1').alias('room_Entire'),\
		is_have('room_type', 't2').alias('room_Private'),\
		is_have('room_type', 't3').alias('room_Shared'),\
		is_have('room_type', 't4').alias('room_Hotel'),\
	)
	df = df.join(room_bional,'id').cache()
	neigh_type = d1.select('id','neighbourhood_cleansed')\
		.withColumn('Arbutus Ridge',lit('Arbutus Ridge'))\
		.withColumn('Downtown',lit('Downtown'))\
		.withColumn('Downtown Eastside',lit('Downtown Eastside'))\
		.withColumn('Dunbar Southlands',lit('Dunbar Southlands'))\
		.withColumn('Fairview',lit('Fairview'))\
		.withColumn('Grandview-Woodland',lit('Grandview-Woodland'))\
		.withColumn('Hastings-Sunrise',lit('Hastings-Sunrise'))\
		.withColumn('Kensington-Cedar Cottage',lit('Kensington-Cedar Cottage'))\
		.withColumn('Kerrisdale',lit('Kerrisdale'))\
		.withColumn('Killarney',lit('Killarney'))\
		.withColumn('Kitsilano',lit('Kitsilano'))\
		.withColumn('Marpole',lit('Marpole'))\
		.withColumn('Mount Pleasant',lit('Mount Pleasant'))\
		.withColumn('Oakridge',lit('Oakridge'))\
		.withColumn('Renfrew-Collingwood',lit('Renfrew-Collingwood'))\
		.withColumn('Riley Park',lit('Riley Park'))\
		.withColumn('Shaughnessy',lit('Shaughnessy'))\
		.withColumn('South Cambie',lit('South Cambie'))\
		.withColumn('Strathcona',lit('Strathcona'))\
		.withColumn('Sunset',lit('Sunset'))\
		.withColumn('Victoria-Fraserview',lit('Victoria-Fraserview'))\
		.withColumn('West End',lit('West End'))\
		.withColumn('West Point Grey',lit('West Point Grey'))
	neigh_binoal = neigh_type.select('id',\
		is_have('neighbourhood_cleansed', 'Arbutus Ridge').alias('Arbutus Ridge'),\
		is_have('neighbourhood_cleansed', 'Downtown').alias('Downtown'),\
		is_have('neighbourhood_cleansed', 'Downtown Eastside').alias('Downtown Eastside'),\
		is_have('neighbourhood_cleansed', 'Dunbar Southlands').alias('Dunbar Southlands'),\
		is_have('neighbourhood_cleansed', 'Fairview').alias('Fairview'),\
		is_have('neighbourhood_cleansed', 'Grandview-Woodland').alias('Grandview-Woodland'),\
		is_have('neighbourhood_cleansed', 'Hastings-Sunrise').alias('Hastings-Sunrise'),\
		is_have('neighbourhood_cleansed', 'Kensington-Cedar Cottage').alias('Kensington-Cedar Cottage'),\
		is_have('neighbourhood_cleansed', 'Kerrisdale').alias('Kerrisdale'),\
		is_have('neighbourhood_cleansed', 'Killarney').alias('Killarney'),\
		is_have('neighbourhood_cleansed', 'Kitsilano').alias('Kitsilano'),\
		is_have('neighbourhood_cleansed', 'Marpole').alias('Marpole'),\
		is_have('neighbourhood_cleansed', 'Mount Pleasant').alias('Mount Pleasant'),\
		is_have('neighbourhood_cleansed', 'Oakridge').alias('Oakridge'),\
		is_have('neighbourhood_cleansed', 'Renfrew-Collingwood').alias('Renfrew-Collingwood'),\
		is_have('neighbourhood_cleansed', 'Riley Park').alias('Riley Park'),\
		is_have('neighbourhood_cleansed', 'Shaughnessy').alias('Shaughnessy'),\
		is_have('neighbourhood_cleansed', 'South Cambie').alias('South Cambie'),\
		is_have('neighbourhood_cleansed', 'Strathcona').alias('Strathcona'),\
		is_have('neighbourhood_cleansed', 'Sunset').alias('Sunset'),\
		is_have('neighbourhood_cleansed', 'Victoria-Fraserview').alias('Victoria-Fraserview'),\
		is_have('neighbourhood_cleansed', 'West End').alias('West End'),\
		is_have('neighbourhood_cleansed', 'West Point Grey').alias('West Point Grey'),\
	)
	df = df.join(neigh_binoal,'id')
	
	
	c1 = calendar.select(calendar['id'], is_avaliable(calendar['available']).alias('available')).cache()
	feature_occupancy = c1.groupBy('id').agg(functions.avg(c1['available']).alias('feature_occupancy')).orderBy('id')
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