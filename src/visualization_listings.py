#!/usr/bin/env python3
#author: Ziyue Cheng
#runuing in terminal as: spark-submit visualization_listings.py listings.csv out_dir

#this code is to create table for plot feature and price

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import sys,os,uuid,gzip,re

from pyspark.sql import SparkSession, functions, types
from datetime import datetime


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
	
	
#this function to covert the price from string to float
@functions.udf(returnType=types.FloatType())
def get_price(price):
	price = price.replace(',', '')
	price = price[1:]
	return float(price)


def main(inputs, output):
	# main logic starts here
	airbnb = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(inputs)
	d1 = airbnb.select(airbnb['id'],airbnb['room_type'],airbnb['accommodates'],airbnb['bathrooms_text'],airbnb['beds'], airbnb['neighbourhood_cleansed'] ,airbnb['price']).cache()
	
	d2 = d1.withColumn('price',get_price(airbnb['price'])).withColumn('baths', bath_count(airbnb['bathrooms_text']))
	
	property = d2.groupBy('room_type').agg(functions.count(d2['price']).alias('count'),functions.avg(d2['price']).alias('price')).orderBy('room_type')
	property.coalesce(1).write.option("header", "true").csv(output+ '/property')
	
	accommodates = d2.groupBy('accommodates').agg(functions.count(d2['price']).alias('count'),functions.avg(d2['price']).alias('price')).orderBy('accommodates')
	accommodates.coalesce(1).write.option("header", "true").csv(output+ '/accommodates')

	beds = d2.groupBy('beds').agg(functions.count(d2['price']).alias('count'),functions.avg(d2['price']).alias('price')).orderBy('beds')
	beds.coalesce(1).write.option("header", "true").csv(output+ '/beds')
	
	baths = d2.groupBy('baths').agg(functions.count(d2['price']).alias('count'),functions.avg(d2['price']).alias('price')).orderBy('baths')
	baths.coalesce(1).write.option("header", "true").csv(output+ '/baths')
	
	neighbourhood = d2.groupBy('neighbourhood_cleansed').agg(functions.count(d2['price']).alias('count'),functions.avg(d2['price']).alias('price')).orderBy('neighbourhood_cleansed')
	neighbourhood.coalesce(1).write.option("header", "true").csv(output+ '/neighbourhood')
if __name__ == '__main__':
	inputs = sys.argv[1]
	output = sys.argv[2]
	spark = SparkSession.builder.appName('airbnb data clearning').getOrCreate()
	assert spark.version >= '3.0' # make sure we have Spark 3.0+
	spark.sparkContext.setLogLevel('WARN')
	sc = spark.sparkContext
	main(inputs, output)