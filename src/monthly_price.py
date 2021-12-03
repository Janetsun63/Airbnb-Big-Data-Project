#!/usr/bin/env python3
##author: Ziyue Cheng
#runuing in terminal as: spark-submit monthly_price.py calendar_4-12.csv month

#this file will return 2 things, the future occupancy for each id and monthly price of the monthly calendar
#As Known as the prices of airBnB will change frequencyly due to difference reason
#So in this file will calculate montly price by the newerly update calendar

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import sys,os,uuid,gzip,re
from pyspark.sql.functions import lit

from pyspark.sql import SparkSession, functions, types
from datetime import datetime


@functions.udf(returnType=types.StringType())
def return_month(date):
	date = datetime.strptime(date, "%Y-%m-%d")
	month = str(date.month)
	return month

#this function to covert the price from string to float
@functions.udf(returnType=types.FloatType())
def get_price(price):
	if price is not None:
		price = price.replace(',', '')
		price = price[1:]
		return float(price)
	
	
def main(inputs, cal_month):
	# main logic starts here
	# calculate the future occupancy by listing Id
	calendar = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(inputs)
			
	d1 =calendar.select(calendar['listing_id'], return_month(calendar['date']).alias('month'), get_price('price').alias('price'))
	d2 = d1.where(d1['month'] == cal_month)
	id_price = d2.groupBy('listing_id').agg(functions.avg(d2['price']).alias('id_price')).cache()
	id_price = id_price.withColumn('month',lit(cal_month))
	month_price = id_price.groupBy('month').agg(functions.avg(id_price['id_price']).alias('price')).cache()
	
	price = month_price.select('price').collect()[0][0]
	
	print( cal_month + "   " + str(price))
	
if __name__ == '__main__':
	inputs = sys.argv[1]
	cal_month = sys.argv[2]

	spark = SparkSession.builder.appName('example code').getOrCreate()
	assert spark.version >= '3.0' # make sure we have Spark 3.0+
	spark.sparkContext.setLogLevel('WARN')
	sc = spark.sparkContext
	main(inputs, cal_month)