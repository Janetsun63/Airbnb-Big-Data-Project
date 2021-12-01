#!/usr/bin/env python3
##author: Ziyue Cheng
#runuing in terminal as: spark-submit feature_occupancy_proce.py calendar_4-12.csv month out_dir

#this file will return 2 things, the future occupancy for each id and monthly price of the monthly calendar
#As Known as the prices of airBnB will change frequencyly due to difference reason
#So in this file will calculate montly price by the newerly update calendar

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import sys,os,uuid,gzip,re
from pyspark.sql.functions import lit

from pyspark.sql import SparkSession, functions, types
from datetime import datetime

# add more functions as necessary
@functions.udf(returnType=types.IntegerType())
def is_avaliable(available):
	if available is not None:
		if available =='f':
			return 1
		else:
			return 0
	else:
		return 0

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
	
	
def main(inputs, cal_month, output):
	# main logic starts here
	# calculate the future occupancy by listing Id
	calendar = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(inputs).cache()
	d1 = calendar.select(calendar['listing_id'], is_avaliable(calendar['available']).alias('available')).cache()
	count_df = d1.groupBy('listing_id').agg(functions.count('available').alias('count')).cache()		
	sum_df = d1.groupBy('listing_id').agg(functions.sum('available').alias('sum')).orderBy("listing_id")
	d2 = sum_df.join(count_df,'listing_id').cache()
	df = d2.withColumn("occupancy",functions.round(d2['sum']/d2['count'],3 ))
	df_id =  df.select(df['listing_id'], df['occupancy'])
	df_id.coalesce(1).write.option("header", "true").csv(output+ '/occumpancy')
	
	## calculate the future occupancy by monthly
	
	d3 =calendar.select(calendar['listing_id'], return_month(calendar['date']).alias('month'), get_price('price').alias('price'))
	d4 = d3.where(d3['month'] == cal_month)
	count_d4 = d4.groupBy('listing_id').agg(functions.count(d4['price']).alias('count')).cache()		
	sum_d4 = d4.groupBy('listing_id').agg(functions.sum(d4['price']).alias('sum')).cache()
	d5 = sum_d4.join(count_d4,'listing_id').cache()
	df = d5.withColumn("month_price",functions.round(d5['sum']/d5['count'],3 ))
	
	df = df.withColumn("month",lit(cal_month))
	count_df = df.groupBy('month').agg(functions.count(df['month_price']).alias('count')).cache()		
	sum_df = df.groupBy('month').agg(functions.sum(df['month_price']).alias('sum')).cache()
	df = sum_df.join(count_df,'month').cache()
	df_month = df.withColumn("month_price",functions.round(df['sum']/df['count'],3 ))
	df_month =  df_month.select(df_month['month'], df_month['month_price'])
	price = df_month.select('month_price').collect()[0][0]
	print('monthly price of '+ cal_month + " is: " + str(price))
	
if __name__ == '__main__':
	inputs = sys.argv[1]
	cal_month = sys.argv[2]
	output = sys.argv[3]
	spark = SparkSession.builder.appName('example code').getOrCreate()
	assert spark.version >= '3.0' # make sure we have Spark 3.0+
	spark.sparkContext.setLogLevel('WARN')
	sc = spark.sparkContext
	main(inputs, cal_month, output)