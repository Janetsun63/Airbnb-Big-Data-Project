#!/usr/bin/env python3

##author: Ziyue Cheng
#runuing in terminal as:weather.py weatherstats_vancouver_daily.csv out_dir

#this file will return the average rain and temperature by month
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import sys,os,uuid,gzip,re
from pyspark.sql.functions import lit

from pyspark.sql import SparkSession, functions, types
from datetime import datetime


	
@functions.udf(returnType=types.IntegerType())
def return_month(date):
	date = datetime.strptime(date, "%Y-%m-%d")
	month = int(date.month)
	return month

@functions.udf(returnType=types.IntegerType())
def return_year(date):
	date = datetime.strptime(date, "%Y-%m-%d")
	year = int(date.year)
	return year


@functions.udf(returnType=types.FloatType())
def to_int(num):
	if num is not None:
		return float(num)
	
def main(inputs, output):
	# main logic starts here
	# calculate the future occupancy by listing Id
	weather = spark.read.option("multiline", "true")\
		.option("quote", '"')\
		.option("header", "true")\
		.option("escape", "\\")\
		.option("escape", '"').csv(inputs).cache()
	
	d1 = weather.select(weather['date'], to_int(weather['avg_temperature']).alias('tem'),to_int(weather['rain']).alias('rain')).withColumn('month', return_month('date')).withColumn('year', return_year('date')).cache()
	
	count_df = d1.groupBy('month').agg(functions.count('tem').alias('count')).cache()	
	sum_df = d1.groupBy('month').agg(functions.sum('rain').alias('sum_rain'),functions.sum('tem').alias('sum_tem'))
	d2 = sum_df.join(count_df,'month').cache()
	df = d2.withColumn("rain",functions.round(d2['sum_rain']/d2['count'],1 )).withColumn("temp",functions.round(d2['sum_tem']/d2['count'],1 ))
	df = df.select('month','rain','temp').orderBy('month')
	df.coalesce(1).write.option("header", "true").csv(output)
	
if __name__ == '__main__':
	inputs = sys.argv[1]
	output = sys.argv[2]
	spark = SparkSession.builder.appName('example code').getOrCreate()
	assert spark.version >= '3.0' # make sure we have Spark 3.0+
	spark.sparkContext.setLogLevel('WARN')
	sc = spark.sparkContext
	main(inputs, output)