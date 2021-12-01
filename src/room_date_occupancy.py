#author: Ziyue Cheng
#runuing in terminal as: spark-submit room_date_occupancy.py listings.csv out_dir

#Clearning listing data on the following columns
#host_since: return the days from the room published until 11/6/2021, which is the time we download this dataset
#room_type: 1: entire, 2: private room, 3:shared room 
#bathrooms_text: return the numbers of baths if the bath is shared, we divided the total number by 2. And if it is missing data, we using the median data 1.
#price: return it as float
#prpperty_type: 1: condo 2: unit 3:suit 4: home 5: loft 6: other
#availability_365: return (365 - availability_365)/360

#Then it will return the table of above clarning data, and table of count & average price of above features


import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import sys,os,uuid,gzip,re

from pyspark.sql import SparkSession, functions, types
from datetime import datetime

# add more functions as necessary
@functions.udf(returnType=types.IntegerType())
def yeartoday(host_since):
    if host_since is not None:
        start_date = datetime.strptime(host_since, "%Y-%m-%d")
        end_date = datetime.strptime('11/6/2021', "%m/%d/%Y")
        return (end_date-start_date).days
    else:
        return 0

#this function to check the room tpye 
@functions.udf(returnType=types.IntegerType())
def is_entire(property_type):
    if property_type.find("Entire") >=0:
        pro_type =1
    elif property_type.find("Private") >=0:
        pro_type = 2
    else:
        pro_type =3
    return pro_type

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


#this function to covert the price from string to float
@functions.udf(returnType=types.FloatType())
def get_price(price):
    price = price.replace(',', '')
    price = price[1:]
    return float(price)

#this function is to check the prpperty. 1: condo 2: unit 3:suit 4: home 5: loft 6: other
@functions.udf(returnType=types.IntegerType())
def prop_type(property_type):
    if property_type.find("condo") >=0:
        type =1
    elif property_type.find("unit") >=0:
        type = 2
    elif property_type.find("suit") >=0:
        type =3
    elif property_type.find("home") >=0:
        type =4
    elif property_type.find("loft") >=0:
        type = 5
    else:
        type = 6
    return type

@functions.udf(returnType=types.FloatType())
def get_occupancy(availability_365):
    days = int(availability_365)
    occupancy = (365 - days)/365
    occupancy = round(occupancy,3)
    return occupancy

def main(inputs, output):
    # main logic starts here
    airbnb = spark.read.option("multiline", "true")\
        .option("quote", '"')\
        .option("header", "true")\
        .option("escape", "\\")\
        .option("escape", '"').csv(inputs)
    d1 = airbnb.select(airbnb['id'],airbnb['host_since'],airbnb['property_type'],airbnb['room_type'],airbnb['accommodates'],airbnb['bathrooms_text'],airbnb['beds'],airbnb['availability_365'], airbnb['price']).cache()
    #check the room type if is entire return 1 if is private return 0
    d2 = d1.withColumn('room_type',is_entire(d1['room_type']))
    d3 = d2.withColumn('occupancy_rate', get_occupancy(airbnb['availability_365']))
    
        
    #cover price from string to float
    d4 = d3.withColumn('price',get_price(airbnb['price']))
    d5 = d4.withColumn('property', prop_type(airbnb['property_type']))
    d5 = d5.withColumn('days', yeartoday(airbnb['host_since']))    
    
    df = d5.select(d5['id'], d5['property'],d5['accommodates'], d5['beds'],d5['days'], d5['room_type'], d5['price'], d5['occupancy_rate'],bath_count(airbnb['bathrooms_text']).alias('baths'))
    
    df.coalesce(1).write.option("header", "true").csv(output+ '/full_table')
    
    neighborhood_overview = airbnb.select(airbnb['neighborhood_overview'])
    neighborhood_overview.coalesce(1).write.option("header", "true").csv(output+ '/neighborhood_overview')
    
    property = df.groupBy('property').agg(functions.count(df['price']).alias('count'),functions.avg(df['price']).alias('average')).orderBy('property')
    property.coalesce(1).write.option("header", "true").csv(output+ '/property')
    
    accommodates = df.groupBy('accommodates').agg(functions.count(df['price']).alias('count'),functions.avg(df['price']).alias('average')).orderBy('accommodates')
    accommodates.coalesce(1).write.option("header", "true").csv(output+ '/accommodates')
    
    beds = df.groupBy('beds').agg(functions.count(df['price']).alias('count'),functions.avg(df['price']).alias('average')).orderBy('beds')
    beds.coalesce(1).write.option("header", "true").csv(output+ '/beds')
    
    room_type = df.groupBy('room_type').agg(functions.count(df['price']).alias('count'),functions.avg(df['price']).alias('average')).orderBy('room_type')
    room_type.coalesce(1).write.option("header", "true").csv(output+ '/room_type')
    
    baths = df.groupBy('baths').agg(functions.count(df['price']).alias('count'),functions.avg(df['price']).alias('average')).orderBy('baths')
    baths.coalesce(1).write.option("header", "true").csv(output+ '/baths')
    
    
    
if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('airbnb data clearning').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)