# the code is to convert amenities to amenities_value by counting the items in the amenities column;
# convert price column from string type to float
# input: listings.csv
# output: amenities_output.csv

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions as f, types

@f.udf(returnType=types.IntegerType())
def amenities_value(lines):
    ls = lines.replace("[","").replace("]","").split(", ")
    return len(ls)

@f.udf(returnType=types.DoubleType())
def price_value(price):
    return float (price.replace("$","").replace(",",""))

def main(inputs):

    airbnb = spark.read.option("multiline", "true")\
        .option("quote", '"')\
        .option("header", "true")\
        .option("escape", "\\")\
        .option("escape", '"').csv(inputs)
    
    df = airbnb.select('id','amenities','price').cache()
   
    # to calculate amenities measurement value by count; convert price from string to float
    df1 = df.withColumn('amenities_value', amenities_value(df['amenities'])).withColumn('price_value', price_value(df['price']))
    # select id, amenities value, price (float)
    df2 = df1.select('id','amenities_value','price_value')
    # output data
    df2.write.csv('amenities_output.csv', mode='overwrite')

if __name__ == '__main__':
    inputs = sys.argv[1]
    # output = sys.argv[2]
    spark = SparkSession.builder.appName('amenities data').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs)