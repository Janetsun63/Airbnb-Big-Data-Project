# the code is to convert amenities to amenities_value by counting the items in the amenities column;
# convert price column from string type to float;
# generate wordcloud of amenities column;
# calculate correlation coefficient between price and amenities_value;
# plot scatter of price and amenities value;
# input: listings.csv
# output: amenities.jpg, wordcloud.jpg, r-square, amenities count dictionary

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from collections import Counter
from pyspark.sql import SparkSession, functions as f, types
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

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

    # to print amenities count dictionary
    amenities_full = df.select('amenities').rdd\
        .flatMap(lambda x: x[0].replace("[","").replace("]","").lower().split(", ")).coalesce(1)
    print(Counter(amenities_full.collect()))
    
    # to generate wordcloud
    df_amenities = spark.createDataFrame(amenities_full, types.StringType()).toPandas()
    text = " ".join(amenity for amenity in df_amenities.value)
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud.jpg')
    
    # to calculate amenities measurement value by count; convert price from string to float
    df1 = df.withColumn('amenities_value', amenities_value(df['amenities'])).withColumn('price_value', price_value(df['price']))
    # delete extremely high price (outliers)
    df2 = df1.select('amenities_value','price_value').where(df1['price_value']<1000)
    # df2.show(5)
    
    # to calculate correlation coefficient between price and amenities value
    r = df2.corr('amenities_value','price_value')
    print(r)
   
    df3 = df2.toPandas()
    # plot scatter of price and amenities value
    plt.figure(figsize=(15,10))
    plt.scatter(df3['amenities_value'],df3['price_value'])
    #plt.yticks(np.arange(min(df3['price_value']), max(df3['price_value'])+20, 500))
    plt.yticks(np.arange(0,1000,100))
    plt.xlabel("Amenities_count")
    plt.ylabel("Price")
    plt.savefig('amenities.jpg')
  
    # df1.write.csv(output, mode='overwrite')

if __name__ == '__main__':
    inputs = sys.argv[1]
    # output = sys.argv[2]
    spark = SparkSession.builder.appName('amenities analysis').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs)

    


