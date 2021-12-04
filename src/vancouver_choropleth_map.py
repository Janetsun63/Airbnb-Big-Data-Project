
"""
Generate a choropleth map of average daily price in Vancouver.

Usage:
spark-submit ./src/vancouver_choropleth_map.py ./datasets/cleaned_listing.csv ./datasets/Vancouver_2021-11-06_neighbourhoods.geojson ./plots/
"""

import sys, folium
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from os import path

airbnb_cleaned_schema = types.StructType([
    types.StructField('id', types.LongType()), 
    types.StructField('latitude', types.DoubleType()), 
    types.StructField('longitude', types.DoubleType()), 
    types.StructField('num_bus_2km', types.IntegerType()), 
    types.StructField('num_subway_2km', types.IntegerType()), 
    types.StructField('num_shop_2km', types.IntegerType()), 
    types.StructField('nearest_shop(m)', types.FloatType()), 
    types.StructField('num_restaurant_2km', types.IntegerType()), 
    types.StructField('neighbourhood_cleansed', types.StringType()), 
    types.StructField('room_type', types.StringType()), 
    types.StructField('accommodates', types.IntegerType()), 
    types.StructField('baths', types.FloatType()), 
    types.StructField('beds', types.IntegerType()), 
    types.StructField('days', types.IntegerType()), 
    types.StructField('amenities', types.IntegerType()), 
    types.StructField('price', types.FloatType()), 
    types.StructField('future_occupancy', types.FloatType()), 
])

def main(airbnb, geojson, output_path):
    airbnb_data = spark.read.csv(
        airbnb, 
        header = True, 
        schema = airbnb_cleaned_schema
    )

    # Average Price
    avg_price = airbnb_data.groupby('neighbourhood_cleansed') \
        .agg(functions.mean('Price').alias('Price')) \
        .toPandas()
    print(avg_price)

    # Base map
    van_map = folium.Map([49.25, -123.1], zoom_start = 12)

    # Choropleth map
    folium.Choropleth(
        geo_data = geojson, 
        name = 'Choropleth Map', 
        data = avg_price, 
        columns = ['neighbourhood_cleansed', 'Price'],
        key_on = 'feature.properties.neighbourhood', 
        fill_color = 'YlOrRd', 
        fill_opacity = 0.7, 
        line_opacity = 0.5, 
        legend_name = 'Average Price (USD/day)'
    ).add_to(van_map)

    van_map.save(path.join(output_path, 'vancouver_choropleth_map.html'))

if __name__ == '__main__':
    airbnb = sys.argv[1]
    geojson = sys.argv[2]
    output_path = sys.argv[3]
    spark = SparkSession.builder.appName('OSM processing').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(airbnb, geojson, output_path)
