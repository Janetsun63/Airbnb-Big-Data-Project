
"""
Calculate the distances between Airbnb listings and the amenities 
in the city. New features are created after this step (e.g., the 
number of subway stations within a 2km radius of an Airbnb listing).

Input:
    - OSM-feature dataset from `osm_04_extract_features.py` 
    - Airbnb raw dataset (e.g., `listings.csv`)
Output:
    - CSV table containing `listing id` and some distance-related features

How to use it:
    spark-submit ./src/calc_osm_airbnb_distances.py [OSM-feature dataset] [Airbnb raw dataset] [Output folder]

Example:
    spark-submit ./src/calc_osm_airbnb_distances.py ./datasets/osm_features/ ./datasets/listings.csv ./datasets/distances
"""

import numpy as np
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row

osm_schema = types.StructType([
    types.StructField('osm_id', types.LongType(), nullable = False), 
    types.StructField('lat', types.DoubleType(), nullable = False), 
    types.StructField('lon', types.DoubleType(), nullable = False), 
    types.StructField('category', types.StringType(), nullable = False), 
    types.StructField('name', types.StringType(), nullable = False)
])

# Number of partitions
partition_num = 40

def main(osm_input, airbnb_input, output):

    # Load osm data
    osm_data = spark.read.json(osm_input, schema = osm_schema).repartition(partition_num)

    # Latitude and longitude of OSM data points
    osm_locations = osm_data.select(
        osm_data['osm_id'], 
        osm_data['category'], 
        osm_data['lat'].alias('lat_2'), 
        osm_data['lon'].alias('lon_2')
    )

    # Load Airbnb data
    airbnb_data = spark.read.option("multiline", "true") \
        .option("quote", '"') \
        .option("header", "true") \
        .option("escape", "\\") \
        .option("escape", '"') \
        .csv(airbnb_input).repartition(partition_num)

    # Latitude and longitude of listings
    listing_locations = airbnb_data.select(
        airbnb_data['id'].alias('listing_id'), 
        airbnb_data['latitude'].cast(types.DoubleType()).alias('lat_1'), 
        airbnb_data['longitude'].cast(types.DoubleType()).alias('lon_1')
    )


    # Take the cross product of the two location tables
    listing_locations.createOrReplaceTempView("Listings")
    osm_locations.createOrReplaceTempView("OSM")
    cross_prod = spark.sql("SELECT * FROM OSM, Listings")

    """
    Compute distance between any two lat-lon pairs
    Reference: https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
    """

    # An UDF that converts degrees to radians
    @functions.udf(returnType = types.DoubleType())
    def deg2Rad(deg):
        return deg * (np.pi / 180)

    # Radius of Earth in meters
    R = 6378137

    distances = cross_prod.select(
        cross_prod['osm_id'], 
        cross_prod['category'], 
        cross_prod['listing_id'], 
        cross_prod['lat_1'], 
        cross_prod['lon_1'], 
        cross_prod['lat_2'], 
        cross_prod['lon_2'], 
        (deg2Rad(cross_prod['lat_1']) - deg2Rad(cross_prod['lat_2'])).alias('dlat'), 
        (deg2Rad(cross_prod['lon_1']) - deg2Rad(cross_prod['lon_2'])).alias('dlon')
    )

    distances = distances.withColumn(
        'a', 
        functions.sin(distances['dlat'] / 2) ** 2 \
        + functions.cos(deg2Rad(distances['lat_1'])) \
            * functions.cos(deg2Rad(distances['lat_2'])) \
            * functions.sin(distances['dlon'] / 2) ** 2
    )

    distances = distances.withColumn(
        'dist', 
        2 * R * functions.atan2(
            functions.sqrt(distances['a']), 
            functions.sqrt(1 - distances['a'])
        )
    )

    distances = distances[['osm_id', 'listing_id', 'category', 'dist']]
    distances.cache()

    num_bus_2km = distances.select('listing_id', 'dist') \
        .where((distances['category'] == 'bus') & (distances['dist'] <= 2000.0)) \
        .groupby('listing_id').agg(functions.count('dist').alias('num_bus_2km')).cache()

    num_subway_2km = distances.select('listing_id', 'dist') \
        .where((distances['category'] == 'subway') & (distances['dist'] <= 2000.0)) \
        .groupby('listing_id').agg(functions.count('dist').alias('num_subway_2km')).cache()

    num_shop_5km = distances.select('listing_id', 'dist') \
        .where((distances['category'] == 'shopping') & (distances['dist'] <= 5000.0)) \
        .groupby('listing_id').agg(functions.count('dist').alias('num_shop_2km')).cache()

    nearest_shop = distances.select('listing_id', 'dist') \
        .where(distances['category'] == 'shopping') \
        .groupby('listing_id').agg(functions.min('dist').alias('nearest_shop(m)')).cache()

    num_restaurant_2km = distances.select('listing_id', 'dist') \
        .where((distances['category'] == 'restaurant') & (distances['dist'] <= 2000.0)) \
        .groupby('listing_id').agg(functions.count('dist').alias('num_restaurant_2km')).cache()

    # Result table
    res_table = num_bus_2km.join(num_subway_2km, on = ['listing_id']) \
        .join(num_shop_5km, on = ['listing_id']) \
        .join(nearest_shop, on = ['listing_id']) \
        .join(num_restaurant_2km, on = ['listing_id'])

    # Write result table to file
    res_table.write.csv(
        output, 
        mode = 'overwrite', 
        compression = 'gzip', 
        header = True
    )

if __name__ == '__main__':
    osm_input = sys.argv[1]
    airbnb_input = sys.argv[2]
    output = sys.argv[3]
    spark = SparkSession.builder.appName('Clean and Merge OSM').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(osm_input, airbnb_input, output)


