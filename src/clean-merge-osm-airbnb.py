
"""


How to use it:
    - spark-submit 
"""

import numpy as np
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row

nodes_schema = types.StructType([
    types.StructField('id', types.LongType(), nullable = False), 
    types.StructField('timestamp', types.StringType(), nullable = False), 
    types.StructField('lat', types.DoubleType(), nullable = False), 
    types.StructField('lon', types.DoubleType(), nullable = False), 
    types.StructField('tags', types.MapType(types.StringType(), types.StringType()), nullable = False)
])

def main(inputs, output):

    # Load Airbnb data
    airbnb_data = spark.read.option("multiline", "true") \
        .option("quote", '"') \
        .option("header", "true") \
        .option("escape", "\\") \
        .option("escape", '"').csv('../datasets/listings.csv')

    # Latitude and longitude of listings
    listing_locations = airbnb_data.select(
        airbnb_data['id'].alias('listing_id'), 
        airbnb_data['latitude'].cast(types.DoubleType()).alias('a_lat'), 
        airbnb_data['longitude'].cast(types.DoubleType()).alias('a_lon')
    ).cache()

    # Load osm data
    osm_data = spark.read.json(inputs, schema = nodes_schema)

    # Add columns that are related to public transportation
    osm_data = osm_data.withColumn('public_transport', osm_data['tags']['public_transport'])
    osm_data = osm_data.withColumn('bus', osm_data['tags']['bus'])
    osm_data = osm_data.withColumn('subway', osm_data['tags']['subway'])

    # Extract 'shop' tag from the 'tags' column
    # Focus on 'mall', 'convenience', 'supermarket'
    osm_data = osm_data.withColumn('shop', osm_data['tags']['shop'])

    # Extract 'amenity' tag from the 'tags' column
    osm_data = osm_data.withColumn('amenity', osm_data['tags']['shop'])

    # Keep only the features that are needed
    osm_data = osm_data[[
        'id', 'lat', 'lon', 
        'public_transport', 'bus', 'subway', 
        'shop', 'amenity'
    ]]
    osm_data.cache()

    # Extract all subway stations
    subway_stations = osm_data.filter(
        (osm_data['public_transport'] == 'station') \
        & (osm_data['subway'] == 'yes')
    )

    # Extract all bus stops
    bus_stations = osm_data.filter(
        (osm_data['public_transport'] == 'platform') \
        & (osm_data['bus'] == 'yes')
    )

    # Take the cross product of the two tables
    osm_data.createOrReplaceTempView("OSM")
    listing_locations.createOrReplaceTempView("Listings")
    cross_prod = spark.sql("""
        SELECT *
        FROM OSM AS O, Listings AS L
    """).cache()

    # Compute distance between two lat-lon pairs
    # Reference: https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters
    @functions.udf(returnType = types.DoubleType())
    def deg2Rad(deg):
        return deg * (np.pi / 180)

    R = 6378137     # Radius of Earth in meters

    distances = cross_prod.select(
        cross_prod['id'], 
        cross_prod['listing_id'], 
        cross_prod['lat'].alias('lat_1'), 
        cross_prod['lon'].alias('lon_1'), 
        cross_prod['a_lat'].alias('lat_2'), 
        cross_prod['a_lon'].alias('lon_2'), 
        (deg2Rad(cross_prod['lat']) - deg2Rad(cross_prod['a_lat'])).alias('dlat'), 
        (deg2Rad(cross_prod['lon']) - deg2Rad(cross_prod['a_lon'])).alias('dlon')
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
    distances[['id', 'listing_id', 'dist']].show()

    # df = df.coalesce(1)
    # pandas_df = df.toPandas()
    # pandas_df.to_json(
    #     output_filename, 
    #     orient = 'records', 
    #     lines = True
    # )

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('Clean and Merge OSM').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)


