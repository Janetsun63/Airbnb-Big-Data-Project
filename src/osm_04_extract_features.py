
"""
Extract useful features from the tags in the extracted OSM data.
More specifically, this step will extract the records that are
related to one of the following: bus, subway, shops, and restaurants.

How to use it:
    spark-submit osm_04_extract_features.py [input folder/file] [output folder]

Example:
    spark-submit ./src/osm_04_extract_features.py ./datasets/osm_vancouver_data.json ./datasets/osm_features
"""

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql.functions import array


nodes_schema = types.StructType([
    types.StructField('id', types.LongType(), nullable = False), 
    types.StructField('timestamp', types.StringType(), nullable = False), 
    types.StructField('lat', types.DoubleType(), nullable = False), 
    types.StructField('lon', types.DoubleType(), nullable = False), 
    types.StructField('tags', types.MapType(types.StringType(), types.StringType()), nullable = False)
])

@functions.udf(returnType = types.StringType())
def categorize(arr):
    # arr: ['public_transport', 'bus', 'subway', 'shop', 'amenity']
    if arr[0] == 'station' and arr[2] == 'yes':
        return 'subway'
    elif arr[0] == 'platform' and arr[1] == 'yes':
        return 'bus'
    elif arr[3] == 'mall' \
            or arr[3] == 'convenience' \
            or arr[3] == 'supermarket':
        return 'shopping'
    elif arr[4] == 'restaurant' \
            or arr[4] == 'fast_food' \
            or arr[4] == 'food_court':
        return 'restaurant'
    else:
        return None

def main(inputs, output):

    # Load osm data
    osm_data = spark.read.json(inputs, schema = nodes_schema).repartition(40)

    """
    Extract tags that we are interested in from the 'tags' 
    column and convert them into some separate columns to 
    make the table easier to be filtered by Spark.
    """

    # Tags related to:
    #   - public transportation: 'public_transport', 'bus', 'subway'
    #   - Shopping mall: 'shop'
    #   - Restaurant: 'amenity'
    osm_data = osm_data.withColumn('osm_id', osm_data['id']) \
                    .withColumn('public_transport', osm_data['tags']['public_transport']) \
                    .withColumn('bus', osm_data['tags']['bus']) \
                    .withColumn('subway', osm_data['tags']['subway']) \
                    .withColumn('shop', osm_data['tags']['shop']) \
                    .withColumn('amenity', osm_data['tags']['amenity']) \
                    .withColumn('name', osm_data['tags']['name'])


    # Select only the rows that we are interested in.
    # Put them into 4 categories:
    #     - bus; subway; shopping; restaurant
    categorized = osm_data.withColumn(
        'category', 
        categorize(array(
            'public_transport', 'bus', 'subway', 
            'shop', 'amenity'
        ))
    )

    # Discard the rows that are not needed
    categorized = categorized.filter(categorized['category'].isNotNull())

    # Keep only the features that are needed
    categorized = categorized[[
        'osm_id', 'lat', 'lon', 'category', 'name'
    ]]

    # Write results to file
    categorized.write.json(
        output, 
        mode = 'overwrite', 
        compression = 'gzip'
    )

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('OSM processing').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)

