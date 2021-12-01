
"""
Combine the extracted osm data files into one json file. This
step is optional.

How to use it:
    - spark-submit osm_combiner.py ../datasets/osm_vancouver_data/ ../datasets/osm_vancouver_data.json
"""


import pandas as pd
from lxml import etree
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


def main(inputs, output_filename):
    df = spark.read.json(inputs, schema = nodes_schema)
    df = df.coalesce(1)
    pandas_df = df.toPandas()
    pandas_df.to_json(
        output_filename, 
        orient = 'records', 
        lines = True
    )

if __name__ == '__main__':
    inputs = sys.argv[1]
    output_filename = sys.argv[2]
    spark = SparkSession.builder.appName('Combine OSM').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output_filename)

