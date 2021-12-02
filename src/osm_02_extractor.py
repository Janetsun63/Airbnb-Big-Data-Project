
"""
Extract points of interest in the Greater Vancouver area from 
the splitted OSM dataset.

How to use it:
    spark-submit osm_02_extractor.py [input folder] [output folder]

Example:
    spark-submit osm_02_extractor.py /courses/datasets/openstreetmaps osm_vancouver_data

Note:
    - The dataset is also available on the computing cluster.
"""


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

# Lat-lon range for the Greater Vancouver area
lat_lon_range = {
    'lat': (49, 49.5), 
    'lon': (-123.5, -122)
}

def extract_nodes(line):
    root = etree.fromstring(line)
    if root.tag != 'node':
        return
    lat = float(root.get('lat'))
    lon = float(root.get('lon'))
    if (lat_lon_range['lat'][0] <= lat <= lat_lon_range['lat'][1]) \
            and (lat_lon_range['lon'][0] <= lon <= lat_lon_range['lon'][1]):
        tags = { tag.get('k'): tag.get('v') for tag in root.iter('tag') }
        if 'amenity' in tags \
                or 'public_transport' in tags \
                or 'shop' in tags \
                or 'building' in tags:
            yield Row(
                id = int(root.get('id')), 
                timestamp = root.get('timestamp'), 
                lat = lat, 
                lon = lon, 
                tags = tags
            )

def main(inputs, output):
    lines = sc.textFile(inputs)
    nodes = lines.flatMap(extract_nodes)
    df = spark.createDataFrame(nodes, schema = nodes_schema)
    df.write.json(
        output, 
        mode = 'overwrite', 
        compression = 'gzip'
    )

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('OSM extractor').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)
