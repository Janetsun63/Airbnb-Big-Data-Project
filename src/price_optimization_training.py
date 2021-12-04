

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

training_data_schema = types.StructType([
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

def main(inputs, output):
    data = spark.read.csv(
        inputs, 
        header = True, 
        schema = training_data_schema
    )

    data = data.na.fill(value = 0, subset = ['beds'])

    train, validation = data.randomSplit([0.8, 0.2])
    train = train.cache()
    validation = validation.cache()

    word_indexer_lst = [
        StringIndexer(inputCol = "neighbourhood_cleansed", outputCol = "neighbourhood_index"), 
        StringIndexer(inputCol = "room_type", outputCol = "room_type_index")
    ]

    input_column_lst = list(
        set(data.schema.names) - set([
            'id', 
            'neighbourhood_cleansed', 
            'room_type', 
            'future_occupancy'
        ])
    ) + ['neighbourhood_index', 'room_type_index']
    label_column = 'future_occupancy'

    assembler = VectorAssembler(
        inputCols = input_column_lst, 
        outputCol = 'features'
    )

    regressor = GBTRegressor(
        featuresCol = 'features', 
        labelCol = label_column, 
        predictionCol = 'prediction', 
        maxDepth = len(input_column_lst)
    )

    print(input_column_lst)

    # Pipeline
    pipeline = Pipeline(stages = word_indexer_lst + [
        assembler, 
        regressor
    ])

    # Training
    model = pipeline.fit(train)
    predictions = model.transform(validation)
    predictions.cache()

    # Select example rows to display.
    predictions.show(5)

    # Evaluation
    rmse_evaluator = RegressionEvaluator(
        labelCol = label_column, 
        predictionCol = "prediction", 
        metricName = 'rmse'
    )
    rmse = rmse_evaluator.evaluate(predictions)

    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('Model training').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)
