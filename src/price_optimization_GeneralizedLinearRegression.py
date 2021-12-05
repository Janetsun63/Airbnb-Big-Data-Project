

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
import pyspark.ml.regression as regression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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

    # GeneralizedLinearRegression
    regressor = regression.GeneralizedLinearRegression(
        featuresCol = 'features', 
        labelCol = label_column, 
        predictionCol = 'prediction', 
        family = "gaussian", 
        link = "identity", 
        maxIter = 50, 
        regParam = 0.2
    )

    # Hyperparameters for GeneralizedLinearRegression
    paramGrid = ParamGridBuilder() \
        .addGrid(regressor.maxIter, range(0, 51, 5)) \
        .addGrid(regressor.regParam, [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.15, 0.2, 0.3]) \
        .build()


    # Pipeline
    pipeline = Pipeline(stages = word_indexer_lst + [
        assembler, 
        regressor
    ])

    # Define evaluator
    rmse_evaluator = RegressionEvaluator(
        labelCol = label_column, 
        predictionCol = "prediction", 
        metricName = 'rmse'
    )

    crossval = CrossValidator(
        estimator = pipeline, 
        estimatorParamMaps = paramGrid, 
        evaluator = rmse_evaluator, 
        numFolds = 5, 
        parallelism = 8
    )

    # Training
    cvModel = crossval.fit(train)

    # Get optimal model
    best_model = cvModel.bestModel
    # print(best_model.stages[-1].extractParamMap())
    print("Best Param (maxIter): %g" % best_model.stages[-1].getMaxIter())
    print("Best Param (regParam): %d" % best_model.stages[-1].getRegParam())

    # Make predictions
    predictions = best_model.transform(validation)
    predictions.cache()

    # # Select example rows to display.
    # predictions.show(5)

    # Testing error
    test_rmse = rmse_evaluator.evaluate(predictions)

    print("Root Mean Squared Error (RMSE) on test data = %g" % test_rmse)

    # Save the trained model
    best_model.write().overwrite().save(output)

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('Model training').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)
