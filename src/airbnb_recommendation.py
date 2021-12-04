'''
runuing in terminal as: spark-submit airbnb_recommendation.py review_scores.csv output 10349410

'''
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql import types
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def main(inputs, output, userid):

    reviews_schema = types.StructType([
                                    types.StructField('id', types.IntegerType(), True),
                                    types.StructField('listing_id', types.IntegerType(), True),
                                    types.StructField('reviewer_id', types.IntegerType(), True),
                                    types.StructField('comments', types.StringType(), True,),
                                    types.StructField('score', types.DecimalType(32,18), True)])
    
    df = spark.read.option('multiLine', 'True') \
        .option('escape', '"') \
        .option("mode", "DROPMALFORMED")\
        .csv(inputs, header=True,schema = reviews_schema)

    data = df.select('reviewer_id', 'listing_id', 'score')

    # split and cache training, validation and test datasets
    seed = 5
    (split_60_df, split_a_20_df, split_b_20_df) = data.randomSplit([6.0, 2.0, 2.0],seed)
    training = split_60_df.cache()
    validation = split_a_20_df.cache()
    test = split_b_20_df.cache()

    # Train ALS model

    reg_parameters = [0.1, 0.01, 0.001]
    ranks = [5, 10, 15]
    min_error = float('inf')
    best_rank = 0
    best_regularizer = reg_parameters[0]

    for rank in ranks:
        for reg_parameter in reg_parameters:
            als = ALS(
            maxIter= 5,
            regParam = reg_parameter,
            rank = rank,
            userCol="reviewer_id", 
            itemCol="listing_id",
            ratingCol="score", 
            nonnegative = True, 
            implicitPrefs = False,
            coldStartStrategy = 'drop')
            model = als.fit(training)
            predictions=model.transform(validation)
            evaluator=RegressionEvaluator(metricName="rmse",labelCol="score",predictionCol="prediction")
            error=evaluator.evaluate(predictions)
            print ('For rank %s and regularization parameter %s the RMSE is %s' % (rank, reg_parameter, error))
            if error < min_error:
                min_error = error
                best_rank = rank
                best_regularizer = reg_parameter
    print ('The best model was trained with rank %s, regularization parameter %s and minimum RMSE %s' % (best_rank,best_regularizer,min_error))

    #The best model was trained with rank 5, regularization parameter 0.1 and minimum RMSE 1.7979098914056308

    als = ALS(
            maxIter= 5,
            regParam = best_regularizer,
            rank = best_rank,
            userCol="reviewer_id", 
            itemCol="listing_id",
            ratingCol="score", 
            nonnegative = True, 
            implicitPrefs = False,
            coldStartStrategy = 'drop')
    best_model = als.fit(training)

    # test on the best model
    predictions_test = best_model.transform(test)
    test_RMSE = evaluator.evaluate(predictions_test)
    print('The model had a RMSE on the test set of {0}'.format(test_RMSE))
    # generate 3 listing recommendations for all users
    recommendations = best_model.recommendForAllUsers(10)
    recommendations1 = recommendations.withColumn("rec_exp", functions.explode("recommendations"))\
        .select('reviewer_id', functions.col("rec_exp.listing_id"), functions.col("rec_exp.rating"))
    recommendations2 = recommendations1.filter(recommendations1.reviewer_id == userid)
    recommendations2.coalesce(1).write.option("header", "true").csv(output+'/recommendations '+str(userid), mode='overwrite')
   


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    userid = int(sys.argv[3])
    spark = SparkSession.builder.appName('airbnb recommendation').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output, userid)