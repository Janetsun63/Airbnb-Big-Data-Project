

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types, Row
import numpy as np
import pandas as pd

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

# Input: param: dict
# param = {
#     'address': str, 
#     'postcode': str, 
#     'house_start_date': str/int
#     'neighbourhood_cleansed': str, 
#     'room_type': str, 
#     'accommodates': int, 
#     'baths': int,  
#     'beds': int, 
#     'amenities': int
# }
# 
# Return: Pandas Dataframe
# | 'price' | 'future_occupancy' | 'profit' |

# def get_prediction_api(param: dict):
#     pass

price_prediction_model = './models/predict_price_GLM'

def get_prediction_api(param: dict):

    spark = SparkSession.builder.appName('Make predictions').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext

    # load the model
    model = PipelineModel.load(price_prediction_model)

    param_df = spark.createDataFrame([{
        'latitude': 49.2690, 
        'longitude': -123.139, 
        'num_bus_2km': 250, 
        'num_subway_2km': 6, 
        'num_shop_2km': 140, 
        'nearest_shop(m)': 70.254, 
        'num_restaurant_2km': 400, 
        'neighbourhood_cleansed': 'Fairview', 
        'room_type': 'Entire home/apt', 
        'accommodates': 4, 
        'baths': 4, 
        'beds': 3, 
        'days': 1550, 
        'amenities': 20
    }])

    occupancy = spark.createDataFrame(
        pd.DataFrame(
            np.arange(1, 1001) / 1000, 
            columns = ['future_occupancy']
        )
    )

    input_features = occupancy.join(param_df)
    # input_features.show()

    predictions = model.transform(input_features)


    # 1000 values, small enough to proceed with Pandas
    pd_pred = predictions[['future_occupancy', 'prediction']].toPandas()

    # Transform log_price to price and cast to int
    pd_pred['price'] = np.exp(pd_pred['prediction']).astype(int)

    # Infer data for missing price range
    missing = pd.DataFrame({
        'price': np.arange(0, pd_pred['price'].min() + 1), 
        'future_occupancy': 1.0
    })

    pd_pred = pd.concat([missing, pd_pred], axis = 0).reset_index(drop = True)

    # Merge data points with the same price
    pd_pred = pd_pred[['price', 'future_occupancy']].groupby('price').agg('mean').reset_index()

    # Compute profit
    pd_pred['profit'] = pd_pred['price'] * pd_pred['future_occupancy'] * 30

    pd_pred.to_csv(
        './datasets/predictions_tmp.csv', 
        index = False, 
        header = True
    )



if __name__ == '__main__':
    get_prediction_api({'a': 1})
