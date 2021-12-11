# Execution & Run

[TOC]

## Preparations

**Cloning**:

```bash
git clone https://csil-git1.cs.surrey.sfu.ca/jsa381/airbnb.git
cd airbnb
```

**To get the cleaned dataset**:

```bash
sh clearing_data.sh
```

Or:

```bash
spark-submit ./src/osm_04_extract_features.py ./datasets/osm_vancouver_data.json ./datasets/osm_features
spark-submit ./src/calc_osm_airbnb_distances.py ./datasets/osm_features/ ./datasets/listings.csv ./datasets/distances
spark-submit ./src/listing_data_clearing.py ./datasets/listings.csv ./datasets/calendar.csv ./datasets/distances ./datasets/airbnb_out
```

For the simplicity of testing, we have included the resulting dataset from the above commands in the dataset folder (i.e., `./datasets/cleaned_listing.csv`).



## Analysis

- **To get tables needed by visualization and plots**:

  ```bash
  spark-submit ./src/visualization_listings.py ./datasets/listings.csv ./plots
  ```

- **To get the monthly price**:

  ```bash
  sh monthly_price_cluster.sh
  ```

  Alternatively, you may run the following to get the data for a specific month

  ```bash
  spark-submit ./src/monthly_price.py ./datasets/calendar_4-12.csv 4
  ```



## Price Optimization Tool

**To train and save the regression model**:

```bash
spark-submit ./src/predict_price_GLM_train.py ./datasets/cleaned_listing.csv ./models/predict_price_GLM
```

**To tune hyperparameters for the regression model**:

```bash
spark-submit ./src/predict_price_GLM_tuning.py ./datasets/cleaned_listing.csv
```

**To test the saved regression model**:

- Refer to `prediction_profit.ipynb` for further details



## Recommender System

**To get sentiment scores on the review data**:

```bash
spark-submit ./src/reviews_sentiment.py ./datasets/reviews.csv ./datasets/review_scores
```

**To get recommendation for a given `userid`**:

```bash
# Some userids can be used for testing: 10349410, 146224, 1676647, 5028719, 6000823, 7480215
spark-submit airbnb_recommendation.py ./datasets/review_scores [output] [userid]
```



## Web App

**To start the back-end server**:

```bash
unzip ./airbnb_webapp/Airbnb_webapp_final.zip && cd airbnb_app
python3 index.py
```

**To access the front-end page**:

- Enter the following link in the browser: [http://127.0.0.1:8050/apps/analysis](http://127.0.0.1:8050/apps/analysis) 

