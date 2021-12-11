# Airbnb Big Data Project

In this project, we have two target groups: the hosts that plan to rent out their properties on Airbnb and the Airbnb users who want to book properties. The purpose of this project is to analyze the residential housing data and listing reviews data in Vancouver from Inside Airbnb to develop two tools: a price optimization tool to help Airbnb hosts to maximize their profits,  and a recommendation engine to facilitate the Airbnb users to find their favorable listings. With this price optimization tool, we will be able to predict how the occupancy rate of a residential property in a certain city will change with respect to its price, so that when house owners provide us with their housing information, we can help them to decide what range of price may potentially yield the maximum monthly profit. The recommender system will recommend 10 property listings to each user based on their past listing reviews when provided with the user’s id.

## Project Folder Structure

- `./datasets/`: Contains original datasets
  - Note: the data folder is included in the `.gitignore` file
- `./src/`: Source code for use in this project
  - Getting the plotting table of features and price\
  spark-submit visualization_listings.py listings.csv out_dir
  
  - Getting the monthly price for Airbnb\
  Please make sure all of your datasets are saved in 1 file
  run : sh  monthly_price_cluster.sh  #get all the price\
  OR you can run : spark-submit monthly_price.py calendar_4-12.csv 4 #get each monthly price
  
  - Getting clearing data 
  run : sh clearing_data.sh  \
  OR you can run \
  spark-submit osm_04_extract_features.py [input folder/file] [output folder]\
  spark-submit ./src/calc_osm_airbnb_distances.py [OSM-feature dataset] [Airbnb raw dataset] [Output folder]\
  spark-submit listing_data_clearing.py listings.csv  [Output folder from calc_osm_airbnb_distances.py ] out_dir

  - Sentiment analysis on reviews data  \
  run: spark-submit reviews_sentiment.py reviews.csv out_dir

  - Get recommendation for a input userid \
  run: spark-submit airbnb_recommendation.py review_scores.csv output 10349410 \
  Some other userids can be used for testing: \
  146224, 1676647, 5028719, 6000823, 7480215\

  - pip3 install nltk, textblob, langid
  
- `./plots/`: Generated plots and images to be used in the report
- `./airbnb_webapp/`: UI & Visualization 



