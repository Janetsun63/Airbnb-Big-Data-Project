# Airbnb Big Data Project

[TOC]

## Project Folder Structure

- `./datasets/`: Contains original datasets
  - Note: the data folder is included in the `.gitignore` file
- `./src/`: Source code for use in this project
  - Getting the plotting table of features and price\
  spark-submit visualization_listings.py listings.csv out_dir
  
  - Getting the monthly price for Airbnb\
  run : sh  monthly_price_cluster.sh  #get all the price\
  OR you can run : spark-submit monthly_price.py calendar_4-12.csv 4 #get each monthly price
  
  - Getting clearing data 
  run : sh clearing_data.sh  \
  OR you can run \
  spark-submit osm_04_extract_features.py [input folder/file] [output folder]\
  spark-submit ./src/calc_osm_airbnb_distances.py [OSM-feature dataset] [Airbnb raw dataset] [Output folder]\
  spark-submit listing_data_clearing.py listings.csv calendar.csv [Output folder from calc_osm_airbnb_distances.py ] out_dir
  
- `./plots/`: Generated plots and images to be used in the report



