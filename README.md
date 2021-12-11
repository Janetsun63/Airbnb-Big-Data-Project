# Airbnb Big Data Project

In this project, we have two target groups: the hosts that plan to rent out their properties on Airbnb and the Airbnb users who want to book a visit. The purpose of this project is to analyze the residential housing data and listing reviews data in Vancouver from Inside Airbnb to develop two tools: a price optimization tool to help Airbnb hosts to maximize their profits,  and a recommendation engine to facilitate the Airbnb users to find their favorable listings. With this price optimization tool, we will be able to predict how the occupancy rate of a residential property in a certain city will change with respect to its price, so that when house owners provide us with their housing information, we can help them to decide what range of price may potentially yield the maximum monthly profit. The recommender system will recommend 10 property listings to each user based on their past listing reviews when provided with the user id.



## Prerequisites

**Environment**:

- `python3` 
- `Spark` 



**Installation of required libraries**:

```bash
pip3 install -r requirements.txt
```



## Project Folder Structure

- `./airbnb_webapp/`: UI & Visualization (Web application)
- `./datasets/`: Contains some of the datasets (raw and processed)
- `./models/`: Saved machine learning models
- `./plots/`: Generated plots and images to be used in the report
- `./src/`: Source code for use in this project



## Data Sources

- [Inside Airbnb](http://insideairbnb.com/get-the-data.html) 
  - Raw and processed datasets are available in `./datasets` 
- [Vancouver Weather](https://vancouver.weatherstats.ca/download.html) 
  - Raw and processed datasets are available in `./datasets` 
- [OpenStreetMap](https://download.geofabrik.de/) 
  - Raw dataset was more than 20GB
  - The processed dataset is available in `./datasets` 



## How to run

Refer to `RUNNING.md` for further details





