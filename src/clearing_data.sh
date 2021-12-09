echo "Please put all you code in a directory and all dataset in a directory\n"
echo "Please enter the path you of all your python code(e.g. /Users/airbnb/src)"
read path

echo "Please enter the path you of all your dataset(e.g. /Users/airbnb/datasets)"
read data_path

echo "where you run this code? 1 for cluster and 2 for local computer"
read var
if [[ $var -eq 2 ]];
then
	${SPARK_HOME}/bin/spark-submit ${path}/osm_04_extract_features.py ${data_path}/osm_vancouver_data.json ${data_path}/osm_features
	${SPARK_HOME}/bin/spark-submit ${path}/calc_osm_airbnb_distances.py ${data_path}/osm_features/ ${data_path}/listings.csv ${data_path}/distances
	${SPARK_HOME}/bin/spark-submit ${path}/listing_data_clearing.py ${data_path}/listings.csv ${data_path}/distances ${data_path}/airbnb_out
	
else
	spark-submit ${path}/osm_04_extract_features.py ${data_path}/osm_vancouver_data.json ${data_path}/osm_features
	spark-submit ${path}/calc_osm_airbnb_distances.py ${data_path}/osm_features/ ${data_path}/listings.csv ${data_path}/distances
	spark-submit ${path}/listing_data_clearing.py ${data_path}/listings.csv ${data_path}/calendar.csv ${data_path}/distances ${data_path}/airbnb_out

fi

rm -r ${data_path}/distances
rm -r ${data_path}/osm_features

echo "Your final clearing data saved at ${data_path}/airbnb_out"
