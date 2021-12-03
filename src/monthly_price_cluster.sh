echo "Please enter the path which for all of your python codes(e.g. /Users/airbnb/src)"
read path

echo "Please enter the path which for all of your datasets(e.g. /Users/airbnb/datasets)"
read data_path

echo "where you run this code? 1 for cluster and 2 for local computer"
read var 


if [[ $var -eq 2 ]];
then
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_1-12.csv 1  > ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_2-9.csv 2  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_3-5.csv 3  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_4-12.csv 4  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_6-8.csv 6  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_7-6.csv 7  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_8-8.csv 8  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_9-10.csv 9  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_10-10.csv 10  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_11-6.csv 11  >> ${data_path}/out.txt
	${SPARK_HOME}/bin/spark-submit ${path}/monthly_price.py ${data_path}/calendar_12-16.csv 12  >> ${data_path}/out.txt
else
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_1-12.csv 1  > ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_2-9.csv 2  >> ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_3-5.csv 3  >> ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_4-12.csv 4  >> ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_6-8.csv 6  >> ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_7-6.csv 7  >> ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_8-8.csv 8  >> ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_9-10.csv 9  >> ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_11-6.csv 11  >> ${data_path}/out.txt
	spark-submit ${path}/monthly_price.py ${data_path}/calendar_12-16.csv 12  >> ${data_path}/out.txt
fi



echo "the monthly price saved at out.txt"

#/Users/jennifer/Downloads/2021fall/732/project
#/Users/jennifer/Downloads/2021fall/732/project/calendar