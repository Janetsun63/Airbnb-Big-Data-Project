
"""
Download Airbnb datasets for a specified city (including the 
historical (archived) datasets) from [Inside 
Airbnb](http://insideairbnb.com/get-the-data.html).

How to use it: 
    - python3 download_airbnb_datasets.py [City Name] [Output Folder]

Example: 
    - python3 download_airbnb_datasets.py Vancouver ../datasets/Airbnb_Vancouver

Note:
    - City name is case insensitive
    - Output folder will be created if not exists
"""

import os, sys, requests
from bs4 import BeautifulSoup
import pandas as pd

target_url = "http://insideairbnb.com/get-the-data.html"


def download(table_content, output_path):
    for row in table_content:
        try:
            filename = row[2]
            if not os.path.exists(filename):
                print("Downloading file \"%s\"" % filename)
                r = requests.get(row[-1], timeout = 50)
                with open(os.path.join(output_path, filename),'wb') as f:
                    f.write(r.content)
                    f.close()
            else:
                print("Warning: File \"%s\" already exists. (Operation: Skipped)" % filename)
        except BaseException as e:
            print(e)

def main(city: str, output_path: str) -> None:

    # Create output folder
    # Exit if output folder already exists
    if os.path.exists(output_path):
        print("Warning: Output folder \"%s\" already exists." % output_path)
    else:
        os.makedirs(output_path)
        print("Created output folder ./%s/" % output_path)

    # Request target web page
    page = requests.get(target_url)
    if page.status_code != 200:
        print("Failed to request target page: status code = %d" % page.status_code)
        return

    # Parse raw data
    soup = BeautifulSoup(page.content, 'html.parser')

    # Find the table for the corresponding city
    table = soup.find('table', class_ = 'table table-hover table-striped %s' % city)
    if not table:
        print("Cannot find data for city \"%s\"" % city)
        return

    # Extract information from table
    table_content = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 4:
            date_compiled = cells[0].text.strip()
            country_city = cells[1].text.strip()
            file_name = cells[2].text.strip()
            description = cells[3].text.strip()
            link = cells[2].find('a', href = True)['href']
            table_content += [[
                date_compiled, 
                country_city, 
                '_'.join([
                    country_city, 
                    link.split('/')[-3], 
                    file_name
                ]), 
                description, 
                link
            ]]

    # Convert table into Pandas DataFrame
    table_df = pd.DataFrame(
        data = table_content, 
        columns = [
            "date_compiled", 
            "country_city", 
            "file_name", 
            "description", 
            "link"
        ]
    )

    # print(table_df)

    # Save table content as csv
    table_df.to_csv(
        "./%s/Airbnb_%s_download_list.csv" % (output_path, city), 
        index = False
    )

    # Download data files
    download(table_content, output_path)


if __name__ == '__main__':
    city = sys.argv[1]
    output_path = sys.argv[2]
    main(city.lower(), output_path)

