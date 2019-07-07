# simfin data example 1
# https://simfin.com/data/access/download
# output-mixeddet-quarters-gaps-publish-semicolon-wide

# SW API key
# 6BEqsSZGmXpbrRjS06PoHU8l78R3gBqS

# https://github.com/SimFin/api-tutorial


# import pandas
# location = 'C:/Users/SW/Downloads/output-mixeddet-quarters-gaps-publish-semicolon-wide/'
# firsttest = pandas.read_csv(location + 'output-semicolon-wide.csv', 
#     sep=';', nrows = 5)

# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install("requests")

import requests

api_key = "6BEqsSZGmXpbrRjS06PoHU8l78R3gBqS"
tickers = ["AAPL","NVDA","WMT"]

sim_ids = []
for ticker in tickers:
    request_url = f'https://simfin.com/api/v1/info/find-id/ticker/{ticker}?api-key={api_key}'
    content = requests.get(request_url)
    data = content.json()
    if "error" in data or len(data) < 1:
        sim_ids.append(None)
    else:
        sim_ids.append(data[0]['simId'])
print(sim_ids)

# define time periods for financial statement data
statement_type = "pl"
time_periods = ["Q1","Q2","Q3","Q4"]
year_start = 2013
year_end = 2018

# prep writer
install("pandas")
import pandas as pd
install("xlsxwriter")
import xlsxwriter

writer = pd.ExcelWriter("simfin_data.xlsx", engine='xlsxwriter')
data = {}

# get standardized financial statement
data = {}
for idx, sim_id in enumerate(sim_ids):
    d = data[tickers[idx]] = {"Line Item": []}
    if sim_id is not None:
        for year in range(year_start, year_end + 1):
            for time_period in time_periods:
                period_identifier = time_period + "-" + str(year)
                if period_identifier not in d:
                    d[period_identifier] = []
                request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/statements/standardised?stype={statement_type}&fyear={year}&ptype={time_period}&api-key={api_key}'
                content = requests.get(request_url)
                statement_data = content.json()
                # collect line item names once, they are the same for all companies with the standardised data
                if len(d['Line Item']) == 0:
                    d['Line Item'] = [x['standardisedName'] for x in statement_data['values']]
                if 'values' in statement_data:
                    for item in statement_data['values']:
                        d[period_identifier].append(item['valueChosen'])
                else:
                    # no data found for time period
                    d[period_identifier] = [None for _ in d['Line Item']]

        # saving to xlsx
        # convert to pandas dataframe
        df = pd.DataFrame(data=d)
        # save in the XLSX file configured earlier
        df.to_excel(writer, sheet_name=tickers[idx])
        writer.save()
writer.close()


