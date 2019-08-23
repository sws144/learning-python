# https://stackoverflow.com/questions/39835770/read-data-from-pyodbc-to-pandas

import pyodbc
import pandas
cnxn = pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                      r'DBQ=C:\users\bartogre\desktop\data.mdb;')
sql = "Select sum(CYTM), sum(PYTM), BRAND From data Group By BRAND"
data = pandas.read_sql(sql,cnxn)

# not working yet