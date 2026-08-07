import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(r'data_handler')

from muf_load_to_db import DbRequests



def MUF_interpolation(exp_muf_set, year, month, day):

    min_time = min(exp_muf_set.time)
    date_start = pd.Series(pd.to_datetime(min_time, format='%H:%M:%S'))
    min_time = pd.to_datetime(f'00:00:{date_start.dt.second.values[0]} {year}-{month}-{day}', format='%H:%M:%S %Y-%m-%d')
    max_time = pd.to_datetime(f'23:55:{date_start.dt.second.values[0]} {year}-{month}-{day}', format='%H:%M:%S %Y-%m-%d')

    origin_time = pd.date_range(min_time, max_time, freq='5min')

    exp_muf_set_ = exp_muf_set.copy()
    exp_muf_set_['date'] =  f' {year}-{month}-{day}'
    exp_muf_set_['datetime'] = exp_muf_set_['time'].astype(str) + exp_muf_set_['date']

    exp_muf_set_ = exp_muf_set_.set_index(pd.to_datetime(exp_muf_set_['datetime'], format='%H:%M:%S %Y-%m-%d'))
    exp_muf_set_ = exp_muf_set_.drop(['time', 'date', 'datetime'], axis = 1)
    exp_muf_set_ = exp_muf_set_.sort_index()

    exp_muf_set_ = pd.concat([exp_muf_set_, pd.DataFrame([], index = origin_time)], ignore_index=True, axis=1, sort=True)

    exp_muf_set_.columns = ['muf']

    exp_muf_set_['muf'] = exp_muf_set_['muf'].astype(float)
    
    exp_muf_set_ = exp_muf_set_['muf'].interpolate(method='akima', limit_direction='both').interpolate(method='linear')
    exp_muf_set_ = exp_muf_set_.round(3)

    return exp_muf_set_


def main():
    db = DbRequests(filename='database.ini')

    year = 2023
    print('start')

    schemas = db.get_muf_schemas(year)

    for schema in schemas:
        print('month = ', schema[0])
        tables_names =  db.get_muf_table_names(schema[0], year)
        
        for table_name in tables_names:
            print(table_name)
            muf_table = db.get_muf_table_content(table_name[0], schema[0], year)
            if len(muf_table) < 220:
                continue
            muf_table = pd.DataFrame(muf_table)
            muf_table.columns = ['time', 'muf']
            
            interpolated_muf = MUF_interpolation(muf_table, year, schema[0].split('_')[1], table_name[0])

            del muf_table
            
            db.write_interpolated_muf_to_db(
                table_name[0], 
                schema[0], 
                interpolated_muf, year)

main()
