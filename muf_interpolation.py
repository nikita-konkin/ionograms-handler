import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy.stats
from scipy import signal
import datetime
import calendar
import psycopg2

sys.path.append(r'data_handler')

from local_config import config
# from muf_load_to_db import get_muf_schemas
from muf_load_to_db import get_origin_muf_tables_names as get_muf_tables_names
from muf_load_to_db import get_muf_transmit_station
from muf_load_to_db import get_muf_tables
from muf_load_to_db import check_db_muf_interpolated_table_exist
from muf_load_to_db import write_interpolated_muf_to_db


from muf_load_to_db import DbRequests


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
#     probability_density_function = t.ppf(1-confidence/2, n-1)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    # print(pow(se, 2))
    return se

def Exprmental_muf():
    
    x = pd.read_csv('E:\\MyData\\MyRadio\\ND\\prediction\\data_db.csv', sep = ',', header = 0)
    mask = x['transmit_station'] == 'cyprus1'
    
#     print(MUF)
    df = x[mask]
    y = x[mask].muf
    
#     base = datetime.datetime(00:00:00)
#     arr = numpy.array([base + datetime.timedelta(minute=i) for i in xrange(1440)])

    y_filtered_median = signal.medfilt(y, 7)
    y_filtered_savgol = signal.savgol_filter(y, 17, 2)

    y_ci = mean_confidence_interval(y_filtered_savgol)
#     plt.plot(y_filtered_savgol[:])
#     plt.plot(y_filtered_median[:])
#     plt.plot(y[:])
#     # print(y_ci[:])
#     # plt.errorbar(x[:], y[:], yerr=y_ci[:])

#     plt.legend(loc=4)
#     plt.xlabel('time')
#     plt.ylabel('muf, MGz')
#     plt.show()
    
    return df, y, y_filtered_savgol, y_filtered_median

def MUF_interpolation(exp_muf_set, year, month, day):

    min_time = min(exp_muf_set.time)
    date_start = pd.Series(pd.to_datetime(min_time, format='%H:%M:%S'))
    min_time = pd.to_datetime(f'00:00:{date_start.dt.second.values[0]} {year}-{month}-{day}', format='%H:%M:%S %Y-%m-%d')
    max_time = pd.to_datetime(f'23:55:{date_start.dt.second.values[0]} {year}-{month}-{day}', format='%H:%M:%S %Y-%m-%d')

    origin_time = pd.date_range(min_time, max_time, freq='5min')
    print(origin_time)

    exp_muf_set_ = exp_muf_set.copy()
    exp_muf_set_['date'] =  f' {year}-{month}-{day}'
    exp_muf_set_['datetime'] = exp_muf_set_['time'].astype(str) + exp_muf_set_['date']

    exp_muf_set_ = exp_muf_set_.set_index(pd.to_datetime(exp_muf_set_['datetime'], format='%H:%M:%S %Y-%m-%d'))
    exp_muf_set_ = exp_muf_set_.drop(['time', 'date', 'datetime'], axis = 1)

    exp_muf_set_ = pd.concat([exp_muf_set_, pd.DataFrame([], index = origin_time)], ignore_index=True, axis=1, sort=True)

    exp_muf_set_.columns = ['muf']

    exp_muf_set_['muf'] = exp_muf_set_['muf'].astype(float)
    exp_muf_set_ = exp_muf_set_['muf'].interpolate(method='akima', limit_direction='both')
    print(exp_muf_set_)

    sys.exit()
    return exp_muf_set_


def main():
    db = DbRequests(filename='database.ini')
    # df, y, y_filtered_savgol, y_filtered_median = Exprmental_muf()
    # MUF_interpolation(df)
    year = 2023
    print('start')

    schemas = db.get_muf_schemas(year)

    for schema in schemas:
        print('month = ', schema[0])
        tables_names =  db.get_muf_table_names(schema[0], year)
        print(tables_names)
        for table_name in tables_names:
            muf_table = db.get_muf_table_content(table_name[0], schema[0], year)
            muf_table = pd.DataFrame(muf_table)
            muf_table.columns = ['time', 'muf']

            interpolated_muf = MUF_interpolation(muf_table, year, schema[0].split('_')[1], table_name[0])
            print(interpolated_muf)
            sys.exit()
            del muf_table
            print('loading interpolated muf data to database...')
            break
            for row in range(len(interpolated_muf.index)):
                # print(interpolated_muf.loc[row, :].interpolated)
                write_interpolated_muf_to_db(
                    table_name[0], 
                    schema[0], 
                    transmit_station[0], 
                    interpolated_muf.loc[row, :].time, 
                    round(interpolated_muf.loc[row, :].muf, 2), 
                    interpolated_muf.loc[row, :].interpolated,
                    year)
                # print(row)
            print('done loading interpolated muf data to database.')
            print('-----------------------------------------------')





main()
