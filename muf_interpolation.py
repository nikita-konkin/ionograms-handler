import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy.stats
from scipy import signal
import datetime
import calendar
import psycopg2

sys.path.append(r'C:/data_handler')

from config import config
from muf_load_to_db import get_muf_schemas
from muf_load_to_db import get_origin_muf_tables_names as get_muf_tables_names
from muf_load_to_db import get_muf_transmit_station
from muf_load_to_db import get_muf_tables
from muf_load_to_db import check_db_muf_interpolated_table_exist
from muf_load_to_db import write_interpolated_muf_to_db


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

def MUF_interpolation(exp_muf_set):
    print(exp_muf_set)
    # exp_muf_set.plot()
    # plt.show()
    min_time = min(exp_muf_set.time)
    max_time = max(exp_muf_set.time)
    # print(min_time, max_time)
    date_start = pd.to_datetime(min_time, format='%H:%M:%S')
    date_stop = pd.to_datetime(max_time, format='%H:%M:%S')
    diff = date_stop - date_start
#     print((diff.seconds)/60)
    
    arr_time = date_start + pd.to_timedelta(np.arange(0, diff.seconds/60 + 5, 5), 'm')
    time = arr_time.strftime("%H:%M:%S")
    # print(time)

    time_df = pd.DataFrame(time, columns=['time'])
    time_df['time'] = pd.to_datetime(time_df['time'], format='%H:%M:%S')
    exp_muf_set_ = exp_muf_set.time
    exp_muf_set_['time'] = pd.to_datetime(exp_muf_set['time'], format='%H:%M:%S')

    time_df_index = time_df.time.isin(exp_muf_set_.time)

    true_index = time_df_index.index[time_df_index == True]
#     print(true_index)
    true_index = pd.DataFrame({'ind':true_index})
    # print('true_index0 = ', true_index)
    if true_index.empty == False:


        exp_muf_set = exp_muf_set.set_index(true_index.ind)
        exp_muf_set['interpolated'] = False

        exp_muf_set = exp_muf_set.drop(columns=['time'])
        exp_muf_set = pd.concat([time_df, exp_muf_set], ignore_index=False, axis=1)
        interpolated_true_col = exp_muf_set['interpolated'].fillna(value=True, axis=0)
        exp_muf_set = exp_muf_set.drop(columns=['interpolated'])
        exp_muf_set = pd.concat([exp_muf_set, interpolated_true_col], ignore_index=False, axis=1)
        exp_muf_set.muf = pd.to_numeric(exp_muf_set.muf)
        interpolated_muf_col = exp_muf_set['muf'].interpolate(method='cubic')
        exp_muf_set = exp_muf_set.drop(columns=['muf'])
        exp_muf_set = pd.concat([exp_muf_set, interpolated_muf_col], ignore_index=False, axis=1)
        # print(exp_muf_set.muf.dtype)



        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        # # print(df)
        #     print('NEW = ', exp_muf_set)

        concat = exp_muf_set
        # time_df_frop_by_true_index = time_df.drop(true_index.ind)
        # exp_muf_set.time.index.astype('int64')
        # exp_muf_set.muf.index.astype('int64')

        # exp_muf_set.muf = pd.to_numeric(exp_muf_set.muf)

        # interpol = np.interp(time_df_frop_by_true_index.index, exp_muf_set.time.index, exp_muf_set.muf, period=288)

        # interpol_set = pd.DataFrame({'time': time_df_frop_by_true_index.time, 'muf':interpol})
        # interpol_set['interpolated'] = True
            
        # exp_muf_set_droped = exp_muf_set.set_index(true_index.ind)
        # exp_muf_set_droped['interpolated'] = False
        # print('interpol_set = ',interpol_set)
        
        # concat = pd.concat([exp_muf_set_droped, interpol_set])
        # concat = concat.sort_index()
    else:
        print('muf do not need interpolation')
        concat = exp_muf_set
        concat['interpolated'] = False

    return concat


def main():
    
    # df, y, y_filtered_savgol, y_filtered_median = Exprmental_muf()
    # MUF_interpolation(df)
    year = 2023
    print('start')
    schemas = get_muf_schemas(year)

    for schema in schemas:
        print('month = ', schema[0])
        tables_names = get_muf_tables_names(schema[0], year)
        print(tables_names)
        for table_name in tables_names:
            split = table_name[0].split('_')
            if len(split) == 1:
                print('day = ', table_name[0])
                transmit_stations = get_muf_transmit_station(table_name[0], schema[0], year)
                for transmit_station in transmit_stations:
                    print('transmit_station = ', transmit_station[0])
                    # if check_db_muf_interpolated_table_exist(table_name[0], schema[0], transmit_station[0], year) == False:
                    muf_table = get_muf_tables(table_name[0], schema[0], transmit_station[0], year)
                    muf_table = pd.DataFrame(muf_table)
                    muf_table.columns = ['time', 'muf']
                    
                    interpolated_muf = MUF_interpolation(muf_table)
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
                else:
                    print('muf table already in database!')
                    print('------------------------------')
                    # write_interpolated_muf_to_db(interpolated_muf.time, )





main()
