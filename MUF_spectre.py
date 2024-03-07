import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
from scipy import signal
import datetime
import calendar
from sklearn.metrics import r2_score
from mini_muf import miniMUF
from miniMUF_interpolation import miniMUF_interpolation
from muf_load_to_db import get_muf_schemas
from muf_load_to_db import get_muf_tables_names
from muf_load_to_db import get_time_muf_from_interpolated_muf
from muf_load_to_db import write_minimuf_sunspots_to_db
from muf_load_to_db import check_db_minimuf_sunspots_table_exist
# from muf_interpolation import main as

from scipy.fft import fft, fftfreq
from scipy.signal import blackman

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return se


def savgol1_filter(interp_time_muf):

    x = interp_time_muf
    time = x.time
    y = x.muf
    x = x.index
    # y_filtered_median = signal.medfilt(y, 7)
    y_filtered_savgol1 = signal.savgol_filter(y, 17, 2)

    savgol_filter_coef = signal.savgol_coeffs( 27, 8)

    print(savgol_filter_coef)

    y_filtered_savgol1 = signal.savgol_filter(y_filtered_savgol1, 27, 8)
    y_ci = mean_confidence_interval(y_filtered_savgol1)
    
    return savgol_filter_coef, x, y_filtered_savgol1

def poly1d(y):

    print('y.index = ', len(y))
    myline = np.linspace(0, len(y), len(y))
    mymodel = np.poly1d(np.polyfit(myline, y, 2))
    r2 = r2_score(y, mymodel(myline))
    print('r2 = ',r2)
    poly1d_df = pd.DataFrame({'time': myline, 'muf':mymodel(myline)})

    return poly1d_df

def interpolation_miniMUF_time():

    date_start = pd.to_datetime('00:00:00', format='%H:%M:%S')
    date_stop = pd.to_datetime('23:55:00', format='%H:%M:%S')
    diff = date_stop - date_start
    arr_time = date_start + pd.to_timedelta(np.arange(0, diff.seconds/60 + 5, 60), 'm')
    
    time = arr_time.strftime("%H:%M:%S")
    date_stop = pd.DataFrame({'time':['23:55:00']})
    time_df = pd.DataFrame(time, columns=['time'])
    time_df = time_df.append(date_stop, ignore_index = True)

    return time_df

def main():
    
    print('start')
    schemas = get_muf_schemas()
    m = 0
    for schema in schemas:

        tables_names = get_muf_tables_names(schema[0])
        for table_name in tables_names:
            split = table_name[0].split('_')
            if len(split) > 1 and split[1] == 'interp':

                interp_time_muf = get_time_muf_from_interpolated_muf(table_name[0], schema[0])

                interp_time_muf = pd.DataFrame(interp_time_muf, columns = ['time', 'muf'])

                if len(interp_time_muf) > 149:

                    savgol_filter_coef, x, y_filtered_savgol = savgol1_filter(interp_time_muf)
                    y_filtered_savgol = pd.DataFrame(y_filtered_savgol, columns=['filtered_savgol_MUF'])
                    y_filtered_median = pd.DataFrame({'y_filtered_median_MUF': signal.medfilt(signal.medfilt(interp_time_muf.muf, 9), 5)})


                    N = 288
                    # sample spacing
                    T = 1 / (288*3)

                    yf = fft(interp_time_muf.muf)
                    xf = fftfreq(int(N), T)

                    w = blackman(int(N))
                    w = pd.DataFrame(w, columns=['b'])

                    print(y_filtered_savgol.filtered_savgol_MUF)
                    y_fft_orig = fft(np.array(w.b * pd.to_numeric(interp_time_muf.muf, downcast="float")))

                    y_fft_savgol = fft(np.array(w.b * pd.to_numeric(y_filtered_savgol.filtered_savgol_MUF, downcast="float")))

                    y_fft_median = fft(np.array(w.b * pd.to_numeric(y_filtered_median.y_filtered_median_MUF, downcast="float")))

                    y_fff_diff_savgol = y_fft_orig - y_fft_savgol
                    y_fff_diff_median = y_fft_orig - y_fft_median

                    y_fff_diff_filtered_savgol1 = signal.savgol_filter(y_fff_diff_savgol, 27, 16)
                    y_fff_diff_filtered_median = signal.savgol_filter(y_fff_diff_median, 27, 16)

                    # xf = fftfreq(N, T)

                    w, h = signal.freqz(savgol_filter_coef)
                    # angles = np.unwrap(np.angle(h))

                    # plt.plot(w, angles, 'g')
                    print(xf)

                    plt.grid()
                    plt.plot(w, 20 * np.log10(abs(h)), 'b', label='savgol_filter_afc')
                    plt.legend()
                    plt.show()

                    plt.grid()

                    print('xf')

                    plt.plot(xf, 20 * np.log10(abs(y_fff_diff_filtered_savgol1)), '-r', label='y_fff_diff_filtered_savgol1')
                    plt.plot(xf, 20 * np.log10(abs(y_fff_diff_filtered_median)), '-b', label='y_fff_diff_filtered_median')

                    plt.legend()
                    plt.show()





                    # transmit_station = table_name[0].split('_')[0]

                    # if len(y_filtered_savgol) == 288:

                    #     if m < 3:
                    #         interp_time_muf_0 = pd.DataFrame(columns=['time', 'muf'])
                    #         y_filtered_savgol_0 = pd.DataFrame(columns=['filtered_savgol_MUF'])
                    #         y_filtered_median_0 = pd.DataFrame(columns=['y_filtered_median_MUF'])


                    #     interp_time_muf_0 = interp_time_muf_0.append(interp_time_muf, ignore_index=True)
                    #     y_filtered_savgol_0 = y_filtered_savgol_0.append(y_filtered_savgol, ignore_index=True)
                    #     y_filtered_median_0 = y_filtered_median_0.append(y_filtered_median, ignore_index=True)
                    #     # print(len(interp_time_muf_0))
                    #     time_extrap_df = pd.DataFrame(np.nan, index=range(0, int(len(interp_time_muf_0)), 8), columns=['time'])
                    #     time_extrap_df_1 = pd.DataFrame(np.nan, index=range(0, int(len(interp_time_muf_0)), 1), columns=['time'])

                    #     for i in range(0, int(len(interp_time_muf_0)), 8):
                    #         time_extrap_df.loc[i] = interp_time_muf_0.time.loc[i]


                    #     if m > 2:
                    #         plt.plot(y_filtered_median_0.index, y_filtered_median_0.y_filtered_median_MUF, ms=3, color='red', marker='o', linestyle='dashed', label="MUF median")
                    #         plt.plot(interp_time_muf_0.index, interp_time_muf_0.muf, color='dimgray', label="MUF experimental")
                    #         plt.plot(y_filtered_savgol_0.index, y_filtered_savgol_0.filtered_savgol_MUF, ms=3, color='black', marker='o', linestyle='dashed', label="MUF savgol_filter")
                            
                    #         plt.xticks(np.arange(0, len(interp_time_muf_0), 8), time_extrap_df.time, rotation=90, fontsize=12)
                    #         plt.xlim(0, len(interp_time_muf_0.time))
                    #         plt.xlabel('time, hour', fontsize=18)
                    #         plt.ylabel('MUF, MHz', fontsize=18)
                    #         plt.legend(prop={'size': 18})
                    #         plt.show()

                    #     m = m + 1




main()