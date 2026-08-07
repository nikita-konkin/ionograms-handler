from muf_load_to_db import get_muf_schemas
from muf_load_to_db import get_muf_tables_names
import pandas as pd
import psycopg2
from config import config
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest
import numpy as np
from scipy import signal

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
import scipy.stats as st

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    down, up = st.t.interval(alpha=0.95,  df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
    print((up-down))
    print(up-((down-up)/2))
    # print(down)
    # print((down-m)+(m-up))

    return (up-down)


def data_lag_and_lag_diff(df, shifts, test_size, col_name = 'muf', lag = True, lag_diff = True):
	
	if lag == True or lag_diff == True:
		# lags_count = int((df.size * test_size)/4)
		lags_count = int(test_size)
		# lags_count = int(0)
		for i in range(lags_count-1, lags_count-1 + shifts):
			# print(i)
			if lag_diff:
				df["lag_diff_{}".format(i)] = df[col_name].shift(i) - df[col_name].shift(i+1)
			if lag:
				df["lag_{}".format(i)] = df[col_name].shift(i)
		return df
	return df

def timeseries_train_test_split(X, y, test_size):

	train_index = int(len(X)-test_size)

	X_train = X.iloc[:train_index]
	y_train = y.iloc[:train_index]
	X_test = X.iloc[train_index:]
	y_test = y.iloc[train_index:]

	return X_train, X_test, y_train, y_test

def Create_time_predictors(df):

	df['datetime'] = df.index
	df['hour'] = df.datetime.dt.hour
	df['dayofweek'] = df.datetime.dt.dayofweek
	df['quarter'] = df.datetime.dt.quarter
	df['month'] = df.datetime.dt.month
	df['dayofyear'] = df.datetime.dt.dayofyear
	df['day'] = df.datetime.dt.day
	df['weekofyear'] = df.datetime.dt.isocalendar().week

	return df

def savgol1_filter(muf):

	y = muf.muf
	y_filtered_savgol1 = signal.savgol_filter(y, 19, 1) #window(65) & order(5)

	return y_filtered_savgol1

def IsolationForestOutfitDetection(df): #Детектор пиков

	outliers_fraction = float(.01) #Изменить в пределах до 0,01 до 0,05

	np_scaled = scaler.fit_transform(df['muf'].values.reshape(-1, 1))
	data = pd.DataFrame(np_scaled)

	model =  IsolationForest(contamination=outliers_fraction)
	model.fit(data)

	df['anomaly'] = model.predict(data)

	a = df.loc[df['anomaly'] == -1, ['muf']] #anomaly

	return a

def IsolationForestOutfitDetectionRes(df, col_name):
    df = df.dropna()
    
    outliers_fraction = float(.05) #Изменить в пределах до 0,01 до 0,05
    np_scaled = scaler.fit_transform(df[col_name].values.reshape(-1, 1))
    data = pd.DataFrame(np_scaled)
    
    model =  IsolationForest(contamination=outliers_fraction)
    model.fit(data)

    df['Anomaly'] = model.predict(data)
    
    anomaly = df.loc[df['Anomaly'] == -1, [col_name]] #anomaly

    return anomaly, df

def AnomalyFilterAndInterpolation(f_absCB_resid, col_name):

	anomaly_df, df = IsolationForestOutfitDetectionRes(f_absCB_resid, col_name)
	df.loc[anomaly_df[col_name].index, col_name] = np.NaN
	df2 = pd.DataFrame()
	df2[col_name] = df[col_name].interpolate(method='time', order=3, limit_direction='forward')
	return anomaly_df, df2

def AnomalyFilterAndInterpolationAllColumns(df, anomaly_filter = True, trend_filter = True):

	df_anom_filtered = pd.DataFrame()
	for column in df.columns:
		if anomaly_filter:

			result = seasonal_decompose(df[column], model='additive', period=48, extrapolate_trend=3)
			f_absCB_resid = pd.DataFrame(result.resid)
			f_absCB_trend = pd.DataFrame(result.trend)
			f_absCB_sesonal = pd.DataFrame(result.seasonal)

			anomaly_df, f_absCB_resid_filtered = AnomalyFilterAndInterpolation(f_absCB_resid, 'resid')
			anomaly_df, f_absCB_resid_filtered = AnomalyFilterAndInterpolation(f_absCB_resid_filtered, 'resid')
			anomaly_df, f_absCB_resid_filtered = AnomalyFilterAndInterpolation(f_absCB_resid_filtered, 'resid')
			anomaly_df, df22 = AnomalyFilterAndInterpolation(f_absCB_resid_filtered, 'resid')
			gdp_cycle, f_absCB_trend_filtered = sm.tsa.filters.hpfilter(f_absCB_trend, 50000)

			sum_df = pd.concat([f_absCB_trend_filtered, f_absCB_sesonal, df22['resid']], axis=1)
			sum_df = pd.DataFrame(sum_df.sum(axis=1), columns = [column])
		else:
			sum_df = df

		if trend_filter:
			gdp_cycle, trend_filtered = sm.tsa.filters.hpfilter(sum_df[column], 10)
			trend_filtered = pd.DataFrame(trend_filtered)
			trend_filtered = trend_filtered.rename({str(column)+'_trend': column}, axis=1)
		else:
			trend_filtered = sum_df

		df_anom_filtered = pd.concat([df_anom_filtered, trend_filtered], axis=1)

		del gdp_cycle, trend_filtered, sum_df, result, f_absCB_resid_filtered

	return df_anom_filtered

def get_muf_data(table_name, schema):

	conn = None

	try:

		# print('Connecting to the PostgreSQL database...')
		params = config()
		conn = psycopg2.connect(**params)
		# print(conn)
		select_reqest = conn.cursor()

		select_reqest.execute('SELECT time, muf FROM "%s"."%s"' % (str(schema), str(table_name)))
		select_reqest_rows = select_reqest.fetchall()
		select_reqest.close()
		conn.commit()

	except (Exception, psycopg2.DatabaseError) as error:
		print(error) 
		pass

	return select_reqest_rows

def data_handler(days_for_predict, lag = False, shifts = 6):
	# print('start')
	schemas = get_muf_schemas()
	n = 1
	conf_interval = 0
	df_total = pd.DataFrame()
	for schema in schemas:
		# print('month = ', schema[0])
		if schema[0] == 'month_10':

			tables_names = get_muf_tables_names(schema[0])
			for table_name in tables_names:
				split = table_name[0].split('_')
				if len(split) > 1 and split[1] == 'interp' and split[0] == 'cyprus1':

					df = pd.DataFrame(data=get_muf_data(table_name[0], schema[0]), columns=['time', 'muf'])

					if len(df) > 280:

						year = 2021
						df['date'] = pd.to_datetime(str(year) + '-' + str(schema[0].split('_')[1]) + '-' + str(table_name[0].split('_')[4]), format='%Y-%m-%d').strftime('%Y-%m-%d')
						df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')
						df.index = df['datetime'].apply(pd.to_datetime)
						# print(df)
						df = df.drop(['time', 'datetime', 'date'], axis=1).apply(pd.to_numeric, errors='coerce')
						# conf_interval += mean_confidence_interval(df)
						y_filtered_decomposition = AnomalyFilterAndInterpolationAllColumns(df, anomaly_filter = True, trend_filter = True)
						y_filtered_savgol = savgol1_filter(y_filtered_decomposition)
						y_filtered_savgol = pd.DataFrame({'muf': y_filtered_savgol})
						y_filtered_savgol.index = df.index
						print('day '+ str(n))

						n += 1
						df_total = pd.concat([df_total, y_filtered_savgol])
						# df_total = pd.concat([df_total, y_filtered_decomposition])

	# print('data conf int = ',str(conf_interval/n))
	pd.set_option("display.max_rows", None, "display.max_columns", None)

	df_total = df_total.sort_index()
	print(df_total)

	# mean_confidence_interval(df_total.muf)

	# days_for_predict = 4
	time_step_minute = 5
	hour = 24
	minute = 60
	test_size = (days_for_predict*((hour*minute)/time_step_minute))
	# test_size = (days_for_predict*((hour*minute)/time_step_minute))/len(df_total)
	# print(test_size)
	# size = 0.1
	inter_filter_order = 3 # polynom order in range (1-5)

	for value in range(3):
	 	#Детектирование аномалий (выбросы) (Начало)
		a = IsolationForestOutfitDetection(df_total)

		df_total.loc[a.muf.index, 'muf'] = np.NaN
		df_total['muf'] = df_total['muf'].interpolate(method='polynomial', order=inter_filter_order, limit_direction='both')
		# df_total = df_total.drop(a.muf.index)

	if lag:
		# for i in range(int(df_total.size * size), int(df_total.size * size + 30)):
		# 	df_total["lag_{}".format(i)] = df_total.muf.shift(i)
		# shifts = 6
		df_total = data_lag_and_lag_diff(df_total, shifts, test_size, lag = True, lag_diff = True)

	df_total = Create_time_predictors(df_total)
	df_total = df_total.dropna()
	
	# print(df_total)
	y = df_total.muf.copy()
	X = df_total.drop(['muf', 'anomaly', 'datetime'], axis=1).copy()

	# mean_confidence_interval(y)

	X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	df_total = df_total.drop(['anomaly', 'datetime'], axis=1)

	return X_train_scaled, X_test_scaled, y_train, y_test, df_total