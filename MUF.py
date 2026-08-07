#!/usr/bin/python

# -*- coding: utf-8 -*-

import csv
import gc
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
import math

import stuffr
from optparse import OptionParser

sys.path.append(r'C:/data_handler')
from muf_load_to_db import Load_muf_data_to_muf_db as MUF_to_db

from lfs_header import header
from lfs_data import data

NOISE_COEF = 4*np.log(2)

def medians(S):
		medians = np.zeros(S.shape[0])
		for i in np.arange(S.shape[0]):
				medians[i] = NOISE_COEF*np.median(S[i,:])
		return medians

def medianEqualize(S, medians):
		S_M = np.zeros(S.shape)
		for i in np.arange(S.shape[0]):
				S_M[i,:] = S[i,:]/medians[i]
		return S_M

def plot_wo_muf(filename, window=8192, shift_window=0, plot_shift=0):
		file_header = header(filename)
		(file_name, file_ext) = os.path.splitext(filename)
		dir_name = options.dirname
		f = open(filename)
		f_name = os.path.basename(f.name)
		f.close()
		png_filename = "%s/%s_z%d_f%d.png"%(dir_name, f_name, options.zero_periods, options.freq_dec_factor)

		if os.path.isfile(png_filename) and options.reanalyze == 0:
				return 0		

		# file_header = header(filename)
		#print file_header
		file_data = data(filename)
		cr = file_header['rate'][0]
		sr = file_header['sample_rate'][0] / file_header['dec'][0]
		dur = file_header['dur'][0]
		freq_start	= (file_header['cf'][0] - file_header['sample_rate'][0]/2) / 1e6 
		freq_stop	 = freq_start + (dur * cr) / 1e6
		# print(freq_start, freq_stop)
		# rmin = 2500##file_header['rmin']
		# rmax = 3200##file_header['rmax']
		rmin = 2000 ##file_header['rmin']
		rmax = 4000 ##file_header['rmax']
		tx_name = ''.join(file_header['tx_name'][0:7])
		rx_name = ''.join(file_header['rx_name'][0:11])
		sound_type	= 'oblique' if (tx_name != rx_name) else 'vertical'
		div_coef		= 2.0 if (sound_type == 'oblique') else 4.0
		S					 = stuffr.spectrogram(file_data,window=window,zero_periods=options.zero_periods) #	,shift_window=shift_window,plot_shift=plot_shift)
		M = medians(S)
		M_log = 10.0*np.log10(M)
		S_M	= medianEqualize(S, M)
		# S_M	= S
		print("S_M_shape = ", S_M.shape[0], S_M.shape[1])
 
		freq_dec_factor = options.freq_dec_factor
		num=((S.shape[0])//(freq_dec_factor))
		vrange			= np.linspace( 3e8*(-(sr/div_coef))/cr, 3e8*(sr/div_coef)/cr , num=S.shape[1])/1e3
		step = round((freq_stop - freq_start)/num, 4)
		freq_stop = freq_start + num*step
		# MUF, vrng, muf_column = stuffr.filter2_np_nb_MUF(S_M, step, freq_dec_factor, freq_start)

		muf_time		= "%02d:%02d:%02d"\
							%(file_header['start_hour'][0], file_header['start_minute'][0], file_header['start_second'][0])
		muf_date = "%d%02d%02d"%(file_header['start_year'][0], file_header['start_month'][0],	file_header['start_day'][0])
		station = file_header['sample_rate'][0]
		muf_filename = '%s\MUF_%s_%s.csv'%(dir_name, str(tx_name[0:7]), muf_date)

		vrange_step = (((3e8*(sr/div_coef)/cr)*2)/1e3)/(S.shape[1])
		print('virange_step = ', vrange_step)

		print("plotting...")

		freq = np.linspace(freq_start, freq_stop, num)
		# print((freq))
		# vrange = np.linspace( 3e8*(-(sr/div_coef))/cr, 3e8*(sr/div_coef)/cr , num=S.shape[1])/1e3
		# print(vrange)
		# mycmap = plt.get_cmap('jet')

		font = {'family' : 'Times New Roman',
				'weight' : 'normal',
				'size'	 : 15}
		plt.rc('font', **font)

		fig, axes = plt.subplots(figsize=(20,8) )


		trace	 = "%s - %s"%(tx_name,rx_name)
		date		= "%d/%02d/%02d %02d:%02d:%02d "\
									%(file_header['start_year'][0], file_header['start_month'][0],	file_header['start_day'][0], 
									file_header['start_hour'][0], file_header['start_minute'][0], file_header['start_second'][0]) 


		axes.set_title("%s\n%s"%(trace, date), fontsize=20)

		axes.set_xlabel("Frequency (MHz)", fontsize=20)
		axes.set_ylabel("Virtual range (km)", fontsize=20)

		pcm3 = axes.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='nearest', cmap="jet", vmin=20.0)
		# pcm3 = axes.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='nearest', cmap="jet", vmin=0, vmax=16)
		# pcm3 = axes.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='nearest', cmap="jet", vmin=-1)
		axes.set_xlim(freq_start, freq_stop)
		axes.set_xticks(np.arange(freq_start, freq_stop, 0.5))
		axes.set_xticklabels(axes.get_xticks(), rotation = 90)
		axes.set_ylim(rmin, rmax)
		# axes.set_xlim(freq_start, 22)
		cbar = plt.colorbar(pcm3)
		cbar.set_label("Power (dB)")


		fig.savefig(png_filename, dpi=150) 

		fig.clf()
		plt.close()
		gc.collect()


		del M, M_log, S, S_M, freq, vrange

def plot(filename, window=8192, shift_window=0, plot_shift=0):
		file_header = header(filename)
		(file_name, file_ext) = os.path.splitext(filename)
		dir_name = options.dirname
		f = open(filename)
		f_name = os.path.basename(f.name)
		f.close()
		png_filename = "%s/%s_z%d_f%d.png"%(dir_name, f_name, options.zero_periods, options.freq_dec_factor)

		if os.path.isfile(png_filename) and options.reanalyze == 0:
				return 0		

		# file_header = header(filename)
		#print file_header
		file_data = data(filename)
		cr = file_header['rate'][0]
		sr = file_header['sample_rate'][0] / file_header['dec'][0]
		dur = file_header['dur'][0]
		freq_start	= (file_header['cf'][0] - file_header['sample_rate'][0]/2) / 1e6 
		freq_stop	 = freq_start + (dur * cr) / 1e6
		# print(freq_start, freq_stop)
		# rmin = 2500##file_header['rmin']
		# rmax = 3200##file_header['rmax']
		rmin = 2000 ##file_header['rmin']
		rmax = 4000 ##file_header['rmax']
		tx_name = ''.join(file_header['tx_name'][0:7])
		rx_name = ''.join(file_header['rx_name'][0:11])
		sound_type	= 'oblique' if (tx_name != rx_name) else 'vertical'
		div_coef		= 2.0 if (sound_type == 'oblique') else 4.0
		S					 = stuffr.spectrogram(file_data,window=window,zero_periods=options.zero_periods)#,shift_window=shift_window,plot_shift=plot_shift)
		M = medians(S)
		M_log = 10.0*np.log10(M)
		# S_M	= medianEqualize(S, M)
		S_M	= S
		print("S_M_shape = ", S_M.shape[0], S_M.shape[1])
 
		freq_dec_factor = options.freq_dec_factor
		num=((S.shape[0])//(freq_dec_factor))

		step = round((freq_stop - freq_start)/num, 4)
		freq_stop = freq_start + num*step
		MUF, vrng, muf_column = stuffr.filter2_np_nb_MUF(S_M, step, freq_dec_factor, freq_start)

		muf_time		= "%02d:%02d:%02d"\
							%(file_header['start_hour'][0], file_header['start_minute'][0], file_header['start_second'][0])
		muf_date = "%d%02d%02d"%(file_header['start_year'][0], file_header['start_month'][0],	file_header['start_day'][0])
		station = file_header['sample_rate'][0]
		muf_filename = '%s\MUF_%s_%s.csv'%(dir_name, str(tx_name[0:7]), muf_date)
		# print(muf_filename)
		# print('S = ', S.shape[1])
		# print('vrng = ',vrng)
		print('vrng = ', vrng)
		vrange_step = (((3e8*(sr/div_coef)/cr)*2)/1e3)/(S.shape[1])
		print('virange_step = ', vrange_step)
		# print(3e8*(-(sr/div_coef))/cr)
		vrng =	(3e8*((sr/div_coef))/cr)/1e3 - vrng*vrange_step
		muf_row = vrng
		# vrng = np.linspace( 3e8*(-(sr/div_coef))/cr, 3e8*(sr/div_coef)/cr , num=S.shape[1])/1e3

		print('MUF = ', MUF, 'vrange = ',vrng, 'time = ', muf_time)

		with open(muf_filename, mode='a') as muf_file:
				muf_file_write = csv.writer(muf_file, delimiter=' ')
				# for i, f in enumerate(freq):
				muf_file_write.writerow([round(MUF, 2), muf_time, vrng]) 

		print("plotting...")

		freq = np.linspace(freq_start, freq_stop, num)
		# print(type(freq))
		vrange = np.linspace( 3e8*(-(sr/div_coef))/cr, 3e8*(sr/div_coef)/cr , num=S.shape[1])/1e3
		# print(vrange)
		mycmap = plt.get_cmap('jet')

		font = {'family' : 'Times New Roman',
				'weight' : 'normal',
				'size'	 : 15}
		plt.rc('font', **font)

		fig, axes = plt.subplots(constrained_layout=True, figsize=(20,8) )

		# plt.subplots_adjust(hspace=0.001, wspace=0.05)

		# ax1, ax3 = axes[:, 0]
		# axes[0,1].remove()
		# axes[0,0].remove()
		# freq_stop = 18
		trace	 = "%s - %s"%(tx_name,rx_name)
		date		= "%d/%02d/%02d %02d:%02d:%02d	MUF = %.3f"\
									%(file_header['start_year'][0], file_header['start_month'][0],	file_header['start_day'][0], 
									file_header['start_hour'][0], file_header['start_minute'][0], file_header['start_second'][0], MUF) 


		axes.set_title("%s\n%s"%(trace, date), fontsize=20)

		axes.set_xlabel("Frequency (MHz)", fontsize=20)
		axes.set_ylabel("Virtual range (km)", fontsize=20)
		
		pcm3 = axes.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='nearest', cmap="jet", vmin=0, vmax=60)
		# pcm3 = axes.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='nearest', cmap="jet", vmin=-1)
		# ax3.set_xlim(freq_start, freq_stop)
		axes.set_xticks(np.arange(freq_start, freq_stop, 0.5))
		axes.set_xticklabels(axes.get_xticks(), rotation = 90)
		axes.set_ylim(rmin, rmax)
		axes.set_xlim(freq_start, 22)
		plt.colorbar(pcm3)


		fig.savefig(png_filename, dpi=150) 

		fig.clf()
		plt.close()
		gc.collect()


		del M, M_log, S, S_M, freq, vrange

def MUF(filename, window=8192, shift_window=0, plot_shift=0):

		print('MUF processing...')
		dir_name = options.dirname
		file_header = header(filename)
		file_data = data(filename)
		cr = file_header['rate'][0]
		sr = file_header['sample_rate'][0] / file_header['dec'][0]
		dur = file_header['dur'][0]
		tx_name = ''.join(file_header['tx_name'][0:7])
		rx_name = ''.join(file_header['rx_name'][0:11])
		sound_type	= 'oblique' if (tx_name != rx_name) else 'vertical'
		div_coef		= 2.0 if (sound_type == 'oblique') else 4.0

		freq_start	= (file_header['cf'][0] - file_header['sample_rate'][0]/2) / 1e6 
		freq_stop	 = freq_start + (dur * cr) / 1e6
		

		S					 = stuffr.spectrogram(file_data,window=window,zero_periods=options.zero_periods)#,shift_window=shift_window,plot_shift=plot_shift)
		M = medians(S)
		M_log = 10.0*np.log10(M)
		S_M	= medianEqualize(S, M)


		# print("S_M_shape = ", S_M.shape[0], S_M.shape[1])
 
		freq_dec_factor = options.freq_dec_factor
		num=((S.shape[0])//(freq_dec_factor))
		step = round((freq_stop - freq_start)/num, 2)
		freq_stop = freq_start + num*step

		MUF, vrng, muf_column = stuffr.filter2_np_nb_MUF(S_M, step, freq_dec_factor, freq_start)

		muf_time		= "%02d:%02d:%02d"\
							%(file_header['start_hour'][0], file_header['start_minute'][0], file_header['start_second'][0])
		muf_date = "%d%02d%02d"%(file_header['start_year'][0], file_header['start_month'][0],	file_header['start_day'][0])
		station = file_header['sample_rate'][0]
		muf_filename = '%s\MUF_%s_%s.csv'%(dir_name, str(tx_name[0:7]), muf_date)
		# print('vrng = ', vrng)
		vrange_step = (((3e8*(sr/div_coef)/cr)*2)/1e3)/(S.shape[1])
		# print(vrng, vrange_step)
		muf_row = vrng
		vrng =	(3e8*((sr/div_coef))/cr)/1e3 - vrng*vrange_step
		if vrng < 0:
			# print(vrng, vrange_step)
			vrng = (3e8*((sr/div_coef))/cr)/1e3 + vrng

		print('MUF = ', MUF, 'vrange = ',vrng, 'time = ', muf_time)
		MUF = round(MUF, 2)
		vrng = round(vrng, 2)


		MUF_to_db(file_header['start_year'][0], file_header['start_month'][0], 
			file_header['start_day'][0], muf_time, MUF, vrng, rx_name, tx_name,
				muf_column, muf_row)

		print('--------------------------------------------------------------')
		# with open(muf_filename, mode='a') as muf_file:
		#		 muf_file_write = csv.writer(muf_file, delimiter=' ')
		#		 # for i, f in enumerate(freq):
		#		 muf_file_write.writerow([muf_time,	round(MUF, 2), vrng])

def plot_wo_axes(filename, folder, window=16384, shift_window=0, plot_shift=0):

		file_header = header(filename)
		(file_name, file_ext) = os.path.splitext(filename)
		dir_name = options.dirname
		f = open(filename)
		f_name = os.path.basename(f.name)
		f.close()

		file_data = data(filename)
		cr = file_header['rate'][0]
		sr = file_header['sample_rate'][0] / file_header['dec'][0]
		dur = file_header['dur'][0]
		freq_start	= (file_header['cf'][0] - file_header['sample_rate'][0]/2) / 1e6 
		freq_stop	= (freq_start) + (dur * cr) / 1e6
		# print(freq_start, '---', freq_stop)
		# print(dur, '---', cr)
		rmin = 2500 #file_header['rmin']
		rmax = 4000 #file_header['rmax']
		tx_name = ''.join(file_header['tx_name'][0:7])
		rx_name = ''.join(file_header['rx_name'][0:11])
		sound_type	= 'oblique' if (tx_name != rx_name) else 'vertical'
		div_coef		= 2.0 if (sound_type == 'oblique') else 4.0
		print(options.zero_periods)
		S				= stuffr.spectrogram(file_data, window=window, zero_periods=options.zero_periods)#,shift_window=shift_window,plot_shift=plot_shift)
		M = medians(S)
		M_log = 10.0*np.log10(M)
		S_M	= medianEqualize(S, M)
		# S_M	= S
		print("S_M_shape = ", S_M.shape[0], S_M.shape[1])
 
		freq_dec_factor = options.freq_dec_factor
		num=((S.shape[0])//(freq_dec_factor))
		step = round((freq_stop - freq_start)/num, 4)
		freq_stop = freq_start + num*step

		png_filename = "%s/%s/%s_z%d_f%d_wo_axes_startf_%d_stopf_%d_num_%d.png"%(dir_name, folder, f_name.split('.')[0], options.zero_periods, options.freq_dec_factor, freq_start, freq_stop, num)
		
		if os.path.isfile(png_filename) and options.reanalyze == 0:
				print(png_filename, ' - already exist')
				return 0	 

		# MUF, vrng, muf_column = stuffr.filter2_np_nb_MUF(S_M, step, freq_dec_factor, freq_start)

		# muf_time		= "%02d:%02d:%02d"\
		# 					%(file_header['start_hour'][0], file_header['start_minute'][0], file_header['start_second'][0])
		# muf_date = "%d%02d%02d"%(file_header['start_year'][0], file_header['start_month'][0],	file_header['start_day'][0])
		# station = file_header['sample_rate'][0]
		# muf_filename = '%s\MUF_%s_%s.csv'%(dir_name, str(tx_name[0:7]), muf_date)

		# print('vrng = ', vrng)
		# vrange_step = (((3e8*(sr/div_coef)/cr)*2)/1e3)/(S.shape[1])
		# print('virange_step = ', vrange_step)
		# # print(3e8*(-(sr/div_coef))/cr)
		# vrng =	(3e8*((sr/div_coef))/cr)/1e3 - vrng*vrange_step
		# muf_row = vrng
		# # vrng = np.linspace( 3e8*(-(sr/div_coef))/cr, 3e8*(sr/div_coef)/cr , num=S.shape[1])/1e3

		# print('MUF = ', MUF, 'vrange = ',vrng, 'time = ', muf_time)

		# with open(muf_filename, mode='a') as muf_file:
		# 		muf_file_write = csv.writer(muf_file, delimiter=' ')
		# 		# for i, f in enumerate(freq):
		# 		muf_file_write.writerow([round(MUF, 2), muf_time, vrng, MUF]) 

		print("plotting...")

		freq = np.linspace(freq_start, freq_stop, num)
		# print(type(freq))
		vrange = np.linspace( 3e8*(-(sr/div_coef))/cr, 3e8*(sr/div_coef)/cr , num=S.shape[1])/1e3

		
		# print(vrange)
		fig, ax = plt.subplots(constrained_layout=True, figsize=(20,8))
		# ax.set_xticks(np.arange(freq_start, freq_stop, 0.5))
		# ax.set_ylim(rmin, rmax)
		# ax.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='auto')
		pcm3 = ax.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='auto', cmap="jet")
		# axes.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='nearest', cmap="jet", vmin=20.0)
		# axes.pcolormesh(freq, vrange, np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor,::-1])), shading='nearest', cmap="jet", vmin=0, vmax=60)

		fig.colorbar(pcm3, ax=ax)
		fig.delaxes(fig.axes[1])
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.set_ylim(rmin, rmax)
		ax.set_xlim(freq_start, freq_stop, 0.5)
		# ax.set_xticks(np.arange(freq_start, freq_stop, 0.5))
		plt.axis('off')
		plt.box(False)
		plt.savefig(png_filename, bbox_inches='tight', transparent="True", pad_inches=0, dpi=160)
		# cd.remove()
		plt.clf()
		fig.clf()
		plt.close()
		gc.collect()

def Process_data_by_folders(dirname):

		list_dir=os.listdir(dirname)
		list_dir.sort()
		print("Found "+str(len(list_dir))+" folders")
		print(*[list((i, list_dir[i])) for i in range(len(list_dir))], sep=', ')
		print("Input range of folders to process")
		print("For example '1-12'")
		folders_count=input()
		folders_start_stop=folders_count.split('-')
		start_folder=int(folders_start_stop[0])
		stop_folder=int(folders_start_stop[1])
		if stop_folder > len(list_dir):
				print("Error: Max "+str(len(list_dir))+" folders!")
		else:
				folders_range=stop_folder-start_folder
				print(str(folders_range+1) + " folders will be processed")
				new_list_dir = list_dir[start_folder:stop_folder+1]
				print(*new_list_dir, sep=', ')
				print("-----------------------------")
				print("Choose mode:")
				print("1 - plot")
				print("2 - plot_wo_muf")
				print("3 - plot_wo_axes")
				print("4 - MUF")
				fnc = input()

				for folder in new_list_dir:
						print(folder)
						current_folder = str(os.path.abspath(dirname))+'\\'+str(folder)
						print("Work directory: %s"%current_folder)
						files = glob.glob("%s/*.lfs"%(current_folder))
						file_count = len(files)
						print("Files founded: %d"%file_count)
						files.sort()
						if len(files) > 0:
								file_num = 1
								for filename in files:
										print("Processed file: %s"%os.path.basename(filename))
										fnc_choose_and_start(fnc, filename, str(folder), options.fftlen, options.shift_window, options.plot_shift)
										print("Processed files: %d of %d"%(file_num, file_count))
										file_num += 1

		return 0
			
def Process_one_file(dirname):

		list_dir=os.listdir(dirname)
		list_dir.sort()
		print("Found "+str(len(list_dir))+" files")
		print(*[list((i, list_dir[i])) for i in range(len(list_dir))], sep=', ')
		print("Input range of files to process")
		print("For example '1-12'")
		folders_count=input()
		folders_start_stop=folders_count.split('-')
		start_folder=int(folders_start_stop[0])
		stop_folder=int(folders_start_stop[1])
		if stop_folder > len(list_dir):
				print("Error: Max "+str(len(list_dir))+" files!")
		else:
				folders_range=stop_folder-start_folder
				print(str(folders_range+1) + " files will be processed")
				new_list_dir = list_dir[start_folder:stop_folder+1]
				print(*new_list_dir, sep=', ')
				print("-----------------------------")
				print("Choose mode:")
				print("1 - plot")
				print("2 - plot_wo_muf")
				print("3 - plot_wo_axes")
				print("4 - MUF")
				fnc = input()
				
				for folder in new_list_dir:
						print(folder)
						current_file = str(os.path.abspath(dirname))+'\\'+str(folder)

						print("Work file: %s"%current_file)

						filename = current_file
						print("Processed file: %s"%os.path.basename(filename))
						fnc_choose_and_start(fnc, filename, options.fftlen, options.shift_window, options.plot_shift)



def fnc_choose_and_start(fnc, filename, folder, window, shift_window, plot_shift):

	if fnc == '1':
		plot(filename=filename, window=window, 
					 shift_window=shift_window,	plot_shift=plot_shift)
	elif fnc == '2':
		plot_wo_muf(filename=filename, window=window, 
					 shift_window=shift_window,	plot_shift=plot_shift)
	elif fnc == '3':
		plot_wo_axes(filename=filename, window=window, folder=folder,
					 shift_window=shift_window,	plot_shift=plot_shift)
	elif fnc == '4':
		MUF(filename=filename, window=window, 
					 shift_window=shift_window,	plot_shift=plot_shift)


def main():

		dirname=options.dirname

		if options.folders_range:
				Process_data_by_folders(dirname)
		if options.one_file:
				Process_one_file(dirname)
		else:
				files = glob.glob("%s/*.lfs"%(os.path.abspath(dirname)))
				file_count = len(files)
				print("Files founded: %d"%file_count)
				files.sort()

				print("Choose mode:")
				print("1 - plot")
				print("2 - plot_wo_muf")
				print("3 - plot_wo_axes")
				print("4 - MUF")
				fnc = input()

				if len(files) > 0:
						file_num = 1
						for filename in files:
							fnc_choose_and_start(fnc, filename, options.fftlen, options.shift_window, options.plot_shift)
							print("Processed files: %d of %d"%(file_num, file_count))
							file_num += 1 
		return 0

global options
global args

parser = OptionParser(conflict_handler="resolve")

parser.add_option("-d", "--dirname",dest="dirname", action="store", type="string",
															default=os.path.dirname(os.path.realpath(sys.argv[0])),
															help="LFS file's directory name (default: current directory)") 

parser.add_option("-f", "--freq_dec_factor",dest="freq_dec_factor", action="store", type="int",
															default=1,
															help="Frequency decimation factor (default: 1)")

parser.add_option("-r", "--reanalyze",dest="reanalyze", action="store", type="int",
															default=0,
															help="Reanalyze results? (default 0)")

parser.add_option("-l", "--fftlen",dest="fftlen", action="store", type="int",
															default=8192,
															help="FFT length (default: 8192)")
																																																																																																										
parser.add_option("-w", "--shift_window",dest="shift_window", action="store", type="int",
															default=800,
															help="First shift window (default: 800")

parser.add_option("-p", "--plot_shift",dest="plot_shift", action="store", type="int",
															default=0,
															help="Plot shift in pixels (default: 0)")

parser.add_option("-z", "--zero_periods",dest="zero_periods", action="store", type="int",
															default=0,
															help="Zero periods in spectrum (default: 0)")

parser.add_option("-t", "--folders_range",dest="folders_range", action="store", type="int",
															default=0,
															help="Process data from some folders in a row")
parser.add_option("-o", "--one_file",dest="one_file", action="store", type="int",
															default=0,
															help="Process data from some folders in a row")


(options, args) = parser.parse_args()

if __name__ == "__main__":
		main()																																	
