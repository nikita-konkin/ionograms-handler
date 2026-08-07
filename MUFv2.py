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

# sys.path.append(r'C:/data_handler')
# from muf_load_to_db import Load_muf_data_to_muf_db as MUF_to_db

from lfs_header import header
from lfs_data import data

NOISE_COEF = 4 * np.log(2)
STOCK_DPI = 100
STOCK_DEFAULT_FFTLEN = 32768


class MUFProcessor:
    """Класс для обработки MUF данных"""
    
    def __init__(self, options):
        self.options = options
    
    @staticmethod
    def calculate_medians(S):
        """Вычисляет медианы для каждой строки спектрограммы"""
        medians = np.zeros(S.shape[0])
        for i in range(S.shape[0]):
            medians[i] = NOISE_COEF * np.median(S[i, :])
        return medians

    @staticmethod
    def median_equalize(S, medians):
        """Нормализует спектрограмму по медианам"""
        S_M = np.zeros(S.shape)
        for i in range(S.shape[0]):
            S_M[i, :] = S[i, :] / medians[i]
        return S_M

    def _get_file_info(self, filename):
        """Извлекает основную информацию из файла"""
        file_header = header(filename)
        file_data = data(filename)

        sample_rate_raw = file_header['sample_rate'][0]
        dec_raw = file_header['dec'][0]
        cr_raw = file_header['rate'][0]
        cf_raw = file_header['cf'][0]

        # CLI overrides are optional; header values remain default behavior.
        sample_rate_eff = self.options.sample_rate if self.options.sample_rate is not None else sample_rate_raw
        dec_eff = self.options.dec if self.options.dec is not None else dec_raw
        cr = self.options.cr if self.options.cr is not None else cr_raw

        if dec_eff == 0:
            raise ValueError("Effective decimation cannot be zero")

        sr = sample_rate_eff / dec_eff
        dur = file_header['dur'][0]

        if self.options.f_start is not None:
            freq_start = self.options.f_start
        else:
            freq_start = (cf_raw - sample_rate_eff / 2) / 1e6
        freq_stop = freq_start + (dur * cr) / 1e6

        sample_rate_src = "override" if self.options.sample_rate is not None else "header"
        dec_src = "override" if self.options.dec is not None else "header"
        cr_src = "override" if self.options.cr is not None else "header"
        f_start_src = "override" if self.options.f_start is not None else "header-derived"

        print(
            "[%s] params: sample_rate=%s (%s), dec=%s (%s), sr=%s, cr=%s (%s), f_start=%.6f MHz (%s)" % (
                os.path.basename(filename),
                sample_rate_eff,
                sample_rate_src,
                dec_eff,
                dec_src,
                sr,
                cr,
                cr_src,
                freq_start,
                f_start_src,
            )
        )
        
        tx_name = ''.join(file_header['tx_name'][0:7])
        rx_name = ''.join(file_header['rx_name'][0:11])
        sound_type = 'oblique' if (tx_name != rx_name) else 'vertical'
        div_coef = 2.0 if (sound_type == 'oblique') else 4.0
        
        muf_time = "%02d:%02d:%02d" % (
            file_header['start_hour'][0], 
            file_header['start_minute'][0], 
            file_header['start_second'][0]
        )
        
        muf_date = "%d%02d%02d" % (
            file_header['start_year'][0], 
            file_header['start_month'][0], 
            file_header['start_day'][0]
        )
        
        return {
            'header': file_header,
            'data': file_data,
            'cr': cr,
            'sr': sr,
            'dur': dur,
            'freq_start': freq_start,
            'freq_stop': freq_stop,
            'tx_name': tx_name,
            'rx_name': rx_name,
            'sound_type': sound_type,
            'div_coef': div_coef,
            'muf_time': muf_time,
            'muf_date': muf_date
        }

    def _setup_plot_style(self):
        """Настраивает стиль графиков"""
        font = {
            'family': 'Times New Roman',
            'weight': 'normal',
            'size': 15
        }
        plt.rc('font', **font)

    def plot_without_muf(self, filename, window=8192, shift_window=0, plot_shift=0):
        """Строит график без расчета MUF"""
        file_info = self._get_file_info(filename)
        out_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "png")
        os.makedirs(out_dir, exist_ok=True)

        # Проверяем, существует ли уже файл
        f_name = os.path.basename(filename)
        png_filename = os.path.join(out_dir, "%s_z%d_f%d.png" % (
            f_name, self.options.zero_periods, self.options.freq_dec_factor
        ))

        if os.path.isfile(png_filename) and self.options.reanalyze == 0:
            return 0

        # Создаем спектрограмму
        S = stuffr.spectrogram(
            file_info['data'], 
            window=window, 
            zero_periods=self.options.zero_periods
        )
        
        M = self.calculate_medians(S)
        S_M = self.median_equalize(S, M)

        freq_dec_factor = self.options.freq_dec_factor
        num = S.shape[0] // freq_dec_factor
        
        # Подготавливаем данные для графика
        step = round((file_info['freq_stop'] - file_info['freq_start']) / num, 4)
        freq_stop = file_info['freq_start'] + num * step
        freq = np.linspace(file_info['freq_start'], freq_stop, num)
        
        vrange = np.linspace(
            3e8 * (-(file_info['sr'] / file_info['div_coef'])) / file_info['cr'],
            3e8 * (file_info['sr'] / file_info['div_coef']) / file_info['cr'],
            num=S.shape[1]
        ) / 1e3

        # Строим график
        self._setup_plot_style()
        fig, axes = plt.subplots(figsize=(20, 8))

        trace = "%s - %s" % (file_info['tx_name'], file_info['rx_name'])
        date = "%d/%02d/%02d %02d:%02d:%02d" % (
            file_info['header']['start_year'][0],
            file_info['header']['start_month'][0],
            file_info['header']['start_day'][0],
            file_info['header']['start_hour'][0],
            file_info['header']['start_minute'][0],
            file_info['header']['start_second'][0]
        )

        axes.set_title("%s\n%s" % (trace, date), fontsize=20)
        axes.set_xlabel("Frequency (MHz)", fontsize=20)
        axes.set_ylabel("Virtual range (km)", fontsize=20)

        pcm3 = axes.pcolormesh(
            freq, vrange, 
            np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor, ::-1])),
            shading='nearest', cmap="jet", vmin=20.0
        )
        
        axes.set_xlim(file_info['freq_start'], freq_stop)
        axes.set_xticks(np.arange(file_info['freq_start'], freq_stop, 0.5))
        axes.set_xticklabels(axes.get_xticks(), rotation=90)
        axes.set_ylim(1000, 6000)  # rmin, rmax
        
        cbar = plt.colorbar(pcm3)
        cbar.set_label("Power (dB)")

        fig.savefig(png_filename, dpi=150)
        self._cleanup_plot(fig)
        
        return 1

    def plot_with_muf(self, filename, window=8192, shift_window=0, plot_shift=0):
        """Строит график с расчетом MUF"""
        file_info = self._get_file_info(filename)
        out_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "png")
        os.makedirs(out_dir, exist_ok=True)

        # Проверяем, существует ли уже файл
        f_name = os.path.basename(filename)
        png_filename = os.path.join(out_dir, "%s_z%d_f%d.png" % (
            f_name, self.options.zero_periods, self.options.freq_dec_factor
        ))

        if os.path.isfile(png_filename) and self.options.reanalyze == 0:
            return 0

        # Создаем спектрограмму и рассчитываем MUF
        S = stuffr.spectrogram(
            file_info['data'], 
            window=window, 
            zero_periods=self.options.zero_periods
        )
        
        M = self.calculate_medians(S)
        S_M = S  # Используем ненормализованные данные

        freq_dec_factor = self.options.freq_dec_factor
        num = S.shape[0] // freq_dec_factor
        
        step = round((file_info['freq_stop'] - file_info['freq_start']) / num, 4)
        freq_stop = file_info['freq_start'] + num * step
        
        MUF, vrng, muf_column = stuffr.filter2_np_nb_MUF(
            S_M, step, freq_dec_factor, file_info['freq_start']
        )

        # Сохраняем MUF в файл
        muf_filename = '%s\\MUF_%s_%s.csv' % (
            dir_name, str(file_info['tx_name'][0:7]), file_info['muf_date']
        )
        
        vrange_step = (((3e8 * (file_info['sr'] / file_info['div_coef']) / file_info['cr']) * 2) / 1e3) / S.shape[1]
        vrng = (3e8 * ((file_info['sr'] / file_info['div_coef'])) / file_info['cr']) / 1e3 - vrng * vrange_step

        with open(muf_filename, mode='a') as muf_file:
            muf_file_write = csv.writer(muf_file, delimiter=' ')
            muf_file_write.writerow([round(MUF, 2), file_info['muf_time'], vrng])

        # Строим график
        self._create_plot(file_info, S_M, freq_stop, MUF, png_filename)
        return 1

    def _create_plot(self, file_info, S_M, freq_stop, MUF, png_filename):
        """Создает график с данными"""
        freq_dec_factor = self.options.freq_dec_factor
        num = S_M.shape[0] // freq_dec_factor
        
        freq = np.linspace(file_info['freq_start'], freq_stop, num)
        vrange = np.linspace(
            3e8 * (-(file_info['sr'] / file_info['div_coef'])) / file_info['cr'],
            3e8 * (file_info['sr'] / file_info['div_coef']) / file_info['cr'],
            num=S_M.shape[1]
        ) / 1e3

        self._setup_plot_style()
        fig, axes = plt.subplots(constrained_layout=True, figsize=(20, 8))

        trace = "%s - %s" % (file_info['tx_name'], file_info['rx_name'])
        date = "%d/%02d/%02d %02d:%02d:%02d MUF = %.3f" % (
            file_info['header']['start_year'][0],
            file_info['header']['start_month'][0],
            file_info['header']['start_day'][0],
            file_info['header']['start_hour'][0],
            file_info['header']['start_minute'][0],
            file_info['header']['start_second'][0],
            MUF
        )

        axes.set_title("%s\n%s" % (trace, date), fontsize=20)
        axes.set_xlabel("Frequency (MHz)", fontsize=20)
        axes.set_ylabel("Virtual range (km)", fontsize=20)

        pcm3 = axes.pcolormesh(
            freq, vrange,
            np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor, ::-1])),
            shading='nearest', cmap="jet", vmin=0, vmax=60
        )
        
        axes.set_xticks(np.arange(file_info['freq_start'], freq_stop, 0.5))
        axes.set_xticklabels(axes.get_xticks(), rotation=90)
        axes.set_ylim(2000, 4000)
        axes.set_xlim(file_info['freq_start'], 22)
        
        plt.colorbar(pcm3)
        fig.savefig(png_filename, dpi=150)
        self._cleanup_plot(fig)

    def calculate_muf(self, filename, window=8192, shift_window=0, plot_shift=0):
        """Рассчитывает MUF и сохраняет в базу данных"""
        print('MUF processing...')
        file_info = self._get_file_info(filename)
        
        # Создаем спектрограмму
        S = stuffr.spectrogram(
            file_info['data'], 
            window=window, 
            zero_periods=self.options.zero_periods
        )
        
        M = self.calculate_medians(S)
        S_M = self.median_equalize(S, M)

        freq_dec_factor = self.options.freq_dec_factor
        num = S.shape[0] // freq_dec_factor
        
        step = round((file_info['freq_stop'] - file_info['freq_start']) / num, 2)
        freq_stop = file_info['freq_start'] + num * step

        MUF, vrng, muf_column = stuffr.filter2_np_nb_MUF(
            S_M, step, freq_dec_factor, file_info['freq_start']
        )

        # Рассчитываем виртуальный диапазон
        vrange_step = (((3e8 * (file_info['sr'] / file_info['div_coef']) / file_info['cr']) * 2) / 1e3) / S.shape[1]
        muf_row = vrng
        vrng = (3e8 * ((file_info['sr'] / file_info['div_coef'])) / file_info['cr']) / 1e3 - vrng * vrange_step
        
        if vrng < 0:
            vrng = (3e8 * ((file_info['sr'] / file_info['div_coef'])) / file_info['cr']) / 1e3 + vrng

        MUF = round(MUF, 2)
        vrng = round(vrng, 2)

        print('MUF = ', MUF, 'vrange = ', vrng, 'time = ', file_info['muf_time'])

        # Сохраняем в базу данных
        MUF_to_db(
            file_info['header']['start_year'][0],
            file_info['header']['start_month'][0],
            file_info['header']['start_day'][0],
            file_info['muf_time'],
            MUF,
            vrng,
            file_info['rx_name'],
            file_info['tx_name'],
            muf_column,
            muf_row
        )

        print('--------------------------------------------------------------')
        return MUF, vrng

    def plot_without_axes(self, filename, folder, window=16384, shift_window=0, plot_shift=0):
        """Строит график без осей"""
        file_info = self._get_file_info(filename)
        dir_name = self.options.dirname
        
        f_name = os.path.basename(filename)

        # Создаем спектрограмму
        S = stuffr.spectrogram(
            file_info['data'], 
            window=window, 
            zero_periods=self.options.zero_periods
        )
        
        M = self.calculate_medians(S)
        S_M = self.median_equalize(S, M)

        freq_dec_factor = self.options.freq_dec_factor
        num = S.shape[0] // freq_dec_factor
        
        step = round((file_info['freq_stop'] - file_info['freq_start']) / num, 4)
        freq_stop = file_info['freq_start'] + num * step

        out_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "png")
        os.makedirs(out_dir, exist_ok=True)
        png_filename = os.path.join(out_dir, "%s_z%d_f%d_wo_axes_startf_%d_stopf_%d_num_%d.png" % (
            f_name.split('.')[0], self.options.zero_periods, self.options.freq_dec_factor,
            file_info['freq_start'], freq_stop, num
        ))

        if os.path.isfile(png_filename) and self.options.reanalyze == 0:
            print(png_filename, ' - already exists')
            return 0

        # Строим график без осей
        freq = np.linspace(file_info['freq_start'], freq_stop, num)
        vrange = np.linspace(
            3e8 * (-(file_info['sr'] / file_info['div_coef'])) / file_info['cr'],
            3e8 * (file_info['sr'] / file_info['div_coef']) / file_info['cr'],
            num=S.shape[1]
        ) / 1e3

        fig, ax = plt.subplots(constrained_layout=True, figsize=(20, 8))
        pcm3 = ax.pcolormesh(
            freq, vrange,
            np.transpose(stuffr.comprz_dB(S_M[::freq_dec_factor, ::-1])),
            shading='gouraud', cmap="jet"
        )
        
        fig.colorbar(pcm3, ax=ax)
        fig.delaxes(fig.axes[1])
        
        # Убираем оси
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(2500, 3500)
        ax.set_xlim(file_info['freq_start'], freq_stop)
        plt.axis('off')
        plt.box(False)
        
        plt.savefig(
            png_filename, 
            bbox_inches='tight', 
            transparent="True", 
            pad_inches=0, 
            dpi="figure"
        )
        
        self._cleanup_plot(fig)
        return 1

    def plot_stock_wo_axes(self, filename):
        """Build and save stock ionogram image for neural-network datasets."""
        file_info = self._get_file_info(filename)

        # Keep MUF defaults for other modes, but match plot_ionogram sizing in stock mode.
        stock_fftlen = STOCK_DEFAULT_FFTLEN if self.options.fftlen_is_default else self.options.fftlen

        out_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "png")
        os.makedirs(out_dir, exist_ok=True)

        f_name = os.path.basename(filename)
        png_filename = os.path.join(
            out_dir,
            "%s_stock_w%d_y%.1f_%.1f.png" % (
                f_name.split('.')[0],
                stock_fftlen,
                self.options.stock_ylim_min,
                self.options.stock_ylim_max,
            )
        )

        if os.path.isfile(png_filename) and self.options.reanalyze == 0:
            return 0

        S = stuffr.spectrogram_kaiser(
            file_info['data'],
            window=stock_fftlen,
        )

        if not self.options.stock_no_median:
            S = stuffr.median_equalize_rows(S)

        freq_stop = file_info['freq_start'] + (len(file_info['data']) / file_info['sr']) * file_info['cr'] / 1e6
        freq = np.linspace(file_info['freq_start'], freq_stop, num=S.shape[0])

        frange_hz = np.linspace(-file_info['sr'] / 2.0, file_info['sr'] / 2.0, num=S.shape[1])
        delay_ms = frange_hz / file_info['cr'] * 1e3

        ms_to_km = 120e3 / frange_hz.shape[0]
        window_size_in_km = (self.options.stock_ylim_max - self.options.stock_ylim_min) / 1e6 * 3e8
        width_px = max(freq.shape[0], 1)
        height_px = max(window_size_in_km / ms_to_km, 1)

        fig = plt.figure(figsize=(width_px / STOCK_DPI, height_px / STOCK_DPI), dpi=STOCK_DPI)
        ax = fig.add_axes([0, 0, 1, 1])

        power_db = stuffr.power_to_db_floor(S[:, ::-1])
        ax.pcolormesh(
            freq,
            delay_ms,
            np.transpose(power_db),
            cmap="jet",
            vmin=self.options.stock_vmin,
            shading="auto",
        )

        ax.set_xlim(freq[0], freq[-1])
        ax.set_ylim(self.options.stock_ylim_min, self.options.stock_ylim_max)
        ax.axis('off')

        fig.savefig(png_filename, dpi=STOCK_DPI, bbox_inches='tight', pad_inches=0)
        self._cleanup_plot(fig)
        return 1

    def _cleanup_plot(self, fig):
        """Очищает память после построения графика"""
        fig.clf()
        plt.close()
        gc.collect()


class FileProcessor:
    """Класс для обработки файлов и папок"""
    
    def __init__(self, options, muf_processor):
        self.options = options
        self.muf_processor = muf_processor

    def process_folders(self, dirname):
        """Обрабатывает данные из нескольких папок"""
        list_dir = os.listdir(dirname)
        list_dir.sort()
        
        print(f"Found {len(list_dir)} folders")
        self._print_numbered_list(list_dir)
        
        folders_range = self._get_user_range_input(len(list_dir), "folders")
        if not folders_range:
            return 0
            
        start_folder, stop_folder = folders_range
        new_list_dir = list_dir[start_folder:stop_folder + 1]
        
        print(f"{len(new_list_dir)} folders will be processed")
        print(*new_list_dir, sep=', ')
        
        mode = self._get_processing_mode()
        
        for folder in new_list_dir:
            self._process_folder(folder, dirname, mode)
            
        return 1

    def process_single_file(self, dirname):
        """Обрабатывает один файл"""
        list_dir = os.listdir(dirname)
        list_dir.sort()
        
        print(f"Found {len(list_dir)} files")
        self._print_numbered_list(list_dir)
        
        files_range = self._get_user_range_input(len(list_dir), "files")
        if not files_range:
            return 0
            
        start_file, stop_file = files_range
        new_list_dir = list_dir[start_file:stop_file + 1]
        
        print(f"{len(new_list_dir)} files will be processed")
        print(*new_list_dir, sep=', ')
        
        mode = self._get_processing_mode()
        
        for file_name in new_list_dir:
            current_file = os.path.join(os.path.abspath(dirname), file_name)
            print(f"Processing file: {os.path.basename(current_file)}")
            self._execute_processing_function(mode, current_file, file_name)
            
        return 1

    def process_files_in_directory(self, dirname):
        """Обрабатывает все файлы в директории"""
        files = glob.glob("%s/*.lfs" % os.path.abspath(dirname))
        file_count = len(files)
        print(f"Files found: {file_count}")
        files.sort()

        mode = self._get_processing_mode()

        if files:
            for file_num, filename in enumerate(files, 1):
                self._execute_processing_function(mode, filename, "")
                print(f"Processed files: {file_num} of {file_count}")
                
        return 1

    def _print_numbered_list(self, items):
        """Печатает пронумерованный список"""
        for i, item in enumerate(items):
            print(f"{i}: {item}")

    def _get_user_range_input(self, max_items, item_type):
        """Получает диапазон от пользователя"""
        print(f"Input range of {item_type} to process (e.g., '1-12')")
        user_input = input()
        
        try:
            start_stop = user_input.split('-')
            start = int(start_stop[0])
            stop = int(start_stop[1])
            
            if stop >= max_items:
                print(f"Error: Maximum {max_items} {item_type}!")
                return None
                
            return start, stop
        except (ValueError, IndexError):
            print("Invalid input format")
            return None

    def _get_processing_mode(self):
        """Получает режим обработки от пользователя"""
        print("Choose mode:")
        print("1 - plot")
        print("2 - plot_wo_muf")
        print("3 - plot_wo_axes")
        print("4 - MUF")
        print("5 - plot_stock_wo_axes")
        return input()

    def _process_folder(self, folder, base_dir, mode):
        """Обрабатывает одну папку"""
        print(folder)
        current_folder = os.path.join(os.path.abspath(base_dir), folder)
        print(f"Work directory: {current_folder}")
        
        files = glob.glob("%s/*.lfs" % current_folder)
        file_count = len(files)
        print(f"Files found: {file_count}")
        files.sort()
        
        if files:
            for file_num, filename in enumerate(files, 1):
                print(f"Processing file: {os.path.basename(filename)}")
                self._execute_processing_function(mode, filename, folder)
                print(f"Processed files: {file_num} of {file_count}")

    def _execute_processing_function(self, mode, filename, folder):
        """Выполняет соответствующую функцию обработки"""
        if mode == '1':
            self.muf_processor.plot_with_muf(
                filename=filename,
                window=self.options.fftlen,
                shift_window=self.options.shift_window,
                plot_shift=self.options.plot_shift
            )
        elif mode == '2':
            self.muf_processor.plot_without_muf(
                filename=filename,
                window=self.options.fftlen,
                shift_window=self.options.shift_window,
                plot_shift=self.options.plot_shift
            )
        elif mode == '3':
            self.muf_processor.plot_without_axes(
                filename=filename,
                folder=folder,
                window=self.options.fftlen,
                shift_window=self.options.shift_window,
                plot_shift=self.options.plot_shift
            )
        elif mode == '4':
            self.muf_processor.calculate_muf(
                filename=filename,
                window=self.options.fftlen,
                shift_window=self.options.shift_window,
                plot_shift=self.options.plot_shift
            )
        elif mode == '5':
            self.muf_processor.plot_stock_wo_axes(filename=filename)


def main():
    """Основная функция программы"""
    options = parse_arguments()
    muf_processor = MUFProcessor(options)
    file_processor = FileProcessor(options, muf_processor)
    
    dirname = options.dirname

    if options.folders_range:
        file_processor.process_folders(dirname)
    elif options.one_file:
        file_processor.process_single_file(dirname)
    else:
        file_processor.process_files_in_directory(dirname)
        
    return 0


def parse_arguments():
    """Парсит аргументы командной строки"""
    parser = OptionParser(conflict_handler="resolve")

    parser.add_option("-d", "--dirname", dest="dirname", action="store", type="string",
                      default=os.path.dirname(os.path.realpath(sys.argv[0])),
                      help="LFS file's directory name (default: current directory)")

    parser.add_option("-f", "--freq_dec_factor", dest="freq_dec_factor", action="store", type="int",
                      default=1, help="Frequency decimation factor (default: 1)")

    parser.add_option("-r", "--reanalyze", dest="reanalyze", action="store", type="int",
                      default=0, help="Reanalyze results? (default 0)")

    parser.add_option("-l", "--fftlen", dest="fftlen", action="store", type="int",
                      default=32768, help="FFT length (default: 32768)")

    parser.add_option("--sample-rate", dest="sample_rate", action="store", type="float",
                      default=None, help="Override sample rate from header (Hz)")

    parser.add_option("--dec", dest="dec", action="store", type="int",
                      default=None, help="Override decimation from header")

    parser.add_option("--cr", dest="cr", action="store", type="float",
                      default=None, help="Override chirp rate from header (Hz/s)")

    parser.add_option("--f-start", dest="f_start", action="store", type="float",
                      default=None, help="Override start frequency (MHz)")

    parser.add_option("-w", "--shift_window", dest="shift_window", action="store", type="int",
                      default=800, help="First shift window (default: 800)")

    parser.add_option("-p", "--plot_shift", dest="plot_shift", action="store", type="int",
                      default=0, help="Plot shift in pixels (default: 0)")

    parser.add_option("-z", "--zero_periods", dest="zero_periods", action="store", type="int",
                      default=0, help="Zero periods in spectrum (default: 0)")

    parser.add_option("-t", "--folders_range", dest="folders_range", action="store", type="int",
                      default=0, help="Process data from some folders in a row")
                      
    parser.add_option("-o", "--one_file", dest="one_file", action="store", type="int",
                      default=0, help="Process files from folder in a row")

    parser.add_option("--stock-vmin", dest="stock_vmin", action="store", type="float",
                      default=0.0, help="Minimum dB colormap value for plot_stock (default: 0)")

    parser.add_option("--stock-ylim-min", dest="stock_ylim_min", action="store", type="float",
                      default=8.0, help="Minimum delay (ms) for plot_stock Y axis (default: 8.0)")

    parser.add_option("--stock-ylim-max", dest="stock_ylim_max", action="store", type="float",
                      default=12.0, help="Maximum delay (ms) for plot_stock Y axis (default: 12.0)")

    parser.add_option("--stock-no-median", dest="stock_no_median", action="store_true",
                      default=False, help="Disable median equalization for plot_stock")

    options = parser.parse_args()[0]

    # Track whether --fftlen was explicitly provided by user.
    options.fftlen_is_default = True
    for arg in sys.argv[1:]:
        if arg in ("-l", "--fftlen") or arg.startswith("--fftlen="):
            options.fftlen_is_default = False
            break

    return options


if __name__ == "__main__":
    main()