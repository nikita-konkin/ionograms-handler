#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Построение ионограммы по файлу разностного сигнала (.out).
Код взят из ноутбука Chirp_calc_vert-50bit-corr-yalchik-Copy3.ipynb.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


# Параметры по умолчанию (из ноутбука)
DEFAULT_SAMPLE_RATE = 25e6
DEFAULT_DEC = 625
DEFAULT_CR = 0.1e6   # скорость ЛЧМ, Гц/с
DEFAULT_F_START = 7.5   # МГц

# Параметры для сохранения
DATA_DIR = '/Users/w/lfs/ionograms/эксперимент 15.10 - 04.11'
FILE_TYPE = 'lfs'
IMG_DIR = 'png'
DPI = 100


def spectrogram(x, window):
    """Спектрограмма: разбиение на окна, FFT, мощность.
    Окно Кайзера (beta=4), как в ноутбуке.
    """
    wfv = np.kaiser(int(window), 4)
    Nwindow = int(np.floor(len(x) / window))
    res = np.zeros([Nwindow, window], dtype=np.float64)
    for i in range(Nwindow):
        segment = wfv * x[i * window + np.arange(window)]
        res[i, :] = np.abs(np.fft.fftshift(np.fft.fft(segment))) ** 2
    return res


def median_equalize(S):
    """Нормировка по медиане по каждой строке (частоте)."""
    M = np.zeros_like(S, dtype=S.dtype)
    for i in range(S.shape[0]):
        med = np.median(S[i, :])
        if med > 0:
            M[i, :] = S[i, :] / med
        else:
            M[i, :] = S[i, :]
    return M


def load_out_file(path):
    """Загрузка комплексного сигнала из файла .out."""
    data = np.fromfile(path, dtype=np.complex64, offset=512)
    return data


def plot_ionogram(
    data_2d,
    data_file_len,
    sr,
    cr,
    f_start,
    title=None,
    vmin=0,
    vmax=None,
    ylim_ms=(8, 11),
    out_path=None,
):
    """
    Построение 2D-ионограммы: частота (МГц) vs задержка (мс).
    Цветовая шкала мощности — от 0 дБ.
    data_2d: матрица спектрограммы (n_profiles x window)
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Ось X: частота (МГц)
    freq_mhz = np.linspace(
        f_start,
        f_start + data_file_len / sr * cr / 1e6,
        num=data_2d.shape[0],
    )
    # Ось Y: задержка (мс). tau = f_diff / cr => delay_ms = f_diff / cr * 1e3
    frange_hz = np.linspace(-sr / 2.0, sr / 2.0, num=data_2d.shape[1])
    delay_ms = frange_hz / cr * 1e3

    # Отрисовка: data[:, ::-1] — разворот по частоте
    power_db = 10.0 * np.log10(np.maximum(data_2d[:, ::-1], 1e-20))
    img = ax1.pcolormesh(
        freq_mhz,
        delay_ms,
        np.transpose(power_db),
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    ax1.set_ylim(ylim_ms)
    ax1.set_xlabel("Частота, МГц")
    ax1.set_ylabel("Задержка, мс")

    # Второй x-axis: время (с)
    ax2 = ax1.twiny()
    ax2.set_xlabel("Время, с")
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(
        [round((x - f_start) * 1e6 / cr, 2) for x in ax1.get_xticks()]
    )

    # Второй y-axis: разностная частота (Гц). delay_ms = f_diff/cr*1e3 => f_diff = delay_ms*cr/1e3
    ax3 = ax1.twinx()
    ax3.set_ylabel("Разностная частота, Гц")
    ax3.set_yticks(ax1.get_yticks())
    ax3.set_ybound(ax1.get_ybound())
    ax3.set_yticklabels([int(x * cr / 1e3) for x in ax3.get_yticks()])

    # Оставляем место справа для оси «Разностная частота» и цветовой шкалы
    plt.subplots_adjust(right=0.70, top=0.92)
    cbar_ax = fig.add_axes([0.86, 0.12, 0.03, 0.75])
    fig.colorbar(img, cax=cbar_ax, label="Мощность, дБ")

    if title:
        ax1.set_title(title, y=1.12)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_ionogram_image(
    data_2d,
    data_file_len,
    sr,
    cr,
    f_start,
    vmin=0,
    vmax=None,
    ylim_ms=(0, 10),
    out_path=None,
):
    """
    Та же картинка, что и plot_ionogram, но без осей, подписей и цветовой шкалы —
    только само изображение ионограммы.
    """
    freq_mhz = np.linspace(
        f_start,
        f_start + data_file_len / sr * cr / 1e6,
        num=data_2d.shape[0],
    )

    frange_hz = np.linspace(-sr / 2.0, sr / 2.0, num=data_2d.shape[1])
    delay_ms = frange_hz / cr * 1e3

    ms_to_km = (120e3 / frange_hz.shape[0])
    window_size_in_km = (ylim_ms[1] - ylim_ms[0]) / 1e6 * 3e8

    width_px, height_px = freq_mhz.shape[0], window_size_in_km / ms_to_km

    fig, ax = plt.subplots(figsize=(width_px/DPI, height_px/DPI), dpi=DPI)

    ax = fig.add_axes([0,0,1,1])

    power_db = 10.0 * np.log10(np.maximum(data_2d[:, ::-1], 1e-20))
    ax.pcolormesh(
        freq_mhz,
        delay_ms,
        np.transpose(power_db),
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    ax.set_xlim(freq_mhz[0], freq_mhz[-1])
    ax.set_ylim(ylim_ms)
    ax.axis("off")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Построение ионограммы по файлу (комплексный разностный сигнал)"
    )

    abs_data_path = os.path.join(os.getcwd(), DATA_DIR)
    # abs_files_path = os.path.join(abs_data_path, 'records', FILE_TYPE)
    files_list = os.listdir(abs_data_path)

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.path.join(abs_data_path, 'images', IMG_DIR),
        help="Путь для сохранения рисунка (если не задан — показ в окне)",
    )
    parser.add_argument(
        "--no-median",
        action="store_true",
        help="Не применять нормировку по медиане (medianEqualize)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help="Частота дискретизации до децимации (по умолч. %s)" % DEFAULT_SAMPLE_RATE,
    )
    parser.add_argument(
        "--dec",
        type=int,
        default=DEFAULT_DEC,
        help="Децимация (по умолч. %s)" % DEFAULT_DEC,
    )
    parser.add_argument(
        "--cr",
        type=float,
        default=DEFAULT_CR,
        help="Скорость ЛЧМ, Гц/с (по умолч. %s)" % DEFAULT_CR,
    )
    parser.add_argument(
        "--f-start",
        type=float,
        default=DEFAULT_F_START,
        help="Начальная частота, МГц (по умолч. %s)" % DEFAULT_F_START,
    )
    parser.add_argument(
        "-w", "--window",
        type=int,
        default=32768,
        help="Размер окна FFT (по умолч. 32768)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=[8, 12],
        metavar=("MIN_MS", "MAX_MS"),
        help="Диапазон оси Y: задержка от и до, мс (по умолч. 0 10)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0,
        help="Минимум цветовой шкалы мощности, дБ (по умолч. 0)",
    )

    for i in range(len(files_list)):
        parser.add_argument(
            "path",
            type=str,
            nargs="?",
            # default=os.path.join(abs_files_path, files_list[i]),
            default=os.path.join(abs_data_path, files_list[i]),
            help="Путь к файлу c записями (если не задан — ищется один файл в текущей папке; по умолчанию .lfs)",
        )
        args = parser.parse_args()

        path = args.path
        if path is None:
            out_files = [f for f in os.listdir(".") if f.lower().endswith(".out")]
            if len(out_files) == 1:
                path = out_files[0]
                print("Используется файл: %s" % path)
            elif not out_files:
                parser.error("Не задан path и в текущей папке нет файлов .out. Укажите: plot_ionogram.py <файл.out>")
            else:
                parser.error(
                    "Не задан path, в папке несколько .out: %s. Укажите файл явно."
                    % ", ".join(out_files)
                )
        if not os.path.isfile(path):
            raise SystemExit("Файл не найден: %s" % path)

        sr = args.sample_rate / args.dec
        window = args.window
        cr = args.cr
        f_start = args.f_start

        data_file = load_out_file(path)
        n_window = int(np.floor(len(data_file) / window))
        if n_window == 0:
            raise SystemExit(
                "Недостаточно отсчётов в файле: %d, нужно минимум %d (window=%d)"
                % (len(data_file), window, window)
            )

        S = spectrogram(data_file, window)
        if not args.no_median:
            S = median_equalize(S)

        # print(f"S.shape: {S.shape}")

        title = os.path.basename(path)
        if title.lower().endswith('.' + FILE_TYPE):
            title = title[:-4]

        # plot_ionogram(
        #     data_2d=S,
        #     data_file_len=len(data_file),
        #     sr=sr,
        #     cr=cr,
        #     f_start=f_start,
        #     title=title,
        #     vmin=args.vmin,
        #     ylim_ms=tuple(args.ylim),
        #     out_path=args.output,
        # )

        fig = plot_ionogram_image(
            data_2d=S,
            data_file_len=len(data_file),
            sr=sr,
            cr=cr,
            f_start=f_start,
            vmin=args.vmin,
            ylim_ms=tuple(args.ylim),
        )

        if args.output:
            fig.savefig(os.path.join(args.output, f'{files_list[i].split('.')[0]}_{args.window}_{args.ylim[0]}_{args.ylim[1]}'), dpi=DPI, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f'Ионограмма № {i+1} сохранена под названием: %s' % f'{files_list[i].split('.')[0]}_{args.window}_{args.ylim[0]}_{args.ylim[1]}')
        else:
            plt.show()


if __name__ == "__main__":
    main()
