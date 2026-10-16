sounders = [
[{'name':'cyprus1','cf':20000e3,'chirpt':10,'dur':250,'lat':35,'lon':34,'pdp_max':10,'rate':100e3,'rep':300,'rx':False,'snr_max':50}]
]

rx_station = {'name':'yoshkar-ola','lat':56.38,'lon':47.53}

colormap_gradient = True
colormap_index = 8
data_dir = "/tmp/igs/"
dec = 625
direct_signal_cutting = False
fft_count_index = 5
ig_colormap_index = 1
ig_list_count = 2
ig_vertical_scale_index = 1
lfsr_polynome_degree = 0
sample_rate = 25000e3
if_rate = sample_rate/dec
sample_rate_index = 3
ssas = 1
tb = 
whiten = False
whiten_len = 8192
whiten_n = 20000

def get_all_sounders():
    sounder_list = []
    for sounder_thread in sounders:
        for sounder in sounder_thread:
            sounder_list.append(sounder)
    return(sounder_list)