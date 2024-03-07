#!/usr/bin/python
import numpy as np
from optparse import OptionParser
import struct

lfs_header_type = np.dtype([
                ('format', np.str_, 4), 
                ('format_ver', np.float32), 
                ('header_id', np.str_, 4), 
                ('header_size', np.uint16),         
                ('tx_name', np.str_, 64),
				('tx_latitude', np.float32),
				('tx_longitude', np.float32),
                ('rx_name', np.str_, 64),
				('rx_latitude', np.float32),
				('rx_longitude', np.float32),
				('start_year', np.uint16),
				('start_daynumber', np.uint16),
				('start_month', np.uint16),
				('start_day', np.uint16),
				('start_hour', np.uint16),
				('start_minute', np.uint16),
				('start_second', np.uint16),
				('start_epoch', np.uint32),
				('chirpt', np.uint32),
				#('sf', np.uint32),
				('cf', np.uint32),
				('dur', np.uint16),
				('rate', np.uint32),
				('rep',  np.uint32),
				('rmin', np.int32),
				('rmax', np.int32),
				('dec', np.uint32),
				('sample_rate', np.uint32),
				('whiten', np.uint16),
				('whiten_len', np.uint32),
				('whiten_n', np.uint32),
				('reserved', np.str_, 288)
                ])

#
# read file's header
# 
def header(filename):
    # file_header = np.fromfile(filename, dtype=lfs_header_type, count=1)
    # header = {}
    # for key in file_header.dtype.names:
    #     header[key] = file_header[key][0]  
    #     print("%s: %s"%(key, header[key]))
    # return header

	header = {}

	f=open(filename, "rb")

	f.seek(0, 0)
	format_ = f.read(4)
	# print('format: ',format_.decode('ascii'))

	header['format'] = format_

	f.seek(4, 0)
	format_ver = f.read(4)
	format_ver = struct.unpack('f', format_ver)
	# print('format_ver:',format_ver)

	header['format_ver'] = format_ver

	f.seek(8, 0)
	header_id = f.read(4)
	# print('format: ',header_id.decode('ascii'))
	header['header_id'] = header_id.decode('ascii')

	f.seek(12, 0) 
	header_size = f.read(2)
	header_size = struct.unpack('H', header_size)
	# print('header_size:',header_size)
	header['header_size'] = header_size

	f.seek(14, 0) 
	tx_name = f.read(64)
	# print('tx_name: ',tx_name.decode('ascii'))
	header['tx_name'] = tx_name.decode('ascii')

	f.seek(78, 0) 
	tx_latitude = f.read(4)
	tx_latitude = struct.unpack('f', tx_latitude)
	# print('tx_latitude:',tx_latitude)
	header['tx_latitude'] = tx_latitude

	f.seek(82, 0)
	tx_longitude = f.read(4)
	tx_longitude = struct.unpack('f', tx_longitude)
	# print('tx_longitude:',tx_longitude)
	header['tx_longitude'] = tx_longitude

	f.seek(86, 0) 
	rx_name = f.read(64)
	# print('rx_name: ',rx_name.decode('ascii'))
	header['rx_name'] = rx_name.decode('ascii')

	f.seek(150, 0) 
	rx_latitude = f.read(4)
	rx_latitude = struct.unpack('f', rx_latitude)
	# print('rx_latitude:',rx_latitude)
	header['rx_latitude'] = rx_latitude	

	f.seek(150, 0)
	rx_longitude = f.read(4)
	rx_longitude = struct.unpack('f', rx_longitude)
	# print('rx_longitude:',rx_longitude)
	header['rx_longitude'] = rx_longitude

	f.seek(158, 0) 
	start_year = f.read(2)
	start_year = struct.unpack('H', start_year)
	# print('start_year:',start_year)
	header['start_year'] = start_year

	f.seek(160, 0) 
	start_daynumber = f.read(2)
	start_daynumber = struct.unpack('H', start_daynumber)
	# print('start_daynumber:',start_daynumber)
	header['start_daynumber'] = start_daynumber

	f.seek(162, 0) 
	start_month = f.read(2)
	start_month = struct.unpack('H', start_month)
	# print('start_month:',start_month)
	header['start_month'] = start_month

	f.seek(164, 0) 
	start_day = f.read(2)
	start_day = struct.unpack('H', start_day)
	# print('start_day:',start_day)
	header['start_day'] = start_day

	f.seek(166, 0) 
	start_hour = f.read(2)
	start_hour = struct.unpack('H', start_hour)
	# print('start_hour:',start_hour)
	header['start_hour'] = start_hour

	f.seek(168, 0) 
	start_minute = f.read(2)
	start_minute = struct.unpack('H', start_minute)
	# print('start_minute:',start_minute)
	header['start_minute'] = start_minute

	f.seek(170, 0) 
	start_second = f.read(2)
	start_second = struct.unpack('H', start_second)
	# print('start_second:',start_second)
	header['start_second'] = 	start_second

	f.seek(172, 0) 
	start_epoch = f.read(4)
	start_epoch = struct.unpack('I', start_epoch)
	# print('start_epoch:',start_epoch)
	header['start_epoch'] = start_epoch

	f.seek(176, 0) 
	chirpt = f.read(4)
	chirpt = struct.unpack('I', chirpt)
	# print('chirpt:',chirpt)
	header['chirpt'] = chirpt

	f.seek(180, 0) 
	cf = f.read(4)
	cf = struct.unpack('I', cf)
	# print('сf:',сf)
	header['cf'] = cf

	f.seek(184, 0) 
	dur = f.read(2)
	dur = struct.unpack('H', dur)
	# print('dur:',dur)
	header['dur'] = dur

	f.seek(186, 0) 
	rate = f.read(4)
	rate = struct.unpack('I', rate)
	# print('rate:',rate)
	header['rate'] = rate

	f.seek(190, 0) 
	rep = f.read(4)
	rep = struct.unpack('I', rep)
	# print('rep:',rep)
	header['rep'] = rep

	f.seek(194, 0) 
	rmin = f.read(4)
	rmin = struct.unpack('i', rmin)
	# print('rmin:',rmin)
	header['rmin'] = rmin

	f.seek(198, 0) 
	rmax = f.read(4)
	rmax = struct.unpack('i', rmax)
	# print('rmax:',rmax)
	header['rmax'] = rmax

	f.seek(202, 0) 
	dec = f.read(4)
	dec = struct.unpack('I', dec)
	# print('dec:',dec)
	header['dec'] = dec

	f.seek(206, 0) 
	sample_rate = f.read(4)
	sample_rate = struct.unpack('I', sample_rate)
	# print('sample_rate:',sample_rate)
	header['sample_rate'] = sample_rate

	f.seek(210, 0) 
	whiten = f.read(2)
	whiten = struct.unpack('H', whiten)
	# print('whiten:',whiten)
	header['whiten'] = whiten

	f.seek(212, 0) 
	whiten_len = f.read(4)
	whiten_len = struct.unpack('I', whiten_len)
	# print('whiten_len:',whiten_len)
	header['whiten_len'] = whiten_len

	f.seek(216, 0) 
	whiten_n = f.read(4)
	whiten_n = struct.unpack('I', whiten_n)
	# print('whiten_n:',whiten_n)
	header['whiten_n'] = whiten_n

	f.seek(216, 0)
	reserved = f.read(288)
	# print('reserved: ',reserved.decode('ascii'))
	header['reserved'] = reserved.decode('ascii')

	f.close()

	return header

def main():                     
    parser = OptionParser()

    parser.add_option("-f", "--filename",
                        dest="filename", 
                        action="store", 
                        type="string",
                        default="",
                        help="LFS filename")

    (op, args) = parser.parse_args()

    header(op.filename)

if __name__ == "__main__":
    	main()
