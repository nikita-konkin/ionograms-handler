#!/usr/bin/python
"""This script get data part from lfs-file"""

import numpy as np
import os
from optparse import OptionParser

LFS_HEADER_SIZE = 512

def data(filename):
    """This function get data part from lfs-file"""
    lfs_file = open(filename, 'rb')
    lfs_file.seek(LFS_HEADER_SIZE, os.SEEK_SET)
    lfs_data = np.fromfile(lfs_file, dtype=np.complex64)
    return lfs_data

def main():
    PARSER = OptionParser()

    PARSER.add_option("-f", "--filename",
                      dest="filename",
                      action="store",
                      type="string",
                      default="",
                      help="LFS's filename")

    (OP, ARGS) = PARSER.parse_args()

    data(OP.filename)

if __name__ == "__main__":
    main()
