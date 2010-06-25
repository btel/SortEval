#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

from spike_sort import io
import spike_sort.io.hdf5, spike_sort.io.bakerlab

datasets = [
     "/Joy/s3349a16/el7/cell1"
     ]

spt_dir = "./Data/join_spt/"
hdf5_dest = "hdf5/data_microel.h5"
DATAPATH = os.environ.get("DATAPATH")

if __name__ == "__main__":

    h5_fname = os.path.join(DATAPATH, hdf5_dest)
    
    for d in datasets:
        fname = d.replace("/", "_")
        spt = io.bakerlab.read_spt(spt_dir, fname)
        spt_orig = io.hdf5.read_spt(h5_fname, d)
        spt['events'] = spt_orig['events']
        io.hdf5.write_spt(spt, h5_fname, d+"_corrected")

