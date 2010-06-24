#!/usr/bin/env python
#coding=utf-8

import numpy as np

import os
import spike_sort as sort
import spike_sort.io.bakerlab
import spike_sort.io.hdf5
import spike_sort.ui.manual_sort

DATAPATH = os.environ.get("DATAPATH")

if __name__ == "__main__":
    #main
    h5_fname = os.path.join(DATAPATH, "hdf5/data_microel.h5")
    dataset = "/Joy/s3349a16/el7/cell1"
    out_dir = "./Data/find_missed_spikes/23-06-2010/Sim_15"
    sp_win = [-0.2, 0.8]

    spt_fname = "cluster"

    spt = sort.io.bakerlab.read_spt(out_dir, spt_fname)
    sp = sort.io.hdf5.read_sp(h5_fname, dataset)
    
    sp_waves = sort.extract.extract_spikes(sp, spt, sp_win)

    features = sort.features.combine(
            (
            sort.features.fetP2P(sp_waves),
            sort.features.fetPCs(sp_waves)))


    clust_idx = sort.ui.manual_sort.show(features, sp_waves, [0,2])

    clust, rest = sort.ui.manual_sort.cluster_spt(spt, clust_idx)

    if len(clust)>0:
        print "Exporting."
        sort.io.bakerlab.write_spt(clust, out_dir, spt_fname+"cluster")
        sort.io.bakerlab.write_spt(rest, out_dir, spt_fname+"rest")

    else: 
        print "Exiting."
