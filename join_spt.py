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
    spt_orig = sort.io.hdf5.read_spt(h5_fname, dataset)
    sp = sort.io.hdf5.read_sp(h5_fname, dataset)
    
    spt_new = np.concatenate((spt, spt_orig))
    clust_idx = np.concatenate((np.repeat(spt_fname,len(spt)),
                                np.repeat("original", len(spt_orig))))
    i = spt_new.argsort()
    spt_new = spt_new[i]
    clust_idx = clust_idx[i]
    
    spt_new = sort.extract.align_spikes(sp, spt_new, sp_win,
            type="min", resample=10)
    sp_waves_all = sort.extract.extract_spikes(sp,
            spt_new, sp_win)
   
    sort.plotting.plot_spikes(sp_waves_all, clust_idx, n_spikes=100)

    features_dict = sort.features.combine(
            (
            sort.features.fetSpIdx(sp_waves_all),
            sort.features.fetP2P(sp_waves_all),
            sort.features.fetPCs(sp_waves_all)))

    sort.plotting.figure()
    sort.plotting.plot_features(features_dict, clust_idx, size=1)

    sort.plotting.show()

    #sort.io.bakerlab.write_spt(clust, out_dir, "all_spikes")

