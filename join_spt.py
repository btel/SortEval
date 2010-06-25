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
    h5_fname = os.path.join(DATAPATH, "hdf5/test.h5")
    dataset = "/Joy/s3349a16/el7/cell1"
    spt_dir = "./Data/find_missed_spikes/25-06-2010/Sim_9"
    out_dir = "./Data/join_spt/"
    sp_win = [-0.2, 0.8]
    SAVE = 1 

    spt_fname = "missed_rest_rest"

    spt = sort.io.bakerlab.read_spt(spt_dir, spt_fname)
    spt_orig = sort.io.hdf5.read_spt(h5_fname, dataset)
    sp = sort.io.hdf5.read_sp(h5_fname, dataset)

    
    spt_new = np.concatenate((spt['data'], spt_orig['data']))
    clust_idx = np.concatenate((np.repeat(spt_fname, len(spt['data'])),
                                np.repeat("original", len(spt_orig['data']))))
    i = spt_new.argsort()
    spt_new = {"data":spt_new[i]}
    clust_idx = clust_idx[i]
    
    spt_new = sort.extract.align_spikes(sp, spt_new, sp_win,
            type="min", resample=1)
    sp_waves_all = sort.extract.extract_spikes(sp,
            spt_new, sp_win)
   
    features_dict = sort.features.combine(
            (
            sort.features.fetSpIdx(sp_waves_all),
            sort.features.fetP2P(sp_waves_all),
            sort.features.fetPCs(sp_waves_all)),
            normalize=True)

    fig1 = sort.plotting.figure()
    sort.plotting.plot_spikes(sp_waves_all, clust_idx,
            n_spikes=100)
    fig2 = sort.plotting.figure()
    sort.plotting.plot_features(features_dict, clust_idx, size=1)
   
    if SAVE:
        fig_dir = os.path.join(out_dir, "figures")
        fname = dataset.replace("/","_")
        try:
            os.mkdir(fig_dir)
        except OSError:
            pass

        fig1.savefig(os.path.join(fig_dir, fname + "_spikes.pdf"))
        fig2.savefig(os.path.join(fig_dir, fname + "_features.pdf"))

        spt_new['events'] = spt_orig['events']
        sort.io.bakerlab.write_spt(spt_new, out_dir,fname)
    
    sort.plotting.show()
    
    #sort.io.hdf5.write_spt(h5_fname, dataset+spt_fname, spt_new)
    #sort.io.hdf5.write_spt(h5_fname, dataset+spt_fname, spt_new)


