#!/usr/bin/env python
#coding=utf-8

"""
Based on raw recordings detect spikes, calculate features and do automatic 
clustering with k-means.

TODO:
After clustering the spike times are exported back to HDF5 (cell_kmeansX, where 
X is cluster index)
"""

import numpy as np
import os, sys

import spike_sort as sort
import spike_sort.ui.manual_sort
import tables

import time
from spike_sort.io.filters import PyTablesFilter, BakerlabFilter
from spike_sort.io import export

DATAPATH = "../data" 

if __name__ == "__main__":

    dataset = "/Gollum/s39gollum01/el1"
    sp_win = [-0.2, 0.8]
    
    start = time.time()
    io_filter = BakerlabFilter("../data/gollum.inf")
    sp = io_filter.read_sp(dataset,memmap="tables")
    spt = sort.extract.detect_spikes(sp,  contact=3,
                                     thresh='auto')
    
    spt = sort.extract.align_spikes(sp, spt, sp_win, type="max", resample=10)
    sp_waves = sort.extract.extract_spikes(sp, spt, sp_win)
    features = sort.features.combine(
            (
            sort.features.fetP2P(sp_waves,contacts=[0,1,2,3]),
            sort.features.fetPCs(sp_waves)),
            normalize=True
    )


    clust_idx = sort.cluster.cluster("gmm",features,5)
    
    features = sort.features.combine(
            (sort.features.fetSpIdx(sp_waves), features))
    spike_sort.ui.plotting.plot_features(features, clust_idx)
    spike_sort.ui.plotting.figure()
    spike_sort.ui.plotting.plot_spikes(sp_waves, clust_idx, n_spikes=200)
    
    spt_cells = sort.cluster.split_cells(spt, clust_idx)
    features_cells = sort.features.split_cells(features, clust_idx)
    spikes_cells = sort.extract.split_cells(sp_waves, clust_idx)
    stim = io_filter.read_spt(dataset)
    
    from patterns import show, plotPSTH, figure,legend
    
    color_map = spike_sort.ui.plotting.label_color(np.unique(spt_cells.keys()))
    figure()
    [plotPSTH(spt_cells[i]['data'], stim['data'], 
              color=color_map(i),
              label="cell {0}".format(i)) 
            for i in spt_cells.keys()]
    legend()
    show()
    #cell_templ = "/Gollum/s39gollum01/el1/cell{cell_id}"
    #export.export_cells(io_filter, cell_templ, spt)
    
    #io_filter.close()
    #from IPython.Shell import IPShellEmbed 

    #ipshell = IPShellEmbed(['--colors NoColor'])

    #ipshell()

    #TODO: export
    #sort.io.hdf5.write_spt(clust, h5f, cell_node+"_clust",
    #                           overwrite=True)
    #sort.io.hdf5.write_spt(rest, h5f, cell_node+"_rest",
    #                           overwrite=True)

