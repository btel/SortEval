#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append("../modules/")

import evaluate as eval
import spike_sort as sort
from spike_sort.io.filters import PyTablesFilter, BakerlabFilter
import spike_sort.ui.manual_sort
import tables

import time
from spike_sort.ui import spike_browser


def calc_metrics(features, clust_idx):
    
    uni_metric = eval.univariate_metric(eval.mutual_information, 
                                          features, clust_idx)
    multi_metric = eval.k_nearest(features, clust_idx, n_pts=1000)
    
    return uni_metric, multi_metric

if __name__ == "__main__":

    h5_fname = "simulated.h5"
    h5filter = PyTablesFilter(h5_fname)

    dataset = "/TestSubject/sSession01/el1"
    sp_win = [-0.4, 0.8]
    f_filter=None
    thresh = 'auto'
    type='max'
    
    sp = h5filter.read_sp(dataset)
    spt_orig = h5filter.read_spt(dataset+"/cell1_orig")
    stim = h5filter.read_spt(dataset+"/stim")
    
    sp = eval.filter_data(sp, f_filter)

    spt, clust_idx, n_missing = eval.spike_clusters(sp, spt_orig,
                                                    stim,
                                                    thresh,
                                                    type, sp_win) 
    
    features = eval.calc_features(sp, spt, sp_win, ["P2P", "PCs"], [0,1])
    
    single_metric, knearest_metric = calc_metrics(features, clust_idx)
    
    
    n_total = len(spt_orig['data'])
    print "Total n/o spikes:", n_total
    print "Number of undetected spikes: %d (%f)" % (n_missing,
                                                    n_missing*1./n_total)
    print "Univariate MI:", single_metric
    print "K-nearest class. rate:", knearest_metric
    
    spike_sort.ui.plotting.plot_features(features, clust_idx)
    #plt.figure()
    #spike_sort.ui.plotting.plot_spikes(sp_waves, clust_idx)

    spike_sort.ui.plotting.show()
    h5filter.close()
