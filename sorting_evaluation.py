#!/usr/bin/env python
#coding=utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tables
import patterns
import eegtools
import utils
from cluster import MetricEuclidean

from NeuroTools.parameters import ParameterSet as NTParameterSet

from DataFrame import DataFrame

datapath = os.environ.get("DATAPATH")

def spike_in_win(spt, stim, win):

    i = np.searchsorted(stim, spt)-1

    sp_pst = spt-stim[i]

    bool = ((sp_pst>win[0]) &( sp_pst<win[1]))

    return bool

def cluster_distance(dist, clust, min_trials=1):
   
    clust_lab = np.unique(clust)
    n_clust = np.asarray([np.sum(clust==i) for i in np.unique(clust)])
    clust_lab = clust_lab[n_clust>min_trials]
    n_clust = n_clust[n_clust>min_trials]

    within_dist = np.mean([dist[clust==i,:][:,clust==i].sum()/(n*(n-1))
                          for n,i in zip(n_clust, clust_lab)])
    between_dist = np.mean([dist[clust==cl1,:][:, clust==cl2].mean()
                           for i, cl1 in enumerate(clust_lab)
                           for cl2 in clust_lab[i+1:]])
   
    n = dist.shape[0]
    total_dist = np.sum(dist)/(n*(n-1))
    return (between_dist - within_dist)/total_dist


def which_window(spt, stim, ev):

    bWin = np.vstack([spike_in_win(spt, stim, [ev[i], ev[i+1]]) 
                   for i in range(len(ev)-1)])

    if len(ev)>2:
        cl = bWin.argmax(0)
    else:
        cl = bWin[0,:]*1

    return cl


def shuffle_gen(x,n):

    for i in xrange(n):
        i = np.random.rand(len(x)).argsort()
        yield x[i]


def bootstrap_clust_dist(sp_dist, clust, shuffles):
    
    boot_clust_dist = [cluster_distance(sp_dist, x) for x in shuffles]
    bott_clust_dist = np.asarray(boot_clust_dist)

    orig_clust_dist = cluster_distance(sp_dist, clust)

    return np.mean(orig_clust_dist>boot_clust_dist)




def run_analysis(h5f_in, cell, sp_win = [-0.2, 0.8], n_shuffles=100.,
        trials_per_cl=1):
    """
    Parmaeters:
    h5f_in: pytables object
    cell: cell name
    sp_win: spike waveform window

    """
    
    #Read the data from HDF5 file
    dataset = "/".join(cell.split('/')[:3])
    electrode = "/".join(cell.split('/')[:4])

    cell_node = h5f_in.getNode(cell)
    data_node = h5f_in.getNode(dataset)
    electrode_node = h5f_in.getNode(electrode)

    stim = data_node.stim.read()
    spt = cell_node.read()
    sp_raw = electrode_node.raw.read()
    FS = electrode_node.raw.attrs['sampfreq']
    sp_events = cell_node.attrs['events']

    trains=patterns.SortSpikes(spt,stim,[sp_events[0], sp_events[-1]])
    cl = patterns.FindClasses(trains, sp_events)
    cl_lab = np.unique(cl)

    spt_reduced = spt[spike_in_win(spt, stim, [sp_events[0],
        sp_events[-1]])]

    sp_waves, sp_time = eegtools.GetEEGTrials(sp_raw,
                               spt_reduced/1000.*FS, win=sp_win,Fs=FS)
    spt_reduced = spt_reduced[:sp_waves.shape[1]]
    sp_dist = MetricEuclidean(sp_waves.T)

    sp_trial_idx = np.searchsorted(stim[:-1], spt_reduced)-1
    sp_pattern_lab = cl[sp_trial_idx]
    pattern_cluster = cluster_distance(sp_dist, sp_pattern_lab,
            trials_per_cl)
    pval_pattern_cluster = bootstrap_clust_dist(sp_dist,
            sp_pattern_lab, shuffle_gen(sp_pattern_lab, n_shuffles))

    sp_window_lab = which_window(spt_reduced, stim, sp_events)
    #sp_window_lab = sp_window_lab[np.random.rand(len(sp_window_lab)).argsort()]
    window_cluster = cluster_distance(sp_dist, sp_window_lab)
    pval_window_cluster = bootstrap_clust_dist(sp_dist,
            sp_window_lab, shuffle_gen(sp_window_lab, n_shuffles))

    thresh = np.median(np.abs(sp_raw))/0.6745
    mean_sp_amp = np.median(np.abs(sp_waves).max(0))

    snr = mean_sp_amp/thresh

    plt.figure()

    trains_win = [patterns.SortSpikes(spt,stim,[sp_events[i], sp_events[i+1]])
            for i in xrange(len(sp_events)-1)]
    colors = ['r', 'b', 'g', 'y']
    n_rows = n_cols = np.ceil(np.sqrt(len(cl_lab)))
    for i,c in enumerate(cl_lab[:]):
        for col,tr in zip(colors,trains_win): 
            sp_index = np.concatenate([stim[j] + tr[j] for j in
                xrange(len(cl)) if cl[j]==c])
            if len(sp_index)>1:
                plt.subplot(n_rows,n_cols,i)
                (sp_class,sp_time)=eegtools.GetEEGTrials(sp_raw, sp_index/1000*FS, win=sp_win,Fs=FS)
                plt.plot(sp_class, col)
                plt.title("Pattern %s" % utils.dec2binstr(c,len(sp_events)-1))




    ret_dict =  {"Dataset": cell, 
                 "SNR": snr,
                 "Pattern ClustCoef": pattern_cluster, 
                 "Pattern ClustCoef (p)": pval_pattern_cluster, 
                 "Window ClustCoef": window_cluster,
                 "Window ClustCoef (p)": pval_window_cluster,
                 }
    print ret_dict

    return ret_dict

def to_str(x):
    return "%f +/- %f" % x

def range_str(x):
    return "%f -- %f" % (np.min(x), np.max(x))

if __name__=="__main__":
    parameter_file = sys.argv[1]
    parameters = NTParameterSet(parameter_file)
    params = parameters.process_params

    h5f = tables.openFile(datapath+parameters.in_datafile, 'r')
    df = DataFrame({"Dataset":[], "SNR":[], 
        "Pattern ClustCoef" : [], "Pattern ClustCoef (p)" : [],
        "Window ClustCoef" : [], "Window ClustCoef (p)" : []},
        ["Dataset", "SNR", "Pattern ClustCoef","Pattern ClustCoef (p)",
        "Window ClustCoef", "Window ClustCoef (p)"])
    out_dir = utils.create_new_dir("Data/", "sorting_evaluation")

    data_list = parameters.cells
    
    try: 
        for node_name in data_list:
            data = run_analysis(h5f, node_name, params.sp_win,
                    params.n_shuffles, params.trials_per_class)
            df.insert_row(data, new_fields_ok=False)
            fig_fname = os.path.join(out_dir, node_name[1:].replace('/', "_")) 
            plt.savefig(fig_fname)
    except ZeroDivisionError:
        raise
    mean_dict =  {
            "Dataset": "mean +/- sem", 
            "SNR": to_str(df.mean_and_sem("SNR")),
            "Pattern ClustCoef": to_str(df.mean_and_sem("Pattern ClustCoef")), 
            "Pattern ClustCoef (p)": np.NaN, 
            "Window ClustCoef": to_str(df.mean_and_sem("Window ClustCoef")),
            "Window ClustCoef (p)": np.NaN,
                 }
    range_dict =  {
            "Dataset": "range", 
            "SNR": range_str(df[:]["SNR"]),
            "Pattern ClustCoef": range_str(df[:]["Pattern ClustCoef"]), 
            "Pattern ClustCoef (p)": np.NaN, 
            "Window ClustCoef": range_str(df[:]["Window ClustCoef"]),
            "Window ClustCoef (p)": np.NaN,
                 }

    df.insert_row(mean_dict)
    df.insert_row(range_dict)
    df.write_csv(os.path.join(out_dir, "results.csv"))


    
