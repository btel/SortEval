#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import os, sys

import spike_sort as sort
from spike_sort.io.filters import PyTablesFilter, BakerlabFilter
import spike_sort.ui.manual_sort
import tables
from scipy.stats import spearmanr, kde
from scipy.spatial import KDTree

import time
from spike_sort.ui import spike_browser

def univariate_metric(metric_func, features, clust_idx):
    n_feats = features['data'].shape[1]

    metric_vals = {}
    feat_data = features['data']
    for i, feature in enumerate(features['names']):
        metric_vals[str(feature)] = metric_func(feat_data[:,i], clust_idx)

    return metric_vals

def multivariate_metric(metric_func, features, clust_idx):
    feat_data = features['data']
    feature1 = feat_data[clust_idx==0,:]
    feature2 = feat_data[clust_idx==1,:]

    return metric_func(feature1, feature2)



def check_intvs(times, centers, win):
    intvs = centers[:,np.newaxis]+np.array(win)
    b =(
        (times[:,None]>=intvs[None,:,0])& 
        (times[:,None]<intvs[None,:,1])
       )
    return b.any(1)



#metric functions
def corrcoef(x,y):
    return spearmanr(x,y)[0]

def mutual_information(features, classes, n_bins=100):
    def entropy(counts):
        '''Compute entropy.'''
        ps = counts/np.float(np.sum(counts)) # coerce to float and normalize
        ps = ps[np.nonzero(ps)] # toss out zeros
        H = -np.sum(ps*np.log2(ps)) # compute entropy
        
        return H
    feat_bins = np.linspace(features.min(), features.max(), n_bins)
    cls_bins = np.arange(classes.max()+2)
    counts_2d,_,_ = np.histogram2d(features, classes, [feat_bins,cls_bins])
    counts_feats,_ = np.histogram(features, feat_bins)
    counts_cls,_ = np.histogram(classes, cls_bins)
    H_xy = entropy(counts_2d)
    H_x = entropy(counts_feats)
    H_y = entropy(counts_cls)
    
    return H_x + H_y - H_xy

        
def kullback_leibler(x, y, nbins=20):

    def _est_hist(x, nbins, range, eps=0.01):
        counts, edges = np.histogramdd(x, bins=nbins,
                                       range=range)
        counts = counts+eps
        return counts/np.float(np.sum(counts))
    
    def _pdf_kde(x, nbins, range, eps=0.01):
        k = kde.gaussian_kde(y.T)
        #pdf = k.evaluate(x)
        return counts/np.float(np.sum(counts))


    xmin, xmax = np.min(x,0), np.max(x,0)
    ymin, ymax = np.min(y,0), np.max(y,0)
    min = np.minimum(xmin, ymin)
    max = np.maximum(xmax, ymax)
    range = zip(min, max)
    p1 = _est_hist(x, nbins, range)
    p2 = _est_hist(y, nbins, range)
   
    aux = (p1*np.log(p1/p2))
    aux = aux[~np.isnan(aux)]
    KL_div = np.sum(aux)

    return KL_div


def k_nearest(features, clust_idx, K=5, n_pts='all', eps=0.05):

    def _calc_frac(l):
        return np.sum((real_cl==l) & (cl==l))*1./np.sum(real_cl==l)
    data = features['data']

    k_tree = KDTree(data)
    
    if n_pts=='all':
        samp_pts = data
        real_cl = clust_idx
    else:
        i = np.argsort(np.random.rand(data.shape[0]))
        samp_pts = data[i[:n_pts],:]
        real_cl = clust_idx[i[:n_pts]]

    dist, neigh_idx = k_tree.query(samp_pts, K+1, eps)

    cl_neigh = clust_idx[neigh_idx]
    #remove the point itself from the neighbours
    cl_neigh = cl_neigh[:,1:]
    cl = 1*(cl_neigh.sum(1)>=K/2)

    return (_calc_frac(0)+_calc_frac(1))/2.

def combine_spikes(spt1_dict, spt2_dict, tol=0.5):
    """
    spt1, spt2 - spike times dicts
    tol - tolerance in miliseconds
    """
    spt1 = spt1_dict['data']
    spt2 = spt2_dict['data']

    overlap = check_intvs(spt1, spt2, [-tol, tol])
    missing = ~(check_intvs(spt2, spt1, [-tol, tol]))
    n_missing = np.sum(missing)

    spt1 = spt1[~overlap]
    
    spt_comb = np.concatenate((spt1, spt2))
    i = np.argsort(spt_comb)
    spt_comb = spt_comb[i]

    clust_idx = np.concatenate((np.zeros(len(spt1)),
                                np.ones(len(spt2))))
    clust_idx = clust_idx[i]


    spt_out = spt1_dict.copy()
    spt_out['data'] = spt_comb
    return spt_out, clust_idx, n_missing

def remove_stimulus(spt_dict, stim_dict,tol=2):
    spt = spt_dict['data']
    stim = stim_dict['data']

    is_stim = check_intvs(spt, stim, [-tol, tol])
    spt = spt[~is_stim]
    
    spt_out = spt_dict.copy()
    spt_out['data']=spt
    return spt_out


if __name__ == "__main__":

    h5_fname = "simulated.h5"
    h5filter = PyTablesFilter(h5_fname)

    dataset = "/TestSubject/sSession01/el1"
    sp_win = [-0.4, 0.8]
    f_filter=None
    thresh = '4'
    type='max'
    
    start = time.time()
    sp = h5filter.read_sp(dataset)
    
    #filter data
    if f_filter is not None:
        filter = sort.extract.Filter("ellip", *f_filter)
        sp = sort.extract.filter_proxy(sp, filter)
    
    spt_orig = h5filter.read_spt(dataset+"/cell1_orig")
    stim = h5filter.read_spt(dataset+"/stim")
    spt_det = sort.extract.detect_spikes(sp,  contact=0, thresh=thresh)
    
    spt_det = sort.extract.align_spikes(sp, spt_det, sp_win, 
                                        type=type, resample=10)
    spt_det = remove_stimulus(spt_det, stim)

    #spt_orig = sort.extract.align_spikes(sp, spt_orig, sp_win, 
    #                                     type=type, resample=10)
    
    spt, clust_idx, n_missing = combine_spikes(spt_det, spt_orig,
                                               tol=1.5)
    spt = sort.extract.align_spikes(sp, spt, sp_win, 
                                    type=type, resample=10,
                                    remove=False)
    
    sp_waves = sort.extract.extract_spikes(sp, spt, sp_win,
                                           contacts=[0,1])
    features = sort.features.combine(
            (
            sort.features.fetP2P(sp_waves),
            sort.features.fetPCs(sp_waves, ncomps=2)),
            norm=True
    )

    n_total = len(spt_orig['data'])
    single_metric = univariate_metric(mutual_information, 
                                      features, clust_idx)
    #multi_metric = multivariate_metric(kullback_leibler, 
    #                                  features, clust_idx)
    knearest_metric = k_nearest(features, clust_idx, n_pts=1000)

    #spike_browser.browse_data_tk(sp, spt, clust_idx)

    print "Total n/o spikes:", n_total
    print "Number of undetected spikes: %d (%f)" % (n_missing,
                                                    n_missing*100./n_total)
    print "Univariate MI:", single_metric
    #print "Mulitvariate KL divergence:", multi_metric
    print "K-nearest class. rate:", knearest_metric
    
    plt_features = sort.features.combine((sort.features.fetSpIdx(sp_waves),
                                          features),norm=True)
    spike_sort.ui.plotting.plot_features(plt_features, clust_idx)
    plt.figure()
    spike_sort.ui.plotting.plot_spikes(sp_waves, clust_idx)

    spike_sort.ui.plotting.show()
    h5filter.close()
