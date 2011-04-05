#!/usr/bin/env python
#coding=utf-8

import numpy as np
from spike_analysis import basic
import spike_sort as sort
from scipy.spatial import KDTree
from scipy.stats import spearmanr, kde


#simulate dataset

def read_data(filter, dataset, cell_id=None):
   
    raw = filter.read_sp(dataset)
    stim = filter.read_spt(dataset + '/stim')
    
    if cell_id:
        spt = filter.read_spt(dataset + '/' + cell_id)
        return raw, stim, spt
    return raw, stim

def mix_cellbg(filter, cell_src, background_src, sp_win,
               pow_frac, binsz=0.01):
    #read data from files

    spike_src, cell_id = cell_src.rsplit('/', 1)
    raw_cell, stim_cell, spt_cell = read_data(filter, spike_src,
                                              cell_id)
    raw_bg, stim_bg = read_data(filter, background_src)
    
    #get spike waveshapes
    avg_spike = calc_avg_spike(raw_cell, spt_cell, sp_win)
   
    #estimate firing rate and simulate trains
    spt_sim = generate_spt(spt_cell, stim_cell, stim_bg, 
                                 [0,300], binsz)
    #add spikes
    sp_sim = add_spikes(raw_bg, spt_sim, avg_spike, pow_frac)

    return sp_sim, stim_bg, spt_sim


def calc_avg_spike(raw_spikes, spt_spikes, sp_win):

    sp_waves = sort.extract.extract_spikes(raw_spikes, spt_spikes, sp_win)
    avg_spike = sp_waves['data'].mean(1)
    return avg_spike

def generate_spt(spt_spikes, stim_spikes, stim_background,  win, binsz):

    psth_spikes, time = basic.CalcPSTH(spt_spikes['data'], 
                                       stim_spikes['data'],
                                       win=[0,300],
                                       bin=binsz,
                                       norm=True)
    n_trials = len(stim_background['data'])
    trains_simulated = binom_generator(psth_spikes,
                                       n_trials, bin=binsz)
    spt_simulated = np.concatenate(
                       map(np.add, stim_background['data'], trains_simulated)
                    )

    return {'data':spt_simulated}
    

def binom_generator(psth, n_trials, bin=0.25):
    def _trial():
        i, = np.where(np.random.rand(len(psth)) < (psth*bin/1000))
        return i*bin

    trials = [_trial() for i in range(n_trials)]

    return trials

def add_spikes(sp_data, spt_dict, sp_wave, pow_frac):
    spt = spt_dict['data']
    fs = sp_data['FS']
    center_idx = spt/1000.*fs
    n_pts = sp_wave.shape[0]
    data = sp_data['data'].copy()
    data_std = np.std(data[:,:10*fs],1)
    spike_std = np.std(sp_wave,0)
    frac = data_std/spike_std*pow_frac
    sp_wave = sp_wave*frac

    for t in center_idx:
        data[:,(t-n_pts/2):(t+n_pts/2)] += sp_wave.T
    ret_dict = sp_data.copy()
    ret_dict['data']=data
    return ret_dict


#data pre-processing

def calc_features(sp, spt, sp_win, fet_list, contacts=[0]):

    def _get_feature(name):
        return getattr(sort.features, "fet"+name)(sp_waves)
    
    sp_waves = sort.extract.extract_spikes(sp, spt, sp_win,
                                           contacts=contacts)
    feature_data = map(_get_feature, fet_list)
    features = sort.features.combine(feature_data, norm=True)
    return features

def filter_data(sp, f_filter):
    #filter data
    if f_filter is not None:
        filter = sort.extract.Filter("ellip", *f_filter)
        sp = sort.extract.filter_proxy(sp, filter)
    return sp

def spike_clusters(sp, spt_orig, stim, thresh, type, sp_win, tol=1.5):

    spt_det = sort.extract.detect_spikes(sp,  contact=0, thresh=thresh)
    
    spt_det = sort.extract.align_spikes(sp, spt_det, sp_win, 
                                        type=type, resample=10)
    spt_det = remove_stimulus(spt_det, stim)

    spt, clust_idx, n_missing = combine_spikes(spt_det, spt_orig,
                                               tol=tol)
    spt = sort.extract.align_spikes(sp, spt, sp_win, 
                                    type=type, resample=10,
                                    remove=False)
    return spt, clust_idx, n_missing

def combine_spikes(spt1_dict, spt2_dict, tol=0.5):
    """
    spt1, spt2 - spike times dicts
    tol - tolerance in miliseconds
    """
    spt1 = spt1_dict['data']
    spt2 = spt2_dict['data']

    overlap = check_intvs(spt1, spt2, [-tol, tol])
    missing = ~(check_intvs(spt2, spt1, [-tol, tol]))
    n_missing = int(np.sum(missing))

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

def check_intvs(times, centers, win):
    intvs = centers[:,np.newaxis]+np.array(win)
    b =(
        (times[:,None]>=intvs[None,:,0])& 
        (times[:,None]<intvs[None,:,1])
       )
    return b.any(1)

#metrics

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


