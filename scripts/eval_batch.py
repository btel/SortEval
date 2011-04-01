#!/usr/bin/env python
#coding=utf-8

import numpy as np
import os, sys
import json
sys.path.append("../modules/")

from spike_sort.io.filters import  BakerlabFilter

import pymongo

connection = pymongo.Connection('localhost', 27017)

#np.random.seed(11111)
np.random.seed(99999)

def single_run(filter, spk_src, bg_src, params):
    import evaluate as eval

    sp_win    = params['sp_win']
    spadd_win = params['sp_add_win']
    pow_frac  = params['pow_frac']
    f_filter  = params['f_filter']
    type      = params['sp_type']
    thresh    = params['thresh']
    feats     = params['features']
    contacts  = params['contacts']
    n_pts     = params['n_pts']

    sp, stim, spt_real = eval.mix_cellbg(filter, spk_src, bg_src, 
                                         spadd_win, pow_frac)
    sp = eval.filter_data(sp, f_filter)

    spt, clust_idx, n_missing = eval.spike_clusters(sp, spt_real,
                                                    stim,
                                                    thresh,
                                                    type, sp_win) 
    
    features = eval.calc_features(sp, spt, sp_win, feats, contacts)
    
    uni_metric = eval.univariate_metric(eval.mutual_information, 
                                        features, clust_idx)

    multi_metric = eval.k_nearest(features, clust_idx, n_pts=n_pts)

    n_total = len(spt_real['data'])

    result_dict= {"cell" : spk_src,
                  "electrode:" : bg_src,
                  "spikes_total" : n_total,
                  "spikes_missed" : n_missing,
                  "mutual_information" : uni_metric,
                  "k_nearest" : multi_metric}

    result_dict.update(params)

    return result_dict

def local_run(filter, datasets, params, db_out):

    for spk_src, bg_src in datasets:
        out = single_run(filter, spk_src, bg_src, params)
        db_out.insert(out)

def main():
    _, param_file = sys.argv


    with open(param_file) as f:
        params = json.load(f)

    cell_list = params['cells']
    bg_list   = params['electrodes']
    conf_file = params['rec_conf']
    target    = params['target']
    process   = params['process_params']
    db_name   = params['database']

    src_pairs = [(cell, bg) for cell in cell_list for bg in bg_list]

    filter = BakerlabFilter(conf_file)
    db = connection[db_name]
    collection = db['test']

    if target == 'local':
        local_run(filter, src_pairs, process, collection)


if __name__ == "__main__":

    main()
