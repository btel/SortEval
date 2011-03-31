#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from spike_sort.io import filters
import spike_sort as sort
from spike_analysis import basic


spike_src = "/Gollum/s39gollum03/el1"
cell_id = "cell1"
background_src = "/Gollum/s39gollum03/el3"
sp_win = [-2, 2]
binsz=0.01
pow_frac = 1

def binom_generator(psth, n_trials, bin=0.25):
    def _trial():
        i, = np.where(np.random.rand(len(psth)) < (psth*bin/1000))
        return i*bin

    trials = [_trial() for i in range(n_trials)]

    return trials

def add_spikes(sp_data, spt, sp_wave, pow_frac):
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
    return data

if __name__ == "__main__":
    in_filter = filters.BakerlabFilter("../data/gollum.inf")
    out_filter = filters.PyTablesFilter("simulated.h5")

    #read data from files
    spt_spikes = in_filter.read_spt(spike_src + '/' + cell_id)
    raw_spikes = in_filter.read_sp(spike_src)
    stim_spikes = in_filter.read_spt(spike_src + '/stim')

    sp_background = in_filter.read_sp(background_src)
    stim_background = in_filter.read_spt(background_src+"/stim")
    
    #get spike waveshapes
    sp_waves = sort.extract.extract_spikes(raw_spikes, spt_spikes, sp_win)
    avg_spike = sp_waves['data'].mean(1)

    sp_simulated = sp_background.copy()
   

    #estimate firing rate and simulate trains
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
    
    #add spikes
    sp_simulated['data'] = add_spikes(sp_simulated, spt_simulated,
                                      avg_spike, pow_frac)

    

    out_filter.write_spt({'data':spt_simulated},
                         "/TestSubject/sSession01/el1/cell1_orig",
                        overwrite=True)
    out_filter.write_spt(stim_background,
                         "/TestSubject/sSession01/el1/stim",
                        overwrite=True)
    out_filter.write_sp(sp_simulated,
                        "/TestSubject/sSession01/el1/raw",
                       overwrite=True)
    
    basic.plotPSTH(spt_spikes['data'],  stim_spikes['data'],
                   win=[0,300])
    basic.plotPSTH(spt_simulated, stim_background['data'], win=[0,300])
    plt.figure()
    plt.plot(sp_waves['time'], avg_spike)
    plt.figure()
    plt.plot(sp_background['data'][0,:25E3])
    plt.plot(sp_simulated['data'][0,:25E3])
    plt.show()
