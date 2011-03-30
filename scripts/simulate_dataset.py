#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from spike_sort.io import filters
import spike_sort as sort
from spike_analysis import basic


spike_src = "/Gollum/s39gollum03/el1"
cell_id = "cell1"
background_src = "/Gollum/s39gollum03/el1"
sp_win = [-0.2, 0.8]


def cell2stim(node):
    return "/".join(node.split("/")+['stim'])

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

    sp_simulated = sp_background

    psth_spikes, time = basic.CalcPSTH(spt_spikes['data'], 
                                       stim_spikes['data'],
                                      win=[0,300])
    
    out_filter.write_spt(spt_spikes,
                         "/TestSubject/sSession01/el1/cell1",
                        overwrite=True)
    out_filter.write_spt(stim_background,
                         "/TestSubject/sSession01/el1/stim",
                        overwrite=True)
    out_filter.write_sp(sp_simulated,
                        "/TestSubject/sSession01/el1/raw",
                       overwrite=True)
    
    plt.plot(time, psth_spikes)
    plt.show()
