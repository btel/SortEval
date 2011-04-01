#!/usr/bin/env python
#coding=utf-8
import sys
sys.path.append("../modules/")
import numpy as np
import matplotlib.pyplot as plt

from spike_sort.io import filters
import spike_sort as sort
from spike_analysis import basic
import evaluate as eval

spike_src = "/Gollum/s39gollum03/el1/cell1"
background_src = "/Gollum/s39gollum03/el3"
sp_win = [-2, 2]
pow_frac = 1
out_dataset = "/TestSubject/sSession01/el1"


if __name__ == "__main__":
    in_filter = filters.BakerlabFilter("../data/gollum.inf")
    out_filter = filters.PyTablesFilter("simulated.h5")

    sp_sim, stim_bg, spt_sim = eval.mix_cellbg(in_filter, spike_src,
                                               background_src,
                                              sp_win, pow_frac) 
    #export
    out_filter.write_spt(spt_sim, out_dataset+"/cell1_orig",
                        overwrite=True)
    out_filter.write_spt(stim_bg,out_dataset+ "stim",
                        overwrite=True)
    out_filter.write_sp(sp_sim, out_dataset+"/raw",
                       overwrite=True)
   
    #plotting
    raw_bg, _ = eval.read_data(in_filter, background_src)
    basic.plotPSTH(spt_sim['data'], stim_bg['data'], win=[0,300])
    plt.figure()
    plt.plot(raw_bg['data'][0,:25E3])
    plt.plot(sp_sim['data'][0,:25E3])
    plt.show()
