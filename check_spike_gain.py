#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tables
import patterns
import eegtools
import utils

from NeuroTools.parameters import ParameterSet as NTParameterSet

datapath = os.environ.get("DATAPATH")

def which_window(spt, stim, ev):

    bWin = np.vstack([spike_in_win(spt, stim, [ev[i], ev[i+1]]) 
                   for i in range(len(ev)-1)])

    if len(ev)>2:
        cl = bWin.argmax(0)
    else:
        cl = bWin[0,:]*1

    return cl


def spike_in_win(spt, stim, win):

    i = np.searchsorted(stim, spt)-1

    sp_pst = spt-stim[i]

    bool = ((sp_pst>win[0]) &( sp_pst<win[1]))

    return bool

def main_plot(h5f_in, cell, sp_win, n_spikes=100):

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
    
       
    spt_reduced = spt[spike_in_win(spt, stim, [sp_events[0],
        sp_events[-1]])]

    sp_waves, sp_time = eegtools.GetEEGTrials(sp_raw,
                               spt_reduced/1000.*FS, win=sp_win,Fs=FS)
    sp_waves = (sp_waves - sp_waves.min())/(sp_waves.max()-sp_waves.min())
    spt_reduced = spt_reduced[:sp_waves.shape[1]]

    amp = sp_waves.max(0)

    
    idx_changed = 164275.75
    plt.figure()
    plt.plot(spt_reduced, amp)
    plt.figure()
    plt.subplot(211)
    i = np.abs(spt_reduced-idx_changed).argmin()
    plt.plot(spt_reduced[(i-100):(i+100)]-idx_changed, 
                     amp[(i-100):(i+100)])

    i = np.arange(idx_changed/1000.*FS-5000,
            idx_changed/1000.*FS+5000, dtype=int) 
    plt.subplot(212)
    plt.plot(np.arange(-5000, 5000)*1000./FS, sp_raw[i])
    plt.show()

if __name__ == "__main__":
    parameter_file = sys.argv[1]
    parameters = NTParameterSet(parameter_file)

    h5f = tables.openFile(datapath+parameters.in_datafile, 'r')
    
    cell = parameters.sample_cell_detail.Dataset
    params = parameters.process_params

    main_plot(h5f, cell, params.sp_win)

    h5f.close()
