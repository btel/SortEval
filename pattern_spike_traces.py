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

#scale=1.0
#font_size= 10
#fig_width = 4 / scale  # width in inches
#fig_height= 5.4 / scale # height in inches
#fig_size =  [fig_width,fig_height]
#params = {'backend': 'GTKAgg',
#          'axes.labelsize': font_size,
#          'axes.titlesize': 10,
#          'text.fontsize': font_size,
#          'xtick.labelsize': font_size,
#          'ytick.labelsize': font_size,
#          'text.usetex': False,
#          'figure.figsize': fig_size,
#          'figure.dpi': 600,
#          'savefig.dpi' : 600,
#          #'font.family': 'sans-serif',
#          'lines.linewidth': 0.5}
#plt.rcParams.update(params)


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

def main_plot(h5f_in, cell, sp_win, labels, n_spikes=20):

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
    
    sp_waves, sp_time = eegtools.GetEEGTrials(sp_raw,
                               stim/1000.*FS, win=sp_win,Fs=FS)
    #sp_waves = (sp_waves - sp_waves.min())/(sp_waves.max()-sp_waves.min())

    trains=patterns.SortSpikes(spt,stim,[sp_events[0], sp_events[-1]])
    cl = patterns.FindClasses(trains, sp_events)
    
    colors = ['r', 'b']
    pattern_labels = map(utils.binstr2dec, labels)

    for i, pattern in enumerate(pattern_labels):
        plt.subplot(len(pattern_labels), 1, i+1)
        sp_win_waves = sp_waves[:, cl==pattern]
        idx = np.random.rand(sp_win_waves.shape[1]).argsort()
        traces_to_plot = sp_win_waves[:,idx[:n_spikes]]
        plt.plot(sp_time, traces_to_plot, colors[i],
                alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        ylims = traces_to_plot.min(), traces_to_plot.max() 
        #plt.text(0.1, 0.9, labels[i], transform=ax.transAxes,
        #        va="top", size=9)
        plt.vlines(sp_events, *ylims, linestyle='-', zorder=10)
        plt.twinx()
        stim_pattern = stim[cl==pattern]
        patterns.plotraster(spt, np.sort(stim_pattern[idx[:n_spikes]]), sp_win)

    #plt.plot([sp_win[1], sp_win[1]-0.3], [1., 1.], 'k', lw=1.5)
    #plt.text(sp_win[1]-0.15, 1.06, "0.3 ms", va="bottom", ha="center",
    #        size=9)
    #plt.text(0.0, 0.9, "A", transform=axes_list[0].transAxes,
    #        size=10, weight='bold')


if __name__ == "__main__":
    parameter_file = sys.argv[1]
    parameters = NTParameterSet(parameter_file)

    h5f = tables.openFile(datapath+parameters.in_datafile, 'r')
    
    cell = parameters.sample_cell_detail    
    params = parameters.process_params

    main_plot(h5f, cell.Dataset, [5, 15], cell.pattern)

    h5f.close()

    plt.show()
#savefig(sys.argv[1], transparent=True)
