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


def read_data(h5f_in, cell):

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
    #is_spike = np.array([len(t)>0 for t in trains])
    cl = patterns.FindClasses(trains, sp_events)
    
    return {"stim": stim, "spt" : spt, "sp_raw": sp_raw, "FS": FS,
            "sp_events" : sp_events, "cl": cl}
i = -1

def find_spikes(data_dict,win, is_spike, sp_win=[-0.2, 0.8]):

    def next_plot(inc=1):
        global i
        ax.cla()
        i=i+inc
        if i < sp_waves.shape[1]:
            print i
            ax.plot(sp_time, sp_waves[:,i], picker=5)
            ax.vlines(sp_events,0,1)
            if len(trains_new[i])>0:
                ax.plot(trains_new[i],np.ones(len(trains_new[i]))*0.8,
                        "*")
            if i in missed_trains:
                ax.plot(missed_trains[i],np.ones(len(missed_trains[i]))*0.8,
                        "r*")
            ax.set_ylim(0,1)
            ax.set_title("%d/%d" % (i, sp_waves.shape[1]))
            fig.canvas.draw()
        else:
            print "end"

    def onclick(event):
        if event.button == 1:
            if event.inaxes is None:
                next_plot()
            else:
                print "bla"
        elif event.button == 3:
            next_plot(-1)


    def onpick(event):
        if event.mouseevent.button == 1:
            #append a new spike
            _win = (np.array(sp_win)/1000.*FS).astype(int)
            sp_idx = (spt_idx[i] + win[0]/1000.*FS + event.ind[0])
            spike = sp_raw[int(sp_idx)+_win[0]:int(sp_idx)+_win[1]]
        
            missed_spikes.append(sp_idx/FS*1000.)
            if not i in missed_trains:
                missed_trains[i] = []
            missed_trains[i].append(sp_idx/FS*1000-stim[i])
            ax2.plot(spike, 'k')
            fig2.canvas.draw()
            next_plot()
        elif event.mouseevent.button == 3:
            #implement deleting a spike
            #for now just go to the previous trial
            next_plot(-1)



    stim = data_dict['stim']
    spt = data_dict['spt']
    sp_raw = data_dict['sp_raw']
    FS = data_dict['FS']
    sp_events = data_dict['sp_events']
   
    spt_idx = stim[is_spike]/1000.*FS
    trains_new=patterns.SortSpikes(spt,stim[is_spike], win)
    
    sp_waves, sp_time = eegtools.GetEEGTrials(sp_raw, spt_idx,
            win=win,Fs=FS)
    sp_waves = (sp_waves -
            sp_waves.min())/(sp_waves.max()-sp_waves.min())

    missed_spikes = []
    missed_trains = {}
    fig = plt.figure()
    ax = plt.subplot(111)
    next_plot()
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect('pick_event', onpick)

    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    plt.show()

    missed_spikes = np.array(missed_spikes)

    return missed_spikes
   
def plot_spikes(data, spt, sp_win=[-0.2, 0.8]):
    sp_raw = data['sp_raw']
    FS = data['FS']
    sp_waves, sp_time = eegtools.GetEEGTrials(sp_raw, spt/1000*FS,
            win=sp_win,Fs=FS)

    plt.plot(sp_time, sp_waves)
    plt.show()

if __name__ == "__main__":
    parameter_file = "params/params.cfg"
    parameters = NTParameterSet(parameter_file)

    out_dir = utils.create_new_dir("Data/", "find_missed_spikes")
    h5f = tables.openFile(datapath+parameters.in_datafile, 'r')
    
    cell = parameters.sample_cell_detail    
    params = parameters.process_params

    data = read_data(h5f, cell.Dataset)
    missed_spikes = find_spikes(data, [8,13], data['cl']>=0)

    export_spikes = (missed_spikes*200).astype(np.int32)

    export_spikes.tofile(os.path.join(out_dir, "missed.spt"))

    plot_spikes(data,export_spikes/200.)

    print "Data saved to:", out_dir

    h5f.close()

#savefig(sys.argv[1], transparent=True)
