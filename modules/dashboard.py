#!/usr/bin/env python
#coding=utf-8
from matplotlib import rcParams

font_size= 10
params = {'backend': 'WxAgg',
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'text.fontsize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'text.usetex': False,
          'font.size' : font_size,
          }
rcParams.update(params)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tables
import os, sys
import patterns
import hdf5tools

from NeuroTools.parameters import ParameterSet as NTParameterSet

DATAPATH = os.environ['DATAPATH']

def plot_psth(dataset, **kwargs):
    spt = dataset['spt']
    stim = dataset['stim']
    ev = dataset['ev']

    patterns.plotPSTH(spt, stim, **kwargs)
    ymin, ymax = plt.ylim()
    plt.vlines(ev, ymin, ymax)

def plot_isi(dataset, win=[0,5], bin=0.1):
    spt = dataset['spt']
    stim = dataset['stim']

    isi = np.diff(spt)
    intvs = np.arange(win[0], win[1], bin)
    counts, bins = np.histogram(isi, intvs)
    mode,n = stats.mode(isi)
    
    plt.plot(intvs[:-1], counts, drawstyle='steps-post')
    plt.axvline(mode, color='k')
    
    ax = plt.gca()
    plt.text(0.95, 0.9,"mode: %.2f ms" % (mode,),
             transform=ax.transAxes,
            ha='right')
    plt.xlabel("interval (ms)")
    plt.ylabel("count")

def plot_trains(dataset, **kwargs):
    spt = dataset['spt']
    stim = dataset['stim']
    ev = dataset['ev']
    
    patterns.plotraster(spt, stim, **kwargs)
    ymin, ymax = plt.ylim()
    plt.vlines(ev, ymin, ymax)

def plot_nspikes(dataset, win=[0,30]):
    spt = dataset['spt']
    stim = dataset['stim']
    
    trains = patterns.SortSpikes(spt, stim, win)
    n_spks = np.array([len(t) for t in trains])
    count, bins = np.histogram(n_spks, np.arange(10))
    plt.bar(bins[:-1]-0.5, count) 

    burst_frac = np.mean(n_spks>1) 
    ax = plt.gca()
    plt.text(0.95, 0.9,"%d %% bursts" % (burst_frac*100,),
             transform=ax.transAxes,
            ha='right')
    plt.xlabel("no. spikes")
    plt.ylabel("count")


def cell_dashboard(h5f, cell):

    dataset = hdf5tools.read_hdf5_dataset(h5f, cell)

    plt.subplots_adjust(hspace=0.3)

    plt.subplot(2,2,1)
    plot_psth(dataset)
    plt.title("PSTH")
    plt.subplot(2,2,2)
    plot_isi(dataset)
    plt.title("ISIH")
    plt.subplot(2,2,3)
    plot_trains(dataset)
    plt.title("raster")
    plt.subplot(2,2,4)
    plot_nspikes(dataset)
    plt.title("burst order")

def main(param_fname, out_dir):
    
    params = NTParameterSet(param_fname)
    h5fname = DATAPATH[:-1]+ params.in_datafile
    h5f = tables.openFile(h5fname,'r')
    cells = params.cells
    
    summary_dir = os.path.join(out_dir, "summary")
    os.mkdir(summary_dir)
    
    for cell in cells:
        cell_dashboard(h5f, cell)
        cell_id = "-".join(cell.split('/')[2:])
        plt.savefig(os.path.join(summary_dir, cell_id+".png"))
        plt.close()

if __name__ == "__main__":

    _, param_fname, out_dir = sys.argv
    main(param_fname, out_dir)

