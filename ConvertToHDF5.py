import tables
import bakerlab


import numpy as np


def import_subject(subject, h5f, dummy=False):
    """add data of subject to the hdf5 file
    
    Arguments:
        subject - the prefix of the inf file ([subject]Inf.py)
        h5f - open HDF5 table
        dummy - create dummy variables (zeros) if the recording does
                not exist
    """
    descr = __import__(subject + "Inf")
    rec_dict = bakerlab.GetDataset(descr.recordings)
    subject_node = h5f.createGroup('/', subject)
    for rec in rec_dict:
        try:
            new_session=h5f.createGroup(subject_node, 's'+rec['session'])
            stim = bakerlab.readspt(descr.fstim % rec)
            stim_range = rec['stim_range']
            print stim_range
            if stim_range:
                stim = stim[stim_range[0]:stim_range[1]]
            h5f.createArray(new_session, 'stim', stim/200.)
            for key in chans:
                rec['chan']=descr.chan[key]
                try:
                    data = bakerlab.readsp(descr.fcore % rec)
                    data = data/32768.*5/descr.gains[key]
                    new_data=h5f.createArray(new_session,key, data)
                    new_data._v_attrs.sampfreq=descr.FS
                except IOError:
                    if dummy:
                        print "Warning: Creating dummy %s data for %s" % (key,
                            new_session)
                        data = np.zeros(stim.max()/200.*descr.FS/1000.)
                        new_data=h5f.createArray(new_session,key, data)
                        new_data._v_attrs.sampfreq=descr.FS
                        new_data._v_attrs.comment='Dummy'
                    else:
                        print "File %s not found" % descr.fcore % rec
                    
        except tables.exceptions.NodeError:
            new_session = h5f.getNode('/%s/s%s' % (subject, rec['session']))
        try:
            electrode_node=h5f.createGroup(new_session,'el%d' % rec['electrode'])
        except:
            electrode_node = h5f.getNode(new_session, 'el%d' % rec['electrode'])
   
        try:
            data_name = "%(session)s-%(electrode)d" % rec
            sp = bakerlab.readsp(descr.fspike % rec)
            #correct gains gains
            if data_name in descr.spike_gains:
                FS=descr.FS_spike
                gains, times = descr.spike_gains[data_name]
                for i in range(len(gains)-1):
                    sp[times[i]/1000.*FS:times[i+1]/1000*FS]/=gains[i]
                sp[times[-1]/1000.*FS:]/=gains[-1]

            raw_spikes = h5f.createArray(electrode_node, 'raw', sp)
            raw_spikes._v_attrs.sampfreq=descr.FS_spike
        except tables.exceptions.NodeError:
            pass
        spt = bakerlab.readspt(descr.fspt % rec)
        cell_node=h5f.createArray(electrode_node, 'cell%d' % rec['cellid'], spt/200.)
        cell_node.attrs.events = rec['EventBorders']

import os

datapath = os.environ.get("DATAPATH")+'/hdf5/'
chans = ["Ball", "Pt"]
subjects = ["Joy", "Poppy"] 
h5f = tables.openFile(datapath + 'data_microel.h5', 'a', title='Data')

for subject in subjects:
    import_subject(subject, h5f, dummy=False)

h5f.close()
