#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import pymongo
import pprint
db_name = 'eval_batch'
col_name = 'snr_test'
connection = pymongo.Connection('localhost', 27017)
db = connection[db_name]
collection = db[col_name]

reduce = '''function(obj, prev) {
            prev.missed.push(100*obj.spikes_missed/obj.spikes_total);
            prev.k_nearest.push(obj.k_nearest)}'''

def group_by(field):
    return collection.group([field], criteria,
                     {'missed':[], 'k_nearest':[]}, reduce)

def print_group_by(field):
    print "Group by %s" % field
    pprint.pprint(collection.group([field], criteria,
                                   {'missed':[], 'k_nearest':[]}, reduce))

def group_boxplot(grp_field, output):
    result = group_by(grp_field)

    
    vectors = [r[output] for r in result]
    labels =  [str(r[grp_field]) for r in result]
    
    ax = plt.subplot(111)
    plt.boxplot(vectors)

    ax.xaxis.set_ticklabels(labels)
    plt.xlabel(grp_field)
    plt.ylabel(output)
    

if __name__ == "__main__":
    criteria = {}


    print "Available cells:", collection.distinct("cell")
    print "Available electrodes:", collection.distinct("electrode:")
    print_group_by('cell') 
    print_group_by('electrode:') 
    print_group_by('pow_frac') 

    group_boxplot('features', 'k_nearest')
    plt.show()
    
