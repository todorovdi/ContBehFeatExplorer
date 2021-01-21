#!/usr/bin/python3

import os
import mne
from mne.preprocessing import ICA
import multiprocessing as mpr
import numpy as np
import utils
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils_preproc as upre
import matplotlib as mpl
import globvars as gv
from globvars import gp

from os.path import join as pjoin


#subjinds = [1,2,3]
#tasks = ['hold' , 'move', 'rest']
#medstates = ['on','off']
rawnames = ['S01_off_move', 'S01_on_hold']

#ann_types = ['beh_states', 'MEGartif', 'LFPartif']
ann_types = ['beh_states', 'MEGartif']

import sys
import getopt

min_duration_remaining = 30  # in sec

print('sys.argv is ',sys.argv)
effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

helpstr = 'Usage example\nrun_collect_artifacts.py --rawname <rawname_naked> '
opts, args = getopt.getopt(effargv,"hr:", ["ann_types=","rawname="])
print(sys.argv, opts, args)

for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt in ('-r','--rawname'):
        if len(arg) < 5:
            print('Empty raw name provided, exiting {}'.format(arg) )
            sys.exit(1)
        rawnames = arg.split(',')  #lfp of msrc
        if len(rawnames) > 1:
            print('Using {} datasets at once'.format(len(rawnames) ) )
        #rawname_ = arg
    elif opt == "--mods":
        ann_types = arg.split(',')


print('ann_types', ann_types)

raws_permod_both_sides = upre.loadRaws(rawnames,['EMG'], None, None, None)

for rawname_ in rawnames:

    raw = raws_permod_both_sides[rawname_]['EMG']
    duration =raw.times[-1]
    sfreq = int(raw.info['sfreq'])

    anns_fnames = []
    if 'beh_states' in ann_types:
        fname = '{}_anns.txt'.format(rawname_)
        anns_fnames += [fname]
    if 'MEGartif' in ann_types:
        fname = '{}_ann_MEGartif.txt'.format(rawname_)
        anns_fnames += [fname]
    if 'MEGartif_flt' in ann_types:
        fname = '{}_ann_MEGartif_flt.txt'.format(rawname_)
        anns_fnames += [fname]
    if 'LFPartif' in ann_types:
        fname = '{}_ann_LFPartif.txt'.format(rawname_)
        anns_fnames += [fname]

    ann_list = []
    for ann_fname in anns_fnames:
        anns_cur = mne.read_annotations(pjoin(gv.data_dir,ann_fname ) )
        if len(anns_cur) == 0:
            continue
        ann_list += [anns_cur]
        print(ann_fname, anns_cur,anns_cur.onset,anns_cur.duration,anns_cur.description)

    merged_anns = utils.mergeAnns(ann_list,duration,sfreq,out_descr='BAD_intervals')
    print(merged_anns,merged_anns.onset,merged_anns.duration)

    assert( duration -  np.sum(  merged_anns.duration ) > min_duration_remaining )

    fn = '{}_ann_srcrec_exclude.txt'.format(rawname_)
    fn_full = pjoin(gv.data_dir,fn)
    print('Saving ',fn_full)
    merged_anns.save( fn_full  )
