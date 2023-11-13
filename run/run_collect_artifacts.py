#!/usr/bin/python3
# it collects the data that will NOT be used for source reconstruction
import sys, os
sys.path.append( os.path.expandvars('$OSCBAGDIS_DATAPROC_CODE') )

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

print(f'Starting {__file__}')


#subjinds = [1,2,3]
#tasks = ['hold' , 'move', 'rest']
#medstates = ['on','off']
rawnames = ['S01_off_move', 'S01_on_hold']

# annotations types to be excluded, except beh_state_types_not_to_exclude
#ann_types = ['beh_states', 'MEGartif', 'LFPartif']
#ann_types = ['beh_states', 'MEGartif']
ann_types = ['beh_states', 'MEGartif_ICA', 'MEGartif_muscle']

beh_state_types_not_to_exclude = ['notrem_{}']

import sys
import getopt

min_duration_remaining = 30  # in sec
exclude_artifacts_only = True
overwrite = True

output_subdir = ""

print('sys.argv is ',sys.argv)
effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three


helpstr = 'Usage example\nrun_collect_artifacts.py --rawname <rawname_naked> '
opts, args = getopt.getopt(effargv,"hr:", ["ann_types=","rawname=","min_dur=","output_subdir=" ])
print(sys.argv, opts, args)

for opt, arg in opts:
    print(opt,arg)
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
    elif opt == "--ann_types":
        ann_types = arg.split(',')
    elif opt == "--output_subdir":
        output_subdir = arg
    #elif opt == "--exclude_artifacts_only":
    #    exclude_artifacts_only = int(arg)
    elif opt == "--min_dur":
        min_duration_remaining = float(arg)
    elif opt == "--overwrite":
        overwrite = int(arg)
    else:
        raise ValueError('Unknown option {} with arg {}'.format(opt,arg) )


data_dir_output = pjoin(gv.data_dir, output_subdir)
if not os.path.exists(data_dir_output):
    print('Creating {}'.format(data_dir_output) )
    os.makedirs(data_dir_output)

print('ann_types', ann_types)

raws_permod_both_sides = upre.loadRaws(rawnames,['EMG'], None, None, None)

# if I use notrem from both sides, I am still merging all non-quiet periods
# together from both sides. So the remaining will be essentially an
# intersection of both untrem_L and untrem_R
# if I use only one notrem, then I will essentially subtract one notrem
# from another and it is not what I want
keep_only_main_side = False

for rawname_ in rawnames:

    raw = raws_permod_both_sides[rawname_]['EMG']
    duration =raw.times[-1]
    sfreq = int(raw.info['sfreq'])

    anns_fnames = []
    # TODO: maybe add filtering of beh_states (e.g. specifically tremor and so)
    if 'beh_states' in ann_types:
        fname = '{}_anns.txt'.format(rawname_)
        anns_fnames += [fname]
    if 'MEGartif' in ann_types:
        fname = '{}_ann_MEGartif.txt'.format(rawname_)
        anns_fnames += [fname]
    if 'MEGartif_ICA' in ann_types:
        fname = '{}_ann_MEGartif_ICA.txt'.format(rawname_)
        anns_fnames += [fname]
    if 'MEGartif_muscle' in ann_types:
        fname = '{}_ann_MEGartif_muscle.txt'.format(rawname_)
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


    subj_cur,medcond_cur,task_cur  = utils.getParamsFromRawname(rawname_)


    # define main_side. If move_side is present, we use it, otherwise we use
    # tremor side
    sinfo = gv.gen_subj_info[subj_cur]
    mainmoveside_cur = sinfo.get('move_side',None)
    maintremside_cur = sinfo.get('tremor_side',None)
    main_side = None
    if mainmoveside_cur  is not None:
        main_side = mainmoveside_cur
    else:
        main_side = maintremside_cur
    assert main_side is not None

    # description to skip from being included for exclusion. So we will KEEP
    # only these desriptions
    descr_to_skip = []
    for bst in beh_state_types_not_to_exclude :
        if keep_only_main_side:
            descr_to_skip += [ bst.format(main_side[0].upper() ) ]
        else:
            descr_to_skip += [ bst.format('L') ]
            descr_to_skip += [ bst.format('R') ]
    assert len(descr_to_skip) > 0
    print('descr_to_skip (the only ones that will not be included in the list for exclusion) = ',descr_to_skip)


    # merge all annotations, except non-tremor ones
    merged_anns = utils.mergeAnns(ann_list,duration,sfreq,out_descr='BAD_intervals',
                    descr_to_skip=descr_to_skip)
    print('Anns to be excluded for src reconstruction',merged_anns,merged_anns.onset,merged_anns.duration)

    dur_merged = np.sum(  merged_anns.duration )
    print('Duration of the excluded data {}, of total {}'.format(dur_merged,duration) )

    # skip the assertiong for testing data (it is shorter)
    if subj_cur != 'S99':
        assert( duration -  np.sum(  merged_anns.duration ) > min_duration_remaining ), \
            (duration,dur_merged,min_duration_remaining)

    fn = '{}_ann_srcrec_exclude.txt'.format(rawname_)
    fn_full = pjoin(data_dir_output,fn)
    print('Saving ',fn_full)
    merged_anns.save( fn_full, overwrite=overwrite  )
