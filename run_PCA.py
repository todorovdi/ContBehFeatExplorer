import os
import sys
import mne
import json
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mpr
import matplotlib as mpl
import time
import gc;
import scipy.signal as sig

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif

import globvars as gv
import utils
import utils_tSNE as utsne
import utils_preproc as upre

import re
import sys, getopt
import copy
from globvars import gp
import datetime

from xgboost import XGBClassifier

if os.environ.get('DATA_DUSS') is not None:
    data_dir = os.path.expandvars('$DATA_DUSS')
else:
    data_dir = '/home/demitau/data'

#nPCA_comp = 0.95
nPCA_comp = 0.95
n_channels = 7
skip = 32
windowsz = 256

nPCAcomponents_to_plot = 5
nfeats_per_comp_LDA = 50
nfeats_per_comp_LDA_strongred = 5

show_plots = 0

discard = 1e-2
qshift = 1e-2
force_single_core = False
##########################

feat_types_all = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr']
data_modalities_all = ['LFP', 'msrc']

features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr']
data_modalities = ['LFP', 'msrc']
msrc_inds = np.arange(8,dtype=int)  #indices appearing in channel (sources) names, not channnel indices
use_main_LFP_chan = False
use_lfp_HFO = 1

fband_names_crude = ['tremor', 'beta', 'gamma']
fband_names_fine = ['tremor', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma' ]
fband_names_crude_inc_HFO = fband_names_crude + ['HFO']
fband_names_fine_inc_HFO = fband_names_fine + ['HFO1', 'HFO2', 'HFO3']
fbands_to_use = fband_names_fine_inc_HFO
bands_type = 'fine'

prefix = ''

#do_impute_artifacts = 1
artifact_handling = 'impute' #or 'discard' or 'do_nothing'
do_outliers_discard = 1

def_sources_type = 'HirschPt2011'
sources_type = def_sources_type

load_only = 0   # load and preproc to be precise
do_LDA = 1
n_feats = 609  # this actually depends on the dataset which may have some channels bad :(
do_XGB = 1

remove_crossLFP = 1
##############################

effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

print(effargv)
n_feats_set_explicitly  = False

crop_end,crop_start = None,None
parcel_types = ['all'] # here it means 'all contained in the feature file'

src_file_grouping_ind = 10
src_grouping = 0  # src_grouping is used to get info from the file

LFP_related_only = 0
parcel_group_names = []

subskip_fit = 1   # > 1 means that I fit to subsampled dataset
feat_variance_q_thr = [0.6, 0.75, 0.9]  # I will select only features that have high variance according to PCA (high contributions to highest components)
use_low_var_feats_for_heavy_fits = True # whether I fit XGB and min feat sel only to highly varing features (according to feat_variance_q_thr)
search_best_LFP = 1

save_output = 1
rescale_feats = 1

cross_couplings_only=0
mainLFPchan_new_name_templ = 'LFP{}007'    # to make datasets consistent
remove_corr_self_couplings = 1

#groupings_to_use = gp.groupings
groupings_to_use = [ 'merge_all_not_trem', 'merge_movements', 'merge_nothing' ]
groupings_to_use = [ 'merge_nothing' ]
#int_types_to_use = gp.int_types_to_include
int_types_to_use = [ 'basic', 'trem_vs_quiet' ]

helpstr = 'Usage example\nrun_PCA.py --rawnames <rawname_naked1,rawnames_naked2> '
opts, args = getopt.getopt(effargv,"hr:n:s:w:p:",
        ["rawnames=", "n_channels=", "skip_feat=", "windowsz=", "pcexpl=",
         "show_plots=","discard=", 'feat_types=', 'use_HFO=', 'mods=',
         'prefix=', 'load_only=', 'fbands=', 'n_feats=', 'single_core=', 'sources_type=',
         'bands_type=', 'crop=', 'parcel_types=', "src_grouping=", "src_grouping_fn=",
         'groupings_to_use=', 'int_types_to_use=', 'skip_XGB=', 'LFP_related_only=',
         'parcel_group_names=', "subskip_fit=", "search_best_LFP=", "save_output=",
        'rescale_feats=', "cross_couplings_only=", "LFPchan=", "heavy_fit_red_featset=" ])
print(sys.argv)
print(opts)
print(args)
for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == "--n_channels":
        n_channels = int(arg)
    elif opt == "--n_feats":
        n_feats = int(arg)
        n_feats_set_explicitly = True
    elif opt == "--skip_XGB":
        do_XGB = not bool(int(arg))
    elif opt == "--heavy_fit_red_featset":
        use_low_var_feats_for_heavy_fits = int(arg)
    elif opt == "--parcel_types":  # names of the roi_labels (not groupings)
        parcel_types = arg.split(',')
    elif opt == "--parcel_group_names":
        parcel_group_names = arg.split(',')
    elif opt == "--crop":
        cr =  arg.split(',')
        crop_start = float(cr[0])
        crop_end = float(cr[1] )
    elif opt == '--int_types_to_use':
        int_types_to_use = arg.split(',')
    elif opt == '--groupings_to_use':
        groupings_to_use = arg.split(',')
    elif opt == "--search_best_LFP":
        search_best_LFP = int(arg)
    elif opt == "--src_grouping":
        src_grouping = int(arg)
    elif opt == "--src_grouping_fn":
        src_file_grouping_ind = int(arg)
    elif opt == "--rescale_feats":
        rescale_feats = int(arg)
    elif opt == "--cross_couplings_only":
        cross_couplings_only = int(arg)
    elif opt == "--skip_feat":
        skip = int(arg)
    elif opt == "--save_output":
        save_output = int(arg)
    elif opt == "--bands_type":
        bands_type = arg
    elif opt == "--sources_type":
        if len(arg):
            sources_type = arg
    elif opt == "--single_core":
        force_single_core = int(arg)
    elif opt == '--load_only':
        load_only = int(arg)
    elif opt == '--show_plots':
        show_plots = int(arg)
    elif opt == '--subskip_fit':
        subskip_fit = int(arg)
    elif opt == "--windowsz":
        windowsz = int(arg)
    elif opt == "--LFP_related_only":
        LFP_related_only = int(arg)
    elif opt == "--discard":
        discard = float(arg)
    elif opt == "--prefix":
        prefix = arg + '_'
    elif opt == "--msrc_inds":
        msrc_inds = [ int(indstr) for indstr in arg.split(',') ]
        msrc_inds = np.array(msrc_inds, dtype=int)
    elif opt == "--feat_types":
        features_to_use = arg.split(',')
        for ftu in features_to_use:
            assert ftu in feat_types_all, ftu
    elif opt == "--mods":
        data_modalities = arg.split(',')   #lfp of msrc
    elif opt == '--fbands':
        fbands_to_use = arg.split(',')
    elif opt == '--LFPchan':
        if arg == 'main': # use only main channel
            use_main_LFP_chan = 1
        elif arg == 'all':   # use all channels
            use_main_LFP_chan = 0
        elif arg.find('LFP') >= 0:   # maybe I want to specify expliclity channel name
            raise ValueError('to be implemented')
    elif opt == '--use_HFO':
        use_lfp_HFO = int(arg)
    elif opt == "--pcexpl":
        nPCA_comp = float(arg)  #crude of fine
        if nPCA_comp - int(nPCA_comp) < 1e-6:
            nPCA_comp = int(nPCA_comp)
    elif opt in ('-r','--rawnames'):
        rawnames = arg.split(',')
    else:
        print('Unrecognized option {} with arg {}, exiting'.format(opt,arg) )
        sys.exit(1)

#print('test exit', parcel_types); sys.exit(1) ;

if bands_type == 'fine':
    fbands_def = fband_names_fine_inc_HFO
else:
    fbands_def = fband_names_crude_inc_HFO

print('nPCA_comp = ',nPCA_comp)

############################
rn_str = ','.join(rawnames)

with open('subj_info.json') as info_json:
    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)

##############################

tasks = []
X_pri = []
feature_names_pri = []
selected_feat_inds_pri = []

feat_fnames = []
feat_file_pri = []
Xtimes_pri = []
subjs_pri = []
mts_letters_pri = []
src_rec_info_pri = []
rawtimes_pri = []
wbd_pri = []
main_side_pri = []
new_main_side_pri = []
feat_info_pri = []
fname_feat_full_pri = []
chnames_src_pri = []
chnames_LFP_pri = []

# Load everything
for rawname_ in rawnames:
    subj,medcond,task  = utils.getParamsFromRawname(rawname_)
    subjs_pri += [subj]
    #8,9 -- more LFP chanenls
    #if subj in ['S08', 'S09' ] and not n_feats_set_explicitly:
    #    raise ValueError('need to be finished')
    #    n_feats #= some special number

    ############### load stuff

    crp_str = ''
    if crop_end is not None:
        crp_str = '_crop{}-{}'.format(int(crop_start),int(crop_end) )

    if sources_type == def_sources_type:
        a = '{}_feats_{}chs_nfeats{}_skip{}_wsz{}_grp{}-{}{}.npz'.\
            format(rawname_,n_channels, n_feats, skip, windowsz,
                   src_file_grouping_ind, src_grouping, crp_str)
        feat_fnames += [a]
        fname_feat_full = os.path.join( data_dir,a)
    else:
        regex = '{}_feats_{}_{}chs_nfeats{}_skip{}_wsz{}_grp{}-{}{}.npz'.\
            format(rawname_,sources_type,'[0-9]+', '[0-9]+', skip, windowsz,
                    src_file_grouping_ind, src_grouping, crp_str)
            #format(rawname_, prefix, regex_nrPCA, regex_nfeats, regex_pcadim)

        fnfound = utsne.findByPrefix(data_dir, rawname_, prefix, regex=regex)
        if len(fnfound) > 1:
            fnt = [0] * len(fnfound)
            for fni in range(len(fnt) ):
                fnfull = os.path.join(data_dir, fnfound[fni])
                fnt[fni] = os.path.getmtime(fnfull)
            fni_max = np.argmax(fnt)
            fnfound = [ fnfound[fni_max] ]


        assert len(fnfound) == 1, 'For {} found not single fnames {}'.format(rawname_,fnfound)
        fname_feat_full = os.path.join( data_dir, fnfound[0] )

    fname_feat_full_pri += [fname_feat_full]

    modtime = datetime.datetime.fromtimestamp(os.path.getmtime(fname_feat_full)  )
    print('Loading feats from {}, modtime {}'.format(fname_feat_full, modtime ))
    f = np.load(fname_feat_full, allow_pickle=True)
    feat_file_pri += [f]

    ################## extract stuff


    sfreq = f['sfreq']
    X_allfeats =  f['X']
    Xtimes = f['Xtimes']
    rawtimes = f['rawtimes']
    skip_ =f['skip']
    wbd =f.get('wbd',None)
    feature_names_all = f['feature_names_all']
    chnames_src = f['chnames_src']
    chnames_LFP = f['chnames_LFP']

    chnames_src_pri += [chnames_src]
    chnames_LFP_pri += [chnames_LFP]

    feat_info = f.get('feat_info',None)[()]
    mts_letter = gen_subj_info[subj]['tremor_side'][0].upper()
    nedgeBins = feat_info['nedgeBins']
    assert skip_ == skip

    # this is what is actually in features already, so it is needed only for
    # annotations. Body side is meant, not brain side
    new_main_side = feat_info.get('new_main_side','L')
    main_side = feat_info.get('main_side_before_switch','L')
    main_side_pri += [main_side]
    new_main_side_pri += [new_main_side]

    feat_info_pri += [feat_info]
    rawtimes_pri += [rawtimes]

    if Xtimes[1] > 0.5:  # then we have saved the wrong thing
        print('Warning: Xtimes {} are too large, dividing by sfreq '.format(Xtimes[:5] ) )
        Xtimes = Xtimes / sfreq
    Xtimes_pri += [Xtimes]

    if wbd is None:
        print('Warning! wbd is not int the feature file! Using Xtimes + wsz')
        wbd = np.vstack([Xtimes,Xtimes]) * sfreq # we need orig bins
        wbd[1] += windowsz

    if X_allfeats.shape[0] < wbd.shape[1]: # this can arise from saveing wbd_H instead of the right thing
        print('Warning, differnt X_allfeats and wbd shapes {},{}, cropping wbd'.
              format( X_allfeats.shape , wbd.shape )  )
        wbd = wbd[:,:X_allfeats.shape[0] ]
    assert wbd.shape[1] == len(Xtimes)   # Xtimes were aligned for sure
    wbd_pri += [wbd]

    tasks += [task]

    # TODO: allow to use other mainLFPchans
    #mainLFPchan = gen_subj_info[subj]['lfpchan_used_in_paper']
    mainLFPchan = gen_subj_info[subj].get('lfpchan_selected_by_pipeline',None)
    if use_main_LFP_chan:
        assert mainLFPchan is not None



    bad_inds = set([] )


    src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.\
        format(rawname_,sources_type, src_file_grouping_ind)
    src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
    src_rec_info_pri += [rec_info]
    roi_labels = rec_info['label_groups_dict'][()]
    srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]

    # remove those that don't have LFP in their names i.e. purely MEG features
    if LFP_related_only:
        regexes = ['.*LFP.*']  # LFP to LFP (all bands) and LFP to msrc (all bands)
        inds_good_ = utsne.selFeatsRegexInds(feature_names_all,regexes)
        inds_bad_ = set(range(len(feature_names_all))) - set(inds_good_)
        bad_inds.update(inds_bad_)

    # remove features involving parcels not of the desired type
    if len(parcel_types) == 1 and parcel_types[0] == 'all' and len(parcel_group_names) == 0:
        print('Using all parcels from the file')
    else:
        assert set(roi_labels.keys() ) == set(srcgrouping_names_sorted )

        all_parcels= roi_labels[gp.src_grouping_names_order[src_file_grouping_ind]]

        if 'motor-related' in parcel_group_names:
            assert 'not_motor-related' not in parcel_group_names
            # since parcel_types can be wihout side
            bad_parcels = []
            for pcl1 in all_parcels:
                for pcl2 in gp.areas_list_aal_my_guess:
                    if pcl1.find(pcl2) < 0:
                        bad_parcels += [pcl1]
        if 'not_motor-related' in parcel_group_names:
            # since parcel_types can be wihout side
            bad_parcels = []
            for pcl1 in all_parcels:
                for pcl2 in gp.areas_list_aal_my_guess:
                    if pcl1.find(pcl2) >= 0:
                        bad_parcels += [pcl1]

        # since parcel_types can be wihout side
        bad_parcels = []
        for pcl1 in all_parcels:
            for pcl2 in parcel_types:
                if pcl1.find(pcl2) < 0:
                    bad_parcels += [pcl1]
        #desired_parcels_inds
        #bad_parcel_inds = set( range(len(all_parcels) ) ) -

        temp = set(bad_parcels)
        assert len(temp) > 0
        print('Parcels to remove ',temp)
        bad_parcel_inds = [i for i, p in enumerate(all_parcels) if p in temp]
        assert len(bad_parcel_inds) > 0, bad_parcels

        regexes = []
        for bpi in bad_parcel_inds:
            regex_parcel_cur = '.*src._{}_{}_.*'.format(src_grouping, bpi)
            regexes += [regex_parcel_cur]
        inds_bad_parcels = utsne.selFeatsRegexInds(feature_names_all,regexes)
        bad_inds.update(inds_bad_parcels)

        #print('test exit', len(inds_bad_parcels) ); sys.exit(1) ;
    if remove_corr_self_couplings:
        regex_same_LFP = r'.?.?corr.*(LFP.[0-9]+),.*\1.*'
        regex_same_src = r'.?.?corr.*(msrc._[0-9]+_[0-9]+_c[0-9]+),.*\1.*'
        regexs = [regex_same_LFP, regex_same_src]
        inds_self_coupling = utsne.selFeatsRegexInds(feature_names_all,regexs)
        bad_inds.update(inds_self_coupling )

    if remove_crossLFP:
        # we want to keep same LFP but remove cross
        regex_same_LFP = r'.*(LFP.[0-9]+),.*\1.*'
        inds_same_LFP = utsne.selFeatsRegexInds(feature_names_all,[regex_same_LFP])

        regex_biLFP = r'.*(LFP.[0-9]+),.*(LFP.[0-9]+).*'
        inds_biLFP = utsne.selFeatsRegexInds(feature_names_all,[regex_biLFP])

        inds_notsame_LFP = set(inds_biLFP) - set(inds_same_LFP)
        if len(inds_notsame_LFP):
            print('Removing cross LFPs {}'.format( inds_notsame_LFP) )
            #print( np.array(feature_names_all)[list(inds_notsame_LFP)] )
            bad_inds.update(inds_notsame_LFP  ) #same LFP are fine, it is just power

    if cross_couplings_only:
        regex_same_LFP = r'.*(LFP.[0-9]+),.*\1.*'
        regex_same_src = r'.*(msrc._[0-9]+_[0-9]+_c[0-9]+),.*\1.*'
        regex_same_Hs = [ r'H_act.*', r'H_mob.*', r'H_compl.*'   ]
        regexs = [regex_same_LFP, regex_same_src] + regex_same_Hs
        inds_self_coupling = utsne.selFeatsRegexInds(feature_names_all,regexs)

        if len(inds_self_coupling):
            #print('Removing self-couplings of LFP and msrc {}'.format( inds_self_coupling) )
            bad_inds.update(inds_self_coupling )

    if len(fbands_to_use) < len(fband_names_fine_inc_HFO):
        fbnames_bad = set(fbands_def) - set(fbands_to_use)
        print('Removing bands ',fbnames_bad)
        regexs = []
        for fbname in fbnames_bad:
            regexs += [ '.*{}.*'.format(fbname)  ]
        inds_bad_fbands = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_bad_fbands)
    # here 'bad' means nothing essentially, just something I want to remove
    if set(feat_types_all) != set(features_to_use):
        badfeats = set(feat_types_all) - set(features_to_use)
        print('Removing features ',badfeats)
        regexs = [ '{}.*'.format(feat_name) for feat_name in  badfeats]
        inds_badfeats = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_badfeats)

    if set(data_modalities_all) != set(data_modalities):
        badmod = list( set(data_modalities_all) - set(data_modalities) )
        print('Removing modalities ',badmod)
        assert len(badmod) == 1
        badmod = badmod[0]
        regexs = [ '.*{}.*'.format(badmod) ]
        inds_badmod = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_badmod)

    if use_main_LFP_chan:
        # take mainLFPchan, extract <side><number>,  select names where there
        # is LFP with other <side><number>
        # similar with msrc_inds

        chnames_bad_LFP = set(chnames_LFP) - set([mainLFPchan] )

        regexs = [ '.*{}.*'.format(chname) for chname in  chnames_bad_LFP]
        inds_bad_LFP = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)

        regexs = [ '.*{}.*'.format(mainLFPchan) ]
        inds_mainLFPchan_rel = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        # TODO: if I reverse side forcefully, it should be done differently
        mainLFPchan_sidelet = mainLFPchan[3]
        assert mainLFPchan_sidelet in ['R', 'L']
        for ind in inds_mainLFPchan_rel:
            s = feature_names_all[ind].replace(mainLFPchan,
                                               mainLFPchan_new_name_templ.format(mainLFPchan_sidelet) )
            feature_names_all[ind] = s

        print('Removing non-main LFPs ',chnames_bad_LFP)
        bad_inds.update(inds_bad_LFP)

    # collecting indices of all msrc that we have used
    regex = 'msrc._([0-9]+)'
    res = []
    for fn in feature_names_all:
        r = re.findall(regex,fn)
        res += r
        #print(r)
    #     if r is not None:
    #         print(fn, r.groups() )
    res = list( map(int,res) )
    res = np.unique(res)

    msrc_inds_undesired = np.setdiff1d( msrc_inds, res)
    if len(msrc_inds_undesired):
        #chnames_bad_src =

        regexs = [ '.*msrc.{}.*'.format(ind) for ind in msrc_inds_undesired]
        inds_bad_src = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_bad_src)

    # removing HFO-related feats if needed
    if not use_lfp_HFO:
        regexs = [ '.*HFO.*' ]
        inds_HFO = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update( inds_HFO )

    # keeping good features
    print('Removing {} features out of {}'.format( len(bad_inds) , len(feature_names_all) ) )
    selected_feat_inds = set( range(len(feature_names_all) ) ) - bad_inds
    selected_feat_inds = list( sorted(selected_feat_inds) )

    good_feats = feature_names_all[ selected_feat_inds]
    if len(good_feats) == 0:
        print('!!!!!!!!!!!!--------  We got zero features! Exiting')
        sys.exit(5)
    X = X_allfeats[:, selected_feat_inds]

    X_pri += [ X ]
    feature_names_pri += [ good_feats ]
    selected_feat_inds_pri += [selected_feat_inds]


    mts_letters_pri += [mts_letter]

# check all feature names are the same
assert np.all ( [ (set(featns) == set(good_feats) ) for featns in feature_names_pri ] )
featnames = feature_names_pri[0]  # already filtered

##################################################

#main_side_let = 'L'
int_type_pri = ['notrem_{}'.format(main_side[0].upper() ) for main_side in main_side_pri  ]

if rescale_feats:
    print('Rescaling features')
    # we need to give movement annotations that contain notrem_ of this main side
    X_pri = upre.rescaleFeats(rawnames, X_pri, featnames, wbd_pri,
                    sfreq, rawtimes_pri, int_type_pri = int_type_pri,
                    main_side_pri = None,
                    minlen_bins = 5 * sfreq // skip, combine_within='no')



if len(set(main_side_pri)  ) > 1:
    print( set(main_side_pri) )
    raise ValueError('STOP!  we have datasets with different main sides here! Remapping is needed!')

#if len(set(mts_letters_pri)  ) > 1:
#    print( set(mts_letters_pri) )
#    raise ValueError('STOP!  we have datasets with different main tremor sides here! Remapping is needed!')

Xconcat = np.concatenate(X_pri,axis=0)

if do_outliers_discard:
    out_bininds, qvmult, discard_ratio = \
        utsne.findOutlierLimitDiscard(Xconcat,discard=discard,qshift=1e-2)
    good_inds = np.setdiff1d( np.arange(Xconcat.shape[0] ), out_bininds)
else:
    good_inds = np.arange(Xconcat.shape[0])    #everything

nbins_total =  sum( [ len(times) for times in rawtimes_pri ] )
# merge wbds
cur_zeroth_bin = 0
wbds = []
for dati in range(len(X_pri) ):
    wbd = wbd_pri [dati]
    times = rawtimes_pri[dati]
    wbds += [wbd + cur_zeroth_bin]
    #cur_zeroth_bin += len(times)  NO! because we concat only windows (thus unused end of the window should be lost)
    cur_zeroth_bin += wbd[1,-1] + skip
wbd_merged = np.hstack(wbds)
d = np.diff(wbd_merged, axis=1)
assert np.min(d)  > 0, d
# collect movement annotations first

side_switch_happened_pri = [ fi.get('side_switched',False) for fi in feat_info_pri ]
anns, anns_pri, times, dataset_bounds = utsne.concatAnns(rawnames,
                                                          rawtimes_pri, crop=(crop_start,crop_end),
                                                          side_rev_pri = side_switch_happened_pri,
                                                         wbd_pri = wbd_pri, sfreq=sfreq)
ivalis = utils.ann2ivalDict(anns)
ivalis_tb_indarrays_merged = \
    utils.getWindowIndicesFromIntervals(wbd_merged,ivalis,
                                        sfreq,ret_type='bins_contig',
                                        ret_indices_type = 'window_inds',
                                        nbins_total=nbins_total )
#ivalis_tb, ivalis_tb_indarrays = utsne.getAnnBins(ivalis, Xtimes, nedgeBins, sfreq, skip, windowsz, dataset_bounds)
#ivalis_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_tb_indarrays)


# collect artifacts now annotations first
suffixes = []
if 'LFP' in data_modalities:
    suffixes +=  ['_ann_LFPartif']
if 'msrc' in data_modalities:
    suffixes += ['_ann_MEGartif']
anns_artif, anns_artif_pri, times_, dataset_bounds_ = \
    utsne.concatAnns(rawnames,rawtimes_pri, suffixes,crop=(crop_start,crop_end),
                 allow_short_intervals=True, side_rev_pri =
                 side_switch_happened_pri, wbd_pri = wbd_pri, sfreq=sfreq)
ivalis_artif = utils.ann2ivalDict(anns_artif)
ivalis_artif_tb_indarrays_merged = \
    utils.getWindowIndicesFromIntervals(wbd_merged,ivalis_artif,
                                    sfreq,ret_type='bins_contig',
                                    ret_indices_type =
                                    'window_inds', nbins_total=nbins_total )
#ivalis_artif_tb, ivalis_artif_tb_indarrays = utsne.getAnnBins(ivalis_artif, Xtimes,
#                                                            nedgeBins, sfreq, skip, windowsz, dataset_bounds)
#ivalis_artif_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)
####################

test_mode = int(rawnames[0][1:3]) > 10
Xconcat_artif_nan  = utils.setArtifNaN(Xconcat, ivalis_artif_tb_indarrays_merged, featnames,
                                       ignore_shape_warning=test_mode)
isnan = np.isnan( Xconcat_artif_nan)
if np.sum(isnan):
    artif_bininds = np.where( isnan )[0]
else:
    artif_bininds = []
bininds_noartif = np.setdiff1d( np.arange(len(Xconcat) ) , artif_bininds)
num_nans = np.sum(np.isnan(Xconcat_artif_nan), axis=0)
print('Max artifact NaN percentage is {:.4f}%'.format(100 * np.max(num_nans)/Xconcat_artif_nan.shape[0] ) )

if artifact_handling == 'impute':
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.)
    imp_mean.fit(Xconcat_artif_nan)
    Xconcat_imputed = imp_mean.transform(Xconcat_artif_nan)

if artifact_handling == 'impute':
    Xconcat_to_fit = Xconcat_imputed
    #Xconcat_all    = Xconcat_imputed
elif artifact_handling == 'discard':
    Xconcat_to_fit = Xconcat
    np.setdiff1d(good_inds, artif_bininds)
    #Xconcat_all    = Xconcat
#elif artifact_handling == 'do_nothing':
#    #Xconcat_to_fit = Xconcat
#    Xconcat_all    = Xconcat

lst =  [inds for inds in ivalis_tb_indarrays_merged.values()]
all_interval_inds = np.hstack(lst  )
unset_inds = np.setdiff1d(np.arange(len(Xconcat)), all_interval_inds)

# this is the most bruteforce way when I join all artifacts
lst2 =  [inds for inds in ivalis_artif_tb_indarrays_merged.values()]
if len(lst2):
    all_interval_artif_inds = np.hstack(lst2  )
else:
    all_interval_artif_inds = np.array([])

#sys.exit(0)
#unset_inds = np.setdiff1d(unset_inds, all_interval_artif_inds)

remove_unlabeled = 1
if remove_unlabeled:
    #do somthing
    good_inds_ = np.setdiff1d( good_inds, unset_inds)
    print('Removing {} unlabeled pts before PCA'.format(len(good_inds) - len(good_inds_) ) )
    good_inds = good_inds_
else:
    print('Warning not removing unlabeled before PCA')

print('Outliers selection result: qvmult={:.3f}, len(out_bininds)={} of {} = {:.3f}s, discard_ratio={:.3f} %'.
    format(qvmult, len(out_bininds), Xconcat_imputed.shape[0],
           len(out_bininds)/sfreq,  100 * discard_ratio ) )

if load_only:
    print('Got load_only, exiting!')
    sys.exit(0)

# good_inds  are inds of bin where I have thrown away outliers and removed
# unlabeled

print('Input PCA dimension ', (len(good_inds),Xconcat_imputed.shape[1]) )
pca = PCA(n_components=nPCA_comp)
Xconcat_good = Xconcat_to_fit[good_inds]
pca.fit(Xconcat_good )   # fit to not-outlier data
pcapts = pca.transform(Xconcat_imputed)  # transform outliers as well

print('Output PCA dimension {}, total explained variance proportion {:.4f}'.
      format( pcapts.shape[1] , np.sum(pca.explained_variance_ratio_) ) )
print('PCA First several var ratios ',pca.explained_variance_ratio_[:5])

nfeats_per_comp_LDA_strongred = max(pcapts.shape[1] // 10, 5)


featnames_nice = utils.nicenFeatNames(featnames,
                                    roi_labels,srcgrouping_names_sorted)
n_splits = 4

if do_LDA:
    # don't change the order!
    #int_types_L = ['trem_L', 'notrem_L', 'hold_L', 'move_L', 'undef_L', 'holdtrem_L', 'movetrem_L']
    #int_types_R = ['trem_R', 'notrem_R', 'hold_R', 'move_R', 'undef_R', 'holdtrem_R', 'movetrem_R']
    # these are GLOBAL ids, they should be consistent across everything


    lda_output_pg = {}


    # over groupings of behavioral states
    for grouping_key in groupings_to_use:
        grouping = gp.groupings[grouping_key]

        lda_output_pit = {}
        # over behavioral states to decode (other points get thrown away)
        for int_types_key in int_types_to_use:
            if grouping_key not in gp.group_vs_int_type_allowed[int_types_key]:
                print('Skipping grouping {} for int types {}'.format(grouping,int_types_key) )
                lda_output_pit[int_types_key] = None
                continue
            int_types_to_distinguish = gp.int_types_to_include[int_types_key]

            print('------')
            print('Start classif (grp {}, its {})'.format(grouping_key, int_types_key))

            if int_types_key in gp.int_type_datset_rel:
                class_labels = np.zeros( len(Xconcat_imputed) , dtype=int )

                lens_pri = [ Xcur.shape[0] for Xcur in X_pri ]

                indst = 0
                indend = lens_pri[0]
                #if grouping_key == 'subj_medcond_task':
                #    for rawi,rn in enumerate(rawnames) :
                #        cid = int_types_to_distinguish.index(rn)
                #        class_labels[indst:indend] = cid
                #        if rawind+1 < len(lens_pri):
                #            indst += lens_pri[rawi]
                #            indend += lens_pri[rawi+1]
                #elif grouping_key == 'subj_medcond':
                #    for rawi,rn in enumerate(rawnames) :
                #        sind_str,m,t = utils.getParamsFromRawname(rn)
                #        key_cur = '{}_{}'.format(sind_str,m)
                #        cid = int_types_to_distinguish.index(key_cur)
                #        class_labels[indst:indend] = cid
                #        if rawind+1 < len(lens_pri):
                #            indst += lens_pri[rawi]
                #            indend += lens_pri[rawi+1]
                #elif grouping_key == 'subj':
                class_ids_grouped = {}
                revdict = {}
                for rawi,rn in enumerate(rawnames) :
                    sind_str,m,t = utils.getParamsFromRawname(rn)
                    if int_types_key == 'subj_medcond_task':
                        key_cur = '{}_{}_{}'.format(sind_str,m,t)
                    elif int_types_key == 'subj_medcond':
                        key_cur = '{}_{}'.format(sind_str,m)
                    elif int_types_key == 'subj':
                        key_cur = '{}'.format(sind_str)
                    else:
                        raise ValueError('wrong int_types_key {}'.format(int_types_key) )
                    cid = int_types_to_distinguish.index(key_cur) + \
                        gp.int_types_aux_cid_shift[int_types_key]

                    class_ids_grouped[ key_cur ] = cid
                    revdict[cid] = key_cur

                    class_labels[indst:indend] = cid
                    if rawi+1 < len(lens_pri):
                        indst += lens_pri[rawi]
                        indend += lens_pri[rawi+1]
                # we should have labeled everything
                assert np.sum( class_labels  ==0) == 0
                class_labels_good = class_labels[good_inds]

                rawind_to_test = 0
                sind_str,m,t = utils.getParamsFromRawname(rawnames[rawind_to_test] )
                if int_types_key == 'subj_medcond_task':
                    key_cur = '{}_{}_{}'.format(sind_str,m,t)
                elif int_types_key == 'subj_medcond':
                    key_cur = '{}_{}'.format(sind_str,m)
                elif int_types_key == 'subj':
                    key_cur = '{}'.format(sind_str)
                else:
                    raise ValueError('wrong int_types_key {}'.format(int_types_key) )
                class_to_check = key_cur

                Xconcat_good_cur = Xconcat_good
            else:
                sides_hand = [main_side[0].upper() ]
                rem_neut = 1
                class_labels, class_labels_good, revdict, class_ids_grouped = \
                    utsne.makeClassLabels(sides_hand, grouping,
                                        int_types_to_distinguish,
                                        ivalis_tb_indarrays_merged, good_inds,
                                        len(Xconcat_imputed), rem_neut )
                if rem_neut:
                    neq = class_labels_good != gp.class_id_neut
                    inds_not_neut = np.where( neq)[0]
                    Xconcat_good_cur = Xconcat_good[inds_not_neut]
                else:
                    Xconcat_good_cur = Xconcat_good

                #this is a string label
                class_to_check = '{}_{}'.format(int_types_to_distinguish[0], main_side[0].upper() )
            class_ind_to_check = class_ids_grouped[class_to_check]

            counts = utsne.countClassLabels(class_labels_good, class_ids_grouped)
            print('bincounts are ',counts)
            if counts[class_to_check] < 10:
                cid = class_ids_grouped[class_to_check]
                s = '!!! WARNING: grouping_key,int_types_key: class {} (cid={}) is not present at all! skipping'.format(class_to_check,cid)
                print(s)
                continue

            n_components_LDA = len(set(class_labels_good)) - 1
            print('n_components_LDA =', n_components_LDA)
            if n_components_LDA == 0:
                lda_output_pit[int_types_key] = None
                continue

            print('  Computing MI')
            MI_per_feati = utsne.getMIs(Xconcat_good_cur,class_labels_good,class_ind_to_check)
            high_to_low_MIinds = np.argsort(MI_per_feati)[::-1]

            n_MI_to_show = 8
            for ii in high_to_low_MIinds[:n_MI_to_show]:
                print('  {} MI = {:.5f}'.format(featnames_nice[ii], MI_per_feati[ii]  ) )

            # first axis gives best separation, second does the second best job, etc
            #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
            #y = np.array([1, 1, 1, 2, 2, 2])
            #if ldas_CV[0].solver == 'svd':
            #    xbar_aver = sum(xbars_list) / len(xbars_list)
            #    #xbars = np.vstack(xbars)
            #    #xbar_aver     = np.mean(xbars_list,axis=0)
            #    X_LDA_CV = np.dot(Xconcat_imputed - xbar_aver, scalings_aver)
            #elif ldas_CV[0].solver == 'eigen':
            #    X_LDA_CV = np.dot(Xconcat_imputed, scalings_aver)
            #else:
            #    raise ValueError('Wrong alg of LDA')

            pca_derived_featinds_perthr  = []
            highest_meaningful_thri = -1
            for thri in range(len( feat_variance_q_thr )):
                m = np.max(np.abs(pca.components_),axis=0)
                q = np.quantile(m,feat_variance_q_thr[thri])
                pca_derived_featinds = np.where(m > q)[0]
                if len(pca_derived_featinds) >= 10:
                    highest_meaningful_thri = thri
                pca_derived_featinds_perthr += [ pca_derived_featinds ]

            ##################

            nfeats = Xconcat_good_cur.shape[1]
            if use_low_var_feats_for_heavy_fits and nfeats > 40:
                if highest_meaningful_thri < 0:
                    feat_subset_heavy = np.arange(nfeats )
                else:
                    print('Selecting only {}-q-variance-thresholded features for heavy fits'.format(
                        feat_variance_q_thr[highest_meaningful_thri] ) )
                    feat_subset_heavy = pca_derived_featinds_perthr[highest_meaningful_thri]
            else:
                feat_subset_heavy = np.arange(nfeats )
            X_for_heavy = Xconcat_good_cur[::subskip_fit, feat_subset_heavy]
            class_labels_for_heavy = class_labels_good[::subskip_fit]


            # for DEBUG only
            # continue
            ##################

            lda_analysis_versions = {}
            lda_version_name = 'all_present_features'
            res_all_feats = utsne.calcLDAVersions(Xconcat_good_cur, Xconcat_imputed, class_labels_good,
                                  n_components_LDA, class_ind_to_check, revdict,
                                        calcName=lda_version_name,n_splits=n_splits)
            lda_analysis_versions[lda_version_name] = res_all_feats
            gc.collect()

            if search_best_LFP and (not use_main_LFP_chan) and ('LFP' in data_modalities):
                for chn_LFP in chnames_LFP:

                    # I want to remove features related to this LFP channel and
                    # see what happens to performance
                    regexs = [ '.*{}.*'.format(chn_LFP) ]
                    inds_bad_LFP = utsne.selFeatsRegexInds(featnames, regexs, unique=1)
                    feat_inds_good = set(range(nfeats ) ) - set(inds_bad_LFP)
                    feat_inds_good = list(sorted(feat_inds_good) )

                    lda_version_name = 'all_present_features_but_{}'.format(chn_LFP)
                    res_all_feats =\
                        utsne.calcLDAVersions(Xconcat_good_cur[:,feat_inds_good],
                                          Xconcat_imputed[:,feat_inds_good],
                                          class_labels_good, n_components_LDA,
                                          class_ind_to_check, revdict,
                                          calcName=lda_version_name,n_splits=n_splits)
                    lda_analysis_versions[lda_version_name] = res_all_feats
                    gc.collect()

            # look at different feature subsets based on q threshold from PCA
            for thri in range(len( feat_variance_q_thr )):
                lda_version_name = 'best_PCA-derived_features_{}'.format( feat_variance_q_thr[thri] )
                pca_derived_featinds = pca_derived_featinds_perthr[thri]
                if len(pca_derived_featinds) == 0:
                    continue
                res_all_feats = utsne.calcLDAVersions(Xconcat_good_cur[:,pca_derived_featinds],
                                                    Xconcat_imputed[:,pca_derived_featinds],
                                                    class_labels_good,
                                    n_components_LDA, class_ind_to_check, revdict,
                                            calcName=lda_version_name,n_splits=n_splits)
                lda_analysis_versions[lda_version_name] = res_all_feats
                gc.collect()


            # Important indices only (from LDA scalings)
            r = utsne.getImporantCoordInds(
                res_all_feats['fit_to_all_data']['ldaobj'].scalings_.T,
                nfeats_show = nfeats_per_comp_LDA_strongred,
                q=0.8, printLog = 0)
            inds_important, strong_inds_pc, strongest_inds_pc  = r


            lda_version_name =  'strongest_features_LDA_opinion'
            res_strongest_LDA_feats = utsne.calcLDAVersions(Xconcat_good_cur[:,inds_important],
                                  Xconcat_imputed[:,inds_important],
                                  class_labels_good,
                                  n_components_LDA, class_ind_to_check,
                                  revdict, calcName=lda_version_name,n_splits=n_splits)
            lda_analysis_versions[lda_version_name] = res_strongest_LDA_feats
            gc.collect()

            ldaobj = LinearDiscriminantAnalysis(n_components=n_components_LDA)
            ldaobj.fit(X_for_heavy, class_labels_for_heavy)
            sortinds_LDA = np.argsort( np.max(np.abs(ldaobj.scalings_ ), axis=1) )
            perfs_LDA_featsearch = utsne.selMinFeatSet(ldaobj, X_for_heavy,
                                                       class_labels_for_heavy,
                                class_ind_to_check,sortinds_LDA,n_splits=n_splits,
                                                       verbose=2, check_CV_perf=True, nfeats_step=5,
                                                       nsteps_report=5)
            _, best_inds_LDA , _, _ =   perfs_LDA_featsearch[-1]
            best_inds_LDA = feat_subset_heavy[best_inds_LDA]
            gc.collect()


            lda_version_name =  'strongest_features_LDA_selMinFeatSet'
            res_strongest_LDA_feats = utsne.calcLDAVersions(Xconcat_good_cur[:,best_inds_LDA],
                                  Xconcat_imputed[:,best_inds_LDA],
                                  class_labels_good,
                                  n_components_LDA, class_ind_to_check,
                                  revdict, calcName=lda_version_name,n_splits=n_splits)
            lda_analysis_versions[lda_version_name] = res_strongest_LDA_feats
            gc.collect()

            ##################
            if do_XGB:
                n_jobs_XGB = None
                if force_single_core:
                    n_jobs_XGB = 1
                else:
                    n_jobs_XGB = mpr.cpu_count()-gp.n_free_cores

                from sklearn import preprocessing
                le = preprocessing.LabelEncoder()
                le.fit(class_labels_for_heavy)
                class_labels_good_for_classif = le.transform(class_labels_for_heavy)

                # TODO: XGboost in future release wants set(class labels) to be
                # continousely increasing from zero, they don't want to use
                # sklearn version.. but I will anyway
                add_clf_creopts={ 'n_jobs':n_jobs_XGB, 'use_label_encoder':False }
                model = XGBClassifier(**add_clf_creopts)
                # fit the model to get feature importances
                print('Starting XGB on X.shape ', X_for_heavy.shape)
                add_fitopts = { 'eval_metric':'logloss'}
                model.fit(X_for_heavy, class_labels_good_for_classif, **add_fitopts)
                print('--- main XGB finished')
                importance = model.feature_importances_
                sortinds = np.argsort( importance )
                gc.collect()

                perfs_XGB = utsne.selMinFeatSet(model,X_for_heavy,class_labels_good_for_classif,
                                    list(le.classes_).index(class_ind_to_check),sortinds,n_splits=n_splits,
                                                add_fitopts=add_fitopts, add_clf_creopts=add_clf_creopts,
                                                check_CV_perf=True, nfeats_step=max(5, X_for_heavy.shape[1] // 20),
                                                verbose=2, max_nfeats = X_for_heavy.shape[1] // 2)
                gc.collect()

                perf_inds_to_print = [0,1,2,-1]
                for perf_ind in perf_inds_to_print:
                    if perf_ind >= len(perfs_XGB):
                        continue
                    _, best_inds_XGB , perf_nocv, res_aver =   perfs_XGB[perf_ind]
                    print('XGB CV perf on {} feats : sens {:.2f} spec {:.2f} F1 {:.2f}'.format(
                        len(best_inds_XGB), res_aver[0], res_aver[1], res_aver[2] ) )
                _, best_inds_XGB , perf_nocv, res_aver =   perfs_XGB[-1]
                best_inds_XGB = feat_subset_heavy[best_inds_XGB]

                best_nice = list( np.array(featnames_nice) [best_inds_XGB] )
                print('XGB best feats ',best_nice )

                pca_XGBfeats = PCA(n_components=nPCA_comp )
                pca_XGBfeats.fit( Xconcat_good_cur[:,best_inds_XGB])
                print('Min number of features found by XGB is {}, PCA on them gives {}'.
                        format( len(best_inds_XGB), pca_XGBfeats.n_components_) )
            else:
                model = None
                best_inds_XGB = None
                perfs_XGB = None
                pca_XGBfeats = None

            ##################

            #lda_red = LinearDiscriminantAnalysis(n_components=n_components_LDA )
            #lda_red.fit(ldapts_red, class_labels_good)
            #sens_red,spec_red,F1_red = utsne.getLDApredPower(lda_red,Xconcat[:,inds_important],
            #                                        class_labels, class_ind_to_check, printLog= 0)

            #print( ('--!! LDA on raw training data (grp {}, its {}) all vs {}:' +\
            #        '\n      sens={:.3f}; spec={:.3f};; sens_red={:.3f}; spec_red={:.3f}').
            #    format(grouping_key, int_types_key, class_to_check,sens,spec, sens_red,spec_red))


            if do_XGB:
                lda_version_name =  'strongest_features_XGB_opinion'
                res_strongest_XGB_feats = utsne.calcLDAVersions(Xconcat_good_cur[:,best_inds_XGB],
                                    Xconcat_imputed[:,best_inds_XGB],
                                    class_labels_good,
                                    n_components_LDA, class_ind_to_check,
                                    revdict, calcName=lda_version_name,n_splits=n_splits)
                lda_analysis_versions[lda_version_name] = res_all_feats
                gc.collect()

            #---------- again, not much reduction
            #ldapts_red2 = Xconcat_good_cur[:,best_inds_XGB]
            #lda_red2 = LinearDiscriminantAnalysis(n_components=n_components_LDA )
            #lda_red2.fit(ldapts_red2, class_labels_good)
            #sens_red2,spec_red2,F1_red2 = utsne.getLDApredPower(lda_red2,Xconcat[:,best_inds_XGB],
            #                                        class_labels, class_ind_to_check, printLog= 0)

            #print( ('--!! LDA on raw training data (grp {}, its {}) all vs {}:' +\
            #        '\n      sens={:.3f}; spec={:.3f};; sens_red_XGB={:.3f}; spec_red_XGB={:.3f}').
            #    format(grouping_key, int_types_key, class_to_check,sens,spec, sens_red2,spec_red2))

            # in strong_inds_pc, the last one in each array is the strongest
            # LDA on XGB-selected feats
            if do_XGB:
                perf_red_XGB = lda_analysis_versions['strongest_features_XGB_opinion']['fit_to_all_data']['perfs']
            else:
                perf_red_XGB = None
            results_cur = { 'lda_analysis_versions':  lda_analysis_versions,
                            'ldaobj':lda_analysis_versions['all_present_features']['fit_to_all_data']['ldaobj'],
                           'ldaobj_avCV':lda_analysis_versions['all_present_features']['CV_aver']['ldaobj'],
                            'ldaobjs_CV':lda_analysis_versions['all_present_features']['CV']['ldaobjs'],
                            'transformed_imputed':lda_analysis_versions['all_present_features']['fit_to_all_data']['X_transformed'],
                            'transformed_imputed_CV':lda_analysis_versions['all_present_features']['CV_aver']['X_transformed'],
                            'perf':lda_analysis_versions['all_present_features']['fit_to_all_data']['perfs'],
                            'perf_red':lda_analysis_versions['strongest_features_LDA_opinion']['fit_to_all_data']['perfs'],
                            'perf_red_XGB':perf_red_XGB,
                        'labels_good':class_labels_good,
                            'class_labels':class_labels,
                           'highest_meaningful_thri':highest_meaningful_thri,
                           'pca_derived_featinds_perthr':pca_derived_featinds_perthr,
                           'feat_variance_q_thr':feat_variance_q_thr,
                           'feat_subset_heavy':feat_subset_heavy,
                        'inds_important':inds_important,
                        'strong_inds_pc':strong_inds_pc,
                        'strongest_inds_pc':strongest_inds_pc,
                            'XGBobj':model,
                            'strong_inds_XGB':best_inds_XGB,
                            'perfs_XGB': perfs_XGB,
                            'pca_xgafeats': pca_XGBfeats,
                           'MI_per_feati':MI_per_feati }

            out_name_templ = '_{}_grp{}-{}_{}PCA_nr{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}__({},{})'
            out_name = (out_name_templ ).\
                format(sources_type, src_file_grouping_ind, src_grouping,
                    prefix, len(rawnames),
                    n_channels, Xconcat_imputed.shape[1],
                    pcapts.shape[1], skip, windowsz,grouping_key,int_types_key)
            if use_main_LFP_chan:
                out_name += '_mainLFP'
            fname_PCA_full_intermed = os.path.join( data_dir, '_{}{}{}.npz'.format(len(rawnames),rawnames[0],out_name))
            if save_output:
                print('Saving intermediate result to {}'.format(fname_PCA_full_intermed) )
                np.savez(fname_PCA_full_intermed, results_cur=results_cur)
            else:
                print('Skipping saving intermediate result')

            lda_output_pit[int_types_key] = results_cur
        lda_output_pg[grouping_key] = lda_output_pit
else:
    lda_output_pg = None

    #lda = None
    #class_labels_good = None
    #X_LDA = None
    #sens = np.nan
    #spec = np.nan



###### Save result
lens_pri = [ Xcur.shape[0] for Xcur in X_pri ]

indst = 0
indend = lens_pri[0]
# save PCA output separately
for rawind,rawname_ in enumerate(rawnames):
    # note that we use number of features that we actually used, not that we
    # read

    str_feats = ','.join(features_to_use)
    str_mods = ','.join(data_modalities)
    #use_lfp_HFO
    #use_main_LFP_chan

    # I don't include rawname in template because I want to use it for PDF name
    # as well
    out_name_templ = '_{}_grp{}-{}_{}PCA_nr{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}'
    out_name = (out_name_templ ).\
        format(sources_type, src_file_grouping_ind, src_grouping,
               prefix, len(rawnames),
               n_channels, Xconcat_imputed.shape[1],
               pcapts.shape[1], skip, windowsz)
    if use_main_LFP_chan:
        out_name += '_mainLFP'
    fname_PCA_full = os.path.join( data_dir, '{}{}.npz'.format(rawname_,out_name))

    info = {}
    info['n_channels_featfile'] = 7
    info['features_to_use'] = features_to_use
    info['data_modalities'] = data_modalities
    info['msrc_inds' ]  = msrc_inds
    info['use_main_LFP_chan'] = use_main_LFP_chan
    info['use_lfp_HFO'] = use_lfp_HFO
    info['nPCA_comp'] = nPCA_comp
    info['feat_fnames'] = feat_fnames
    info['selected_feat_inds_pri'] = selected_feat_inds_pri
    info['good_feat_inds_pri'] = selected_feat_inds_pri  # for compatibility only
    info['out_bininds'] = out_bininds
    info['cross_couplings_only'] = cross_couplings_only
    info['qvmult'] = qvmult
    info['discard_ratio'] = discard_ratio
    info['prefix'] = prefix
    info['use_low_var_feats_for_heavy_fits'] = use_low_var_feats_for_heavy_fits
    info['rescale_feats'] = rescale_feats
    info['do_XGB'] = do_XGB
    info['sources_type'] = sources_type
    info['LFP_related_only'] = LFP_related_only
    info['fbands_to_use'] = LFP_related_only
    info['remove_crossLFP'] = remove_crossLFP
    info['features_to_use'] = features_to_use
    info['data_modalities'] = data_modalities
    info['int_types_to_use'] = int_types_to_use
    info['groupings_to_use'] = groupings_to_use
    info['src_grouping'] = src_grouping
    info['src_grouping_fn'] = src_file_grouping_ind
    # I'd prefer to save both the entire list and the feat fname most related
    # to the current output
    info['fname_feat_full'] = fname_feat_full
    info['fname_feat_full_pri'] = fname_feat_full_pri


    lda_output_pg_cur = copy.deepcopy(lda_output_pg)
    sl = slice(indst,indend)
    print(sl, sl.stop - sl.start)
    assert (sl.stop-sl.start) == len(Xtimes_pri[rawind])
    assert (sl.stop-sl.start) == len(X_pri[rawind])

    # we need to restrict the LDA output to the right index range
    #for grouping_key in groupings_to_use:
    for grouping_key in lda_output_pg_cur:
        grouping = gp.groupings[grouping_key]

        lda_output_pit = {}
        #for int_types_key in int_types_to_use:
        for int_types_key in lda_output_pg_cur[grouping_key]:
            r = lda_output_pg_cur[grouping_key][int_types_key]
            if r is not None:
                r['transformed_imputed'] = r['transformed_imputed'][sl]
                r['transformed_imputed_CV'] = r['transformed_imputed_CV'][sl]

                lda_analysis_vers =  r['lda_analysis_versions']
                for ver in lda_analysis_vers:
                    curver = lda_analysis_vers[ver]
                    trkey = 'X_transformed'
                    if trkey  in curver:
                        curver[trkey] = curver[trkey][sl]

    if save_output:
        #before I had 'good_inds' name in the file for what now I call 'selected_feat_inds'
        # here I am atually saving not all features names, but only the
        # filtered ones
        np.savez(fname_PCA_full, pcapts = pcapts[sl], pcaobj=pca,
                X=X_pri[rawind], wbd=wbd_pri[rawind],
                feature_names_all = feature_names_pri[rawind] , selected_feat_inds = selected_feat_inds_pri[rawind],
                info = info, feat_info = feat_info_pri[rawind],
                lda_output_pg = lda_output_pg_cur, Xtimes=Xtimes_pri[rawind], argv=sys.argv,
                Xconcat_imputed=Xconcat_imputed ,  rawtimes=rawtimes_pri[rawind] )
        print('Saving PCA to ',fname_PCA_full)
    else:
        print('Skipng saving')

    if rawind+1 < len(lens_pri):
        indst += lens_pri[rawind]
        indend += lens_pri[rawind+1]



######################## Plotting
if show_plots:
    from matplotlib.backends.backend_pdf import PdfPages
    mpl.use('Agg')

    print('Starting to plot')

    use_main_tremorside = 1 # for now, temporarily
    bands_only = 'fine' # until I code merging



    #str_feats = ','.join(features_to_use)
    #str_mods = ','.join(data_modalities)

    out_name_plot = rn_str + out_name + \
        'mainLFP{}_HFO{}_{}_{}'.\
        format(int(use_main_LFP_chan), int(use_lfp_HFO), str_mods, str_feats)
    #a = out_name_templ.\
    #    format(rn_str,n_channels, Xconcat.shape[1], pcapts.shape[1], skip, windowsz)
    pdf= PdfPages(os.path.join(gv.dir_fig,out_name_plot + '.pdf' ))
    #pdf= PdfPages(   )


    ################  Prep colors
    mrk = ['<','>','o','^','v']
    mrknames = ['_pres','_posts','','_pree','_poste']

    tremcolor = 'r'
    notremcolor = 'g'
    movecolor = 'b'  #c,y
    holdcolor = 'purple'  #c,y
    neutcolor = 'grey'

    color_per_int_type = { 'trem':tremcolor, 'notrem':notremcolor, 'neut':neutcolor,
                        'move':movecolor, 'hold':holdcolor }

    colors,markers =utsne.prepColorsMarkers(main_side[0].upper(), anns, Xtimes,
            nedgeBins, windowsz, sfreq, skip, mrk,mrknames, color_per_int_type )

    utsne.plotPCA(pcapts,pca, nPCAcomponents_to_plot,feature_names_all, colors, markers,
                mrk, mrknames, color_per_int_type, tasks,
                pdf=pdf,neutcolor=neutcolor)

    if do_LDA:

        for int_types_key in int_types_to_use:
            for grouping_key in groupings_to_use:
                grouping = gp.groupings[grouping_key]
                r = lda_output_pg[grouping_key][int_types_key]

                if r is None:
                    continue
                X_LDA = r['transformed_imputed_CV']
                lda = r['ldaobj_avCV']

                s = map( lambda x : '{:.1f}%'.format(x*100), list(r['perf'] ) + list(r['perf_red'] ) )
                s  = list(s)
                s = '{},{}; red {},{}'.format(s[0],s[1],s[2],s[3])

                utsne.plotPCA(X_LDA,lda, 999,feature_names_all, colors, markers,
                            mrk, mrknames, color_per_int_type, tasks,
                            pdf=pdf,neutcolor=neutcolor, nfeats_show=nfeats_per_comp_LDA,
                              title_suffix = '_(grp {}, its {})_{}'.
                              format(grouping_key, int_types_key, s) )

    pdf.close()

gc.collect()
