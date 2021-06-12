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

from os.path import join as pjoin

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

# all possible
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
#artif_handling_before_fit = 'impute' #or 'discard' or 'do_nothing'
artif_handling_before_fit = 'discard'
do_outliers_discard = 1

def_sources_type = 'HirschPt2011'
sources_type = def_sources_type

load_only = 0   # load and preproc to be precise
do_LDA = 1
n_feats = 609  # this actually depends on the dataset which may have some channels bad :(
do_XGB = 1
calc_MI = 1

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

discard_remaining_int_types_during_fit = 1
skip_XGB_aux_intervals = 1
max_XGB_step_nfeats=30

cross_couplings_only=0
mainLFPchan_new_name_templ = 'LFP{}007'    # to make datasets consistent
remove_corr_self_couplings = 1
self_couplings_only =0

plot_types_all = ['pcapoints', 'ldapoints', 'feat_stats' ]
plot_types = plot_types_all

n_splits = 4
input_subdir = ""
output_subdir = ""
scale_feat_combine_type = 'medcond'

allow_CUDA = True
XGB_tree_method = 'hist'  # or 'exact' or 'gpu_hist'

use_smoothened = 0

#groupings_to_use = gp.groupings
groupings_to_use = [ 'merge_all_not_trem', 'merge_movements', 'merge_nothing' ]
groupings_to_use = [ 'merge_nothing' ]
#int_types_to_use = gp.int_types_to_include
int_types_to_use = [ 'basic', 'trem_vs_quiet' ]
featsel_methods = []
featsel_methods_all_possible = ['interpret_EBM', 'XGB_Shapley', 'SHAP_XGB' ]

params_read = {}
params_cmd = {}


helpstr = 'Usage example\nrun_ML.py --rawnames <rawname_naked1,rawnames_naked2> '
opts, args = getopt.getopt(effargv,"hr:n:s:w:p:",
        ["rawnames=", "n_channels=",  "windowsz=", "pcexpl=",
         "show_plots=","discard=", 'feat_types=', 'use_HFO=', 'mods=',
         'prefix=', 'load_only=', 'fbands=', 'n_feats=', 'single_core=',
         'sources_type=', 'bands_type=', 'crop=', 'parcel_types=',
         "src_grouping=", "src_grouping_fn=", 'groupings_to_use=',
         'int_types_to_use=', 'skip_XGB=', 'LFP_related_only=',
         'parcel_group_names=', "subskip_fit=", "search_best_LFP=",
         "save_output=", 'rescale_feats=', "cross_couplings_only=", "LFPchan=",
         "heavy_fit_red_featset=", "n_splits=", "input_subdir=",
         "output_subdir=", "artif_handling=", "plot_types=",
         "skip_XGB_aux_int=", "max_XGB_step_nfeats=", "self_couplings_only=",
         "param_file=", "scale_feat_combine_type=", "use_smoothened=",
         "featsel_methods=", "allow_CUDA=", "XGB_tree_method=", "calc_MI="])
print(sys.argv)
print('Argv str = ',' '.join(sys.argv ) )
print(opts)
print(args)
for opt, arg in opts:
    print(opt,arg)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == "--param_file":
        param_fname_full = pjoin(gv.param_dir,arg)
        params_read = gv.paramFileRead(param_fname_full)
    else:
        if opt.startswith('--'):
            optkey = opt[2:]
        elif opt == '-r':
            optkey = 'rawnames'
        else:
            raise ValueError('wrong opt {}'.format(opt) )
        params_cmd[optkey] = arg

params_read.update(params_cmd)
pars = params_read

for opt,arg in pars.items():
    if opt == "n_channels":
        n_channels = int(arg)
    elif opt == "n_feats":
        n_feats = int(arg)
        n_feats_set_explicitly = True
    elif opt == "scale_feat_combine_type":
        scale_feat_combine_type = arg
    elif opt == "allow_CUDA":
        allow_CUDA = int(arg)
    elif opt == "XGB_tree_method":
        XGB_tree_method = arg
    elif opt == "skip_XGB":
        do_XGB = not bool(int(arg))
    elif opt == "calc_MI":
        calc_MI = int(arg)
    elif opt == "skip_XGB_aux_int":
        skip_XGB_aux_intervals = bool(int(arg))
    elif opt == "self_couplings_only":
        self_couplings_only = int(arg)
    elif opt == "heavy_fit_red_featset":
        use_low_var_feats_for_heavy_fits = int(arg)
    elif opt == "parcel_types":  # names of the roi_labels (not groupings)
        parcel_types = arg.split(',')
    elif opt == "plot_types":  # names of the roi_labels (not groupings)
        plot_types = arg.split(',')
        for pt in plot_types:
            assert pt in plot_types_all
    elif opt == "input_subdir":
        input_subdir = arg
        if len(input_subdir) > 0:
            subdir = pjoin(gv.data_dir,input_subdir)
            assert os.path.exists(subdir ), subdir
    elif opt == "output_subdir":
        output_subdir = arg
        if len(output_subdir) > 0:
            subdir = pjoin(gv.data_dir,output_subdir)
            if not os.path.exists(subdir ):
                print('Creating output subdir {}'.format(subdir) )
                os.makedirs(subdir)
    elif opt == "featsel_methods":
        featsel_methods = arg.split(',')
        for fsh in featsel_methods:
            assert fsh in featsel_methods_all_possible
    elif opt == "parcel_group_names":
        parcel_group_names = arg.split(',')
    elif opt == "artif_handling":
        assert arg in ['impute' , 'discard']
        artif_handling_before_fit = arg
    elif opt == "crop":
        cr =  arg.split(',')
        crop_start = float(cr[0])
        crop_end = float(cr[1] )
    elif opt == 'int_types_to_use':
        int_types_to_use = arg.split(',')
    elif opt == 'groupings_to_use':
        groupings_to_use = arg.split(',')
    elif opt == 'n_splits': # number of splits to be used in cross-validation
        n_splits = int(arg)
    elif opt == "search_best_LFP":
        search_best_LFP = int(arg)
    elif opt == "src_grouping":
        src_grouping = int(arg)
    elif opt == "use_smoothened":
        use_smoothened = int(arg)
    elif opt == "src_grouping_fn":
        src_file_grouping_ind = int(arg)
    elif opt == "max_XGB_step_nfeats":
        max_XGB_step_nfeats = int(arg)
        assert max_XGB_step_nfeats > 0
    elif opt == "rescale_feats":
        rescale_feats = int(arg)
    elif opt == "cross_couplings_only":
        cross_couplings_only = int(arg)
    elif opt == "save_output":
        save_output = int(arg)
    elif opt == "bands_type":
        bands_type = arg
    elif opt == "sources_type":
        if len(arg):
            sources_type = arg
    elif opt == "single_core":
        force_single_core = int(arg)
    elif opt == 'load_only':
        load_only = int(arg)
    elif opt == 'show_plots':
        show_plots = int(arg)
    elif opt == 'subskip_fit':
        subskip_fit = int(arg)
    elif opt == "windowsz":
        windowsz = int(arg)
    elif opt == "LFP_related_only":
        LFP_related_only = int(arg)
    elif opt == "discard":
        discard = float(arg)
    elif opt == "prefix":
        prefix = arg + '_'
    elif opt == "msrc_inds":
        msrc_inds = [ int(indstr) for indstr in arg.split(',') ]
        msrc_inds = np.array(msrc_inds, dtype=int)
    elif opt == "feat_types":
        features_to_use = arg.split(',')
        for ftu in features_to_use:
            assert ftu in feat_types_all, ftu
    elif opt == "mods":
        data_modalities = arg.split(',')   #lfp of msrc
    elif opt == 'fbands':
        fbands_to_use = arg.split(',')
    elif opt == 'LFPchan':
        if arg == 'main': # use only main channel
            use_main_LFP_chan = 1
        elif arg == 'all':   # use all channels
            use_main_LFP_chan = 0
        elif arg.find('LFP') >= 0:   # maybe I want to specify expliclity channel name
            raise ValueError('to be implemented')
    elif opt == 'use_HFO':
        use_lfp_HFO = int(arg)
    elif opt == "pcexpl":
        nPCA_comp = float(arg)  #crude of fine
        if nPCA_comp - int(nPCA_comp) < 1e-6:
            nPCA_comp = int(nPCA_comp)
    elif opt == 'rawnames':
        rawnames = arg.split(',')
    elif opt.startswith('iniAdd'):
        print('skip ',opt)
    else:
        print('Unrecognized option {} with arg {}, exiting'.format(opt,arg) )
        sys.exit('Unrecognized option')


single_fit_type_mode = 0
if len(groupings_to_use) == 1 and len(int_types_to_use) == 1:
    single_fit_type_mode = 1
    grouping_key = groupings_to_use[0]
    int_types_key = int_types_to_use[0]

    grouping = gp.groupings[grouping_key ]
    if grouping_key not in gp.group_vs_int_type_allowed[int_types_key]:
        print('--!!--!!-- runPCA: Exiting without output because grouping {} for int types {}'.
              format(grouping,int_types_key) )
        #sys.exit("wrong grouping vs int_set")
        sys.exit(None)

#print('test exit', parcel_types); sys.exit(1) ;
n_jobs_XGB = None
if force_single_core:
    n_jobs_XGB = 1
else:
    n_jobs_XGB = max(1, mpr.cpu_count()-gp.n_free_cores)
assert n_jobs_XGB > 0

if bands_type == 'fine':
    fbands_def = fband_names_fine_inc_HFO
else:
    fbands_def = fband_names_crude_inc_HFO


allow_CUDA_MNE = mne.utils.get_config('MNE_USE_CUDA')
print('nPCA_comp = ',nPCA_comp)

print(f'''do_XGB={do_XGB}, XGB_tree_method={XGB_tree_method},
          allow_CUDA={allow_CUDA}, allow_CUDA_MNE={allow_CUDA_MNE},
          gpus found={gv.GPUs_list}''')

############################
rn_str = ','.join(rawnames)

##############################
test_mode = ( int(rawnames[0][1:3]) > 10 ) or (prefix == 'test_')
if test_mode:
    print( '!!!!!!!!!!  test_mode   !!!!!!!!!!!!!' )
    print( '!!!!!!!!!!  test_mode   !!!!!!!!!!!!!' )
    print( '!!!!!!!!!!  test_mode   !!!!!!!!!!!!!' )
##########################

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
main_side_pri = [] # before reversal
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

    inp_sub = pjoin(gv.data_dir, input_subdir)
    if sources_type == def_sources_type:
        a = '{}_feats_{}chs_nfeats{}_skip{}_wsz{}_grp{}-{}{}.npz'.\
            format(rawname_,n_channels, n_feats, skip, windowsz,
                   src_file_grouping_ind, src_grouping, crp_str)
        feat_fnames += [a]
        fname_feat_full = pjoin( inp_sub,a)
    else:
        regex = '{}_feats_{}_{}chs_nfeats{}_skip{}_wsz{}_grp{}-{}{}.npz'.\
            format(rawname_,sources_type,'[0-9]+', '[0-9]+', skip, windowsz,
                    src_file_grouping_ind, src_grouping, crp_str)
            #format(rawname_, prefix, regex_nrPCA, regex_nfeats, regex_pcadim)

        fnfound = utsne.findByPrefix(inp_sub, rawname_, prefix, regex=regex)
        if len(fnfound) > 1:
            fnt = [0] * len(fnfound)
            for fni in range(len(fnt) ):
                fnfull = pjoin(inp_sub, fnfound[fni])
                fnt[fni] = os.path.getmtime(fnfull)
            fni_max = np.argmax(fnt)
            fnfound = [ fnfound[fni_max] ]


        assert len(fnfound) == 1, 'For {} found not single fnames {}'.format(rawname_,fnfound)
        fname_feat_full = pjoin( inp_sub, fnfound[0] )

    fname_feat_full_pri += [fname_feat_full]

    modtime = datetime.datetime.fromtimestamp(os.path.getmtime(fname_feat_full)  )
    print('Loading feats from {}, modtime {}'.format(fname_feat_full, modtime ))
    f = np.load(fname_feat_full, allow_pickle=True)
    feat_file_pri += [f]

    ################## extract stuff


    sfreq = f['sfreq']
    if use_smoothened:
        X_allfeats =  f['X_smooth']
    else:
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
    mts_letter = gv.gen_subj_info[subj]['tremor_side'][0].upper()
    nedgeBins = feat_info['nedgeBins']
    assert skip_ == skip

    # this is what is actually in features already, so it is needed only for
    # annotations. Body side is meant, not brain side
    new_main_side = feat_info.get('new_main_side','left')
    main_side_pri += [feat_info.get('main_side_before_switch','left') ]
    new_main_side_pri += [new_main_side]

    feat_info_pri += [feat_info]
    rawtimes_pri += [rawtimes]


    if Xtimes[1]-Xtimes[0] > 0.5:  # then we have saved the wrong thing
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
    mainLFPchan = gv.gen_subj_info[subj].get('lfpchan_selected_by_pipeline',None)
    if use_main_LFP_chan:
        assert mainLFPchan is not None

    ##################

    #if 'brcorr' in data_modalities or 'rbcorr' in data_modalities:
    #search for non-zero components in short msrc names
    b = []
    for fen in feature_names_all:
        r = re.match('.*msrc._[0-9]+_[0-9]+_c[^0][0-9]*', fen)
        if r is not None:
            print(fen)
            b += [fen]
    if len(b):
        raise ValueError('Non zero components found among feature names {}'.format(b) )

    ##################


    bad_inds = set([] )


    src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.\
        format(rawname_,sources_type, src_file_grouping_ind)
    src_rec_info_fn_full = pjoin(gv.data_dir, input_subdir, src_rec_info_fn + '.npz')
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

        final_src_grouping = 9
        regexes = []
        for bpi in bad_parcel_inds:
            regex_parcel_cur = '.*src._[0-9]+_{}_.*'.format( bpi)
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

    if self_couplings_only:
        regex_same_LFP = r'.*(LFP.[0-9]+),.*\1.*'
        regex_same_src = r'.*(msrc._[0-9]+_[0-9]+_c[0-9]+),.*\1.*'
        regex_same_Hs = [ r'H_act.*', r'H_mob.*', r'H_compl.*'   ]
        regexs = [regex_same_LFP, regex_same_src] + regex_same_Hs
        inds_self_coupling = utsne.selFeatsRegexInds(feature_names_all,regexs)

        if len(inds_self_coupling):
            inds_non_self_coupling = set(range(len(feature_names_all) )) - set(inds_self_coupling)
            bad_inds.update( inds_non_self_coupling )

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

    # removing HFO-related feats if needed
    if not use_lfp_HFO:
        regexs = [ '.*HFO.*' ]
        inds_HFO = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update( inds_HFO )

    ############# for HisrchPt2011 only
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
    #################### end


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
side_switch_happened_pri = [ fi.get('side_switched',False) for fi in feat_info_pri ]

if len(set(main_side_pri)  ) > 1:
    print('we have datasets with different main sides here! Remapping was done ! orig_sides {} switched {}'.
          format(main_side_pri, side_switch_happened_pri) )
    #raise ValueError('STOP!  we have datasets with different main sides here! Remapping is needed!')

new_main_side_let = new_main_side[0].upper()
#int_type_baseline_pri = ['notrem_{}'.format(main_side[0].upper() ) for main_side in main_side_pri  ]
baseline_int_type = 'notrem_{}'.format(new_main_side[0].upper() )

plot_feat_stat_scatter = 1

if show_plots:
    from matplotlib.backends.backend_pdf import PdfPages
    mpl.use('Agg')

    out_name_templ = '_{}_grp{}-{}_{}ML_nr{}_{}chs_nfeats{}_skip{}_wsz{}'
    out_name = (out_name_templ ).\
        format(sources_type, src_file_grouping_ind, src_grouping,
            prefix, len(rawnames),
            n_channels, sum([len(X) for X in X_pri] ),
            skip, windowsz)
    if use_main_LFP_chan:
        out_name += '_mainLFP'

    str_mods = ','.join(data_modalities)
    str_feats = ','.join(features_to_use)

    out_name_plot = rn_str + out_name + \
        'mainLFP{}_HFO{}_{}_{}'.\
        format(int(use_main_LFP_chan), int(use_lfp_HFO), str_mods, str_feats)
    pdf= PdfPages(pjoin(gv.dir_fig, output_subdir, out_name_plot + '.pdf' ))

if rescale_feats:
    print('Rescaling features')


    if show_plots and 'feat_stats' in plot_types:
        int_types_to_stat = [it + '_{}'.format(new_main_side_let) for it in gp.int_types_basic]
        upre.plotFeatStatsScatter(rawnames, X_pri, int_types_to_stat,
                            featnames,sfreq,
                            rawtimes_pri,side_switch_happened_pri, wbd_pri=wbd_pri,
                                    save_fig=False, separate_by = 'feat_type' )
        plt.suptitle('Feat stats before rescaling')
        pdf.savefig()
        plt.close()
        gc.collect()

    # we need to give movement annotations that contain notrem_ of this main side
    import copy
    if test_mode:
        X_pri_pre = X_pri
        X_pri = copy.deepcopy(X_pri)
    # rescaling happens in-place
    # we don't need stat file because here we load everything in memory in any
    # case
    X_pri_rescaled, indsets, means, stds = upre.rescaleFeats(rawnames, X_pri,
                    feature_names_pri, wbd_pri,
                    sfreq, rawtimes_pri, int_type = baseline_int_type,
                    main_side = None,
                    side_rev_pri = side_switch_happened_pri,
                    minlen_bins = 5 * sfreq // skip,
                    combine_within=scale_feat_combine_type)
    #X_pri = X_pri_rescaled

    if show_plots and 'feat_stats' in plot_types:
        int_types_to_stat = [it + '_{}'.format(new_main_side_let) for it in gp.int_types_basic]
        upre.plotFeatStatsScatter(rawnames, X_pri, int_types_to_stat,
                            featnames,sfreq,
                            rawtimes_pri,side_switch_happened_pri, wbd_pri=wbd_pri,
                                    save_fig=False, separate_by = 'feat_type' )
        plt.suptitle('Feat stats after rescaling')
        pdf.savefig()
        plt.close()
        gc.collect()
        if len(plot_types) == 1:
            pdf.close()


if len(set(new_main_side_pri)  ) > 1:
    ws = 'STOP!  we have datasets with different main tremor sides here! Remapping is needed!'
    print(ws, set(new_main_side_pri) )
    #raise ValueError(ws)

Xconcat = np.concatenate(X_pri,axis=0)


nbins_total =  sum( [ len(times) for times in rawtimes_pri ] )
# merge wbds
#cur_zeroth_bin = 0
#wbds = []
#for dati in range(len(X_pri) ):
#    wbd = wbd_pri [dati]
#    times = rawtimes_pri[dati]
#    #cur_zeroth_bin += len(times)  NO! because we concat only windows (thus unused end of the window should be lost)
#    wbds += [wbd + cur_zeroth_bin]
#    cur_zeroth_bin += wbd[1,-1] + skip
#wbd_merged = np.hstack(wbds)
# collect movement annotations first

anns, anns_pri, times_concat, dataset_bounds, wbd_merged = utsne.concatAnns(rawnames,
                                                          rawtimes_pri, crop=(crop_start,crop_end),
                                                          side_rev_pri = side_switch_happened_pri,
                                                         wbd_pri = wbd_pri, sfreq=sfreq, ret_wbd_merged=1)
print('times_concat end {} wbd end {}'.format(times_concat[-1] * sfreq, wbd_merged[1,-1] ) )

ivalis = utils.ann2ivalDict(anns)
ivalis_tb_indarrays_merged = \
    utils.getWindowIndicesFromIntervals(wbd_merged,ivalis,
                                        sfreq,ret_type='bins_contig',
                                        ret_indices_type = 'window_inds',
                                        nbins_total=nbins_total )
#ivalis_tb, ivalis_tb_indarrays = utsne.getAnnBins(ivalis, Xtimes, nedgeBins, sfreq, skip, windowsz, dataset_bounds)
#ivalis_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_tb_indarrays)

##############################  Artifacts ####################################

#######################   naive artifacts
bininds_concat_good = np.arange(Xconcat.shape[0])    #everything
if do_outliers_discard:
    artif_naive_bininds, qvmult, discard_ratio = \
        utsne.findOutlierLimitDiscard(Xconcat,discard=discard,qshift=1e-2)
    bininds_concat_good = np.setdiff1d( bininds_concat_good, artif_naive_bininds)

    print('Outliers selection result: qvmult={:.3f}, len(artif_naive_bininds)={} of {} = {:.3f}s, discard_ratio={:.3f} %'.
        format(qvmult, len(artif_naive_bininds), Xconcat.shape[0],
            len(artif_naive_bininds)/sfreq,  100 * discard_ratio ) )


#######################   artifacts by hads
ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt'
# collect artifacts now annotations first
suffixes = []
if 'LFP' in data_modalities:
    suffixes += [ '_ann_LFPartif' ]
if 'msrc' in data_modalities:
    suffixes += [ ann_MEGartif_prefix_to_use ]
anns_artif, anns_artif_pri, times_, dataset_bounds_ = \
    utsne.concatAnns(rawnames,rawtimes_pri, suffixes,crop=(crop_start,crop_end),
                 allow_short_intervals=True, side_rev_pri =
                 side_switch_happened_pri, wbd_pri = wbd_pri, sfreq=sfreq)
wrong_brain_side_let = new_main_side[0].upper()  # ipsilater is wrong
anns_artif = utils.removeAnnsByDescr(anns_artif, 'BAD_LFP{}'.format(wrong_brain_side_let)  )
ivalis_artif = utils.ann2ivalDict(anns_artif)
ivalis_artif_tb_indarrays_merged = \
    utils.getWindowIndicesFromIntervals(wbd_merged,ivalis_artif,
                                    sfreq,ret_type='bins_contig',
                                    ret_indices_type =
                                    'window_inds', nbins_total=nbins_total )
#ivalis_artif_tb, ivalis_artif_tb_indarrays = utsne.getAnnBins(ivalis_artif, Xtimes,
#                                                            nedgeBins, sfreq, skip, windowsz, dataset_bounds)
#ivalis_artif_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)
Xconcat_artif_nan  = utils.setArtifNaN(Xconcat, ivalis_artif_tb_indarrays_merged, featnames,
                                       ignore_shape_warning=test_mode)
isnan = np.isnan( Xconcat_artif_nan)
if np.sum(isnan):
    artif_bininds = np.where( isnan )[0]
else:
    artif_bininds = []
#bininds_noartif = np.setdiff1d( np.arange(len(Xconcat) ) , artif_bininds)
num_nans = np.sum(np.isnan(Xconcat_artif_nan), axis=0)
print('Max artifact NaN percentage is {:.4f}%'.format(100 * np.max(num_nans)/Xconcat_artif_nan.shape[0] ) )

imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.)
imp_mean.fit(Xconcat_artif_nan)
Xconcat_imputed = imp_mean.transform(Xconcat_artif_nan)
#Xconcat_to_fit is NOT about removing points, it is used TOGETHER with bininds_concat_good
if artif_handling_before_fit == 'impute':
    # replacing artifact-related NaNs with zeros
    # mean should be zero since we centered features earlier

    #we DO NOT change bininds_concat_good here
    Xconcat_to_fit = Xconcat_imputed
elif artif_handling_before_fit == 'discard':
    bininds_concat_good = np.setdiff1d(bininds_concat_good, artif_bininds)
    Xconcat_to_fit = Xconcat

    assert not np.any( np.isnan(Xconcat[bininds_concat_good] ) )
else:
    raise ValueError('wrong value of artif_handling_before_fit = {}'.format(artif_handling_before_fit) )
    #Xconcat_all    = Xconcat
#elif artif_handling_before_fit == 'do_nothing':
#    #Xconcat_to_fit = Xconcat
#    Xconcat_all    = Xconcat

# collect all marked behavioral states
lst =  [inds for inds in ivalis_tb_indarrays_merged.values()]
all_interval_inds = np.hstack(lst  )
unset_inds = np.setdiff1d(np.arange(len(Xconcat)), all_interval_inds)

# this is the most bruteforce way when I join all artifacts
#lst2 =  [inds for inds in ivalis_artif_tb_indarrays_merged.values()]
#if len(lst2):
#    all_interval_artif_inds = np.hstack(lst2  )
#else:
#    all_interval_artif_inds = np.array([])

remove_pts_unlabeled_beh_states = 1
if remove_pts_unlabeled_beh_states:
    #do somthing
    bininds_concat_good_yes_label = np.setdiff1d( bininds_concat_good, unset_inds)
    print('Removing {} unlabeled pts before PCA'.
          format(len(bininds_concat_good) - len(bininds_concat_good_yes_label) ) )
    bininds_concat_good = bininds_concat_good_yes_label
else:
    print('Warning not removing unlabeled before PCA')


featnames_nice = utils.nicenFeatNames(featnames,
                                    roi_labels,srcgrouping_names_sorted)

if load_only:
    print('Got load_only, exiting!')
    #if show_plots:
    #    pdf.close()
    sys.exit(0)

# bininds_concat_good  are inds of bin where I have thrown away outliers and removed
# unlabeled

print('Input PCA dimension ', (len(bininds_concat_good),Xconcat.shape[1]) )
pca = PCA(n_components=nPCA_comp)
Xsubset_to_fit = Xconcat_to_fit[bininds_concat_good]
#Xsubset_to_fit = Xsubset_to_fit
pca.fit(Xsubset_to_fit )   # fit to not-outlier data
pcapts = pca.transform(Xconcat_imputed)  # transform outliers as well

print('Output PCA dimension {}, total explained variance proportion {:.4f}'.
      format( pcapts.shape[1] , np.sum(pca.explained_variance_ratio_) ) )
print('PCA First several var ratios ',pca.explained_variance_ratio_[:5])

nfeats_per_comp_LDA_strongred = max(pcapts.shape[1] // 10, 5)

subjs_analyzed = upre.getRawnameListStructure(rawnames)
sind_strs = list(sorted( (subjs_analyzed.keys()) ) )
sind_join_str = ','.join(sind_strs)

############################################
lens_pri = [ Xcur.shape[0] for Xcur in X_pri ]
indst = 0
indend = lens_pri[0]
dataset_bounds_Xbins = [] #skipped bin indices
for rawind,rawname_ in enumerate(rawnames):
    sl = slice(indst,indend)
    dataset_bounds_Xbins += [ (indst,indend) ]

    if rawind+1 < len(lens_pri):
        indst += lens_pri[rawind]
        indend += lens_pri[rawind+1]

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
info['artif_naive_bininds'] = artif_naive_bininds
info['cross_couplings_only'] = cross_couplings_only
info['qvmult'] = qvmult
info['discard_ratio'] = discard_ratio
info['prefix'] = prefix
info['use_low_var_feats_for_heavy_fits'] = use_low_var_feats_for_heavy_fits
info['rescale_feats'] = rescale_feats
info['do_XGB'] = do_XGB
info['skip_XGB_aux_intervals'] =  skip_XGB_aux_intervals
info['sources_type'] = sources_type
info['LFP_related_only'] = LFP_related_only
info['fbands_to_use'] = LFP_related_only
info['remove_crossLFP'] = remove_crossLFP
info['data_modalities'] = data_modalities
info['int_types_to_use'] = int_types_to_use
info['groupings_to_use'] = groupings_to_use
info['src_grouping'] = src_grouping
info['src_grouping_fn'] = src_file_grouping_ind
# I'd prefer to save both the entire list and the feat fname most related
# to the current output
info['fname_feat_full'] = fname_feat_full_pri[rawind] # don't remove!! althought before Jan 25 it is wrong :(
info['fname_feat_full_pri'] = fname_feat_full_pri
info['artif_handling'] = artif_handling_before_fit
PCA_info = info


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
            if int_types_key == 'trem_vs_quiet' and 'merge_nothing' in groupings_to_use and \
                    grouping_key == 'merge_all_not_trem':
                print("We don't want to compute the same thing twice, so skipping {},{}".
                      format(grouping_key, int_types_key) )
                continue

            int_types_to_distinguish = gp.int_types_to_include[int_types_key]

            print('---------------------------------------------')
            print('Start classif (grp {}, its {})'.format(grouping_key, int_types_key))

            # distinguishing datasets
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
                class_labels_good = class_labels[bininds_concat_good]

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

                Xconcat_good_cur = Xsubset_to_fit
            else:
                sides_hand = [new_main_side[0].upper() ]
                class_labels, class_labels_good, revdict, class_ids_grouped = \
                    utsne.makeClassLabels(sides_hand, grouping,
                        int_types_to_distinguish,
                        ivalis_tb_indarrays_merged, bininds_concat_good,
                        len(Xconcat_imputed),
                        rem_neut = discard_remaining_int_types_during_fit)
                if discard_remaining_int_types_during_fit:
                    # then we have to remove the data points as well
                    neq = class_labels_good != gp.class_id_neut
                    inds_not_neut = np.where( neq)[0]
                    Xconcat_good_cur = Xsubset_to_fit[inds_not_neut]
                else:
                    Xconcat_good_cur = Xsubset_to_fit

                #this is a string label
                class_to_check = '{}_{}'.format(int_types_to_distinguish[0], new_main_side[0].upper() )
            class_ind_to_check = class_ids_grouped[class_to_check]

            # check that the labels number don't have holes -- well, they do
            #assert np.max (np.abs( np.diff(sorted(set(class_labels_good) )) )) == 0

            counts = utsne.countClassLabels(class_labels_good, class_ids_grouped=None, revdict=revdict)
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

            if calc_MI:
                print('  Computing MI')
                MI_per_feati = utsne.getMIs(Xconcat_good_cur,class_labels_good,class_ind_to_check,
                                            n_jobs=n_jobs_XGB)
                high_to_low_MIinds = np.argsort(MI_per_feati)[::-1]

                n_MI_to_show = 8
                for ii in high_to_low_MIinds[:n_MI_to_show]:
                    print('  {} MI = {:.5f}'.format(featnames_nice[ii], MI_per_feati[ii]  ) )
            else:
                print('  skipping computation of MI')
                MI_per_feati = None


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
                #thri = highest_meaningful_thri
                thri = 0
                if thri < 0:
                    feat_subset_heavy = np.arange(nfeats )
                else:
                    print('Selecting only {}-q-variance-thresholded features for heavy fits'.format(
                        feat_variance_q_thr[thri] ) )
                    feat_subset_heavy = pca_derived_featinds_perthr[thri]
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

            # labels_pred = lda.predict(Xconcat_good_cur)
            # conf_mat = confusion_matrix(y_true, y_pred)

            if search_best_LFP and (not use_main_LFP_chan) and ('LFP' in data_modalities):
                for chn_LFP in chnames_LFP:

                    # I want to remove features related to this LFP channel and
                    # see what happens to performance
                    regexs = [ '.*{}.*'.format(chn_LFP) ]
                    inds_bad_LFP = utsne.selFeatsRegexInds(featnames, regexs, unique=1)
                    feat_inds_good = set(range(nfeats ) ) - set(inds_bad_LFP)
                    feat_inds_good = list(sorted(feat_inds_good) )

                    lda_version_name = 'all_present_features_but_{}'.format(chn_LFP)
                    res_cur =\
                        utsne.calcLDAVersions(Xconcat_good_cur[:,feat_inds_good],
                                          Xconcat_imputed[:,feat_inds_good],
                                          class_labels_good, n_components_LDA,
                                          class_ind_to_check, revdict,
                                          calcName=lda_version_name,n_splits=n_splits)
                    lda_analysis_versions[lda_version_name] = res_cur
                    gc.collect()

            # look at different feature subsets based on q threshold from PCA
            for thri in range(len( feat_variance_q_thr )):
                lda_version_name = 'best_PCA-derived_features_{}'.format( feat_variance_q_thr[thri] )
                pca_derived_featinds = pca_derived_featinds_perthr[thri]
                if len(pca_derived_featinds) == 0:
                    continue
                res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,pca_derived_featinds],
                                                    Xconcat_imputed[:,pca_derived_featinds],
                                                    class_labels_good,
                                    n_components_LDA, class_ind_to_check, revdict,
                                            calcName=lda_version_name,n_splits=n_splits)
                lda_analysis_versions[lda_version_name] = res_cur
                gc.collect()


            # Important indices only (from LDA scalings)
            r = utsne.getImporantCoordInds(
                res_all_feats['fit_to_all_data']['ldaobj'].scalings_.T,
                nfeats_show = nfeats_per_comp_LDA_strongred,
                q=0.8, printLog = 0)
            inds_important, strong_inds_pc, strongest_inds_pc  = r

            lda_version_name =  'strongest_features_LDA_opinion'
            res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,inds_important],
                                  Xconcat_imputed[:,inds_important],
                                  class_labels_good,
                                  n_components_LDA, class_ind_to_check,
                                  revdict, calcName=lda_version_name,n_splits=n_splits)
            lda_analysis_versions[lda_version_name] = res_cur
            gc.collect()

            #######################
            ldaobj = LinearDiscriminantAnalysis(n_components=n_components_LDA)
            ldaobj.fit(X_for_heavy, class_labels_for_heavy)
            sortinds_LDA = np.argsort( np.max(np.abs(ldaobj.scalings_ ), axis=1) )
            perfs_LDA_featsearch = utsne.selMinFeatSet(ldaobj, X_for_heavy,
                            class_labels_for_heavy, class_ind_to_check,sortinds_LDA,
                                n_splits=n_splits, verbose=2, check_CV_perf=True, nfeats_step=5,
                                                       nsteps_report=5, stop_if_boring=False)
            #_, best_inds_LDA , _, _ =   perfs_LDA_featsearch[-1]
            best_inds_LDA =   perfs_LDA_featsearch[-1]['featinds_present']
            best_inds_LDA = feat_subset_heavy[best_inds_LDA]
            gc.collect()


            lda_version_name =  'strongest_features_LDA_selMinFeatSet'
            res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,best_inds_LDA],
                                  Xconcat_imputed[:,best_inds_LDA],
                                  class_labels_good,
                                  n_components_LDA, class_ind_to_check,
                                  revdict, calcName=lda_version_name,n_splits=n_splits)
            lda_analysis_versions[lda_version_name] = res_cur
            gc.collect()

            ##################
            from sklearn import preprocessing
            lab_enc = preprocessing.LabelEncoder()
            lab_enc.fit(class_labels_for_heavy)
            class_labels_good_for_classif = lab_enc.transform(class_labels_for_heavy)

            do_XGB_cur =  do_XGB and not (int_types_key in gp.int_type_datset_rel and skip_XGB_aux_intervals  )
            if do_XGB_cur:


                # TODO: XGboost in future release wants set(class labels) to be
                # continousely increasing from zero, they don't want to use
                # sklearn version.. but I will anyway

                add_clf_creopts={ 'n_jobs':n_jobs_XGB, 'use_label_encoder':False,
                                 'importance_type': 'total_gain' }
                tree_method = XGB_tree_method
                method_params = {'tree_method': tree_method}

                if (XGB_tree_method in ['hist', 'gpu_hist']) \
                        and allow_CUDA \
                        and len(gv.GPUs_list):
                    tree_method = 'gpu_hist'

                    method_params['gpu_id'] = gv.GPUs_list[0]

                add_clf_creopts.update(method_params)
                clf_XGB = XGBClassifier(**add_clf_creopts)
                # fit the clf_XGB to get feature importances
                print('Starting XGB on X.shape ', X_for_heavy.shape)
                add_fitopts = { 'eval_metric':'logloss'}
                clf_XGB.fit(X_for_heavy, class_labels_good_for_classif, **add_fitopts)
                print('--- main XGB finished')
                importance = clf_XGB.feature_importances_
                sortinds = np.argsort( importance )
                gc.collect()

                step_XGB = min(max_XGB_step_nfeats, max(5, X_for_heavy.shape[1] // 20)  )
                perfs_XGB = utsne.selMinFeatSet(clf_XGB, X_for_heavy, class_labels_good_for_classif,
                                    list(lab_enc.classes_).index(class_ind_to_check), sortinds,
                                                n_splits=n_splits,
                                                add_fitopts=add_fitopts, add_clf_creopts=add_clf_creopts,
                                                check_CV_perf=True, nfeats_step= step_XGB,
                                                verbose=2, max_nfeats = X_for_heavy.shape[1] // 2,
                                                ret_clf_obj=True)
                gc.collect()

                perf_inds_to_print = [0,1,2,-1]
                for perf_ind in perf_inds_to_print:
                    if perf_ind >= len(perfs_XGB):
                        continue
                    smfs_output = perfs_XGB[perf_ind]
                    inds_XGB = smfs_output['featinds_present']
                    perf_nocv = smfs_output['perf_nocv']
                    res_aver = smfs_output['perf_aver']


                    print('XGB CV perf on {} feats : sens {:.2f} spec {:.2f} F1 {:.2f}'.format(
                        len(inds_XGB), res_aver[0], res_aver[1], res_aver[2] ) )

                    shfl = smfs_output.get('fold_type_shuffled',None)
                    if shfl is not None:
                        _,_,perf_shuffled = shfl
                        sens_sh,sepc_sh,F1_sh,confmat_sh = perf_shuffled
                        print('  shuffled: XGB CV perf on {} feats : sens {:.2f} spec {:.2f} F1 {:.2f}'.format(
                            len(inds_XGB), perf_shuffled[0], perf_shuffled[1],
                            perf_shuffled[2] ) )

                best_inds_XGB_among_heavy  =   perfs_XGB[-1]['featinds_present']
                best_inds_XGB = feat_subset_heavy[best_inds_XGB_among_heavy]

                best_nice = list( np.array(featnames_nice) [best_inds_XGB] )
                n_XGB_feats_to_print = 20
                print('XGB best feats (best {}, descending importance)={}'.
                      format(n_XGB_feats_to_print,best_nice[::-1][:n_XGB_feats_to_print] ) )

                pca_XGBfeats = PCA(n_components=nPCA_comp )
                pca_XGBfeats.fit( Xconcat_good_cur[:,best_inds_XGB])
                print('Min number of features found by XGB is {}, PCA on them gives {}'.
                        format( len(best_inds_XGB), pca_XGBfeats.n_components_) )
            else:
                clf_XGB = None
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


            if do_XGB_cur:
                lda_version_name =  'strongest_features_XGB_opinion'
                res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,best_inds_XGB],
                                    Xconcat_imputed[:,best_inds_XGB],
                                    class_labels_good,
                                    n_components_LDA, class_ind_to_check,
                                    revdict, calcName=lda_version_name,n_splits=n_splits)
                lda_analysis_versions[lda_version_name] = res_cur
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
            if do_XGB_cur:
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
                           'perfs_LDA_featsearch':perfs_LDA_featsearch,
                        'labels_good':class_labels_good,
                            'class_labels':class_labels,
                           'highest_meaningful_thri':highest_meaningful_thri,
                           'pca_derived_featinds_perthr':pca_derived_featinds_perthr,
                           'feat_variance_q_thr':feat_variance_q_thr,
                           'feat_subset_heavy':feat_subset_heavy,
                        'inds_important':inds_important,
                        'strong_inds_pc':strong_inds_pc,
                        'strongest_inds_pc':strongest_inds_pc,
                            'XGBobj':clf_XGB,
                            'strong_inds_XGB':best_inds_XGB,
                            'strong_inds_LDA':best_inds_LDA,
                            'perfs_XGB': perfs_XGB,
                            'pca_xgafeats': pca_XGBfeats,
                           'MI_per_feati':MI_per_feati,
                           'revdict':revdict,
                           'counts':counts,
                           'class_ids_grouped':class_ids_grouped}

            #out_name_templ = '_{}_grp{}-{}_{}ML_nr{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}__({},{})'
            #out_name = (out_name_templ ).\
            #    format(sources_type, src_file_grouping_ind, src_grouping,
            #        prefix, len(rawnames),
            #        n_channels, Xconcat_imputed.shape[1],
            #        pcapts.shape[1], skip, windowsz,grouping_key,int_types_key)
            #fname_ML_full_intermed = pjoin( gv.data_dir, output_subdir,
            #                                       '_{}{}.npz'.format(sind_join_str,out_name))

            out_name =  utils.genMLresFn(rawnames,sources_type, src_file_grouping_ind, src_grouping,
                    prefix, n_channels, Xconcat_imputed.shape[1],
                    pcapts.shape[1], skip, windowsz, use_main_LFP_chan, grouping_key,int_types_key )

            fname_ML_full_intermed = pjoin( gv.data_dir, output_subdir, out_name)

            # collect small non-system local variables
            #vv = locals().items()
            #for name,val in vv:
            #    if name[0] == '_':
            #        continue
            #    simple_types = (int,float,str,bool)
            #    is_simple_type = isinstance(val,simple_types)
            #    is_good_list = isinstance(val,(list,np.ndarray)) and \
            #        ((len(val) == 0) or isinstance(val[0], simple_types) )
            #    is_good_dict = isinstance(val,(dict)) and np.all( [isinstance(dval, simple_types) for dval in val.values()] )
            #    if is_simple_type or is_good_list or is_good_dict:
            #        print(name, type(val))

            # save first without Shapley values (because they can take too long
            # to compute and I want output anyway) -- somehow it does not work
            # on HPC :(, I get only first file
            if save_output:
                print('Saving intermediate result to {}'.format(fname_ML_full_intermed) )
                np.savez(fname_ML_full_intermed, results_cur=results_cur,
                         Xconcat_good_cur = Xconcat_good_cur, class_labels_good=class_labels_good,
                        selected_feat_inds_pri=dict(enumerate(selected_feat_inds_pri) ),
                        feature_names_filtered_pri = dict(enumerate(feature_names_pri)),
                          bininds_good = bininds_concat_good,
                         feat_info_pri = dict(enumerate(feat_info_pri)),
                        rawtimes_pri=dict(enumerate(rawtimes_pri)),
                         Xtimes_pri=dict(enumerate(Xtimes_pri)),
                        wbd_pri=dict(enumerate(wbd_pri)),
                        pcapts = pcapts, pcaobj=pca,
                        X_imputed=Xconcat_imputed,
                         cmd=(opts,args), pars=pars )
            else:
                print('Skipping saving intermediate result')

            featsel_per_method = {}

            if do_XGB_cur:
                featsel_per_method[ 'XGB_total_gain'] = {'scores': clf_XGB.feature_importances_ }

            for fsh in featsel_methods:
                featsel_info = {}
                shap_values = None
                explainer = None
                if fsh == 'SHAP_XGB':
                    import shap
                    #X = Ximp_per_raw[rncur][prefix]
                    #X_to_fit = X[gi]
                    #print(X_to_fit.shape)

                    #X_to_analyze_feat_sign = Xconcat_good_cur[:,best_inds_XGB]
                    X_to_analyze_feat_sign = X_for_heavy[:,best_inds_XGB_among_heavy]
                    featnames_sel = list( np.array(featnames_nice)[best_inds_XGB]  )

                    #
                    nsamples = max(200, X_to_analyze_feat_sign.shape[0] // 10 )
                    Xsubset = shap.utils.sample(X_to_analyze_feat_sign, nsamples)

                    print('Start computing Shapley values using Xsubset with shape',Xsubset.shape)

                    import copy
                    add_clf_creopts_ = copy.deepcopy(add_clf_creopts)
                    # SHAP doest not work well if GPU for some reason
                    if XGB_tree_method == 'gpu_hist':
                        add_clf_creopts_['tree_method'] = 'cpu_hist'

                    clf_bestfeats = XGBClassifier(**add_clf_creopts_)
                    #clf_bestfeats.fit(X_for_heavy[:,best_inds_XGB_among_heavy],
                    #                  class_labels_good_for_classif, **add_fitopts)
                    clf_bestfeats.fit(X_to_analyze_feat_sign,
                                      class_labels_good_for_classif, **add_fitopts)

                    #try:
                    explainer= shap.Explainer(clf_bestfeats.predict, Xsubset,feature_names=featnames_sel)
                    #explainer= shap.Explainer(clf_bestfeats.predict, Xsubset)
                    shap_values = explainer(X_to_analyze_feat_sign)

                    featsel_info['explainer'] = explainer
                    featsel_info['scores'] = shap_values
                    #except ValueError as e:
                    #    print(str(e) )
                    #    shap_values = None
                elif fsh == 'XGB_Shapley':
                    import xgboost as xgb
                    X = Xconcat_good_cur[::subskip_fit]
                    #X = X_for_heavy
                    y = class_labels_good_for_classif
                    dmat = xgb.DMatrix(X, y)

                    # TODO: perhaps I should select best hyperparameters above
                    # before doing this
                    clf_XGB2 = XGBClassifier(**add_clf_creopts)
                    clf_XGB2.fit(X, y, **add_fitopts)

                    bst = clf_XGB2.get_booster()

                    if (XGB_tree_method in ['hist', 'gpu_hist']) \
                            and allow_CUDA \
                            and len(gv.GPUs_list):
                        bst.set_param({"predictor": "gpu_predictor"})
                    #TODO: perhaps I should try to predict not the entire training
                    shap_values = bst.predict(dmat, pred_contribs=True)
                    #shap_values.shape

                    featsel_info['explainer'] = clf_XGB2
                    featsel_info['scores'] = shap_values

                elif fsh == 'interpret_EBM':
                    import itertools
                    import interpret
                    from interpret.glassbox import ExplainableBoostingClassifier

                    EBM_result_per_cp= {}
                    indpairs_names = []
                    # since EBM only works for binary, I treat each pair of classes separately
                    uls = list(set(class_labels_for_heavy))
                    class_pairs = list(itertools.combinations(uls, 2))
                    print(class_pairs)
                    EBM_seed = 0


                    info_per_cp = {}

                    #cpi = 0
                    for cpi in range(len(class_pairs)):
                        c1,c2 = class_pairs[cpi]
                        inds = np.where( (class_labels_for_heavy == c1) | (class_labels_for_heavy == c2)  )[0]
                        #inds2 = np.where(class_labels_good_for_classif == c2)[0]

                        #ipo = lab_enc.inverse_transform([c1,c2])
                        #indpair_names = ( revdict[ipo[0]], revdict[ipo[1]] )
                        indpair_names = ( revdict[c1], revdict[c2] )

                        print(f'Starting computing EBM for class pair {indpair_names}, in total {len(inds)}'+
                            f'=({sum(class_labels_for_heavy == c1)}+{sum(class_labels_for_heavy == c2)}) data points')

                        # filter classes
                        X = Xconcat_good_cur[inds]
                        y = class_labels_for_heavy[inds]

                        ebm = ExplainableBoostingClassifier(random_state=EBM_seed, feature_names=featnames_nice, n_jobs=n_jobs_XGB)
                        ebm.fit(X, y)
                        global_exp = ebm.explain_global()

                        sens,spec, F1, confmat  = utsne.getClfPredPower(ebm,X,y,class_ind_to_check, printLog=False)
                        confmat_normalized = utsne.confmatNormalize(confmat) * 100
                        print(f'confmat_normalized_true (pct) = {confmat_normalized}')

                        EBM_result_per_cp[indpair_names] = global_exp
                        indpairs_names += [indpair_names]

                        # extracting data from explainer
                        scores = global_exp.data()['scores']
                        names  = global_exp.data()['names']
                        sis = np.argsort(scores)[::-1]
                        featnames_srt = np.array(names)[sis]
                        print(f'EBM: Strongest feat is {featnames_srt[0]}')


                        info_cur = {}
                        info_cur['scores'] = scores
                        info_cur['explainer'] = global_exp
                        info_cur['perf'] = sens,spec, F1, confmat
                        info_cur['confmat_normalized'] = global_exp

                        info_per_cp[indpair_names ] = info_cur

                    featsel_info['info_per_cp'] = info_per_cp
                else:
                    raise ValueError('not implemented')

                featsel_per_method[fsh] = featsel_info

            results_cur['featsel_per_method'] = featsel_per_method

            if save_output:

                print('Saving intermediate result to {}'.format(fname_ML_full_intermed) )
                np.savez(fname_ML_full_intermed, results_cur=results_cur,
                         Xconcat_good_cur = Xconcat_good_cur, class_labels_good=class_labels_good,
                        selected_feat_inds_pri=dict(enumerate(selected_feat_inds_pri) ),
                        feature_names_filtered_pri = dict(enumerate(feature_names_pri)),
                          bininds_good = bininds_concat_good,
                         feat_info_pri = dict(enumerate(feat_info_pri)),
                        rawtimes_pri=dict(enumerate(rawtimes_pri)),
                         Xtimes_pri=dict(enumerate(Xtimes_pri)),
                        wbd_pri=dict(enumerate(wbd_pri)),
                        pcapts = pcapts, pcaobj=pca,
                        X_imputed=Xconcat_imputed,
                         cmd=(opts,args), pars=pars )
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

if not single_fit_type_mode:
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
        fname_PCA_full = pjoin( gv.data_dir, output_subdir, '{}{}.npz'.format(rawname_,out_name))



        lda_output_pg_cur = copy.deepcopy(lda_output_pg)
        dataset_bounds_Xbins
        indst,indend = dataset_bounds_Xbins[rawind]
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

                    trkey = 'X_transformed'
                    # crop everything
                    lda_analysis_vers =  r['lda_analysis_versions']
                    for featset_name, anver_cur in lda_analysis_vers.items():
                        for fit_type_name, fit_cur in anver_cur.items():
                            if trkey in fit_cur:
                                fit_cur[trkey] = fit_cur[trkey][sl]
                            #else:
                            #    print('{} not in {}:{}, skipping '.format(trkey,featset_name,fit_type_name) )

        if save_output:
            mask = np.zeros( len(Xconcat), dtype=bool )
            mask[bininds_concat_good] = 1
            bininds_good_cur = np.where( mask[sl] )[0]
            #before I had 'bininds_concat_good' name in the file for what now I call 'selected_feat_inds'
            # here I am atually saving not all features names, but only the
            # filtered ones. Now bininds_concat_good is good bin inds (not feautre inds)
            # !! Xconcat_imputed util Jan 25 daytime was the intire array, not the [sl] one
            np.savez(fname_PCA_full, pcapts = pcapts[sl], pcaobj=pca,
                    X=X_pri[rawind], wbd=wbd_pri[rawind], bininds_good = bininds_good_cur,
                    feature_names_filtered = feature_names_pri[rawind] ,
                    selected_feat_inds = selected_feat_inds_pri[rawind],
                    info = PCA_info, feat_info = feat_info_pri[rawind],
                    lda_output_pg = lda_output_pg_cur, Xtimes=Xtimes_pri[rawind], argv=sys.argv,
                    X_imputed=Xconcat_imputed[sl] ,  rawtimes=rawtimes_pri[rawind] )
            print('Saving PCA to ',fname_PCA_full)
        else:
            print('Skipping saving because save_output=0')
else:
    print('Skipping saving because of multi-groupings due to single_fit_type_mode')


######################## Plotting
if show_plots and ( ('pcapoints' in plot_types) or ( 'ldapoints' in plot_types  ) ):
    print('Starting to plot')
    #str_feats = ','.join(features_to_use)
    #str_mods = ','.join(data_modalities)

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

    colors,markers =utsne.prepColorsMarkers( anns, Xtimes,
            nedgeBins, windowsz, sfreq, skip, mrk,mrknames, color_per_int_type,
                                            side_letter= new_main_side[0].upper())

    if 'pcapoints' in plot_types:
        utsne.plotPCA(pcapts,pca, nPCAcomponents_to_plot,feature_names_all, colors, markers,
                    mrk, mrknames, color_per_int_type, tasks ,neutcolor=neutcolor)

        pdf.savefig()
        plt.close()

    if do_LDA and 'ldapoints' in plot_types:

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
                              neutcolor=neutcolor, nfeats_show=nfeats_per_comp_LDA,
                              title_suffix = '_(grp {}, its {})_{}'.
                              format(grouping_key, int_types_key, s) )

                pdf.savefig()
                plt.close()

    pdf.close()

gc.collect()
