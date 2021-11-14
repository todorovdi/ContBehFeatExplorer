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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

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
import xgboost as xgb

from os.path import join as pjoin

from resource import getrusage
import psutil,resource
import utils_postprocess as pp

from featlist import collectFeatTypeInfo
from featlist import filterFeats
from featlist import getFeatIndsRelToOnlyOneLFPchan

import builtins; _print = builtins.print
def print(*args, **kw):
    if 'flush' not in kw:
        _print(*args, **kw, flush=True)
    else:
        _print(*args, **kw)

def getMemUsed():
    #return getrusage(resource.RUSAGE_SELF).ru_maxrss;
    usage = psutil.virtual_memory().used
    print( f'^^^^^^^ mem usage (GB) =  {usage / GB:.4f}')
    return usage

def saveResToFolder_(obj,final_path):
    # create subfolders on the way
    print(f'Saving {final_path}')
    np.savez(final_path,obj);



GB = 1024 ** 3

#nPCA_comp = 0.95
nPCA_comp = 0.95
n_channels = 7
skip = 32
windowsz = 256

nPCAcomponents_to_plot = 5
nfeats_per_comp_LDA = 50
nfeats_per_comp_LDA_strongred = 5

show_plots = 0

discard_outliers_q = 1e-2
qshift = 1e-2
force_single_core = False
##########################

# all possible
feat_types_all = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr', 'Hjorth']
data_modalities_all = ['LFP', 'msrc']

# can be changed later
features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr']
data_modalities = [ 'msrc', 'LFP']  # order is important!
msrc_inds = np.arange(8,dtype=int)  #indices appearing in channel (sources) names, not channnel indices
use_main_LFP_chan = False
use_lfp_HFO = 1

fband_names_crude = ['tremor', 'beta', 'gamma']
fband_names_fine = ['tremor', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma' ]
fband_names_crude_inc_HFO = fband_names_crude + ['HFO']
fband_names_fine_inc_HFO = fband_names_fine + ['HFO1', 'HFO2', 'HFO3']
fbands_to_use = fband_names_fine_inc_HFO
bands_type = 'fine'
fbands_per_mod = {}
fbands_per_mod_set = 0

prefix = ''

#do_impute_artifacts = 1
#artif_handling_before_fit = 'impute' #or 'reject' or 'do_nothing'
artif_handling_before_fit = 'reject'
feat_stats_artif_handling  = 'reject'
do_outliers_discard = 1
do_cleanup = 1

def_sources_type = 'parcel_aal'
old_sources_type = 'HirschPt2011'
sources_type = def_sources_type

load_only = 0   # load and preproc to be precise
do_Classif = 1
n_feats = 609  # this actually depends on the dataset which may have some channels bad :(
do_XGB = 1
do_LDA = 1
calc_MI = 1
calc_VIF     = 1
calc_Boruta = 1
compute_ICA = 0
use_ICA_for_classif = 0

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
#feat_variance_q_thr = [0.6, 0.75, 0.9]  # I will select only features that have high variance according to PCA (high contributions to highest components)
feat_variance_q_thr = [0.87, 0.92, 0.99, 0.995, 0.999, 0.9999]  # I will select only features that have high variance according to PCA (high contributions to highest components)
use_low_var_feats_for_heavy_fits = True # whether I fit XGB and min feat sel only to highly varing features (according to feat_variance_q_thr)
search_best_LFP = ['LDA', 'XGB']
featsel_only_best_LFP = 1

save_output = 1
rescale_feats = 1

discard_remaining_int_types_during_fit = 1
skip_XGB_aux_intervals = 1
max_XGB_step_nfeats=30

cross_couplings_only=0
mainLFPchan_new_name_templ = 'LFP{}007'    # to make datasets consistent
self_couplings_only =0

plot_types_all = ['pcapoints', 'ldapoints', 'feat_stats' ]
plot_types = plot_types_all

n_splits = 4
input_subdir = ""
output_subdir = ""
scale_feat_combine_type = 'medcond'
label_groups_to_use = ['subj']  # subj, subj_medcond, medcond are allowed

allow_CUDA = True
XGB_tree_method = 'hist'  # or 'exact' or 'gpu_hist'
XGB_max_depth  = 4 # def = 6
XGB_min_child_weight = 3  # min num of samples making creating new node
XGB_tune_param = True
load_XGB_params_auto = 1
XGB_params_search_grid = {}
XGB_params_search_grid['max_depth'] = np.arange(3,10,2)
XGB_params_search_grid['min_child_weight'] = np.arange(3,12,2)
XGB_params_search_grid['subsample'] = np.array([0.6,0.75,0.83, 0.9,1.])
XGB_params_search_grid['eta'] = np.array([.3, .2, .1, .05])

VIF_thr = 10
VIF_search_worst = False

EBM_compute_pairwise = 1
EBM_featsel_feats = ['VIFsel']  # 'all','best_LFP', 'heavy' are also possible
featsel_on_VIF = 1
EBM_CV = 0
EBM_tune_param = False
EBM_tune_max_evals = 30
EBM_balancing = 'auto'
EBM_balancing_numfeats_thr = 120
EBM_seed = 0

XGB_featsel_feats = ['VIFsel']

use_smoothened = 0

selMinFeatSet_drop_perf_pct = 2.
selMinFeatSet_conv_perf_pct = 2.
n_samples_SHAP=200

#groupings_to_use = gv.groupings
groupings_to_use = [ 'merge_all_not_trem', 'merge_movements', 'merge_nothing' ]
groupings_to_use = [ 'merge_nothing' ]
#int_types_to_use = gv.int_types_to_include
int_types_to_use = [ 'basic', 'trem_vs_quiet' ]
featsel_methods = []
featsel_methods_all_possible = ['interpret_EBM', 'interpret_DPEBM', 'XGB_Shapley', 'SHAP_XGB' ]
selMinFeatSet_after_featsel = 'XGB_Shapley'
do_XGB_SHAP_twice = 1

params_read = {}
params_cmd = {}

prep_for_clf_only = 0

perf_inds_to_print = [0,1,2,10,-1]

strong_correl_level = 0.9

savefile_rawname_format = 'subj'
custom_rawname_str = None

exit_after = 'end'  #load, rescale, artif_processed


helpstr = 'Usage example\nrun_ML.py --rawnames <rawname_naked1,rawnames_naked2> '
opts, args = getopt.getopt(effargv,"hr:n:s:w:p:",
        ["rawnames=", "n_channels=",  "windowsz=", "pcexpl=",
         "show_plots=","discard_outliers_q=", 'feat_types=', 'use_HFO=', 'mods=',
         'prefix=', 'load_only=', 'prep_for_clf_only=', 'exit_after=',
         'fbands=',   'fbands_mod1=','fbands_mod2=',
         'n_feats=', 'single_core=',
         'sources_type=', 'bands_type=', 'crop=', 'parcel_types=',
         "src_grouping=", "src_grouping_fn=", 'groupings_to_use=',
         'int_types_to_use=', 'skip_XGB=', 'skip_LDA=',
         'LFP_related_only=',
         'parcel_group_names=', "subskip_fit=", "search_best_LFP=",
         "save_output=", 'rescale_feats=', "cross_couplings_only=", "LFPchan=",
         "heavy_fit_red_featset=", "n_splits=", "input_subdir=",
         "output_subdir=", "artif_handling=", "plot_types=",
         "skip_XGB_aux_int=", "max_XGB_step_nfeats=", "self_couplings_only=",
         "param_file=", "scale_feat_combine_type=", "use_smoothened=",
         "featsel_methods=", "allow_CUDA=", "XGB_tree_method=",
         "calc_MI=", "calc_VIF=", "calc_Boruta=",
         "n_samples_SHAP=", "calc_selMinFeatSet=",
          "selMinFeatSet_drop_perf_pct=", "selMinFeatSet_conv_perf_pct=",
          "savefile_rawname_format=",
         "selMinFeatSet_after_featsel=", "n_jobs=", "label_groups_to_use=",
         "SLURM_job_id=", "featsel_only_best_LFP=",
         "XGB_max_depth=", "XGB_min_child_weight=", "XGB_tune_param=",
         "EBM_tune_param=", "EBM_tune_max_evals=",
         "EBM_compute_pairwise=", "EBM_featsel_feats=", 'XGB_featsel_feats=',
         "featsel_on_VIF=", "custom_rawname_str=",
         "do_cleanup=", "VIF_thr=", "VIF_search_worst=", "compute_ICA=",
         "use_ICA_for_classif=", "load_XGB_params_auto=", "EBM_CV="])
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
    if arg.startswith('--'):
        raise ValueError('WRONG VALUE STRING!, argument starts with --')
    if opt == "n_channels":
        n_channels = int(arg)
    elif opt == "SLURM_job_id":
        SLURM_job_id = arg # str
    elif opt == "n_feats":
        n_feats = int(arg)
        n_feats_set_explicitly = True
    elif opt == "scale_feat_combine_type":
        scale_feat_combine_type = arg
    elif opt == "allow_CUDA":
        allow_CUDA = int(arg)
    elif opt == "selMinFeatSet_drop_perf_pct":
        selMinFeatSet_drop_perf_pct = float(arg)
    elif opt == "selMinFeatSet_conv_perf_pct":
        selMinFeatSet_conv_perf_pct = float(arg)
    elif opt == "n_jobs":
        n_jobs = int(arg)
    elif opt == "do_cleanup":
        do_cleanup = int(arg)
    elif opt == "XGB_tree_method":
        XGB_tree_method = arg
    elif opt == "featsel_only_best_LFP":
        featsel_only_best_LFP = int(arg)
    elif opt == "label_groups_to_use":
        label_groups_to_use = arg.split(',')
    elif opt == "skip_XGB":
        do_XGB = not bool(int(arg))
    elif opt == "skip_LDA":
        do_LDA = not bool(int(arg))
    elif opt == "load_XGB_params_auto":
        load_XGB_params_auto = int(arg)
    elif opt == "EBM_CV":
        EBM_CV = int(arg)
    elif opt == "VIF_thr":
        VIF_thr = float(arg)
    elif opt == "VIF_search_worst":
        VIF_search_worst = int(arg)
    elif opt == "featsel_on_VIF":
        featsel_on_VIF = int(arg)
    elif opt == "n_samples_SHAP":
        n_samples_SHAP = int(arg)
    elif opt == "EBM_compute_pairwise":
        EBM_compute_pairwise = int(arg)
    elif opt == "EBM_featsel_feats":
        EBM_featsel_feats = arg.split(',')
    elif opt == "XGB_featsel_feats":
        XGB_featsel_feats_ = arg.split(',')
        XGB_featsel_feats = []
        for ff in XGB_featsel_feats_:
            if len(ff)  :
                XGB_featsel_feats += [ff]
        if not len(XGB_featsel_feats):
            print('!Warning! XGB_featsel_feats is empty')
    elif opt == "calc_MI":
        calc_MI = int(arg)
    elif opt == "XGB_max_depth":
        XGB_max_depth = int(arg)
    elif opt == "XGB_min_child_weight":
        XGB_min_child_weight = int(arg)
    elif opt == "XGB_tune_param":
        XGB_tune_param = int(arg)
    elif opt == "EBM_tune_param":
        EBM_tune_param = int(arg)
    elif opt == "EBM_balancing":
        EBM_balancing = arg
    elif opt == "EBM_balancing_numfeats_thr":
        EBM_balancing_numfeats_thr = int(arg)
    elif opt == "EBM_tune_max_evals":
        EBM_tune_max_evals = int(arg)
    elif opt == "calc_VIF":
        calc_VIF = int(arg)
    elif opt == "savefile_rawname_format":
        savefile_rawname_format = arg
    elif opt == "calc_Boruta":
        calc_Boruta = int(arg)
    elif opt == "compute_ICA":
        compute_ICA = int(arg)
    elif opt == "use_ICA_for_classif":
        use_ICA_for_classif = int(arg)
    elif opt == "skip_XGB_aux_int":
        skip_XGB_aux_intervals = bool(int(arg))
    elif opt == "self_couplings_only":
        self_couplings_only = int(arg)
    elif opt == "heavy_fit_red_featset":
        use_low_var_feats_for_heavy_fits = int(arg)
    elif opt == "parcel_types":  # names of the roi_labels (not groupings)
        # "!<parcel" is also allowed
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
    elif opt == "selMinFeatSet_after_featsel":
        selMinFeatSet_after_featsel = arg
    elif opt == "parcel_group_names":
        parcel_group_names = arg.split(',')
    elif opt == "artif_handling":
        assert arg in ['impute' , 'reject']
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
        search_best_LFP = arg.split(',')
        if max([len(a) for a in  search_best_LFP] ) == 0:
            search_best_LFP = []
    elif opt == "calc_selMinFeatSet":
        calc_selMinFeatSet = int(arg)
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
    elif opt == 'exit_after':
        exit_after = arg
    elif opt == 'show_plots':
        show_plots = int(arg)
    elif opt == 'subskip_fit':
        subskip_fit = int(arg)
    elif opt == "windowsz":
        windowsz = int(arg)
    elif opt == "LFP_related_only":
        LFP_related_only = int(arg)
    elif opt == "discard_outliers_q":
        discard_outliers_q = float(arg)
    elif opt == "prefix":
        prefix = arg + '_'
    elif opt == "msrc_inds":
        msrc_inds = [ int(indstr) for indstr in arg.split(',') ]
        msrc_inds = np.array(msrc_inds, dtype=int)
    elif opt == "feat_types":
        features_to_use_pre = arg.split(',')
        features_to_use = []
        for ftu in features_to_use_pre:
            assert ftu in feat_types_all, ftu
            if ftu == 'Hjorth':
                features_to_use += ['H_mob', 'H_act', 'H_compl' ]
            else:
                features_to_use += [ftu]
    elif opt == "custom_rawname_str":
        custom_rawname_str = arg
    elif opt == "mods":
        data_modalities = arg.split(',')   #lfp of msrc
        #if len(data_modalities) == len(data_modalities_all):
        #    # assert we have the same ordering
        #    assert tuple(data_modalities) == tuple(data_modalities_all)
    elif opt == 'fbands':
        fbands_to_use = arg.split(',')
    elif opt == 'fbands_mod1':
        k = arg.find(':')
        mod_cur = arg[:k]
        fbands_per_mod[mod_cur] = arg[k+1:].split(',')
        fbands_per_mod_set += 1
    elif opt == 'fbands_mod2':
        k = arg.find(':')
        mod_cur = arg[:k]
        fbands_per_mod[mod_cur] = arg[k+1:].split(',')
        fbands_per_mod_set += 1
    elif opt == 'LFPchan':
        if arg == 'main': # use only main channel
            use_main_LFP_chan = 1
        elif arg == 'all':   # use all channels
            use_main_LFP_chan = 0
        elif arg.find('LFP') >= 0:   # maybe I want to specify expliclity channel name
            raise ValueError('to be implemented')
    elif opt == 'use_HFO':
        use_lfp_HFO = int(arg)
    elif opt == 'prep_for_clf_only':
        prep_for_clf_only = int(arg)
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

if use_ICA_for_classif and not compute_ICA:
    raise ValueError('nonsense')

if featsel_only_best_LFP and not use_main_LFP_chan:
    if 'best_LFP' not in XGB_featsel_feats:
        XGB_featsel_feats += ['best_LFP']
else:
    search_best_LFP = []

if featsel_on_VIF and ('VIFsel' not in XGB_featsel_feats) and calc_VIF:
    XGB_featsel_feats += ['VIFsel']

if fbands_per_mod_set > 0:
    assert fbands_per_mod_set == 2
    for mod in fbands_per_mod:
        assert len(fbands_per_mod[mod] )

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
n_jobs = None
if force_single_core:
    n_jobs = 1
else:
    n_jobs = max(1, mpr.cpu_count()-gp.n_free_cores)
assert n_jobs > 0

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

bindict_per_rawn = {}
bindict_hires_per_rawn = {}
anndict_per_intcat_per_rawn    = {}

# Load everything
for rawn in rawnames:
    subj,medcond,task  = utils.getParamsFromRawname(rawn)
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
    if sources_type == old_sources_type:
        a = '{}_feats_{}chs_nfeats{}_skip{}_wsz{}_grp{}-{}{}.npz'.\
            format(rawn,n_channels, n_feats, skip, windowsz,
                   src_file_grouping_ind, src_grouping, crp_str)
        feat_fnames += [a]
        fname_feat_full = pjoin( inp_sub,a)
    else:
        regex = '{}_feats_{}_{}chs_nfeats{}_skip{}_wsz{}_grp{}-{}{}.npz'.\
            format(rawn,sources_type,'[0-9]+', '[0-9]+', skip, windowsz,
                    src_file_grouping_ind, src_grouping, crp_str)
            #format(rawn, prefix, regex_nrPCA, regex_nfeats, regex_pcadim)

        fnfound = utsne.findByPrefix(inp_sub, rawn, prefix, regex=regex)
        if len(fnfound) > 1:
            fnt = [0] * len(fnfound)
            for fni in range(len(fnt) ):
                fnfull = pjoin(inp_sub, fnfound[fni])
                fnt[fni] = os.path.getmtime(fnfull)
            fni_max = np.argmax(fnt)
            fnfound = [ fnfound[fni_max] ]


        assert len(fnfound) == 1, 'For {} found not single fnames {}'.format(rawn,fnfound)
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


    feature_names_list_info = collectFeatTypeInfo(feature_names_all, ext_info = 0 )
    missing_ftypes = set(features_to_use)  - set(feature_names_list_info['ftypes'])
    assert missing_ftypes == set([]), f'Feat file is missing feature types {missing_ftypes}'

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

    anndict_per_intcat = f['anndict_per_intcat'][()]
    anndict_per_intcat_per_rawn[rawn] = anndict_per_intcat
    bindict_per_rawn[rawn] = upre.markedIntervals2Bins(anndict_per_intcat,
                                                       rawtimes,sfreq,wbd=wbd)

    if X_allfeats.shape[0] < wbd.shape[1]: # this can arise from saveing wbd_H instead of the right thing
        print('Warning, differnt X_allfeats and wbd shapes {},{}, cropping wbd'.
              format( X_allfeats.shape , wbd.shape )  )
        wbd = wbd[:,:X_allfeats.shape[0] ]
    assert wbd.shape[1] == len(Xtimes)   # Xtimes were aligned for sure
    wbd_pri += [wbd]

    tasks += [task]

    mainLFPchan = None
    if use_main_LFP_chan:
        best_LFP_info_fname = pjoin(gv.data_dir, 'best_LFP_info.json')
        with open(best_LFP_info_fname, 'r') as f:
            best_LFP_info = json.load(f)

        #onlyH modLFP LFPrel_noself_onlyCon LFPrel_noself_onlyRbcorr LFPrel_noself_onlyBpcorr
        best_LFP_prefix = 'modLFP'

        # it could (and should) be different for different subjects
        mainLFPchan = best_LFP_info[subj][f'{best_LFP_prefix},merge_movements,basic']['best_LFP']
        # also 'best_LFP_spec_only' is possible
        mainLFPchan_used_by_Jan = gv.gen_subj_info[subj]['lfpchan_used_in_paper']  # just for information
        print(f"My best LFP {mainLFPchan}, Jan's best LFP {mainLFPchan_used_by_Jan}")
        #mainLFPchan = gv.gen_subj_info[subj].get('lfpchan_selected_by_pipeline',None)
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
        raise ValueError('Non zero component inds found among feature names {}'.format(b) )

    ##################

    src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.\
        format(rawn,sources_type, src_file_grouping_ind)
    src_rec_info_fn_full = pjoin(gv.data_dir, input_subdir, src_rec_info_fn + '.npz')
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
    src_rec_info_pri += [rec_info]
    roi_labels = rec_info['label_groups_dict'][()]
    srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]

    ###########

    selected_feat_inds =  filterFeats(feature_names_all, chnames_LFP,
                                      LFP_related_only, parcel_types,
                                      remove_crossLFP, cross_couplings_only,
                                      self_couplings_only, fbands_to_use,
                                      features_to_use, fbands_per_mod, feat_types_all,
                                      data_modalities, data_modalities_all,
                                      msrc_inds, parcel_group_names,
                                      roi_labels,srcgrouping_names_sorted,
                                      src_file_grouping_ind, fbands_def,
                                      fband_names_fine_inc_HFO, use_lfp_HFO,
                                      use_main_LFP_chan, mainLFPchan,
                                      mainLFPchan_new_name_templ,
                                      printLog=1)

    good_feats = feature_names_all[ selected_feat_inds]
    ###########

    if len(good_feats) == 0:
        print('!!!!!!!!!!!!--------  We got zero features! Exiting')
        sys.exit(5)
    X = X_allfeats[:, selected_feat_inds]


    canonical_feat_order = 1
    if canonical_feat_order:
        from featlist import sortFeats
        ordinds = sortFeats(good_feats,gv.desired_feature_order)
        good_feats = good_feats[ ordinds ]
        X = X[:,ordinds]


    X_pri += [ X ]
    del X_allfeats
    feature_names_pri += [ good_feats ]
    selected_feat_inds_pri += [selected_feat_inds]


    mts_letters_pri += [mts_letter]

    if use_main_LFP_chan:
        mainLFPchan_sidelet = mainLFPchan[3]
        chnames_LFP = [ mainLFPchan_new_name_templ.format(mainLFPchan_sidelet)  ]
        chnames_LFP_pri[-1] = chnames_LFP


for chnames_LFP in chnames_LFP_pri:
    assert set(chnames_LFP) == set(chnames_LFP_pri[0]), chnames_LFP_pri

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

    subdir_fig = pjoin(gv.dir_fig,output_subdir)
    if not os.path.exists(subdir_fig ):
        print('Creating output subdir {}'.format(subdir_fig) )
        os.makedirs(subdir_fig)

    out_name_plot = rn_str + out_name + \
        'mainLFP{}_HFO{}_{}_{}'.\
        format(int(use_main_LFP_chan), int(use_lfp_HFO), str_mods, str_feats)
    pdf= PdfPages(pjoin(subdir_fig, out_name_plot + '.pdf' ))

if load_only or exit_after == 'load':
    print('Got load_only or exit_after, exiting!')
    #if show_plots:
    #    pdf.close()
    sys.exit(0)

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
    X_pri_rescaled, indsets, means, stds = \
        upre.rescaleFeats(rawnames, X_pri,
            feature_names_pri, wbd_pri,
            sfreq, rawtimes_pri, int_type = baseline_int_type,
            main_side = None,
            side_rev_pri = side_switch_happened_pri,
            artif_handling=feat_stats_artif_handling,
            minlen_bins = 5 * sfreq // skip,
            combine_within=scale_feat_combine_type,
            bindict_per_rawn=bindict_per_rawn  )
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

    Xconcat = np.concatenate(X_pri_rescaled,axis=0)
else:
    Xconcat = np.concatenate(X_pri,axis=0)

lens_pri = [ Xcur.shape[0] for Xcur in X_pri ]
if single_fit_type_mode and not gv.DEBUG_MODE and do_cleanup:
    del X_pri
    del feat_file_pri
    del f

if len(set(new_main_side_pri)  ) > 1:
    ws = 'STOP!  we have datasets with different main tremor sides here! Remapping is needed!'
    print(ws, set(new_main_side_pri) )
    #raise ValueError(ws)


if exit_after == 'rescale':
    print(f'Got exit_after={exit_after}, exiting!')
    #if show_plots:
    #    pdf.close()
    sys.exit(0)


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

# here I need to use side_switched because I need to concatenate anns (I were
# not doing it in the previous processing stages)
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
bininds_noartif_nounlab = np.arange(Xconcat.shape[0])    #everything
if do_outliers_discard:
    artif_naive_bininds, qvmult, discard_ratio = \
        utsne.findOutlierLimitDiscard(Xconcat,discard=discard_outliers_q,qshift=1e-2)
    bininds_noartif_nounlab = np.setdiff1d( bininds_noartif_nounlab, artif_naive_bininds)

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
wrong_brain_side_let = new_main_side[0].upper()  # ipsilater is wrong for LFP
anns_artif = utils.removeAnnsByDescr(anns_artif,
        'BAD_LFP{}'.format(wrong_brain_side_let)  )
ivalis_artif = utils.ann2ivalDict(anns_artif)
ivalis_artif_tb_indarrays_merged = \
    utils.getWindowIndicesFromIntervals(wbd_merged,ivalis_artif,
                                    sfreq,ret_type='bins_contig',
                                    ret_indices_type =
                                    'window_inds', nbins_total=nbins_total )
#ivalis_artif_tb, ivalis_artif_tb_indarrays = utsne.getAnnBins(ivalis_artif, Xtimes,
#                                                            nedgeBins, sfreq, skip, windowsz, dataset_bounds)
#ivalis_artif_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)


Xconcat_artif_nan = utils.setArtifNaN(Xconcat,
                        ivalis_artif_tb_indarrays_merged,
                        featnames, ignore_shape_warning=test_mode)
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
#Xconcat_to_fit is NOT about removing points, it is used TOGETHER with bininds_noartif_nounlab
if artif_handling_before_fit == 'impute':
    # replacing artifact-related NaNs with zeros
    # mean should be zero since we centered features earlier

    #we DO NOT change bininds_noartif_nounlab here
    Xconcat_to_fit = Xconcat_imputed
elif artif_handling_before_fit == 'reject':
    bininds_noartif_nounlab = np.setdiff1d(bininds_noartif_nounlab, artif_bininds)
    Xconcat_to_fit = Xconcat

    assert not np.any( np.isnan(Xconcat[bininds_noartif_nounlab] ) )
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
print( f'Num of indices not assigned to any interval = {len(unset_inds) } ')

# this is the most bruteforce way when I join all artifacts
#lst2 =  [inds for inds in ivalis_artif_tb_indarrays_merged.values()]
#if len(lst2):
#    all_interval_artif_inds = np.hstack(lst2  )
#else:
#    all_interval_artif_inds = np.array([])

remove_pts_unlabeled_beh_states = 1
if remove_pts_unlabeled_beh_states:
    #do somthing
    bininds_concat_good_yes_label = np.setdiff1d( bininds_noartif_nounlab, unset_inds)
    print('Removing {} unlabeled pts before PCA'.
          format(len(bininds_noartif_nounlab) - len(bininds_concat_good_yes_label) ) )
    bininds_noartif_nounlab = bininds_concat_good_yes_label
else:
    print('Warning not removing unlabeled before PCA')


featnames_nice = utils.nicenFeatNames(featnames,
                                    roi_labels,srcgrouping_names_sorted)
assert len(featnames_nice) == len(featnames)

if exit_after == 'artif_processed':
    print(f'Got exit_after={exit_after}, exiting!')
    #if show_plots:
    #    pdf.close()
    sys.exit(0)

# bininds_noartif_nounlab  are inds of bin where I have thrown away outliers and removed
# unlabeled

print('Input PCA dimension ', (len(bininds_noartif_nounlab),Xconcat.shape[1]) )
pca = PCA(n_components=nPCA_comp)
Xsubset_to_fit = Xconcat_to_fit[bininds_noartif_nounlab]
#Xsubset_to_fit = Xsubset_to_fit
pca.fit(Xsubset_to_fit )   # fit to not-outlier data
pcapts = pca.transform(Xconcat_imputed)  # transform outliers as well


ica = None
if compute_ICA:
    from sklearn.decomposition import FastICA
    max_iter = 500
    pcapts_good = pca.transform(Xsubset_to_fit)  # transform outliers as well
    ica = FastICA(n_components=pcapts_good.shape[1],
                    random_state=0, max_iter=max_iter)
    ica.fit(pcapts_good)
    Xsubset_to_fit_ICA =  ica.transform(pcapts_good)
    featnames_ICA = [ f'ica {fi}' for fi in \
                     np.arange(Xsubset_to_fit_ICA.shape[1] ) ]

    Xconcat_imputed = ica.transform(pcapts)

assert not (use_ICA_for_classif and len(search_best_LFP))


print('Output PCA dimension {}, total explained variance proportion {:.4f}'.
      format( pcapts.shape[1] , np.sum(pca.explained_variance_ratio_) ) )
print('PCA First several var ratios ',pca.explained_variance_ratio_[:5])

nfeats_per_comp_LDA_strongred = max(pcapts.shape[1] // 10, 5)

subjs_analyzed = upre.getRawnameListStructure(rawnames)
sind_strs = list(sorted( (subjs_analyzed.keys()) ) )
sind_join_str = ','.join(sind_strs)

############################################
indst = 0
indend = lens_pri[0]
dataset_bounds_Xbins = [] #skipped bin indices
for rawind,rawn in enumerate(rawnames):
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
info['int_types_to_use'] = int_types_to_use
info['groupings_to_use'] = groupings_to_use
info['src_grouping'] = src_grouping
info['src_grouping_fn'] = src_file_grouping_ind
# I'd prefer to save both the entire list and the feat fname most related
# to the current output
info['fname_feat_full'] = fname_feat_full_pri[rawind] # don't remove!! althought before Jan 25 it is wrong :(
info['fname_feat_full_pri'] = fname_feat_full_pri
info['artif_handling'] = artif_handling_before_fit
info['rawnames'] = rawnames
ML_info = info


if do_Classif:
    # don't change the order!
    #int_types_L = ['trem_L', 'notrem_L', 'hold_L', 'move_L', 'undef_L', 'holdtrem_L', 'movetrem_L']
    #int_types_R = ['trem_R', 'notrem_R', 'hold_R', 'move_R', 'undef_R', 'holdtrem_R', 'movetrem_R']
    # these are GLOBAL ids, they should be consistent across everything

    mult_clf_results_pg = {}

    # over groupings of behavioral states
    for grouping_key in groupings_to_use:
        grouping = gp.groupings[grouping_key]

        mult_clf_results_pit = {}
        # over behavioral states to decode (other points get thrown away)
        for int_types_key in int_types_to_use:
            if grouping_key not in gp.group_vs_int_type_allowed[int_types_key]:
                print('Skipping grouping {} for int types {}'.format(grouping,int_types_key) )
                mult_clf_results_pit[int_types_key] = None
                continue
            if int_types_key == 'trem_vs_quiet' and 'merge_nothing' in groupings_to_use and \
                    grouping_key == 'merge_all_not_trem':
                print("We don't want to compute the same thing twice, so skipping {},{}".
                      format(grouping_key, int_types_key) )
                continue

            int_types_to_distinguish = gp.int_types_to_include[int_types_key]

            print('---------------------------------------------')
            print('Start classif (grp {}, its {})'.format(grouping_key, int_types_key))

            filename_to_save =  utils.genMLresFn(rawnames,sources_type, src_file_grouping_ind, src_grouping,
                    prefix, n_channels, Xconcat_imputed.shape[1],
                    pcapts.shape[1], skip, windowsz, use_main_LFP_chan, grouping_key,int_types_key,
                    rawname_format = savefile_rawname_format,
                    custom_rawname_str=custom_rawname_str)

            fname_ML_full_intermed = pjoin( gv.data_dir, output_subdir, filename_to_save)
            fname_ML_full_intermed_light = pjoin( gv.data_dir, output_subdir, '_!' + filename_to_save)
            if save_output:
                print(f'later we will be saving to {filename_to_save}')

            group_labels_dict = {}
            revdict_per_dataset_int_type = {}
            class_ids_grouped_per_dataset_int_type = {}
            # generate group labels for all dataset interval types
            for int_types_key_cur in gp.int_type_datset_rel:
                group_labels = np.zeros( len(Xconcat_imputed) , dtype=int )
                int_types_to_distinguish_cur = gp.int_types_to_include[int_types_key_cur]

                indst = 0
                indend = lens_pri[0]
                #if grouping_key == 'subj_medcond_task':
                #    for rawi,rn in enumerate(rawnames) :
                #        cid = int_types_to_distinguish.index(rn)
                #        group_labels[indst:indend] = cid
                #        if rawind+1 < len(lens_pri):
                #            indst += lens_pri[rawi]
                #            indend += lens_pri[rawi+1]
                #elif grouping_key == 'subj_medcond':
                #    for rawi,rn in enumerate(rawnames) :
                #        sind_str,m,t = utils.getParamsFromRawname(rn)
                #        key_cur = '{}_{}'.format(sind_str,m)
                #        cid = int_types_to_distinguish.index(key_cur)
                #        group_labels[indst:indend] = cid
                #        if rawind+1 < len(lens_pri):
                #            indst += lens_pri[rawi]
                #            indend += lens_pri[rawi+1]
                #elif grouping_key == 'subj':
                class_ids_grouped = {}
                revdict = {}
                for rawi,rn in enumerate(rawnames) :
                    sind_str,m,t = utils.getParamsFromRawname(rn)
                    if int_types_key_cur == 'subj_medcond_task':
                        key_cur = '{}_{}_{}'.format(sind_str,m,t)
                    elif int_types_key_cur == 'subj_medcond':
                        key_cur = '{}_{}'.format(sind_str,m)
                    elif int_types_key_cur == 'subj':
                        key_cur = '{}'.format(sind_str)
                    elif int_types_key_cur == 'medcond':
                        key_cur = '{}'.format(m)
                    else:
                        raise ValueError('wrong int_types_key_cur {}'.
                                         format(int_types_key_cur) )
                    cid = int_types_to_distinguish_cur.index(key_cur) + \
                        gp.int_types_aux_cid_shift[int_types_key_cur]

                    class_ids_grouped[ key_cur ] = cid
                    revdict[cid] = key_cur

                    group_labels[indst:indend] = cid
                    if rawi+1 < len(lens_pri):
                        indst += lens_pri[rawi]
                        indend += lens_pri[rawi+1]
                # we should have labeled everything
                assert np.sum( group_labels  ==0) == 0
                group_labels = group_labels[bininds_noartif_nounlab]
                group_labels_dict[int_types_key_cur] = group_labels.copy()
                revdict_per_dataset_int_type[int_types_key_cur] = revdict
                class_ids_grouped_per_dataset_int_type[int_types_key_cur] = class_ids_grouped

            if int_types_key in gp.int_type_datset_rel:
                # distinguishing datasets
                class_labels = group_labels
                class_labels_good = class_labels[bininds_noartif_nounlab]
                revdict = revdict_per_dataset_int_type[int_types_key]
                class_ids_grouped = class_ids_grouped_per_dataset_int_type[int_types_key]

                rawind_to_test = 0
                sind_str,m,t = utils.getParamsFromRawname(rawnames[rawind_to_test] )
                if int_types_key == 'subj_medcond_task':
                    key_cur = '{}_{}_{}'.format(sind_str,m,t)
                elif int_types_key == 'subj_medcond':
                    key_cur = '{}_{}'.format(sind_str,m)
                elif int_types_key == 'subj':
                    key_cur = '{}'.format(sind_str)
                elif int_types_key == 'medcond':
                    key_cur = '{}'.format(m)
                else:
                    raise ValueError('wrong int_types_key {}'.format(int_types_key) )
                class_to_check = key_cur

                if use_ICA_for_classif:
                    Xconcat_good_cur = Xsubset_to_fit_ICA
                else:
                    Xconcat_good_cur = Xsubset_to_fit
            else:
                sides_hand = [new_main_side[0].upper() ]
                # here I need to use length of entire array, before artifacts
                # got thrown away, because invalis_tb_indarrays are related to
                # the full thing
                class_labels, class_labels_good, revdict, class_ids_grouped,inds_not_neut  = \
                    utsne.makeClassLabels(sides_hand, grouping,
                        int_types_to_distinguish,
                        ivalis_tb_indarrays_merged, bininds_noartif_nounlab,
                        len(Xconcat_imputed),
                        rem_neut = discard_remaining_int_types_during_fit)

                # same but without merging (will need for EBM)
                class_labels_nm, class_labels_good_nm, revdict_nm, class_ids_grouped_nm,inds_not_neut_nm  = \
                    utsne.makeClassLabels(sides_hand, gp.groupings['merge_nothing'],
                        int_types_to_distinguish,
                        ivalis_tb_indarrays_merged, bininds_noartif_nounlab,
                        len(Xconcat_imputed),
                        rem_neut = discard_remaining_int_types_during_fit)

                assert len(class_labels_good_nm) == len(class_labels_good)

                if discard_remaining_int_types_during_fit:
                    # then we have to remove the data points as well
                    #neq = class_labels_good != gp.class_id_neut
                    #inds_not_neut2 = np.where( neq)[0]
                    if use_ICA_for_classif:
                        Xconcat_good_cur = Xsubset_to_fit_ICA[inds_not_neut]
                    else:
                        Xconcat_good_cur = Xsubset_to_fit[inds_not_neut]

                    for itkc,glv in group_labels_dict.items():
                        group_labels_dict[itkc] = glv[inds_not_neut]
                else:
                    if use_ICA_for_classif:
                        Xconcat_good_cur = Xsubset_to_fit_ICA
                    else:
                        Xconcat_good_cur = Xsubset_to_fit

                #this is a string label
                class_to_check = '{}_{}'.format(int_types_to_distinguish[0], new_main_side[0].upper() )
            class_ind_to_check = class_ids_grouped[class_to_check]


            if use_ICA_for_classif:
                featnames_for_fit = featnames_ICA
                featnames_nice_for_fit = featnames_ICA
            else:
                featnames_for_fit = featnames
                featnames_nice_for_fit = featnames_nice

            assert Xconcat_good_cur.shape[0] == len(class_labels_good)
            # check that the labels number don't have holes -- well, they do
            #assert np.max (np.abs( np.diff(sorted(set(class_labels_good) )) )) == 0

            numpoints_per_class_id = utsne.countClassLabels(class_labels_good, class_ids_grouped=None, revdict=revdict)
            print('bincounts are ',numpoints_per_class_id)
            bcthr = 10
            if test_mode:
                bcthr = 1
            if numpoints_per_class_id[class_to_check] <= bcthr:
                cid = class_ids_grouped[class_to_check]
                s = '!!! WARNING: grouping_key,int_types_key: class {} (cid={}) has too few instances all! skipping'.format(class_to_check,cid)
                print(s)
                continue


            n_components_LDA = len(set(class_labels_good)) - 1
            print('n_components_LDA =', n_components_LDA)
            if n_components_LDA == 0:
                mult_clf_results_pit[int_types_key] = None
                continue

            subjdir = ''
            if custom_rawname_str is None:
                subjs = list(set(subjs_pri))
                if len(subjs) == 1:
                    subjdir = subjs[0]
                else:
                    subjdir = ','.join(rawnames)
            else:
                subjdir = custom_rawname_str

            def saveResToFolder(resobj, key, subsubdir = ''):
                #from pathlib import Path
                pl = [ gv.data_dir, output_subdir,
                    subjdir, prefix, int_types_key, grouping_key]
                if len(subsubdir):
                    pl += [subsubdir]
                pl += [key ]

                # make dirs except of the last thing (which is a filename)
                os.makedirs(pjoin(*tuple(pl[:-1] ) ), exist_ok=True)

                final_path = pjoin(*tuple(pl) )
                #path_list = pre_final_path.split(os.path.sep)
                #final_path = pjoin(
                obj = resobj[key]
                if save_output:
                    saveResToFolder_(obj, final_path + '.npz')

            ####################### Feature subsets (PCA) preparation


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
                # for each feature I choose maximal contribution across
                # components (the ones that are responsible for most variance)
                # axis 0 -- index componenets, axis 1 -- features
                old_ver = 0
                if old_ver:
                    m = np.max(np.abs(pca.components_),axis=0)
                    q = np.quantile(m,feat_variance_q_thr[thri])
                    # then I take only strongest-contributing
                    pca_derived_featinds = np.where(m > q)[0]
                else:
                    pca_derived_featinds = pp.selBestColumns(np.abs( pca.components_ ), feat_variance_q_thr[thri])
                    #pca_derived_featinds= []
                    #q = np.quantile(np.abs(pca.components_),feat_variance_q_thr[thri],axis=1)
                    #for i in range(pca.components_.shape[0] ):
                    #    cur_inds = np.where( pca.components_[i] > q[i] )[0]
                    #    pca_derived_featinds += cur_inds.tolist()
                    #pca_derived_featinds = np.unique( pca_derived_featinds )

                if len(pca_derived_featinds) >= 10:
                    highest_meaningful_thri = thri
                pca_derived_featinds_perthr += [ pca_derived_featinds ]

            ##################

            nfeats = Xconcat_good_cur.shape[1]
            assert nfeats == len(featnames_for_fit)
            assert nfeats == len(featnames_nice_for_fit)
            if use_low_var_feats_for_heavy_fits and nfeats > 40:
                #thri = highest_meaningful_thri
                thri = 0
                if thri < 0:
                    feat_inds_for_heavy = np.arange(nfeats )
                else:
                    print('Selecting only {}-q-variance-thresholded features for heavy fits'.format(
                        feat_variance_q_thr[thri] ) )
                    feat_inds_for_heavy = pca_derived_featinds_perthr[thri]
            else:
                feat_inds_for_heavy = np.arange(nfeats )
            X_for_heavy = Xconcat_good_cur[::subskip_fit, feat_inds_for_heavy]
            class_labels_for_heavy = class_labels_good[::subskip_fit]

            if prep_for_clf_only:
                print('Got prep_for_clf_only, exiting!')
                #if show_plots:
                #    pdf.close()
                sys.exit(0)


            # for DEBUG only
            # continue
            #########################   MAIN analysis start
            #muctr = 1
            usage = getMemUsed();

            results_cur = {}
            verfn = 'last_code_ver_synced_with_HPC.txt'
            verstr = 'unk'
            if os.path.exists(verfn):
                with open(verfn ) as f:
                    verstr = f.readlines()[0]
            results_cur['code_ver'] = verstr
            results_cur['group_labels_dict'] = group_labels_dict
            results_cur['feature_names_filtered'] = featnames #=feature_names_pri[0]
            results_cur['feature_names_nice'] = featnames_nice #=feature_names_pri[0]
            results_cur['featnames_for_fit'] = featnames_for_fit
            results_cur['featnames_nice_for_fit'] = featnames_nice_for_fit

            results_cur['icaobj'] = ica

            try:
                # we really want this and not Xconcat good cur here
                C = np.corrcoef(Xsubset_to_fit,rowvar=False)
                absC = np.abs(C)

                C_nocenter = absC - np.diag(np.diag(absC))

                C_flat = C_nocenter.flatten()
                sinds = np.argsort(C_flat)
                hist, bin_edges = np.histogram(C_nocenter.flatten(), bins=20, density=False)

                strong_correl_inds = np.where( C_flat > strong_correl_level )[0]
                print('Num: strongly (>{:.2f}) correl feats (div by 2)={}, of total pairs {} it is {:.4f}'.format(strong_correl_level, len(strong_correl_inds) //2 ,
                                                                                                            len(C_flat)//2, len(strong_correl_inds) / len(C_flat) ) )
                nonsyn_feat_inds = pp.getNotSyn(C,strong_correl_level)
                print(f'Num of feats excluding synonyms for corr. level = {strong_correl_level:.3f}: {len(nonsyn_feat_inds)} of {C.shape[0]}' )
            except ValueError as e:
                print(f'Error during correl matrix comp {e}')
                C = None
                hist,bin_edges = None,None
                nonsyn_feat_inds = None

            results_cur['corr_matrix'] = C
            results_cur['corr_matrix_hist'] = hist, bin_edges
            results_cur['nonsyn_feat_inds'] = nonsyn_feat_inds

            saveResToFolder(results_cur, 'corr_matrix' )
            saveResToFolder(results_cur, 'corr_matrix_hist' )
            saveResToFolder(results_cur, 'nonsyn_feat_inds' )

            ##################

            lab_enc = preprocessing.LabelEncoder()
            # just skipped class_labels_good
            lab_enc.fit(class_labels_for_heavy)
            class_labels_good_for_classif = lab_enc.transform(class_labels_for_heavy)
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = compute_sample_weight('balanced',class_labels_good_for_classif)
            class_label_ids = lab_enc.inverse_transform( np.arange( len(set(class_labels_good_for_classif)) ) )
            class_label_names_ordered = [revdict[cli] for cli in class_label_ids]

            results_cur['class_labels_good_for_classif'] = class_labels_good_for_classif
            results_cur['class_label_ids'] = class_label_ids
            results_cur['class_label_names_ordered'] = class_label_names_ordered
            results_cur['class_weights'] = class_weights


            #indlist = fip_fs
            #indlist = np.arange(C.shape[0])
            #C_subset = C[indlist,:][:,indlist]
            #orig_inds = indlist[nonsyn_feat_inds]

            if exit_after == 'corr_matrix':
                print(f'Got exit_after={exit_after}, exiting!')
                #if show_plots:
                #    pdf.close()
                sys.exit(0)



            if calc_MI:
                print('  Computing MI')
                MI_per_feati = utsne.getMIs(Xconcat_good_cur,class_labels_good,class_ind_to_check,
                                            n_jobs=n_jobs)
                high_to_low_MIinds = np.argsort(MI_per_feati)[::-1]

                n_MI_to_show = 8
                for ii in high_to_low_MIinds[:n_MI_to_show]:
                    print('  {} MI = {:.5f}'.format(featnames_nice[ii], MI_per_feati[ii]  ) )

                results_cur['MI_per_feati'] = MI_per_feati
                saveResToFolder(results_cur, 'MI_per_feati' )
            else:
                print('  skipping computation of MI')
                MI_per_feati = None


            colinds_bad_VIFsel,colinds_good_VIFsel,vfs_list = None,None,None
            if calc_VIF:
                assert not use_ICA_for_classif  # it would not make sense

                VIF_reversed_order = 1
                #revinds = np.arange(Xconcat_good_cur.shape[1] )[::-1]
                X_for_VIF = Xconcat_good_cur[::subskip_fit,:]
                featnames_VIF = featnames_nice_for_fit
                #if VIF_reversed_order:
                #    X_for_VIF = X_for_VIF[:,revinds]
                #    featnames_VIF = list( np.array(featnames_VIF)[revinds] )

                print(f'starting VIF for X_for_VIF.shape={X_for_VIF.shape}, rev={VIF_reversed_order}')
                #colinds_bad_VIFsel_unrev,colinds_good_VIFsel_unrev,VIFs_list, \
                #    VIFsel_featsets_list,VIFsel_linreg_objs,exogs_list  = \
                #    utsne.findBadColumnsVIF(X_for_VIF, VIF_thr=VIF_thr,n_jobs=n_jobs,
                #                            search_worst=VIF_search_worst,
                #                            featnames = featnames_VIF, rev=VIF_reversed_order )
                colinds_bad_VIFsel,colinds_good_VIFsel,VIFs_list, \
                    VIFsel_featsets_list,VIFsel_linreg_objs,exogs_list  = \
                    utsne.findBadColumnsVIF(X_for_VIF,
                                            VIF_thr=VIF_thr,n_jobs=n_jobs,
                                            search_worst=VIF_search_worst,
                                            featnames = featnames_VIF,
                                            rev=VIF_reversed_order )

                #colinds_bad_VIFsel   =  revinds[colinds_bad_VIFsel_unrev]
                #colinds_good_VIFsel  =  revinds[colinds_good_VIFsel_unrev]
                #print('VIF-selected bad features are ',np.array(featnames_VIF)[colinds_bad_VIFsel_unrev] )

                print(f'VIF truncation found {len(colinds_bad_VIFsel)}' +\
                      f'bad indices of total {Xconcat_good_cur.shape[1]}')
                print('VIF-selected bad features are ',np.array(featnames_VIF)[colinds_bad_VIFsel] )
                gc.collect()

                VIF_truncation = {}
                VIF_truncation['VIF_thr'] = VIF_thr
                VIF_truncation['exogs_list']  = exogs_list
                VIF_truncation['VIF_search_worst'] = VIF_search_worst
                VIF_truncation['X_for_VIF_shape'] = X_for_VIF.shape
                VIF_truncation[ 'colinds_bad_VIFsel'] = colinds_bad_VIFsel
                VIF_truncation[ 'colinds_good_VIFsel'] = colinds_good_VIFsel
                VIF_truncation[ 'VIFsel_featsets_list'] = VIFsel_featsets_list
                VIF_truncation[ 'VIFs_list'] = VIFs_list
                VIF_truncation['VIFsel_linreg_objs'] = VIFsel_linreg_objs
                VIF_truncation['reversed_order'] = VIF_reversed_order
                #VIF_truncation[ 'colinds_bad_VIFsel_unrev'] = colinds_bad_VIFsel_unrev
                #VIF_truncation[ 'colinds_good_VIFsel_unrev'] = colinds_good_VIFsel_unrev
                results_cur['VIF_truncation'] = VIF_truncation

                saveResToFolder(results_cur, 'VIF_truncation' )


            if exit_after == 'VIF':
                print(f'Got exit_after={exit_after}, exiting!')
                #if show_plots:
                #    pdf.close()
                sys.exit(0)

            ########################
            usage = getMemUsed();

            LDA_analysis_versions = {}
            if do_LDA:
                lda_version_name = 'all_present_features'
                # here "imputed" only used to compute transform, nothing more
                res_all_feats = utsne.calcLDAVersions(Xconcat_good_cur, Xconcat_imputed, class_labels_good,
                                    n_components_LDA, class_ind_to_check, revdict,
                                            calcName=lda_version_name,n_splits=n_splits)
                LDA_analysis_versions[lda_version_name] = res_all_feats
                gc.collect()
                results_cur['LDA_analysis_versions'] = LDA_analysis_versions

                # labels_pred = lda.predict(Xconcat_good_cur)
                # conf_mat = confusion_matrix(y_true, y_pred)

                if len(search_best_LFP ) >0 :
                    results_cur['best_LFP'] = {}
                #    results_cur['best_LFP'] = dict( (kk,[]) for kk in search_best_LFP)

                if 'LDA' in search_best_LFP and (not use_main_LFP_chan) and \
                        ('LFP' in data_modalities) and (len(chnames_LFP) > 1 ):
                    for chn_LFP in chnames_LFP:

                        # I want to remove features related to this LFP channel and
                        # see what happens to performance

                        feat_inds_curLFP, feat_inds_except_curLFP = \
                            getFeatIndsRelToOnlyOneLFPchan(featnames,
                                chn=chn_LFP, chnames_LFP=chnames_LFP)

                        lda_version_name = 'all_present_features_but_{}'.format(chn_LFP)
                        res_cur =\
                            utsne.calcLDAVersions(Xconcat_good_cur[:,feat_inds_except_curLFP],
                                            Xconcat_imputed[:,feat_inds_except_curLFP],
                                            class_labels_good, n_components_LDA,
                                            class_ind_to_check, revdict,
                                            calcName=lda_version_name,n_splits=n_splits)
                        res_cur['featis'] = feat_inds_except_curLFP
                        LDA_analysis_versions[lda_version_name] = res_cur
                        gc.collect()

                        lda_version_name = 'all_present_features_only_{}'.format(chn_LFP)
                        res_cur =\
                            utsne.calcLDAVersions(Xconcat_good_cur[:,feat_inds_curLFP],
                                            Xconcat_imputed[:,feat_inds_curLFP],
                                            class_labels_good, n_components_LDA,
                                            class_ind_to_check, revdict,
                                            calcName=lda_version_name,n_splits=n_splits)
                        res_cur['featis'] = feat_inds_curLFP
                        LDA_analysis_versions[lda_version_name] = res_cur
                        gc.collect()

                    pdrop,winning_chan = \
                        utsne.selBestLFP(results_cur, 'LDA', chnames_LFP=chnames_LFP)
                    _,winning_chan_spec_only = \
                       utsne.selBestLFP(results_cur, 'LDA', chnames_LFP=chnames_LFP, nperfs=1)
                    results_cur['best_LFP']['LDA'] = {'perf_drop':pdrop ,
                                            'winning_chan':winning_chan,
                                            'winning_chan_spec_only':winning_chan_spec_only }

                if exit_after == 'LDA_best_LFP':
                    print(f'Got exit_after={exit_after}, exiting!')
                    #if show_plots:
                    #    pdf.close()
                    sys.exit(0)

                if not use_ICA_for_classif:
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
                        LDA_analysis_versions[lda_version_name] = res_cur
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
                LDA_analysis_versions[lda_version_name] = res_cur
                gc.collect()


                if colinds_good_VIFsel is not None:
                    lda_version_name =  'after_VF_threshold'
                    res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,colinds_good_VIFsel],
                                        Xconcat_imputed[:,colinds_good_VIFsel],
                                        class_labels_good,
                                        n_components_LDA, class_ind_to_check,
                                        revdict, calcName=lda_version_name,n_splits=n_splits)
                    LDA_analysis_versions[lda_version_name] = res_cur
                    gc.collect()

                usage = getMemUsed();

                for vn in LDA_analysis_versions:
                    #LDA_analysis_versions
                    saveResToFolder(LDA_analysis_versions, vn, 'LDA_analysis_versions' )


                perfs_LDA_featsearch  = None
                best_inds_LDA = None
                if calc_selMinFeatSet:
                    #######################
                    ldaobj = LinearDiscriminantAnalysis(n_components=n_components_LDA)
                    ldaobj.fit(X_for_heavy, class_labels_for_heavy)
                    sortinds_LDA = np.argsort( np.max(np.abs(ldaobj.scalings_ ), axis=1) )
                    perfs_LDA_featsearch = utsne.selMinFeatSet(ldaobj, X_for_heavy,
                        class_labels_for_heavy, class_ind_to_check,sortinds_LDA,
                        drop_perf_pct = selMinFeatSet_drop_perf_pct,
                        conv_perf_pct = selMinFeatSet_conv_perf_pct,
                        stop_cond = ['sens','spec' ],
                        n_splits=n_splits, verbose=2, check_CV_perf=True, nfeats_step=1,
                        nsteps_report=2, stop_if_boring=False,
                        featnames=np.array(featnames_nice_for_fit)[feat_inds_for_heavy], max_nfeats=X_for_heavy.shape[1] )
                    #_, best_inds_LDA , _, _ =   perfs_LDA_featsearch[-1]
                    best_inds_LDA =   perfs_LDA_featsearch[-1]['featinds_present']
                    best_inds_LDA = feat_inds_for_heavy[best_inds_LDA]
                    gc.collect()


                    lda_version_name =  'strongest_features_LDA_selMinFeatSet'
                    res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,best_inds_LDA],
                                        Xconcat_imputed[:,best_inds_LDA],
                                        class_labels_good,
                                        n_components_LDA, class_ind_to_check,
                                        revdict, calcName=lda_version_name,n_splits=n_splits)
                    LDA_analysis_versions[lda_version_name] = res_cur
                    saveResToFolder(LDA_analysis_versions, lda_version_name, 'LDA_analysis_versions' )
                    gc.collect()

            usage = getMemUsed();
            ##################
            #lab_enc = preprocessing.LabelEncoder()
            ## just skipped class_labels_good
            #lab_enc.fit(class_labels_for_heavy)
            #class_labels_good_for_classif = lab_enc.transform(class_labels_for_heavy)



            do_XGB_cur =  do_XGB and not (int_types_key in gp.int_type_datset_rel and skip_XGB_aux_intervals  )
            clf_XGB = None
            best_inds_XGB = None
            perfs_XGB = None
            pca_XGBfeats = None
            if do_XGB_cur:
                results_cur['XGB_analysis_versions'] = {}
                # TODO: XGboost in future release wants set(class labels) to be
                # continousely increasing from zero, they don't want to use
                # sklearn version.. but I will anyway

                total_positive_examples = np.sum(class_labels_good == class_ind_to_check  )
                total_negative_examples = np.sum(class_labels_good != class_ind_to_check  )
                scale_pos_weight = total_negative_examples / total_positive_examples
                # 'scale_pos_weight' :scale_pos_weight  -- only for binary

                #gamma [default=0, alias: min_split_loss]
                # Minimum loss reduction required to make a further partition on
                # a leaf node of the tree. The larger gamma is, the more
                # conservative the algorithm will be.
                add_clf_creopts={ 'n_jobs':n_jobs,
                                 'use_label_encoder':False,
                                 'importance_type': 'gain',
                                 'max_depth': XGB_max_depth,
                                 'min_child_weight': XGB_min_child_weight,
                                 'eta':.3, 'subsample': 1 }
                tree_method = XGB_tree_method
                method_params = {'tree_method': tree_method}

                if (XGB_tree_method in ['hist', 'gpu_hist']) \
                        and allow_CUDA \
                        and len(gv.GPUs_list):
                    tree_method = 'gpu_hist'

                    method_params['gpu_id'] = gv.GPUs_list[0]

                add_clf_creopts.update(method_params)


                #'sample_weight':class_weights
                # fit the clf_XGB to get feature importances
                # set explicitly to avoid warning

                featnames_heavy = list(np.array(featnames_for_fit)[feat_inds_for_heavy] )
                featnames_nice_heavy = list( np.array(featnames_nice_for_fit)[feat_inds_for_heavy] )
                class_ind_to_check_lenc = lab_enc.transform([class_ind_to_check])[0]

                revdict_lenc = {}
                for k,kn in revdict.items():
                    revdict_lenc[ lab_enc.transform([k] )[0]  ] = kn
                ##########################3

                add_fitopts = { 'eval_metric':'mlogloss' }

                ###############
                X_cur = X_for_heavy
                y_cur = class_labels_good_for_classif
                print('Starting XGB on X.shape ', X_cur.shape)

                if XGB_tune_param:
                    add_clf_creopts_tuned, add_fitopts_tuned     = None,None
                    if load_XGB_params_auto and os.path.exists(fname_ML_full_intermed_light):
                        fe = np.load(fname_ML_full_intermed_light, allow_pickle=True)
                        resc_ = fe['results_light'][()]
                        rc = resc_['XGB_analysis_versions']['all_present_features']
                        add_clf_creopts_tuned  = rc.get('add_clf_creopts',None)
                        add_fitopts_tuned      = rc.get('add_fitopts',None)
                        if (add_clf_creopts_tuned is not None) and (add_fitopts_tuned is not None):
                            print('-------- Loaded XGB parameters from file!')

                    if (add_clf_creopts_tuned is None) or add_fitopts_tuned is None:
                        add_clf_creopts_CV = dict(add_clf_creopts.items())
                        if len(numpoints_per_class_id)  > 2:
                            add_clf_creopts_CV['objective'] = 'multi:softprob'
                            add_clf_creopts_CV['num_class'] = len(numpoints_per_class_id)
                            tune_metric = 'mlogloss'
                        else:
                            add_clf_creopts_CV['objective'] = 'binary:logistic'
                            tune_metric = 'logloss'

                        #search_grid['eta'] = np.array([.3, .2, .1, .05, .01, .005])
                        XGB_param_list_search_seq = [ ['max_depth','min_child_weight'],
                                                ['subsample', 'eta'] ]

                        print('Start XGB param tuning, XGB_param_list_search_seq=',XGB_param_list_search_seq)
                        add_clf_creopts_orig = add_clf_creopts
                        add_clf_creopts, best_params_list, cv_resutls_best_list,\
                            num_boost_round_best=\
                            utsne.gridSearchSeq(X_cur,y_cur, add_clf_creopts_CV,
                                            XGB_params_search_grid,
                                            XGB_param_list_search_seq,
                                            num_boost_round=100,
                                early_stopping_rounds=10, nfold=n_splits, seed=0,
                                            sel_num_boost_round=1,
                                                main_metric=tune_metric, printLog=1)
                    else:
                        add_clf_creopts = add_clf_creopts_tuned
                        add_fitopts = add_fitopts_tuned

                    #if num_boost_round_best is not None:
                    #    add_fitopts['n_estimators'] = num_boost_round_best

                    print('Hyperparam tuning gave us ',
                        add_clf_creopts,add_fitopts)
                if calc_selMinFeatSet:
                    clf_XGB = XGBClassifier(**add_clf_creopts)
                    clf_XGB.fit(X_cur, y_cur, **add_fitopts)
                    clf_XGB.get_booster().feature_names = featnames_nice_heavy

                    print('--- main XGB finished')
                    importance = clf_XGB.feature_importances_
                    sortinds = np.argsort( importance )
                    gc.collect()

                    # make sure increasing sort
                    assert np.min( np.diff(importance[sortinds] ) ) >= 0


                    usage = getMemUsed();

                    step_MFS_XGB = min(max_XGB_step_nfeats, max(5, X_cur.shape[1] // 20)  )
                    max_nfeats = max(X_cur.shape[1] // 2, 100)
                    perfs_XGB = utsne.selMinFeatSet(clf_XGB, X_cur, y_cur,
                        class_ind_to_check_lenc,
                        sortinds=sortinds,
                        n_splits=n_splits,
                        drop_perf_pct = selMinFeatSet_drop_perf_pct,
                        conv_perf_pct = selMinFeatSet_conv_perf_pct,
                        stop_cond = ['sens','spec' ],
                        add_fitopts=add_fitopts, add_clf_creopts=add_clf_creopts,
                        check_CV_perf=True, nfeats_step= step_MFS_XGB,
                        verbose=2, max_nfeats = max_nfeats,
                        ret_clf_obj=True,
                        featnames= featnames_nice_heavy )
                    gc.collect()

                    usage = getMemUsed();

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
                            if len(shfl) == 3:
                                _,_,perf_shuffled = shfl
                            else:
                                _,perf_shuffled = shfl
                            sens_sh,sepc_sh,F1_sh,confmat_sh = perf_shuffled
                            print('  shuffled: XGB CV perf on {} feats : sens {:.2f} spec {:.2f} F1 {:.2f}'.format(
                                len(inds_XGB), perf_shuffled[0], perf_shuffled[1],
                                perf_shuffled[2] ) )

                    last_perfobj_XGB = perfs_XGB[-1]
                    best_inds_XGB_among_heavy  =  last_perfobj_XGB['featinds_present']
                    best_inds_XGB = feat_inds_for_heavy[best_inds_XGB_among_heavy]

                    best_nice = list( np.array(featnames_nice_for_fit) [best_inds_XGB] )
                    n_XGB_feats_to_print = 20
                    print('XGB best feats (best {}, descending importance)={}'.
                        format(n_XGB_feats_to_print,best_nice[::-1][:n_XGB_feats_to_print] ) )

                    pca_XGBfeats = PCA(n_components=nPCA_comp )
                    pca_XGBfeats.fit( Xconcat_good_cur[:,best_inds_XGB],
                                     sample_weight=class_weights)
                    print('Min number of features found by XGB is {}, PCA on them gives {}'.
                            format( len(best_inds_XGB), pca_XGBfeats.n_components_) )

                    XGB_version_name = 'all_present_features'
                    rc = {'perf_dict':perfs_XGB[0] ,
                        'importances':perfs_XGB[0]['clf_objs'][0].feature_importances_}
                    results_cur['XGB_analysis_versions'][XGB_version_name] = rc
                    saveResToFolder(results_cur['XGB_analysis_versions'],
                                    XGB_version_name, 'XGB_analysis_versions' )

                    #if last_perfobj_XGB['stop_now']:
                    XGB_version_name = 'strongest_features_XGB_opinion'
                    rc = {'perf_dict':perfs_XGB[-1] ,
                        'importances':perfs_XGB[-1]['clf_objs'][0].feature_importances_}
                    results_cur['XGB_analysis_versions'][XGB_version_name] = rc
                    saveResToFolder(results_cur['XGB_analysis_versions'],
                                    XGB_version_name, 'XGB_analysis_versions' )
                else:
                    XGB_version_name = 'all_present_features'

                    clf_XGB = XGBClassifier(**add_clf_creopts)
                    clf_XGB.fit(X_cur, y_cur, **add_fitopts,
                                sample_weight=class_weights)
                    r0 = utsne.getPredPowersCV(clf_XGB,X_cur,y_cur,
                        class_ind_to_check_lenc, printLog = 0,
                        n_splits=n_splits, add_fitopts=add_fitopts,
                        add_clf_creopts=add_clf_creopts,
                        ret_clf_obj=False, seed=0)
                    rc = {'perf_dict':r0, 'importances':clf_XGB.feature_importances_}
                    rc['add_clf_creopts'] = add_clf_creopts
                    rc['add_fitopts'] = add_fitopts

                    perfstr = utsne.sprintfPerfs(r0['perf_aver'])
                    print(f'<!>  XGB {XGB_version_name} perfs are {perfstr}')
                    print( r0['confmat_aver'] * 100 )

                    # here we do NO want the main interval typ to be of
                    # 'dataset' kind
                    if int_types_key not in gp.int_type_datset_rel:
                        rc['across'] = {}
                        for label_group_name,group_labels in group_labels_dict.items():
                            if label_group_name not in label_groups_to_use:
                                continue

                            if len(set(group_labels)) == 1:
                                print(f'CV across label groups: Skipping {label_group_name}, because it is trivial')
                                rc['across'][label_group_name] = None
                                continue

                            ngroups = len( set(group_labels) )
                            r0_across = utsne.getPredPowersCV(clf_XGB,X_cur,y_cur,
                                class_ind_to_check_lenc, printLog = 0,
                                n_splits=ngroups, add_fitopts=add_fitopts,
                                add_clf_creopts=add_clf_creopts,
                                ret_clf_obj=False, seed=0, group_labels=group_labels[::subskip_fit])
                            rc['across'][label_group_name] = r0_across

                            perfstr = utsne.sprintfPerfs(r0_across['perf_aver'])
                            print(f'{label_group_name} label grouping gave perf {perfstr}')
                            print( r0_across['confmat_aver'] * 100 )

                    #r0_across_medcond = utsne.getPredPowersCV(clf_XGB,X_cur,y_cur,
                    #    class_ind_to_check_lenc, printLog = 0,
                    #    n_splits=n_splits, add_fitopts=add_fitopts,
                    #    add_clf_creopts=add_clf_creopts,
                    #    ret_clf_obj=False, seed=0, label_groups=label_groups_subj_medcond)

                    #r0_across_subj_medcond = utsne.getPredPowersCV(clf_XGB,X_cur,y_cur,
                    #    class_ind_to_check_lenc, printLog = 0,
                    #    n_splits=n_splits, add_fitopts=add_fitopts,
                    #    add_clf_creopts=add_clf_creopts,
                    #    ret_clf_obj=False, seed=0, label_groups=label_groups_medcond)


                    results_cur['XGB_analysis_versions'][XGB_version_name] = rc
                    saveResToFolder(results_cur['XGB_analysis_versions'],
                                    XGB_version_name, 'XGB_analysis_versions' )

                    gc.collect()


                if exit_after == 'XGB_main':
                    print(f'Got exit_after={exit_after}, exiting!')
                    #if show_plots:
                    #    pdf.close()
                    sys.exit(0)


                if 'XGB' in search_best_LFP and (not use_main_LFP_chan) \
                        and ('LFP' in data_modalities) and len(chnames_LFP) > 1:
                    for chn_LFP in chnames_LFP:

                        feat_inds_curLFP, feat_inds_except_curLFP = \
                            getFeatIndsRelToOnlyOneLFPchan(featnames,
                                chn=chn_LFP, chnames_LFP=chnames_LFP)

                        X_cur = Xconcat_good_cur[ :, feat_inds_except_curLFP ]
                        y_cur = class_labels_good_for_classif
                        dtrain = xgb.DMatrix(X_cur, label=y_cur)

                        XGB_version_name = 'all_present_features_but_{}'.format(chn_LFP)
                        print(f'Starting XGB {XGB_version_name} on X.shape = {X_cur.shape}')

                        add_clf_creopts_minus_curLFP, best_params_list, \
                            cv_resutls_best_list, num_boost_round_best =\
                            utsne.gridSearchSeq(X_cur,y_cur,  add_clf_creopts_CV,
                                            XGB_params_search_grid,
                                            XGB_param_list_search_seq,
                                            num_boost_round=100,
                                early_stopping_rounds=10, nfold=n_splits, seed=0,
                                                main_metric=tune_metric)


                        #add_fitopts_cur = dict(add_fitopts.items())
                        #if num_boost_round_best is not None:
                        #    add_fitopts_cur['n_estimators'] = num_boost_round_best

                        clf_XGB_ = XGBClassifier(**add_clf_creopts_minus_curLFP)
                        clf_XGB_.fit(X_cur, y_cur, **add_fitopts, sample_weight=class_weights)
                        r0 = utsne.getPredPowersCV(clf_XGB_,X_cur,y_cur,
                            class_ind_to_check_lenc, printLog = 0,
                            n_splits=n_splits, add_fitopts=add_fitopts,
                            add_clf_creopts=add_clf_creopts_minus_curLFP,
                            ret_clf_obj=True, seed=0)

                        rc = {'perf_dict':r0, 'importances':clf_XGB_.feature_importances_, 'featis':feat_inds_except_curLFP}
                        results_cur['XGB_analysis_versions'][XGB_version_name] = rc

                        saveResToFolder(results_cur['XGB_analysis_versions'],
                                        XGB_version_name, 'XGB_analysis_versions' )

                        ############################3

                        X_cur = Xconcat_good_cur[ :, feat_inds_curLFP ]
                        y_cur = class_labels_good_for_classif
                        dtrain = xgb.DMatrix(X_cur, label=y_cur)

                        XGB_version_name = 'all_present_features_only_{}'.format(chn_LFP)
                        print(f'Starting XGB {XGB_version_name} on X.shape = {X_cur.shape}')

                        add_clf_creopts_curLFP, best_params_list, \
                            cv_resutls_best_list, num_boost_round_best =\
                            utsne.gridSearchSeq(X_cur,y_cur, add_clf_creopts_CV,
                                            XGB_params_search_grid,
                                            XGB_param_list_search_seq,
                                            num_boost_round=100,
                                early_stopping_rounds=10, nfold=n_splits, seed=0,
                                                main_metric=tune_metric)

                        #add_fitopts_cur = dict(add_fitopts.items())
                        #if num_boost_round_best is not None:
                        #    add_fitopts_cur['n_estimators'] = num_boost_round_best


                        clf_XGB_ = XGBClassifier(**add_clf_creopts_curLFP)
                        clf_XGB_.fit(X_cur, y_cur, **add_fitopts,sample_weight=class_weights)
                        r0 = utsne.getPredPowersCV(clf_XGB_,X_cur,y_cur,
                            class_ind_to_check_lenc, printLog = 0,
                            n_splits=n_splits, add_fitopts=add_fitopts,
                            add_clf_creopts=add_clf_creopts_curLFP,
                            ret_clf_obj=True, seed=0)

                        rc = {'perf_dict':r0, 'importances':clf_XGB_.feature_importances_, 'featis':feat_inds_curLFP}
                        results_cur['XGB_analysis_versions'][XGB_version_name] = rc

                        saveResToFolder(results_cur['XGB_analysis_versions'],
                                        XGB_version_name, 'XGB_analysis_versions' )

                        gc.collect()

                    pdrop,winning_chan = \
                        utsne.selBestLFP(results_cur, 'XGB', chnames_LFP=chnames_LFP)
                    # select winning channel not caring about what happens with
                    # spec (it might rise)
                    _,winning_chan_spec_only = \
                       utsne.selBestLFP(results_cur, 'XGB', chnames_LFP=chnames_LFP, nperfs=1)
                    results_cur['best_LFP']['XGB'] = {'perf_drop':pdrop ,
                                            'winning_chan':winning_chan,
                                            'winning_chan_spec_only':winning_chan_spec_only }


                if exit_after == 'XGB_search_LFP':
                    print(f'Got exit_after={exit_after}, exiting!')
                    #if show_plots:
                    #    pdf.close()
                    sys.exit(0)


                #################
                from xgboost.core import XGBoostError

                featinds_good_boruta,featinds_ranking_boruta = None,None
                if calc_Boruta:
                    if calc_VIF:
                        fiboruta = colinds_good_VIFsel
                    else:
                        fiboruta = np.arange(Xconcat_good_cur.shape[1])
                    X_boruta = Xconcat_good_cur[:,fiboruta]
                    try:
                        featinds_good_boruta,featinds_ranking_boruta = \
                            utsne.selFeatsBoruta(X_boruta,
                                class_labels_good_for_classif,
                                verbose = 2,add_clf_creopts=None, n_jobs =
                                n_jobs, random_state=0)
                        print(f'boruta selected {len(featinds_good_boruta)} good indices of total {X_boruta.shape[1]}')
                        gc.collect()
                        results_cur['boruta'] = {}
                        results_cur['boruta']['featinds_good_boruta' ] = \
                            fiboruta[featinds_good_boruta]
                        results_cur['boruta']['featinds_ranking_boruta' ] = featinds_ranking_boruta
                    except XGBoostError as e:
                        print(f'boruta failed with error {e}, saving None')


                    saveResToFolder(results_cur, 'boruta')

                ##############3

                if colinds_good_VIFsel is not None:
                    XGB_version_name = 'after_VF_threshold'
                    X_cur = Xconcat_good_cur[ ::subskip_fit, colinds_good_VIFsel ]
                    y_cur = class_labels_good_for_classif

                    clf_XGB_ = XGBClassifier(**add_clf_creopts)
                    clf_XGB_.fit(X_cur, y_cur, **add_fitopts,sample_weight=class_weights)
                #res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,pca_derived_featinds],
                #                                    Xconcat_imputed[:,pca_derived_featinds],

                    r0 = utsne.getPredPowersCV(clf_XGB_,X_cur,y_cur,
                        class_ind_to_check_lenc, printLog = 0,
                        n_splits=n_splits, add_fitopts=add_fitopts,
                        add_clf_creopts=add_clf_creopts,
                        ret_clf_obj=False, seed=0)

                    rc = {'perf_dict':r0, 'importances':clf_XGB_.feature_importances_}
                    results_cur['XGB_analysis_versions'][XGB_version_name] = rc
                    saveResToFolder(results_cur['XGB_analysis_versions'],
                                    XGB_version_name, 'XGB_analysis_versions' )


                ####################

                if nonsyn_feat_inds is not None and best_inds_XGB is not None:
                    XGB_version_name = 'strongest_features_XGB_opinion_nosyn'
                    indlist = best_inds_XGB
                    C_subset = C[indlist,:][:,indlist]
                    nonsyn_feat_inds = pp.getNotSyn(C_subset,strong_correl_level)

                    X_cur = X_for_heavy[ :, nonsyn_feat_inds ]
                    y_cur = class_labels_good_for_classif

                    clf_XGB_ = XGBClassifier(**add_clf_creopts)
                    clf_XGB_.fit(X_cur, y_cur, **add_fitopts,sample_weight=class_weights)
                #res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,pca_derived_featinds],
                #                                    Xconcat_imputed[:,pca_derived_featinds],

                    r0 = utsne.getPredPowersCV(clf_XGB_,X_cur,y_cur,
                        class_ind_to_check_lenc, printLog = 0,
                        n_splits=n_splits, add_fitopts=add_fitopts,
                        add_clf_creopts=add_clf_creopts,
                        ret_clf_obj=False, seed=0)

                    rc = {'perf_dict':r0, 'importances':clf_XGB_.feature_importances_}
                    results_cur['XGB_analysis_versions'][XGB_version_name] = rc
                    saveResToFolder(results_cur['XGB_analysis_versions'],
                                    XGB_version_name, 'XGB_analysis_versions' )

                # look at different feature subsets based on q threshold from PCA
                if not use_ICA_for_classif:
                    for thri in range(len( feat_variance_q_thr )):
                        XGB_version_name = 'best_PCA-derived_features_{}'.\
                            format( feat_variance_q_thr[thri] )
                        pca_derived_featinds = pca_derived_featinds_perthr[thri]
                        if len(pca_derived_featinds) == 0:
                            continue

                        X_cur = X_for_heavy[ :, pca_derived_featinds ]
                        y_cur = class_labels_good_for_classif
                        clf_XGB_ = XGBClassifier(**add_clf_creopts)
                        clf_XGB_.fit(X_cur, y_cur, **add_fitopts,sample_weight=class_weights)
                    #res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,pca_derived_featinds],
                    #                                    Xconcat_imputed[:,pca_derived_featinds],

                        r0 = utsne.getPredPowersCV(clf_XGB_,X_cur,y_cur,
                            class_ind_to_check_lenc, printLog = 0,
                            n_splits=n_splits, add_fitopts=add_fitopts,
                            add_clf_creopts=add_clf_creopts,
                            ret_clf_obj=False, seed=0)

                        rc = {'perf_dict':r0, 'importances':clf_XGB_.feature_importances_}
                        results_cur['XGB_analysis_versions'][XGB_version_name] = rc
                        saveResToFolder(results_cur['XGB_analysis_versions'],
                                        XGB_version_name, 'XGB_analysis_versions' )

                    gc.collect()



            ##################

            #lda_red = LinearDiscriminantAnalysis(n_components=n_components_LDA )
            #lda_red.fit(ldapts_red, class_labels_good)
            #sens_red,spec_red,F1_red = utsne.getLDApredPower(lda_red,Xconcat[:,inds_important],
            #                                        class_labels, class_ind_to_check, printLog= 0)

            #print( ('--!! LDA on raw training data (grp {}, its {}) all vs {}:' +\
            #        '\n      sens={:.3f}; spec={:.3f};; sens_red={:.3f}; spec_red={:.3f}').
            #    format(grouping_key, int_types_key, class_to_check,sens,spec, sens_red,spec_red))


            # LDA on XGB-selected feats
            perf_red_XGB = None
            best_inds_XGB= None
            if best_inds_XGB is not None and do_XGB_cur:
                lda_version_name =  'strongest_features_XGB_opinion'
                res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,best_inds_XGB],
                                    Xconcat_imputed[:,best_inds_XGB],
                                    class_labels_good,
                                    n_components_LDA, class_ind_to_check,
                                    revdict, calcName=lda_version_name,n_splits=n_splits)
                LDA_analysis_versions[lda_version_name] = res_cur
                saveResToFolder(LDA_analysis_versions, lda_version_name, 'LDA_analysis_versions' )
                gc.collect()
                perf_red_XGB = LDA_analysis_versions['strongest_features_XGB_opinion']['fit_to_all_data']['perfs']


                if calc_Boruta and featinds_good_boruta is not None:
                    lda_version_name =  'boruta_selected'
                    res_cur = utsne.calcLDAVersions(
                        Xconcat_good_cur[:,featinds_good_boruta],
                        Xconcat_imputed[:,featinds_good_boruta],
                        class_labels_good, n_components_LDA,
                        class_ind_to_check, revdict,
                        calcName=lda_version_name,n_splits=n_splits)
                    LDA_analysis_versions[lda_version_name] = res_cur
                    saveResToFolder(LDA_analysis_versions, lda_version_name, 'LDA_analysis_versions' )
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

            #'ldaobj_avCV':LDA_analysis_versions['all_present_features']['CV_aver']['ldaobj'],
            #'ldaobjs_CV':LDA_analysis_versions['all_present_features']['CV']['ldaobjs'],
            # 'ldaobj':LDA_analysis_versions['all_present_features']['fit_to_all_data']['ldaobj'],
            d =   { 'labels_good':class_labels_good,
                            'class_labels':class_labels,
                           'highest_meaningful_thri':highest_meaningful_thri,
                           'feat_variance_q_thr':feat_variance_q_thr,
                           'feat_inds_for_heavy':feat_inds_for_heavy,
                            'PCA_derived_featinds_perthr':pca_derived_featinds_perthr,
                            'XGBobj':clf_XGB,
                            'strong_inds_XGB':best_inds_XGB,
                            'perfs_XGB': perfs_XGB,
                            'PCA_XGBfeats': pca_XGBfeats,
                            'perf_red_XGB':perf_red_XGB,
                           'MI_per_feati':MI_per_feati,
                           'revdict':revdict,
                           'counts':numpoints_per_class_id,
                           'class_ids_grouped':class_ids_grouped,
                   'class_labels_good_for_classif': class_labels_good_for_classif,
                   'pars':pars,
                   'info':ML_info,
                   'cmd':(opts,args) }
            if do_LDA:
                d.update({'LDA_analysis_versions':  LDA_analysis_versions,
                            'transformed_imputed':LDA_analysis_versions['all_present_features']['fit_to_all_data']['X_transformed'],
                            'transformed_imputed_CV':LDA_analysis_versions['all_present_features']['CV_aver']['X_transformed'],
                            'perf':LDA_analysis_versions['all_present_features']['fit_to_all_data']['perfs'],
                            'perf_red':LDA_analysis_versions['strongest_features_LDA_opinion']['fit_to_all_data']['perfs'],
                           'perfs_LDA_featsearch':perfs_LDA_featsearch,
                            'strong_inds_LDA':best_inds_LDA,
                            'inds_important':inds_important,
                            'strong_inds_pc':strong_inds_pc,
                            'strongest_inds_pc':strongest_inds_pc,
                          })


            results_cur.update(d)

            usage = getMemUsed();

            #out_name_templ = '_{}_grp{}-{}_{}ML_nr{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}__({},{})'
            #out_name = (out_name_templ ).\
            #    format(sources_type, src_file_grouping_ind, src_grouping,
            #        prefix, len(rawnames),
            #        n_channels, Xconcat_imputed.shape[1],
            #        pcapts.shape[1], skip, windowsz,grouping_key,int_types_key)
            #fname_ML_full_intermed = pjoin( gv.data_dir, output_subdir,
            #                                       '_{}{}.npz'.format(sind_join_str,out_name))


            results_cur['filename_full'] =  fname_ML_full_intermed

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
                savedict = {'results_cur':results_cur,
                        'label_encoder':lab_enc,
                         'Xconcat_good_cur':Xconcat_good_cur, 'class_labels_good':class_labels_good,
                        'selected_feat_inds_pri':dict(enumerate(selected_feat_inds_pri)),
                        'feature_names_filtered_pri':dict(enumerate(feature_names_pri)),
                         'featnames_nice':featnames_nice,
                          'bininds_good':bininds_noartif_nounlab,
                            'inds_not_neut': inds_not_neut,
                         'feat_info_pri':dict(enumerate(feat_info_pri)),
                        'rawtimes_pri':dict(enumerate(rawtimes_pri)),
                         'Xtimes_pri':dict(enumerate(Xtimes_pri)),
                        'wbd_pri':dict(enumerate(wbd_pri)),
                        'pcapts':pcapts, 'pcaobj':pca,
                            'icaobj':ica,
                        'X_imputed':Xconcat_imputed,
                          'pars':pars,
                            'roi_labels': roi_labels,
                            'ann_related': (anns, anns_pri, times_concat, dataset_bounds, wbd_merged)
                            }
                # using bininds_good and X_imputed one gets Xsubset_to_fit
                np.savez(fname_ML_full_intermed, **savedict  )
            else:
                print('Skipping saving intermediate result, before feat significance')

            featsel_per_method = {}

            if do_XGB_cur:
                featsel_per_method[ 'XGB_total_gain'] = {'scores': clf_XGB.feature_importances_ }

                for fsh in featsel_methods:
                    featis = None
                    featsel_info = {}
                    shap_values = None
                    explainer = None
                    if fsh == 'shapr':
                        for featsel_feat_subset_name in shapr_featsel_feats:
                            featis = np.arange(Xconcat_good_cur.shape[-1])
                            if featsel_feat_subset_name == 'all':
                                featis = np.arange(Xconcat_good_cur.shape[-1])
                            elif featsel_feat_subset_name == 'heavy':
                                featis = feat_inds_for_heavy

                        X_shapr = Xconcat_good_cur[::subskip_fit, featis]
                        y_shapr = class_labels_good_for_classif

                        featnames_ebm = np.array(featnames_nice_for_fit)[featis]
                        expl = utsne.shapr_proxy(X_shapr, y_train, colnames=None, groups = None,
                                        n_samples=200, n_batches=1, class_weights=None,
                                        add_clf_creopts={}, n_combinations=None)
                    elif fsh == 'SHAP_XGB':
                        import shap
                        #X = Ximp_per_raw[rncur][prefix]
                        #X_to_fit = X[gi]
                        #print(X_to_fit.shape)

                        #X_to_analyze_feat_sign = Xconcat_good_cur[:,best_inds_XGB]
                        X_to_analyze_feat_sign = X_for_heavy[:,best_inds_XGB_among_heavy]
                        featnames_sel = list( np.array(featnames_nice_for_fit)[best_inds_XGB]  )

                        #
                        nsamples = max(n_samples_SHAP, X_to_analyze_feat_sign.shape[0] // 10 )
                        Xsubset = shap.utils.sample(X_to_analyze_feat_sign, nsamples)

                        print('Start computing Shapley values using Xsubset with shape',Xsubset.shape)

                        import copy
                        add_clf_creopts_ = copy.deepcopy(add_clf_creopts)
                        # SHAP doest not work well if GPU for some reason
                        if XGB_tree_method == 'gpu_hist':
                            add_clf_creopts_['tree_method'] = 'cpu_hist'

                        clf_bestfeats = XGBClassifier(**add_clf_creopts_,sample_weight=class_weights)
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
                        X = Xconcat_good_cur[::subskip_fit]
                        y = class_labels_good_for_classif
                        some_computed = False

                        for featsel_feat_subset_name in XGB_featsel_feats:
                            featis = np.arange(X.shape[-1])
                            if featsel_feat_subset_name == 'all':
                                featis = np.arange(X.shape[-1])
                            elif featsel_feat_subset_name == 'heavy':
                                featis = feat_inds_for_heavy
                            elif featsel_feat_subset_name == 'nonsyn':
                                featis = results_cur['nonsyn_feat_inds']
                            elif featsel_feat_subset_name == 'best_LFP':
                                if featsel_only_best_LFP and 'LFP' in data_modalities \
                                        and 'XGB' in search_best_LFP and len(chnames_LFP) > 1:
                                    featis, featis_bad = getFeatIndsRelToOnlyOneLFPchan(featnames,
                                            chn=results_cur['best_LFP']['XGB']['winning_chan'],
                                            chnames_LFP= chnames_LFP)
                                else:
                                    print(f'{fsh} for best_LFP -- additional conditions not satisfied, skipping')
                                    continue
                            elif (featsel_feat_subset_name == 'VIFsel' or featsel_on_VIF):
                                if colinds_good_VIFsel is not None:
                                    featis = np.intersect1d(featis,colinds_good_VIFsel)
                                else:
                                    print(f'{fsh} for VIFsel -- additional conditions not satisfied, skipping')
                                    continue
                            else:
                                raise ValueError(f'{fsh} unknown feat subset name {featsel_feat_subset_name}')


                            print(f'Starting computing {fsh} for feat subset {featsel_feat_subset_name}')
                            #if featsel_only_best_LFP and 'LFP' in data_modalities and 'XGB' in search_best_LFP and len(chnames_LFP) > 1:
                            #    featis, featis_bad = getFeatIndsRelToOnlyOneLFPchan(featnames,
                            #            chn=results_cur['best_LFP']['XGB']['winning_chan'],
                            #                                            chnames_LFP=chnames_LFP)
                            #else:
                            #    featis = np.arange(X.shape[-1] )

                            if featsel_on_VIF and colinds_good_VIFsel is not None:
                                featis = np.intersect1d(featis,colinds_good_VIFsel)

                            X = X[:,featis]
                            dmat = xgb.DMatrix(X, y, feature_names = np.array(featnames_nice_for_fit)[featis])

                            #X = X_for_heavy

                            # TODO: perhaps I should select best hyperparameters above
                            # before doing this
                            clf_XGB2 = XGBClassifier(**add_clf_creopts)
                            clf_XGB2.fit(X, y, **add_fitopts,sample_weight=class_weights)

                            bst = clf_XGB2.get_booster()

                            if (XGB_tree_method in ['hist', 'gpu_hist']) \
                                    and allow_CUDA \
                                    and len(gv.GPUs_list):
                                bst.set_param({"predictor": "gpu_predictor"})
                            #TODO: perhaps I should try to predict not the entire training -- no, I want values for all points
                            shap_values = bst.predict(dmat, pred_contribs=True)
                            #shap_values.shape

                            #featsel_info['explainer'] = clf_XGB2
                            #featsel_info['scores'] = shap_values

                            info_cur = {}
                            info_cur['explainer'] = clf_XGB2
                            info_cur['scores'] = shap_values
                            info_cur['feature_indices_used'] = featis
                            featsel_info[featsel_feat_subset_name] = info_cur

                            gc.collect()
                            usage = getMemUsed();
                            some_computed = True
                        if not some_computed:
                            print(f'!Warning!: no XGB featsel computed, XGB_featsel_feats = {XGB_featsel_feats}')
                    elif fsh in [ 'interpret_EBM', 'interpret_DPEBM' ]:
                        import itertools
                        import interpret
                        if fsh == 'interpret_EBM':
                            from interpret.glassbox import ExplainableBoostingClassifier as EBM
                        else:
                            from interpret.privacy import DPExplainableBoostingClassifier as EBM

                        # since EBM only works for binary, I treat each pair of classes separately
                        for featsel_feat_subset_name in EBM_featsel_feats:
                            featis = np.arange(Xconcat_good_cur.shape[-1])
                            if featsel_feat_subset_name == 'all':
                                featis = np.arange(Xconcat_good_cur.shape[-1])
                            elif featsel_feat_subset_name == 'heavy':
                                featis = feat_inds_for_heavy
                            elif featsel_feat_subset_name == 'nonsyn':
                                featis = results_cur['nonsyn_feat_inds']
                            elif featsel_feat_subset_name == 'best_LFP':
                                if featsel_only_best_LFP and 'LFP' in data_modalities \
                                        and 'XGB' in search_best_LFP and len(chnames_LFP) > 1:
                                    featis, featis_bad = getFeatIndsRelToOnlyOneLFPchan(featnames,
                                            chn=results_cur['best_LFP']['XGB']['winning_chan'],
                                            chnames_LFP =chnames_LFP)
                                else:
                                    print(f'{fsh} for best_LFP -- additional conditions not satisfied, skipping')
                                    continue
                            elif (featsel_feat_subset_name == 'VIFsel' or featsel_on_VIF):
                                if colinds_good_VIFsel is not None:
                                    featis = np.intersect1d(featis,colinds_good_VIFsel)
                                else:
                                    print(f'{fsh} for VIFsel -- additional conditions not satisfied, skipping')
                                    continue
                            else:
                                raise ValueError(f'unknown feat subset name {featsel_feat_subset_name}')

                            print(f'Starting {fsh} for featsel_feat_subset_name = {featsel_feat_subset_name}')
                            featnames_ebm = np.array(featnames_nice_for_fit)[featis]

                            ebm_params = {}
                            ebm_params['n_jobs'] =n_jobs
                            #ebm_params['min_samples_leaf']=add_clf_creopts['min_child_weight']
                            ebm_params['random_state']=EBM_seed
                            # outer_bags=8
                            #  validation_size=0.15,
                            #  ebm_params['binning'] = 'quantile'
                            #  min_samples_leaf=2,
                            #  max_leaves=3,
                            # interactions=10,
                            # learning_rate=0.01

                            from hyperopt import hp
                            X_EBM = Xconcat_good_cur[::subskip_fit, featis]
                            y_EBM = class_labels_good_for_classif


                            if fsh == 'interpret_EBM':
                                ebm_params['min_samples_leaf']=3

                                params_space = { 'max_bins': hp.choice('max_bins',[64,128,256,384, 512]),
                                    'outer_bags':hp.choice('outer_bags',[2,4,8,12,16]),
                                    'learning_rate':hp.choice('learning_rate',[1e-3,5e-3,1e-2,2e-2]),
                                    'validation_size':hp.choice('validation_size',[0.10,0.15, 0.2, 0.3]),
                                    'min_samples_leaf':hp.choice('min_samples_leaf',[2,3,4,5,7]),
                                    'max_leaves':hp.choice('max_leaves',[2,3,5,7]),
                                    'binning': hp.choice('binning',['quantile','quantile_humanized', 'uniform'])
                                    }
                                if len(numpoints_per_class_id) > 2:
                                    nfeats = len(featnames_ebm)
                                    interaction_d = {'interactions': hp.quniform('interactions',0, (nfeats*(nfeats-1) // 2),1),
                                    'max_interaction_bins':hp.choice('max_interaction_bins', [16,32,64])
                                    }
                                    params_space.update(interaction_d)

                            elif fsh == 'interpret_DPEBM':
                                ebm_params['min_samples_leaf']=3

                                params_space = {  'max_bins': hp.choice('max_bins',[16,32,64,128,256]),
                                #quantile_humanized
                                    'outer_bags':hp.choice('outer_bags',[1,2,4,8]),
                                    'learning_rate':hp.choice('learning_rate',[1e-3,5e-3,1e-2,2e-2]),
                                    #'validation_size':hp.choice('validation_size',[0., 0.10, 0.2, 0.3]),
                                    'min_samples_leaf':hp.choice('min_samples_leaf',[2,3,4,5,7]),
                                    'max_leaves':hp.choice('max_leaves',[2,3,5,7]),
                                    'bin_budget_frac':hp.choice( 'bin_budget_frac', [0.05, 0.1, 0.2] ) ,
                                    'epsilon': hp.choice('epsilon', [0.75, 1, 1.25]),
                                    'delta': hp.choice('delta', [1e-6, 1e-5, 2e-5, 1e-4, 1e-3])
                                }

                                if len(set(y_EBM) ) != 2:
                                    print(f'Cannot do {fsh} for multiclass, skipping')
                                    continue

                            ebm_creopts = {'feature_names': featnames_ebm}
                            ebm_creopts.update(ebm_params)

                            if EBM_balancing == 'auto':
                                if X_EBM.shape[1] > EBM_balancing_numfeats_thr:
                                    EMB_balancing_cur = 'weighting'
                                else:
                                    EMB_balancing_cur = 'oversample'

                            info_cur = utsne.computeEBM(X_EBM,y_EBM,EBM,ebm_creopts,revdict_lenc,
                                    class_ind_to_check_lenc, class_labels_good_nm, revdict_nm,
                                    n_splits=n_splits,
                                    EBM_compute_pairwise=EBM_compute_pairwise,
                                    EBM_CV=EBM_CV, featnames_ebm=featnames_ebm,
                                    tune_params = EBM_tune_param, params_space=params_space,
                                                        max_evals=EBM_tune_max_evals)
                            info_cur['feature_indices_used'] = featis
                            featsel_info[featsel_feat_subset_name] = info_cur

                        usage = getMemUsed();
                    else:
                        raise ValueError(f'{fsh} not implemented')

                    # results_cur['featsel_per_method']['XGB_Shapley'][sbuset_name][scores]
                    # results_cur['featsel_per_method']['XGB_Shapley'][sbuset_name][info_per_cp][cp][scores]
                    # this one is not informative! left for compatibility
                    #featsel_info['feature_indices_used'] = featis
                    #if fsh ==  'interpret_DPEBM':
                    #    import pdb; pdb.set_trace()
                    featsel_per_method[fsh] = featsel_info
                    saveResToFolder(featsel_per_method, fsh,
                                    'featsel_per_method' )

                results_cur['featsel_per_method'] = featsel_per_method


                if calc_selMinFeatSet and selMinFeatSet_after_featsel is not None \
                        and selMinFeatSet_after_featsel in featsel_per_method:
                    X_cur = Xconcat_good_cur[::subskip_fit]
                    y_cur = class_labels_good_for_classif

                    scores = featsel_per_method[selMinFeatSet_after_featsel]['scores']

                    if scores.ndim == 2:
                        assert len( set(class_labels_good_for_classif) ) == 2

                    scores_per_class = utsne.getScoresPerClass(class_labels_good_for_classif, scores)
                    importance_fs = scores_per_class[class_ind_to_check_lenc]
                    sortinds_fs = np.argsort( importance_fs )

                    # this one uses all features, not just for heavy, because
                    # this is how I run XGB_Shapley
                    clf_XGB_fs = XGBClassifier(**add_clf_creopts)
                    clf_XGB_fs.fit(X_cur, y_cur, **add_fitopts,sample_weight=class_weights)
                    clf_XGB_fs.get_booster().feature_names = list( np.array(featnames_nice_for_fit) )

                    usage = getMemUsed();
                    max_nfeats = max(X_cur.shape[1] // 2, 100)
                    step_MFS_XGB = min(max_XGB_step_nfeats, max(5, X_cur.shape[1] // 20)  )
                    perfs_XGB_fs = utsne.selMinFeatSet(clf_XGB_fs, X_cur ,
                        y_cur, class_ind_to_check_lenc, sortinds_fs,
                        n_splits=n_splits,
                        drop_perf_pct = selMinFeatSet_drop_perf_pct,
                        conv_perf_pct = selMinFeatSet_conv_perf_pct,
                        stop_cond = ['sens','spec' ],
                        add_fitopts=add_fitopts, add_clf_creopts=add_clf_creopts,
                        check_CV_perf=True, nfeats_step= step_MFS_XGB,
                        verbose=2, max_nfeats = max_nfeats,
                        ret_clf_obj=True,
                        featnames=featnames_nice_for_fit)
                    gc.collect()
                    usage = getMemUsed();
                    results_cur['perfs_XGB_fs'] = perfs_XGB_fs
                    results_cur['clf_XGB_fs'] = clf_XGB_fs


                    XGB_version_name = 'strongest_features_XGB_fs_opinion'

                    rc = {'perf_dict':perfs_XGB_fs[-1],
                          'importances':perfs_XGB_fs[-1]['clf_objs'][0].feature_importances_}
                    results_cur['XGB_analysis_versions'][XGB_version_name] = rc


                    for perf_ind in perf_inds_to_print:
                        if perf_ind >= len(perfs_XGB_fs):
                            continue
                        smfs_output = perfs_XGB_fs[perf_ind]
                        inds_XGB = smfs_output['featinds_present']
                        perf_nocv = smfs_output['perf_nocv']
                        res_aver = smfs_output['perf_aver']


                        print('XGB CV perf on {} feats : sens {:.2f} spec {:.2f} F1 {:.2f}'.format(
                            len(inds_XGB), res_aver[0], res_aver[1], res_aver[2] ) )

                        shfl = smfs_output.get('fold_type_shuffled',None)
                        if shfl is not None:
                            if len(shfl) == 3:
                                _,_,perf_shuffled = shfl
                            else:
                                _,perf_shuffled = shfl
                            sens_sh,sepc_sh,F1_sh,confmat_sh = perf_shuffled
                            print('  shuffled: XGB CV perf on {} feats : sens {:.2f} spec {:.2f} F1 {:.2f}'.format(
                                len(inds_XGB), perf_shuffled[0], perf_shuffled[1],
                                perf_shuffled[2] ) )

                    best_inds_XGB_fs  =   perfs_XGB_fs[-1]['featinds_present']
                    #best_inds_XGB = feat_inds_for_heavy[best_inds_XGB_fs]
                    results_cur['best_inds_XGB_fs'] = best_inds_XGB_fs

                    ##################

                    if C is not None:
                        XGB_version_name = 'strongest_features_XGB_fs_opinion_nosyn'
                        indlist = best_inds_XGB_fs
                        C_subset = C[indlist,:][:,indlist]
                        nonsyn_feat_inds = pp.getNotSyn(C_subset,strong_correl_level)

                        X_cur = Xconcat_good_cur[ ::subskip_fit, nonsyn_feat_inds ]
                        y_cur = class_labels_good_for_classif

                        clf_XGB_ = XGBClassifier(**add_clf_creopts)
                        clf_XGB_.fit(X_cur, y_cur, **add_fitopts,sample_weight=class_weights)
                    #res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,pca_derived_featinds],
                    #                                    Xconcat_imputed[:,pca_derived_featinds],

                        r0 = utsne.getPredPowersCV(clf_XGB_,X_cur,y_cur,
                            class_ind_to_check_lenc, printLog = 0,
                            n_splits=n_splits, add_fitopts=add_fitopts,
                            add_clf_creopts=add_clf_creopts,
                            ret_clf_obj=False, seed=0)

                        rc = {'perf_dict':r0, 'importances':clf_XGB_.feature_importances_}
                        results_cur['XGB_analysis_versions'][XGB_version_name] = rc
                        gc.collect()


                    #############################
                    if calc_Boruta:
                        if featinds_good_boruta is not None:
                            X = Xconcat_good_cur[::subskip_fit,featinds_good_boruta]
                            #X = X_for_heavy
                            y = class_labels_good_for_classif
                            dmat = xgb.DMatrix(X, y,
                                feature_names = np.array(featnames_nice_for_fit)[featinds_good_boruta] )
                            vername  = 'XGB_Shapley_VF_boruta'
                        else:
                            assert colinds_good_VIFsel is not None
                            X = Xconcat_good_cur[::subskip_fit,colinds_good_VIFsel]
                            #X = X_for_heavy
                            y = class_labels_good_for_classif
                            dmat = xgb.DMatrix(X, y,
                                feature_names = np.array(featnames_nice_for_fit)[colinds_good_VIFsel] )
                            vername  = 'XGB_Shapley_VF'

                        # TODO: perhaps I should select best hyperparameters above
                        # before doing this
                        clf_XGB3 = XGBClassifier(**add_clf_creopts)
                        clf_XGB3.fit(X, y, **add_fitopts, sample_weight=class_weights)

                        bst = clf_XGB3.get_booster()

                        if (XGB_tree_method in ['hist', 'gpu_hist']) \
                                and allow_CUDA \
                                and len(gv.GPUs_list):
                            bst.set_param({"predictor": "gpu_predictor"})
                        #TODO: perhaps I should try to predict not the entire training
                        shap_values = bst.predict(dmat, pred_contribs=True)
                        #shap_values.shape

                        featsel_info = {}
                        featsel_info['explainer'] = clf_XGB3
                        featsel_info['scores'] = shap_values

                        results_cur['featsel_per_method'][vername] = \
                            featsel_info

                        saveResToFolder(results_cur['featsel_per_method'],
                                        vername, 'featsel_per_method' )
                        gc.collect()


                        #############################  run sel min feat on result
                        X_cur = X
                        y_cur = y

                        scores = shap_values

                        if scores.ndim == 2:
                            assert len( set(class_labels_good_for_classif) ) == 2

                        scores_per_class = utsne.getScoresPerClass(class_labels_good_for_classif, scores)
                        importance_fs = scores_per_class[class_ind_to_check_lenc]
                        sortinds_fs = np.argsort( importance_fs )

                        if calc_selMinFeatSet:
                            #clf_XGB_fs.get_booster().feature_names = None


                            max_nfeats = max(X_cur.shape[1] // 2, 100)
                            step_MFS_XGB = min(max_XGB_step_nfeats, max(5, X_cur.shape[1] // 20)  )
                            perfs_XGB_fs = utsne.selMinFeatSet(clf_XGB3, X_cur,
                                y_cur, class_ind_to_check_lenc, sortinds_fs,
                                n_splits=n_splits,
                                drop_perf_pct = selMinFeatSet_drop_perf_pct,
                                conv_perf_pct = selMinFeatSet_conv_perf_pct,
                                stop_cond = ['sens','spec' ],
                                add_fitopts=add_fitopts, add_clf_creopts=add_clf_creopts,
                                check_CV_perf=True, nfeats_step= step_MFS_XGB,
                                verbose=2, max_nfeats = max_nfeats,
                                ret_clf_obj=True,
                                featnames=featnames_nice_for_fit)
                            gc.collect()
                            usage = getMemUsed();
                            results_cur['perfs_XGB_fs_boruta'] = perfs_XGB_fs
                            results_cur['clf_XGB_fs_boruta'] = clf_XGB_fs

                            saveResToFolder(results_cur, 'perfs_XGB_fs_boruta' )
                            saveResToFolder(results_cur, 'clf_XGB_fs_boruta'   )
                            # in calced ver
                            #results_cur['perfs_XGB_fs'] = perfs_XGB_fs
                            #results_cur['clf_XGB_fs'] = clf_XGB_fs

                    ############################
                    if do_XGB_SHAP_twice:
                        # redo SHAP on selected best features

                        X = Xconcat_good_cur[::subskip_fit,best_inds_XGB_fs]
                        #X = X_for_heavy
                        y = class_labels_good_for_classif
                        dmat = xgb.DMatrix(X, y, feature_names = np.array(featnames_nice_for_fit)[best_inds_XGB_fs] )

                        # TODO: perhaps I should select best hyperparameters above
                        # before doing this
                        clf_XGB4 = XGBClassifier(**add_clf_creopts)
                        clf_XGB4.fit(X, y, **add_fitopts, sample_weight=class_weights)

                        bst = clf_XGB4.get_booster()

                        if (XGB_tree_method in ['hist', 'gpu_hist']) \
                                and allow_CUDA \
                                and len(gv.GPUs_list):
                            bst.set_param({"predictor": "gpu_predictor"})
                        #TODO: perhaps I should try to predict not the entire training
                        shap_values = bst.predict(dmat, pred_contribs=True)
                        #shap_values.shape

                        featsel_info = {}
                        featsel_info['explainer'] = clf_XGB4
                        featsel_info['scores'] = shap_values

                        results_cur['featsel_per_method']['XGB_Shapley2'] = \
                            featsel_info
                        gc.collect()

                    #####


                    lda_version_name =  'strongest_features_XGB_fs_opinion'
                    res_cur = utsne.calcLDAVersions(Xconcat_good_cur[:,best_inds_XGB_fs],
                                        Xconcat_imputed[:,best_inds_XGB_fs],
                                        class_labels_good,
                                        n_components_LDA, class_ind_to_check,
                                        revdict, calcName=lda_version_name,n_splits=n_splits)
                    LDA_analysis_versions[lda_version_name] = res_cur
                    saveResToFolder(LDA_analysis_versions,
                        lda_version_name, 'LDA_analysis_versions' )
                    gc.collect()

            if save_output:

                print('Saving ext intermediate result to {}'.format(fname_ML_full_intermed) )
                # updated results
                results_cur['class_labels_good'] = class_labels_good
                results_cur['pars'] = savedict['pars']
                savedict['results_cur'] =results_cur

                np.savez(fname_ML_full_intermed, **savedict )

                print('!!!!!!!!!!!!!  ',featsel_per_method.keys() )

                results_cur_cleaned = pp.removeLargeItems(results_cur)

                print('Saving LIGHT ext intermediate result to {}'.format(fname_ML_full_intermed_light) )
                np.savez(fname_ML_full_intermed_light,
                         results_light=results_cur_cleaned)
            else:
                print('Skipping saving intermediate result')

            mult_clf_results_pit[int_types_key] = results_cur
        mult_clf_results_pg[grouping_key] = mult_clf_results_pit
else:
    mult_clf_results_pg = None

    #lda = None
    #class_labels_good = None
    #X_LDA = None
    #sens = np.nan
    #spec = np.nan



###### Save result several groups/int_types together

if not single_fit_type_mode:
    # save PCA output separately
    for rawind,rawn in enumerate(rawnames):
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
        fname_PCA_full = pjoin( gv.data_dir, output_subdir, '{}{}.npz'.format(rawn,out_name))



        mult_clf_results_pg_cur = copy.deepcopy(mult_clf_results_pg)
        dataset_bounds_Xbins
        indst,indend = dataset_bounds_Xbins[rawind]
        sl = slice(indst,indend)
        print(sl, sl.stop - sl.start)
        assert (sl.stop-sl.start) == len(Xtimes_pri[rawind])
        assert (sl.stop-sl.start) == lens_pri[rawind]

        # we need to restrict the LDA output to the right index range
        #for grouping_key in groupings_to_use:
        for grouping_key in mult_clf_results_pg_cur:
            grouping = gv.groupings[grouping_key]

            mult_clf_results_pit = {}
            #for int_types_key in int_types_to_use:
            for int_types_key in mult_clf_results_pg_cur[grouping_key]:
                r = mult_clf_results_pg_cur[grouping_key][int_types_key]
                if r is not None:
                    r['transformed_imputed'] = r['transformed_imputed'][sl]
                    r['transformed_imputed_CV'] = r['transformed_imputed_CV'][sl]

                    trkey = 'X_transformed'
                    # crop everything
                    lda_analysis_vers =  r['LDA_analysis_versions']
                    for featset_name, anver_cur in lda_analysis_vers.items():
                        for fit_type_name, fit_cur in anver_cur.items():
                            if trkey in fit_cur:
                                fit_cur[trkey] = fit_cur[trkey][sl]
                            #else:
                            #    print('{} not in {}:{}, skipping '.format(trkey,featset_name,fit_type_name) )

        if save_output:
            mask = np.zeros( len(Xconcat), dtype=bool )
            mask[bininds_noartif_nounlab] = 1
            bininds_good_cur = np.where( mask[sl] )[0]
            #before I had 'bininds_noartif_nounlab' name in the file for what now I call 'selected_feat_inds'
            # here I am atually saving not all features names, but only the
            # filtered ones. Now bininds_noartif_nounlab is good bin inds (not feautre inds)
            # !! Xconcat_imputed util Jan 25 daytime was the intire array, not the [sl] one
            np.savez(fname_PCA_full, pcapts = pcapts[sl], pcaobj=pca,
                    X=X_pri[rawind], wbd=wbd_pri[rawind], bininds_good = bininds_good_cur,
                    feature_names_filtered = feature_names_pri[rawind] ,
                    selected_feat_inds = selected_feat_inds_pri[rawind],
                    info = ML_info, feat_info = feat_info_pri[rawind],
                    lda_output_pg = mult_clf_results_pg_cur, Xtimes=Xtimes_pri[rawind], argv=sys.argv,
                    X_imputed=Xconcat_imputed[sl] ,  rawtimes=rawtimes_pri[rawind],
                     bininds_noartif_nounlab = bininds_noartif_nounlab)
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

    if do_Classif and 'ldapoints' in plot_types:

        for int_types_key in int_types_to_use:
            for grouping_key in groupings_to_use:
                grouping = gv.groupings[grouping_key]
                r = mult_clf_results_pg[grouping_key][int_types_key]

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
