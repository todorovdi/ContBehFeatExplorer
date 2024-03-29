import os,sys
import mne
import utils  #my code
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mpr
import matplotlib as mpl
import time
import gc;
import getopt

from sklearn.preprocessing import RobustScaler
from os.path import join as pjoin

import utils_tSNE as utsne
import utils_genfeats as ugf
import utils_preproc as upre
import globvars as gv
from globvars import gp
from utils_genfeats import computeCorr

if sys.argv[0].find('ipykernel_launcher') < 0:
    mpl.use('Agg')

#############################################################
#######  Main params
exit_after = 'end'

use_LFP_HFO = 1
brain_side_to_use = 'body_move_side'
body_side_for_baseline_int = 'body_move_side'
#use_main_moveside = 1  # 0 both , -1 opposite
use_main_LFP_chan = 0
bands_only = 'crude'  #'fine'  or 'no' # here 'only' means 'instead of using all mini-freq bands output by TFR
#subsample_type = 'half'
subsample_type = 'prescribedSkip'  # or 'desiredNpts'
#desiredNpts = 7000
#desiredNpts = 4000
desiredNpts = 5000
prescribedSkip = 32
data_modalities = ['LFP', 'msrc']
#data_modalities = ['LFP']
#data_modalities = ['msrc']

#features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
#features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr']
#features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl']
features_to_use = gv.feat_types_all
#features_to_use = [ 'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
#assert 'bandcorrel' not in features_to_use, 'Not implemented yet'
#cross_types = [ ('LFP.*','.?src.*'), ('.*src_Cerebellum.*' , '.*motor-related.*' ) ]  # only couplings of LFP to sources
#cross_types = [ ('LFP.*','.?src.*'), ('.?src_.*' , '.?src.*' ) ]  # only couplings of LFP to sources
# LFP to sources and Cerebellum vs everything else

allow_CUDA = gv.CUDA_state == 'ok'

msrc_inds = np.arange(8,dtype=int)  #indices appearing in channel (sources) names, not channnel indices

use_LFP_to_LFP = 0
use_imag_coh = True

#corr_time_lagged = []
#corr_time_lagged = [0.5, 1] # in window size fractions


#################### Exec control params
sources_type = 'HirschPt2011'
#sources_type = 'parcel_aal'
#src_type_to_use  = 'center' # or mean_td
#src_type_to_use  = 'mean_td' # or mean_td
src_type_to_use  = 'parcel_ICA'     # or mean_td

# for 2nd order features
#roi_pairs = [ ('Cerebellum_L', 'notCerebellum_L'), ('Cerebellum_R', 'notCerebellum_L'),
#             ('Cerebellum_R', 'notCerebellum_R'), ('Cerebellum_L', 'notCerebellum_R') ]

only_load_data               = 0
use_preloaded_data           = 0
#
load_TFR                     = 2
load_CSD                     = 2
save_TFR                     = 1 #gp.hostname != gp.hostname_home
save_CSD                     = 1 #gp.hostname != gp.hostname_home
use_existing_TFR             = 1  # for DEBUG in jupyter only
load_feat                    = 0
save_feat                    = 1
do_Kalman                    = 1
do_Wiener                    = 1


load_TFR_max_age_h = 24
load_CSD_max_age_h = 24

##########################  ploting params
show_plots                   = 0
#
do_plot_raw_stats            = 1 * show_plots
do_plot_raw_psd              = 1 * show_plots
do_plot_raw_timecourse       = 1 * show_plots
#do_plot_feat_timecourse_full = 0 * show_plots
#do_plot_feat_stats_full      = 0 * show_plots
do_plot_Hjorth               = 1 * show_plots
do_plot_feat_timecourse      = 1 * show_plots
do_plot_feat_stats           = 1 * show_plots
do_plot_CSD                  = 0 * show_plots
do_plot_stat_scatter         = 1 * show_plots
do_plot_feat_stat_scatter    = 1 * show_plots
extend = 3  # for plotting, in seconds

fmax_raw_psd = 45
dpi = 200

#skip_div_TFR = 2
skip_div_TFR = 1

log_before_bandaver = False
spec_uselog = False
log_during_csd = True
# no need to do across datasets because multiplicative constants should pop out
# and I will normalize resulting features in the end anyway
normalize_TFR = "across_datasets"  # 'across_datasets', 'no' # needed to avoid too small numbers
recalc_stats_multi_band = True  # if false, I will try to load

if normalize_TFR != "no":
    assert not log_before_bandaver
    assert not spec_uselog

n_jobs = None
if n_jobs is None:
    n_jobs = max(1, mpr.cpu_count() - gp.n_free_cores )
elif n_jobs == -1:
    n_jobs = mpr.cpu_count()




##########################

crop_start = None
crop_end   = None

src_file_grouping_ind = 10  # motor-related_vs_CB_vs_rest
src_grouping = 0  # src_grouping is used to get info from the file
newchn_grouping_ind = 9 # output group number

scale_data_combine_type = 'medcond'
#baseline_int_type = 'notrem'
rbcorr_use_local_means = False
rbcorr_use_zero_mean = True  # filtered signals should have zero mean

input_subdir = ""
output_subdir = ""

save_bpcorr = 0
save_rbcorr = 0
load_bpcorr = 0
load_rbcorr = 0
rescale_feats = 0  # individual rescale in the end
prescale_data = 1
logH = 1
feat_stats_artif_handling = 'reject'

#coupling_types = ['self', 'motorlike_vs_motorlike', 'LFP_vs_all']
coupling_types = ['self', 'motorlike_vs_motorlike_within_hemi', 'LFP_vs_all']
DEBUG_shorten_couplings = 'no'   # use small set of channel coupling for faster computations
##############################

print('sys.argv is ',sys.argv)
effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

runstring_ind=-1000

params_cmd = {}
params_read = {}

helpstr = 'Usage example\nrun_genfeats.py --rawnames <rawname_naked> '
opts, args = getopt.getopt(effargv,"hr:",
        ["mods=","msrc_inds=","feat_types=","bands=","rawnames=",
            "show_plots=", "LFPchan=", "useHFO=",
         "plot_types=", "plot_only=", "sources_type=", "crop=" ,
         "src_grouping=", "src_grouping_fn=", "brain_side_to_use=",
         "body_side_for_baseline_int=",
         "load_TFR=", "save_TFR=", "save_CSD=", "load_CSD=", "use_existing_TFR=",
         "Kalman_smooth=", "Wiener_smooth=", "save_bpcorr=", "save_rbcorr=", "load_rbcorr=", "load_bpcorr=",
         "load_TFRCSD_max_age_h=", "load_only=", "input_subdir=", "output_subdir=",
         "rbcorr_use_local_means=", "scale_data_combine_type=", "stats_fn_prefix=",
         "param_file=", "coupling_types=", "use_preloaded_data=", "feat_stats_artif_handling=",
         "prescale_data=", "rescale_feats=", "allow_CUDA=", "n_jobs=", "save_feat=",
         "normalize_TFR=", "SLURM_job_id=", "DEBUG_shorten_couplings=",
        "exit_after=", "baseline_int_type=", "runstring_ind=" ])
print('opts is ',opts)
print('args is ',args)

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
    if opt == "msrc_inds":
        msrc_inds = [ int(indstr) for indstr in arg.split(',') ]
        msrc_inds = np.array(msrc_inds, dtype=int)
    elif opt == "crop":
        if len(arg) > 2:
            cr =  arg.split(',')
            crop_start = float(cr[0])
            crop_end = float(cr[1] )
    elif opt == "runstring_ind":
        runstring_ind = runstring_ind
    elif opt == "exit_after":
        exit_after = arg
    elif opt == "scale_data_combine_type":
        assert arg in gv.rawnames_combine_types + ['no_scaling']
        scale_data_combine_type = arg
    elif opt == "load_TFRCSD_max_age_h":
        load_TFR_max_age_h = int(arg)
        load_CSD_max_age_h = int(arg)
    elif opt == "load_only":
        only_load_data = int(arg)
    elif opt == "prescale_data":
        prescale_data = int(arg)
    elif opt == "rescale_feats":
        rescale_feats = int(arg)
    elif opt == "normalize_TFR":
        normalize_TFR = arg
    elif opt == "baseline_int_type":
        baseline_int_type = arg
    elif opt == "allow_CUDA":
        allow_CUDA = int(arg)
    elif opt == "use_preloaded_data":
        use_preloaded_data = int(arg)
    elif opt == "SLURM_job_id":
        SLURM_job_id = arg
    elif opt == "recalc_stats_multi_band":
        recalc_stats_multi_band = arg
    elif opt == "feat_stats_artif_handling":
        feat_stats_artif_handling = arg
    elif opt == "coupling_types":
        coupling_types = arg.split(',')
    elif opt == "brain_side_to_use":
        brain_side_to_use = arg
    elif opt == "body_side_for_baseline_int":
        body_side_for_baseline_int = arg
    elif opt == "stats_fn_prefix":
        stats_fn_prefix = arg
    elif opt == "DEBUG_shorten_couplings":
        DEBUG_shorten_couplings == arg
    elif opt == "src_grouping":
        src_grouping = int(arg)
    elif opt == "rbcorr_use_local_means":
        rbcorr_use_local_means = int(arg)
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
                import time
                time.sleep(5) # in seconds
                if not os.path.exists(subdir ):
                    print('Creating output subdir {}'.format(subdir) )
                    os.makedirs(subdir)
    elif opt == "src_grouping_fn":
        src_file_grouping_ind = int(arg)
    elif opt == "mods":
        data_modalities = arg.split(',')   #lfp of msrc
    elif opt == "feat_types":
        features_to_use_pre = arg.split(',')
        features_to_use = []
        for ftu in features_to_use_pre:
            assert ftu in gv.feat_types_all, ftu
            if ftu == 'Hjorth':
                features_to_use += ['H_mob', 'H_act', 'H_compl' ]
            else:
                features_to_use += [ftu]

    elif opt == "bands":
        bands_only = arg  #crude of fine
        assert bands_only in ['fine', 'crude']
    elif opt == "sources_type":
        if len(arg):
            sources_type = arg
    elif opt == 'rawnames':
        if len(arg) < 5:
            print('Empty raw name provided, exiting')
            sys.exit(1)
        rawnames = arg.split(',')  #lfp of msrc
        for rn in rawnames:
            assert len(rn) > 3
        if len(rawnames) > 1:
            print('Using {} datasets at once'.format(len(rawnames) ) )
        #rawname_ = arg
    elif opt == 'show_plots':
        show_plots = int(arg)
    #elif opt == 'side':
    #    if arg == 'both':
    #        use_main_moveside == 0
    #    elif arg == 'main':
    #        use_main_moveside == 1
    #    elif arg == 'other':
    #        use_main_moveside == -1
    #    elif arg in ['left','right']:
    #        raise ValueError('to be implemented')
    elif opt == 'LFPchan':
        if arg == 'main':
            use_main_LFP_chan = 1
        elif arg == 'other':
            use_main_LFP_chan = 0
        elif arg.find('LFP') >= 0:   # maybe I want to specify expliclity channel name
            raise ValueError('to be implemented')
    elif opt == 'useHFO':
        use_LFP_HFO = int(arg)
    elif opt == 'plot_only':
        if int(arg):
            load_feat = 1
            save_feat = 0
            show_plots = 1
        else:
            load_feat = 0
            save_feat = 1
    elif opt == 'save_feat':
        save_feat = int(arg)
    elif opt == 'plot_types':
        plot_types = arg.split(',')  #lfp of msrc
        if 'raw_stats' in plot_types:
            do_plot_raw_stats = 1
        else:
            do_plot_raw_stats = 0
        if 'raw_stats_scatter' in plot_types:
            do_plot_stat_scatter = 1
        else:
            do_plot_stat_scatter = 0
        if 'raw_timecourse' in plot_types:
            do_plot_raw_timecourse = 1
        else:
            do_plot_raw_timecourse = 0
        if 'raw_psd' in plot_types:
            do_plot_raw_psd = 1
        else:
            do_plot_raw_psd = 0
        if 'csd' in plot_types:
            do_plot_CSD = 1
        else:
            do_plot_CSD = 0
        if 'feat_stats' in plot_types:
            do_plot_feat_stats = 1
        else:
            do_plot_feat_stats = 0
        if 'feat_stats_scatter' in plot_types:
            do_plot_feat_stat_scatter = 1
        else:
            do_plot_feat_stat_scatter = 0
        if 'feat_timecourse' in plot_types:
            do_plot_feat_timecourse = 1
        else:
            do_plot_feat_timecourse = 0
    elif opt == 'load_TFR':
        load_TFR = int(arg)
    elif opt == 'load_CSD':
        load_CSD = int(arg)
    elif opt == 'save_TFR':
        save_TFR = int(arg)
    elif opt == 'save_CSD':
        save_CSD = int(arg)
    elif opt == 'save_rbcorr':
        save_rbcorr = int(arg)
    elif opt == 'save_bpcorr':
        save_bpcorr = int(arg)
    elif opt == 'load_rbcorr':
        load_rbcorr = int(arg)
    elif opt == 'load_bpcorr':
        load_bpcorr = int(arg)
    elif opt == 'prep_dat_prefix':
        prep_dat_prefix = arg
    elif opt == 'Kalman_smooth':
        do_Kalman = int(arg)
    elif opt == 'Wiener_smooth':
        do_Wiener = int(arg)
    elif opt == 'n_jobs':
        n_jobs = int(arg)
    elif opt == 'use_existing_TFR':
        use_existing_TFR = int(arg)
    elif opt.startswith('iniAdd'):
        print('skip ',opt)
    elif opt.startswith('code_ver'):
        print(f'code ver = {arg}')
    else:
        raise ValueError('Unknown option {},{}'.format(opt,arg) )


assert exit_after in ['load', 'prescale_data', 'TFR_and_CSD',
                        'bandAverage', 'Hjorth', 'end' ]

rawnstr = ','.join(rawnames)
#if src_file_grouping_ind == 9:
#    raise ValueError("AAA")
##############################
test_mode = int(rawnames[0][1:3]) > 50
crop_to_integer_second = False

print('run_genfeats: n_jobs = {}, MNE_USE_CUDA = {}'.\
      format(n_jobs, mne.utils.get_config('MNE_USE_CUDA')) )
if allow_CUDA and gv.CUDA_state == 'ok':
    #mne.utils.set_config('MNE_USE_CUDA', 'true')  # this hast be run once and NOT in parallel, otherwise config file gets currupted
    mne.cuda.init_cuda()
#############
motorlike_parcels = gp.parcel_groupings_post['Sensorimotor']
#motorlike_parcels = gp.areas_list_aal_my_guess)

assert set(coupling_types).issubset(set( gv.data_coupling_types_all ) )

cross_types =  []
if 'self' in coupling_types:
    cross_types += [ ('msrc_self','msrc_self'), ('LFP_self','LFP_self') ]
# couples both within and across hemispheres
if 'LFP_vs_all' in coupling_types:
    cross_types += [ ('LFP.*','.?src.*') ]
if 'CB_vs_all' in coupling_types:
    cross_types += [ ('.?src_((?!Cerebellum).)+', '.?src_Cerebellum.*') ]
    exclude_CB_interactions = 1  # makes sense if we already included CB vs all
else:
    exclude_CB_interactions = 0

if 'motorlike_vs_motorlike' in coupling_types:
    for lbli in range(len(motorlike_parcels)):
        for lblj in range(lbli+1, len(motorlike_parcels)):
            lab1 = motorlike_parcels[lbli]
            lab2 = motorlike_parcels[lblj]
            if exclude_CB_interactions and (lab1.find('Cerebellum') >= 0\
                    or lab2.find('Cerebellum') >= 0):
                continue
            #side = ?
            templ1 = '.?src_{}.*'.format(lab1)
            templ2 = '.?src_{}.*'.format(lab2)
            cross_types += [  (templ1 ,templ2 )  ]
            #for sides

# couples ONLY WITHIN hemispheres
if 'LFP_vs_all_within_hemi' in coupling_types:
    cross_types += [ ('LFPL*','.?srcL*') ]
    cross_types += [ ('LFPR*','.?srcR*') ]
if 'CB_vs_all_within_hemi' in coupling_types:
    raise ValueError('not implemented')
else:
    exclude_CB_interactions = 1


if 'motorlike_vs_motorlike_within_hemi' in coupling_types:
    for sidelet in ['L','R']:
        for lbli in range(len(motorlike_parcels)):
            for lblj in range(lbli+1, len(motorlike_parcels) ):
                lab1 = motorlike_parcels[lbli]
                lab2 = motorlike_parcels[lblj]
                if exclude_CB_interactions and (lab1.find('Cerebellum') >= 0\
                        or lab2.find('Cerebellum') >= 0):
                    continue
                #side = ?
                templ1 = '.?src_{}*'.format(lab1 + sidelet)
                templ2 = '.?src_{}*'.format(lab2 + sidelet)
                cross_types += [  (templ1 ,templ2 )  ]
                #for sides


#ann_MEGartif_prefix_to_use = '_ann_MEGartif'
ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt'

mods_to_load = ['LFP', 'src', 'EMG']
#mods_to_load = ['LFP', 'src', 'EMG', 'SSS','resample', 'FTraw']
#mods_to_load = ['LFP', 'src', 'EMG', 'resample', 'afterICA']
#mods_to_load = ['src', 'FTraw']
if use_LFP_HFO:
    mods_to_load += ['LFP_hires']

# these are just to play and plot because the side can change from patient to
# patient


move_sides = []
tremor_sides = []
for rn in rawnames:
    subj_cur,medcond_cur,task_cur  = utils.getParamsFromRawname(rn)
    mainmoveside_cur = gv.gen_subj_info[subj_cur].get('move_side',None)
    maintremside_cur = gv.gen_subj_info[subj_cur].get('tremor_side',None)
    mainLFPchan_cur  = gv.gen_subj_info[subj_cur]['lfpchan_used_in_paper']
    tremfreq_Jan_cur = gv.gen_subj_info[subj_cur]['tremfreq']

    move_sides  += [ mainmoveside_cur ]
    tremor_sides+= [ maintremside_cur ]


def_main_body_side = 'left'

force_consistent_main_sides = 1 # consistent across datasets, even not loaded ones

if brain_side_to_use == 'body_move_side':
    if len(set(move_sides) ) == 1 and move_sides[0] is not None:
        main_side = move_sides[0]
        new_main_body_side = main_side
    else:
        main_side = 'undef'
        print('Setting new main side to be ',def_main_body_side)
        new_main_body_side = def_main_body_side
elif brain_side_to_use == 'body_tremor_side':
    if len(set(tremor_sides) ) == 1:
        main_side = tremor_sides[0]
        new_main_body_side = main_side
    else:
        main_side = 'undef'
        new_main_body_side = def_main_body_side
        print('Setting new main side to be ',def_main_body_side)
elif brain_side_to_use == 'both':
    main_side = 'undef'
    new_main_body_side = 'both'
elif brain_side_to_use in ['left','right']:  # forcibly
    main_side     = utils.getOppositeSideStr(brain_side_to_use)
    new_main_body_side = main_side
else:
    main_side = 'undef'
    raise ValueError('wrong side_to_use')

if force_consistent_main_sides and brain_side_to_use != 'both':
    new_main_body_side = def_main_body_side


#####

n_jobs_tfr = n_jobs  # CUDA not allowed :(
n_jobs_flt = n_jobs

###################################################

if use_preloaded_data:
    print('DEBUG: USE PRELOADED DATA')
else:
    anndict_per_intcat_per_rawn = {}
    print('Start loading data')
    dat_pri                         = [0]*len(rawnames)
    dat_lfp_hires_pri               = [0]*len(rawnames)
    times_pri                       = [0]*len(rawnames)
    times_hires_pri                 = [0]*len(rawnames)
    subfeature_order_pri            = [0]*len(rawnames)
    subfeature_order_lfp_hires_pri  = [0]*len(rawnames)
    extdat_pri                      = [0]*len(rawnames)
    ivalis_pri                      = [0]*len(rawnames)
    rec_info_pri                    = [0]*len(rawnames)
    anndict_per_intcat_per_rawn    = {}
    fname_dat_full_pri = [0]*len(rawnames)

    aux_info_perraw = {}
    for rawi,rawn in enumerate(rawnames):
        fname = utils.genPrepDatFn(rawn, new_main_body_side, data_modalities,
                                    use_main_LFP_chan, src_file_grouping_ind,
                                    src_grouping, brain_side_to_use, prep_dat_prefix)
        fname_dat_full = pjoin(gv.data_dir, input_subdir, fname)
        f = np.load(fname_dat_full, allow_pickle=True)
        fname_dat_full_pri[rawi] = fname_dat_full
        print('Loading data from ',fname_dat_full)
        # for some reason if I don't do it explicitly, it has int64 type which
        # offends MNE
        sfreq =         int( f['sfreq'] )
        sfreq_hires =   int( f['sfreq_hires'] )
        rec_info_pri[rawi] = dict( f['rec_info'][()] )

        ivalis_pri[rawi] = f['ivalis'][()]

        dat_pri[rawi] = f['dat']
        dat_lfp_hires_pri[rawi] = f['dat_lfp_hires']
        extdat_pri[rawi] = f['extdat']
        # for some reason pickling of ordered dicts does not work well
        #anns_pri[rawi] = f['anns']
        times_pri[rawi] = f['times']
        times_hires_pri[rawi] = f['times_hires']
        subfeature_order_pri[rawi] = list( f['subfeature_order_pri'] )
        subfeature_order_lfp_hires_pri[rawi] = list ( f['subfeature_order_lfp_hires_pri'] )
        aux_info_perraw[rawn] = f['aux_info'][()]

        if 'anndict_per_intcat' in f:
            anndict_per_intcat_per_rawn[rawn] = f['anndict_per_intcat'][()]
        else:
            print('anndict_per_intcat_per_rawn not found, recollecting')
            anndict_per_intcat = upre.collectAllMarkedIntervals( rawn, f['times'],
                new_main_body_side, aux_info_perraw[rn]['side_switched'] ,
                sfreq=sfreq,
                ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt',
                printLog=False, allow_missing_files=False,
                remove_nonmain_artif= brain_side_to_use != 'both' )
            anndict_per_intcat_per_rawn[rawn] = anndict_per_intcat

        assert set(data_modalities) == set(f['data_modalities'])
        #fname = '{}_'.format(rawn) + fn_suffix_dat

print('Convert annotations to bin dicts')
bindict_per_rawn = {}
bindict_hires_per_rawn = {}
for rawi,rawn in enumerate(rawnames):
    anndict_per_intcat = anndict_per_intcat_per_rawn[rawn]
    # here side is not important
    times_cur = times_pri[ rawi ]
    bindict_per_rawn[rawn] = upre.markedIntervals2Bins(anndict_per_intcat,times_cur,sfreq)

    times_hires_cur = times_hires_pri[rawi]
    bindict_hires_per_rawn[rawn] = upre.markedIntervals2Bins(anndict_per_intcat,times_hires_cur,sfreq_hires)

fname_stats = utils.genStatsFn(None, new_main_body_side, data_modalities,
                                use_main_LFP_chan, src_file_grouping_ind,
                                src_grouping, brain_side_to_use, stats_fn_prefix )
fname_stats_full = pjoin( gv.data_dir, input_subdir, fname_stats)
print('Load stats from ',fname_stats_full)

f = np.load(fname_stats_full, allow_pickle=True)
rawnames_stats =  f['rawnames']
assert set(rawnames).issubset(rawnames_stats)
stats_per_ct =  f['stats_per_ct'][()]
stats_HFO_per_ct = f['stats_HFO_per_ct'][()]

if (not recalc_stats_multi_band) and ( 'rbcorr' in features_to_use or 'bpcorr' in features_to_use ):
    fname_stats_multi_band = utils.genStatsMultiBandFn(None, new_main_body_side, data_modalities,
                                    use_main_LFP_chan, src_file_grouping_ind,
                                    src_grouping, bands_only, brain_side_to_use, stats_fn_prefix )
    fname_stats_full = pjoin( gv.data_dir, input_subdir, fname_stats_multi_band)
    print('Load multi band stats from ',fname_stats_full)
    assert os.path.exists(fname_stats_full)

#rec_info_pri = []
#for rawname_ in rawnames:
#    src_rec_info_fn_full = utils.genRecInfoFn(rawname_,sources_type,
#                                         src_file_grouping_ind,
#                                         input_subdir)
#    #src_rec_info_fn_full = pjoin(gv.data_dir, input_subdir,
#    #                                    src_rec_info_fn)
#    print('Load rec_info from ',src_rec_info_fn_full)
#
#    if input_subdir != output_subdir:
#        #src_rec_info_fn_full2 = pjoin(gv.data_dir, output_subdir,
#        #                                    src_rec_info_fn)
#        src_rec_info_fn_full2 = utils.genRecInfoFn(rawname_,sources_type,
#                                         src_file_grouping_ind,
#                                         output_subdir)
#
#        import shutil
#        shutil.copyfile(src_rec_info_fn_full,src_rec_info_fn_full2)
#        print("run_genfeats, copy src_rec_info {} to {}".format( src_rec_info_fn_full,src_rec_info_fn_full2)  )
#    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
#    rec_info_pri += [rec_info]

################
rec_info = rec_info_pri[0]
roi_labels = rec_info['label_groups_dict']     # dict of (orderd) lists with keys from srcgroups_key_order
srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]  # this is usually just one item

# it does not really make sense if I don't work with them together anyway
# probably rescaling features in the very end makes more sense
#if do_rescale_raws:
#    upre.rescaleRaws(raws_permod_both_sides, mod='LFP', combine_within_medcond=True)  # rescales LFP in place
#
#    upre.rescaleRaws(raws_permod_both_sides, mod='src',combine_within_medcond=True,
#                    roi_labels=roi_labels, srcgrouping_names_sorted=srcgrouping_names_sorted)
#    gc.collect()

####################  data processing params
windowsz        =  1 * sfreq
windowsz_hires  =  1 * sfreq_hires

min_freq = 3
#freq_step = 1
#max_freq = 100
#freqs = np.arange(min_freq,max_freq,freq_step)
m = windowsz / sfreq
freqs,n_cycles, Ws, windowsz_max = utils.prepFreqs(min_freq = min_freq, max_freq = 90,
                                                   frmults=[2*m,m,m], sfreq=sfreq )
mh = windowsz_hires / sfreq_hires
freqs_inc_HFO,n_cycles_inc_HFO, Ws_hires, windowsz_hires_max = \
    utils.prepFreqs(min_freq = min_freq, max_freq = 400, frmults=[2*mh,mh,mh],
                    sfreq=sfreq_hires)

def isIntDiv(n1,n2):
    return abs(n1/n2 - n1//n2) < 1e-10


debug_plot_ctr = 0

#assert windowsz == windowsz_
#assert windowsz_hires == windowsz_hires_
#wsz_ = np.max( [ np.max( [len(w) for w in ww] ) for ww in Ws ] ) # it should be the same but if I want to use later larger windows in low freq..

# because my definition of wavelets for low freq has larger window
nedgeBins       = int( windowsz_max  )
nedgeBins_hires = int( windowsz_hires_max )
# (longest window width (right - left) = 0.93s, shortest = 0.233, ~ 1/4 of a sec
# so sampling at 1/8 of a sec is safe

# I want 1 sec window sz
#cf =  windowsz/ ( 5/(2*np.pi) * sfreq  )
#freq2cycles_mult = cf  # 1.2566370614359172
#print('cf= ',cf)

percentileOffset = 25

################################

if show_plots:
    fig_subdir =pjoin(gv.dir_fig, output_subdir )
    if not os.path.exists(fig_subdir):
        os.makedirs(fig_subdir)
    fig_fname = pjoin(fig_subdir,
                             '{}_feat_plots{}_side{}_LFP{}_{}_nmod{}_nfeattp{}.pdf'.format(
                                 rawnstr,show_plots, brain_side_to_use == 'body_move_side',
                                 int(use_main_LFP_chan), bands_only,
                                 len(data_modalities),len(features_to_use)   ))
    from matplotlib.backends.backend_pdf import PdfPages
    pdf= PdfPages(fig_fname  )

#############################################################

# only for plotting
ms_letter = main_side[0].upper()
subj,medcond,task  = utils.getParamsFromRawname(rawnames[0])
int_names = ['{}_{}'.format(task,ms_letter), 'trem_{}'.format(ms_letter), 'notrem_{}'.format(ms_letter)]

###########################################################

fbands = gv.fbands

if bands_only == 'fine':
    fband_names = gv.fband_names_fine
else:
    fband_names = gv.fband_names_crude

if bands_only == 'fine':
    fband_names_inc_HFO = gv.fband_names_fine_inc_HFO
else:
    fband_names_inc_HFO = gv.fband_names_crude_inc_HFO
    fband_names_HFO = fband_names_inc_HFO[len(fband_names):]  # that HFO names go after
#{'tremor': [3,10], 'low_beta':[11,22], 'high_beta':[22,30],
#           'low_gamma':[30,60], 'high_gamma':[60,90],
#          'HFO1':[91,200], 'HFO2':[200,300], 'HFO3':[300,400],
#          'beta':[15,30],   'gamma':[30,100], 'HFO':[91,400]}


#fband_names_crude = ['tremor', 'beta', 'gamma']
#fband_names_fine = ['tremor', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma' ]
#fband_names_HFO_crude = ['HFO']
#fband_names_HFO_fine =  ['HFO1', 'HFO2', 'HFO3']
#fband_names_HFO_all = fband_names_HFO_crude + fband_names_HFO_fine
#fband_names_crude_inc_HFO = fband_names_crude + fband_names_HFO_crude
#fband_names_fine_inc_HFO = fband_names_fine + fband_names_HFO_fine

#######################
if subsample_type == 'half':
    skip = (1 * sfreq) // 2
elif subsample_type == 'desiredNpts':
    ntimebins = None
    skip = ntimebins // desiredNpts
elif subsample_type == 'prescribedSkip':
    skip = prescribedSkip
skip_hires = sfreq_hires //   (sfreq  // skip)

assert isIntDiv(sfreq,skip)


# the lower the smoother
#Tp_Kalman = 1 * sfreq
Tp_Kalman = 0.5 * sfreq/skip
#Tp_Kalman = 1 * sfreq/skip

#############################################################



## the output is dat only from the selected hemisphere
#dat_pri, dat_lfp_hires_pri, extdat_pri, anns_pri, times_pri,\
#times_hires_pri, subfeature_order_pri, subfeature_order_lfp_hires_pri, aux_info_perraw = \
#    ugf.collectDataFromMultiRaws(rawnames, raws_permod_both_sides, sources_type,
#                             src_file_grouping_ind, src_grouping, use_main_LFP_chan,
#                             side_to_use, new_main_body_side, data_modalities,
#                             crop_start,crop_end,msrc_inds, rec_info_pri)

main_sides_pri = [ aux_info_perraw[rn]['main_body_side'] for rn in rawnames]
side_switched_pri = [ aux_info_perraw[rn]['side_switched'] for rn in rawnames]

subfeature_order = list( subfeature_order_pri[0] )
subfeature_order_lfp_hires = list( subfeature_order_lfp_hires_pri[0] )

# we cannot work if we have different LFP channel counts. But we can if we have
# different sources counts (as long as parcel numbers are the same)
for sfo in subfeature_order_lfp_hires_pri:
    a,b = ( set(subfeature_order_lfp_hires), set(sfo) )
    assert a == b, (a-b, b-a)


extend_test = False
if test_mode and extend_test:
    print(['test mode: Extnding dat'])
    for rawind in range(len(dat_pri)):
        datcur = dat_pri[rawind]
        dat_pri[rawind] = np.hstack( [np.zeros( (datcur.shape[0], 2048) ) , datcur] )
        dat_lfp_hires_pri[rawind] = np.hstack( [np.zeros( (dat_lfp_hires_pri[rawind].shape[0], 2048) ) , datcur] )

n_channels_pri = [ datcur.shape[0] for datcur in dat_pri ];
# I don't want to prohibit it at this moment beacuse it can happen if I have
# subjects with different main sides
#assert len(set(n_channels_pri) ) == 1, set(n_channels_pri)
n_channels_str = ','.join(map(str, n_channels_pri) )

#for timescur in times_pri:
#    l = len(timescur)
#    # otherwise we have some issues in windows boundaries for Hjorth, probably
#    # due to the way numpy strides length get computed
#    assert isIntDiv(l,windowsz) , l
#    assert isIntDiv(l,skip)     , l


# we actually don't need several
#roi_labels_pri = [ rec_info_['label_groups_dict'][()] for rec_info_ in rec_info_pri]      # dict of (orderd) lists
#roi_labels = roi_labels_pri[0]
rec_info = rec_info_pri[0]
roi_labels = rec_info['label_groups_dict'][()]
# order of grouping names, usually there is just one element "all_raw" meaning
# that I don't do any grouping of sources to perform post source reconstruction
# processing (like PCA, ICA, mean, etc)
srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]
assert set(roi_labels.keys() ) == set(srcgrouping_names_sorted )
assert len(roi_labels) == 1, 'several groupings in single run -- not implmemented'
# assuming we have only one grouping present
#roi_labels_cur = roi_labels[srcgrouping_names_sorted[src_grouping ]  ]

########### merge interval dictionaries  (not used now)
#times_pri_shifted = [times_pri[0] ]
#ivalis = ivalis_pri[0]
#tshift = 0
#for rawind in range(1, len(rawnames) ):
#    ivalis_cur = ivalis_pri[rawind]
#    if rawind > 0:
#        tshift += len(times_pri[rawind-1] )/ sfreq
#    times_pri_shifted += [ tshift + times_pri[rawind]  ]
#    for it in ivalis_cur:
#        upd = [ (s+tshift,e+tshift,it) for s,e,it in ivalis_cur[it]  ]
#        if it in ivalis:
#            ivalis[it] += upd
#        else:
#            ivalis[it] = upd
#
#times = np.hstack(times_pri_shifted)
#dft = np.diff(times)
#dftmn,dftmx = np.min(dft),np.max(dft)
#assert ( abs(dftmn - 1/sfreq) < 1e-10 ) and ( abs(dftmx - 1/sfreq) < 1e-10 )
#
#print('Total raw data length for {} datasets is {} bins (={}s)'.format(len(rawnames), len(times),
#                                                                       len(times) // sfreq ) )

#########################################################
################# scale raw data
#########################################################

# here we plot even if we don't actually rescale
if show_plots and do_plot_stat_scatter and len(set( n_channels_pri ) ) == 1 :
    main_side_let = new_main_body_side[0].upper()
    artif_handling_ = 'reject'

    dat_T_pri = [0]*len(dat_pri)
    for dati in range(len(dat_pri) ):
        dat_T_pri[dati] = dat_pri[dati].T

    int_types_to_stat = [it + '_{}'.format(main_side_let) for it in gp.int_types_basic]
    upre.plotFeatStatsScatter(rawnames,dat_T_pri, int_types_to_stat,
                        subfeature_order_pri,sfreq,
                        times_pri,side_switched_pri, wbd_pri=None,
                                save_fig=False, separate_by='mod', artif_handling=artif_handling_ )
    plt.suptitle('Stats of nonHFO data before rescaling')
    pdf.savefig()
    plt.close()
    gc.collect()

# WARNING: this will not work well if we work on more than one feature files at
# the same time and they have inconsistent sides!!!
#assert len(dat_pri) == 1
assert len(set(side_switched_pri) ) == 1
#if body_side_for_baseline_int == 'body_move_side':
#    main_side_let = move_sides[0][0].upper()
#elif body_side_for_baseline_int == 'body_tremor_side':
#    main_side_let = tremor_sides[0][0].upper()
#elif body_side_for_baseline_int in ['left','right']:
#    main_side_let = body_side_for_baseline_int[0].upper()
#
#if baseline_int_type != 'entire':
#    baseline_int = '{}_{}'.format(baseline_int_type, main_side_let )
#else:
#    baseline_int = baseline_int_type

baseline_int = upre.getBaselineInt(rawnames[0], body_side_for_baseline_int, baseline_int_type)
main_side_let = baseline_int[-1]

if exit_after == 'load':
    print(f'exit_after={exit_after}, exiting!')
    if show_plots:
        pdf.close()
    sys.exit(0)

# rescales separately, just to make TFR more robust to numerical issues
if scale_data_combine_type not in ['no_scaling', 'no']:

    combine_type = scale_data_combine_type  #'medcond'

    print('Start data prescaling with combine_type={}, main_side={}'.
          format(combine_type,main_side_let ) )

    if gv.DEBUG_MODE:
        dat_pri_unscaled = [0]*len(dat_pri)
        for dati in range(len(dat_pri) ):
            dat_pri_unscaled[dati] = dat_pri[dati].copy()

    dat_T_pri = [0]*len(dat_pri)
    for dati in range(len(dat_pri) ):
        dat_T_pri[dati] = dat_pri[dati].T


    curstatinfo = stats_per_ct[combine_type]

    from utils_genfeats import getIndsetsValid
    indsetis_valid,newindsets,means,stds = getIndsetsValid(rawnames,curstatinfo)

    #indsets =   curstatinfo['indsets']
    #means =   curstatinfo['means']
    #stds =   curstatinfo['stds']
    #rawnames_stats =   curstatinfo['rawnames']

    ##inds = [ rawnames_stats.index(rawn) for rawn in rawnames ]
    ##indsetis = len(rawnames) * [-1000]
    ##for rawi in range(len(rawnames)):
    ##    curind = inds[rawi]
    ##    for indseti,indset in enumerate(indsets):
    ##        if curind in indset:
    ##            # make sure we belong only to one indset
    ##            assert indsetis[rawi] < 0
    ##            indsetis[rawi] = indseti

    #newindsets = []
    #indsetis_valid = []
    #for indseti,indset in enumerate(indsets):
    #    newindset_cur = []
    #    for rawi in range(len(rawnames)):
    #        rawn = rawnames[rawi]
    #        curind = rawnames_stats.index(rawn)
    #        if curind in indset:
    #            newindset_cur += [rawi]
    #            if indseti not in indsetis_valid:
    #                indsetis_valid += [indseti]
    #    if len(newindset_cur):
    #        newindsets += [ newindset_cur ]
    #assert len(newindsets)
    #means = [ means[i] for i in indsetis_valid ]
    #stds = [ stds[i] for i in indsetis_valid ]

    #which indset I current raw belogns to?
    if gv.DEBUG_MODE:
        from inspect import currentframe, getframeinfo
        frameinfo = getframeinfo(currentframe())
        #print(frameinfo.filename, frameinfo.lineno)
        plt.figure()
        plt.plot( dat_pri[0][0])
        plt.title(f'Line {frameinfo.lineno} plot N={debug_plot_ctr}')

    # prescaling is necessary because sometimes values for spectral
    # computations can be below double precision
    if prescale_data:
        # rescaling
        dat_T_scaled, indsets, means_rescaled, stds_rescaled = \
            upre.rescaleFeats(rawnames, dat_T_pri, subfeature_order_pri,
                None, sfreq, times_pri, int_type = baseline_int,
                main_side = None, side_rev_pri = side_switched_pri,
                minlen_bins = 5 * sfreq, combine_within=combine_type,
                artif_handling_statcollect=feat_stats_artif_handling,
                means=means, stds=stds, indsets= newindsets,
                bindict_per_rawn=bindict_per_rawn)
        for dati in range(len(dat_pri) ):
            dat_pri[dati] = dat_T_scaled[dati].T

        # plotting
        if show_plots and do_plot_stat_scatter and len(set( n_channels_pri ) ) == 1 :
            int_types_to_stat = [it + '_{}'.format(main_side_let) for it in gp.int_types_basic]
            upre.plotFeatStatsScatter(rawnames,dat_T_scaled, int_types_to_stat,
                subfeature_order_pri,sfreq,
                times_pri,side_switched_pri,
                wbd_pri=None, save_fig=False, separate_by = 'mod',
                artif_handling=feat_stats_artif_handling)
            plt.suptitle('Stats of nonHFO data after rescaling')
            pdf.savefig()
            plt.close()
            gc.collect()


        if use_LFP_HFO:
            dat_T_pri = [0]*len(dat_lfp_hires_pri)
            for dati in range(len(dat_lfp_hires_pri) ):
                dat_T_pri[dati] = dat_lfp_hires_pri[dati].T

            if gv.DEBUG_MODE:
                dat_lfp_hires_pri_unscaled = [0]*len(dat_pri)
                for dati in range(len(dat_pri) ):
                    dat_lfp_hires_pri_unscaled[dati] = dat_lfp_hires_pri[dati].copy()

            curstatinfo = stats_HFO_per_ct[combine_type]
            indsets =   curstatinfo['indsets']
            means =   curstatinfo['means']
            stds =   curstatinfo['stds']
            means = [ means[i] for i in indsetis_valid ]
            stds = [ stds[i] for i in indsetis_valid ]

            dat_T_scaled, indsets,  \
            means_lfp_hires_rescaled, stds_lfp_hires_rescaled = \
                upre.rescaleFeats(rawnames, dat_T_pri,
                subfeature_order_lfp_hires_pri, None,
                sfreq_hires, times_hires_pri, int_type = baseline_int ,
                main_side = None, side_rev_pri = side_switched_pri,
                minlen_bins = 5 * sfreq_hires, combine_within=combine_type,
                means=means, stds=stds, indsets= newindsets,
                bindict_per_rawn=bindict_hires_per_rawn,
                artif_handling_statcollect=feat_stats_artif_handling)
            for dati in range(len(dat_lfp_hires_pri) ):
                dat_lfp_hires_pri[dati] = dat_T_scaled[dati].T

if exit_after == 'prescale_data':
    print(f'exit_after={exit_after}, exiting!')
    if show_plots:
        pdf.close()
    sys.exit(0)

if only_load_data:
    print(' only_load_data, exiting! ')
    if show_plots:
        pdf.close()
    sys.exit(0)

###############  plot interval data

if do_plot_raw_psd:
    print('Starting plottign raw psd')
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        ivalis = ivalis_pri[0]
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv
            #utsne.plotIntervalData(dat_scaled,subfeature_order,iv, times=times,
            #                    plot_types = ['psd'], fmax=fmax_raw_psd )
            utsne.plotIntervalData(dat_pri[0],subfeature_order,iv, times=times_pri[0],
                                plot_types = ['psd'], fmax=fmax_raw_psd )

            for ax in plt.gcf().get_axes():
                #tremfreq_Jan = 6
                ax.axvline(x=6, ls=':')
            pdf.savefig()
            plt.close()


############## plot raw stats
if do_plot_raw_stats:
    #utsne.plotBasicStatsMultiCh(dat_scaled, subfeature_order, printMeans = 0)
    utsne.plotBasicStatsMultiCh(dat_pri[0], subfeature_order, printMeans = 0)
    plt.tight_layout()
    pdf.savefig()

    if use_LFP_HFO:
        #utsne.plotBasicStatsMultiCh(dat_lfp_hires_scaled, subfeature_order_lfp_hires,
        #                            printMeans = 0)
        utsne.plotBasicStatsMultiCh(dat_lfp_hires_pri[0], subfeature_order_lfp_hires,
                                    printMeans = 0)
        plt.tight_layout()
        pdf.savefig()

############## plot raw



if do_plot_raw_timecourse:
    print('Starting plotting timecourse of scaled data' )
    extdat = np.hstack(extdat_pri)
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv
            #raw_lfponly = raws_permod_both_sides[rawnames[0]]['LFP']
            #tt = utsne.plotIntervalData(dat_scaled,subfeature_order,iv,
            #                            raw=raw_lfponly, plot_types=['timecourse'], dat_ext = extdat,
            #                            extend=extend)
            tt = utsne.plotIntervalData(dat_pri[0],subfeature_order,iv,
                                        raw=None, plot_types=['timecourse'], dat_ext = extdat,
                                        extend=extend)
            ax_list = plt.gcf().axes
            for ax in ax_list:
                ax.set_rasterization_zorder(0)
            pdf.savefig(dpi=dpi)
            plt.close()

##################

##################

rawnstr = ','.join(rawnames)
pre_tfr_fname = '{}_tfr_{}chs.npz'
pre_csd_fname = '{}_csd_{}chs.npz'

pre_rbcorr_fname = '{}_rbcorr_{}chs.npz'
pre_bpcorr_fname = '{}_bpcorr_{}chs.npz'
#a = '{}_tfr_{}chs.npz'.format(rawnstr,n_channels_str)
#fname_tfr_full = pjoin(gv.data_dir, output_subdir, a)
#a = '{}_csd_{}chs.npz'.format(rawnstr,n_channels_str)
#fname_csd_full = pjoin(gv.data_dir, output_subdir,a)

fname_feat_full = ""
######################

# later I would add features that are too long to compute for every time point,
# so I don't want to save X itself

do_cleanup = not test_mode

#if load_feat and os.path.exists(fname_feat_full):
#    print('Loading feats from ',fname_feat_full)
#    f = np.load(fname_feat_full)
#    X =  f['X']
#    Xtimes = f ['Xtimes']
#    skip =f['skip']
#    feature_names_all = f['feature_names_all']
#
#else:
#'''
#if I compute all source couplings first and then do something with them
#it takes a lot of memory perhaps
#'''
#exec( open( '_run_featprep.py').read(), globals() )

# compute diagonal terms
have_TFR = False
try:
    print('tfr (existing) num channels =', tfrres_pri[0][0].shape[0] )
except (NameError,IndexError) as e:
    have_TFR = False
else:
    have_TFR = True #and (tfrres.shape[0] == n_channels)

#if not have_TFR:
#    print('OOO')
#    sys.exit(0)
#def computeTFR():
#    return

##############################################################################
###################### generate filenames for TFR and CSD so that we can old
##############################################################################

gs_tfr = np.zeros(len(rawnames) )    # ages
gs_csd = np.zeros(len(rawnames) )
fname_tfr_full_pri = [0]*len(rawnames)
fname_csd_full_pri = [0]*len(rawnames)

for rawi in range(len(rawnames)):
    rawn = rawnames[rawi]
    tfr_fname = pre_tfr_fname.format(rawn,n_channels_pri[rawi] )
    fname_tfr_full = pjoin(gv.data_dir, output_subdir, tfr_fname)
    g = int( os.path.exists( fname_tfr_full ) )
    if g:
        g += int (upre.getFileAge(fname_tfr_full) < load_TFR_max_age_h)
    gs_tfr[rawi] = g

    csd_fname = pre_csd_fname.format(rawn,n_channels_pri[rawi] )
    fname_csd_full = pjoin(gv.data_dir, output_subdir, csd_fname)
    g = int( os.path.exists( fname_csd_full )  )
    if g:
        g += int (upre.getFileAge(fname_csd_full) < load_CSD_max_age_h)
    gs_csd[rawi] = g

    fname_tfr_full_pri[rawi] = fname_tfr_full
    fname_csd_full_pri[rawi] = fname_csd_full
    del fname_tfr_full
    del fname_csd_full



#sys.exit(0)

#################### extracted from inside TFR to be TFR-indipendent ###########

# we do it just in case we don't have 'con' (otherwise it is left un-init)
from utils_tSNE import selFeatsRegexNames
names_src = selFeatsRegexNames(subfeature_order, ['msrc.*'])
names_lfp = selFeatsRegexNames(subfeature_order, ['LFP.*'] )
chnames_tfr = np.append(names_lfp,names_src)

# new chnames (parcel indices based)
parcels_present = []
pp2side = {}
for chn in chnames_tfr:
    if chn.startswith('LFP'):
        continue
    side1, gi1, parcel_ind1, si1 = utils.parseMEGsrcChnameShort(chn)
    if parcel_ind1 in pp2side:
        assert pp2side[parcel_ind1] == side1, 'Side inconsistency within parcel!'
    pp2side[parcel_ind1] = side1
    parcels_present += [parcel_ind1]

pp = list(sorted(set(parcels_present)))
aa = ['msrc{}_{}_{}_c{}'.format(pp2side[p],newchn_grouping_ind,p,0) for p in pp]
lfpinds = utsne.selFeatsRegexInds(chnames_tfr,'LFP.*')
# note that here we'll have LFP indices in the end, not in the beginning!
newchns = aa + np.array(chnames_tfr)[lfpinds].tolist()

#######################################################

if (not (use_existing_TFR and have_TFR) ) and 'con' in features_to_use:
    from utils_genfeats import prepTFR,prepCSD
    tfrr = prepTFR(rawnames,anndict_per_intcat_per_rawn,
             dat_pri,subfeature_order,sfreq,windowsz,skip,freqs,n_cycles,
             use_LFP_HFO,
             dat_lfp_hires_pri,subfeature_order_lfp_hires,
             sfreq_hires, windowsz_hires,skip_hires,
             skip_div_TFR,freqs_inc_HFO,n_cycles_inc_HFO,
             load_TFR,save_TFR,gs_tfr,fname_tfr_full_pri,
             load_CSD,save_CSD,gs_csd,
             n_jobs_tfr)
    tfrres_pri = tfrr['tfrres_pri']
    tfrres_LFP_HFO_pri = tfrr['tfrres_LFP_HFO_pri']
    chnames_tfr_ = tfrr['chnames_tfr']
    tfrres_wbd_pri = tfrr['tfrres_wbd_pri']
    tfrres_wbd_HFO_pri = tfrr['tfrres_wbd_HFO_pri']
    assert tuple(chnames_tfr_) == tuple(chnames_tfr)
    gc.collect()

    #if 'con' in features_to_use:
    csdr = prepCSD(cross_types,tfrres_pri,tfrres_LFP_HFO_pri,
            tfrres_wbd_pri,
            chnames_tfr,subfeature_order,newchns,
            roi_labels,srcgrouping_names_sorted,sfreq,
            newchn_grouping_ind,
            normalize_TFR, DEBUG_shorten_couplings, log_during_csd,
            load_CSD,save_CSD,gs_csd,fname_csd_full_pri)
    gc.collect()

    csd_pri = csdr['csd_pri']
    csdord_pri = csdr['csdord_pri']
    csd_LFP_HFO_pri = csdr['csd_LFP_HFO_pri']
    csdord_LFP_HFO_pri = csdr['csdord_LFP_HFO_pri']
    res_couplings = csdr['res_couplings']
    ntimebins_pri = csdr['ntimebins_pri']
    parcel_couplings = csdr['parcel_couplings']
    LFP2parcel_couplings = csdr['LFP2parcel_couplings']
    LFP2LFP_couplings = csdr['LFP2LFP_couplings']

    assert chnames_tfr[0].startswith('LFP')
    assert subfeature_order[0].startswith('LFP')
    # is needed for bandFilter, because later computeCorr uses indexing based on chnames_tfr
    assert tuple(chnames_tfr) == tuple(subfeature_order), set(chnames_tfr) ^ set(subfeature_order)
    assert tuple(csdr['newchns'] ) == tuple(newchns)
    assert tuple(csdr['chnames_tfr'] ) == tuple(chnames_tfr)


if exit_after == 'TFR_and_CSD':
    print(f'exit_after={exit_after}, exiting!')
    if show_plots:
        pdf.close()
    sys.exit(0)

if gv.DEBUG_MODE:
    from inspect import getframeinfo, currentframe
    frameinfo = getframeinfo(currentframe())
    debug_plot_ctr += 1
    plt.figure()
    plt.plot( dat_pri[0][0])
    plt.title(f'Line {frameinfo.lineno} plot N={debug_plot_ctr}')


if 'con' in features_to_use:
    print('Averaging over freqs within bands')
    #csdord_strs is longer than csdord because it
    #currently (Jan 5, 2021) bpow_imagcsd is useless
    # here I assume that channel counts are the same across datasets
    bpow_abscsd_pri, bpow_imagcsd, csdord_strs_pri, csdord_strs_HFO_pri,bpow_abscsd_LFP_HFO_pri  = \
        ugf.bandAverage( freqs,freqs_inc_HFO,csd_pri,csdord_pri,csdord_LFP_HFO_pri,
                csd_LFP_HFO_pri, fbands,fband_names, fband_names_inc_HFO,
                newchns, subfeature_order_lfp_hires, log_before_bandaver= log_before_bandaver )
    if do_cleanup:
        del csd_pri
        del csdord_LFP_HFO_pri
gc.collect()

if exit_after == 'bandAverage':
    print(f'exit_after={exit_after}, exiting!')
    if show_plots:
        pdf.close()
    sys.exit(0)

#################################  Plot CSD at some time point
if do_plot_CSD:
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        #intervals = ivalis[int_name]
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv

            #raw_srconly = raws_permod_both_sides[rawnames[0]]['src']
            ts,r, int_names = utsne.getIntervalSurround(start,end,extend, times=times_pri[0], raw=None)

            #timebins = raw_lfponly.time_as_index
            utsne.plotCSD(csd_pri[0], fband_names, chnames_tfr, list(r) , sfreq=sfreq, intervalMode=1,
                        int_names=int_names)
            pdf.savefig()
            plt.close()


if gv.DEBUG_MODE:
    frameinfo = getframeinfo(currentframe())
    debug_plot_ctr += 1
    plt.figure()
    plt.plot( dat_pri[0][0])
    plt.title(f'Line {frameinfo.lineno} plot N={debug_plot_ctr}')
############################## Hjorth

if 'Hjorth' in features_to_use or 'H_act' in features_to_use or 'H_mob' in features_to_use or 'H_compl' in features_to_use:
    print('Computing Hjorth')

    #dat_total = np.hstack( dat_pri )   # it should be already scaled since I did rescale raws
    #dat_src,names_src = utsne.selFeatsRegex(dat_total, subfeature_order, ['msrc.*'])
    #dat_lfp,names_lfp = utsne.selFeatsRegex(dat_total, subfeature_order, ['LFP.*'])
    dat_src_pri,names_src = utsne.selFeatsRegex(dat_pri, subfeature_order, ['msrc.*'])

    #TODO maybe it would be better to make a simple coupling between old names and new names
    #act,mob,compl  = utils.Hjorth(dat_scaled_src, 1/sfreq, windowsz=windowsz)
    #n_newchns_srconly = len(newchns) - dat_scaled_lfp.shape[0]
    act_pri,mob_pri,compl_pri,wbd_H_pri  = ugf.Hjorth(dat_src_pri, 1/sfreq,
                                        windowsz=windowsz, skip=skip,
                                        remove_invalid=True)
    n_newchns_srconly = len(newchns) - len(subfeature_order_lfp_hires)
    actnew_pri   = [0] * len(act_pri)
    mobnew_pri   = [0] * len(act_pri)
    complnew_pri = [0] * len(act_pri)
    for acti in range(len(act_pri)):
        act = act_pri[acti]; mob = mob_pri[acti]; compl = compl_pri[acti];
        actnew =    np.zeros( (n_newchns_srconly, act.shape[1] ) )
        mobnew =    np.zeros( (n_newchns_srconly, act.shape[1] ) )
        complnew =  np.zeros( (n_newchns_srconly, act.shape[1] ) )

        assert newchns[0].startswith('msrc')
        # here I combine activities from different channels within parcel
        for chi in range(len(names_src) ):
            chn = names_src[chi]
            side1, gi1, parcel_ind1, si1 = utils.parseMEGsrcChnameShort(chn)
            newchn = 'msrc{}_{}_{}_c{}'.format(side1,newchn_grouping_ind,parcel_ind1,0)
            newchi = newchns.index(newchn)
            actnew[newchi] += act[chi]
            mobnew[newchi] += mob[chi]
            complnew[newchi] += compl[chi]

        actnew_pri[acti] = actnew
        mobnew_pri[acti] = mobnew
        complnew_pri[acti] = complnew

    act_pri = actnew_pri
    mob_pri = mobnew_pri
    compl_pri = complnew_pri

    act = np.hstack(actnew_pri)
    mob = np.hstack(mobnew_pri)
    compl = np.hstack(complnew_pri)

    #act_lfp,mob_lfp,compl_lfp  = utils.Hjorth(dat_scaled_lfp, 1/sfreq, windowsz=windowsz)
    #act = np.vstack( [ act, act_lfp] )
    #mob = np.vstack( [ mob, mob_lfp] )
    #compl = np.vstack( [ compl, compl_lfp] )

    #if use_LFP_HFO:
    #    dat_for_H = dat_scaled_src
    #else:
    #    dat_for_H = dat_scaled
    #act,mob,compl  = utils.Hjorth(dat_for_H, 1/sfreq, windowsz=windowsz)


    # if we have LFP data we better obtain Hjorth parameter from hires data
    if use_LFP_HFO:
        #dat_lfp_hires_total = np.hstack( dat_lfp_hires_pri )
        #act_lfp,mob_lfp,compl_lfp  = utils.Hjorth(dat_lfp_hires_scaled, 1/sfreq_hires,
        #                            windowsz=int( (windowsz/sfreq)*sfreq_hires ) )
        act_lfp_pri,mob_lfp_pri,compl_lfp_pri,wbd_H_lfp_hires_pri  =\
            ugf.Hjorth(dat_lfp_hires_pri, 1/sfreq_hires, windowsz=int(
                (windowsz/sfreq)*sfreq_hires ),
                skip=int(skip/sfreq*sfreq_hires), remove_invalid=True )

        #act_lfp   = np.hstack(act_lfp  )
        #mob_lfp   = np.hstack(mob_lfp  )
        #compl_lfp = np.hstack(compl_lfp)
        ##act_lfp   =      act_lfp   [:,:: sfreq_hires//sfreq ]
        ##mob_lfp   =      mob_lfp   [:,:: sfreq_hires//sfreq ]
        ##compl_lfp =  compl_lfp [:,:: sfreq_hires//sfreq ]
        #act = np.vstack( [act, act_lfp] )
        #mob = np.vstack( [mob, mob_lfp] )
        #compl = np.vstack( [compl, compl_lfp] )
    else:
        dat_lfp_pri,names_lfp = utsne.selFeatsRegex(dat_pri, subfeature_order, ['LFP.*'])
        #act_lfp,mob_lfp,compl_lfp  = utils.Hjorth(dat_scaled_lfp, 1/sfreq, windowsz=windowsz )
        act_lfp_pri,mob_lfp_pri,compl_lfp_pri,wbd_H_lfp_pri = \
            ugf.Hjorth(dat_lfp_pri, 1/sfreq, windowsz=windowsz, skip=skip,
                        remove_invalid=True )

    for acti in range(len(act_pri) ):
        w1 = wbd_H_pri[acti]
        if use_LFP_HFO:
            w2 = wbd_H_lfp_hires_pri[acti]
            assert w1[0][0] == w2[0][0]
            ml = min( w1.shape[1], w2.shape[1] )
            assert w1[1][ml-1] == int( w2[1][ml-1] / sfreq_hires * sfreq)
        else:
            ml = w1.shape[1]
        subsl = slice(0,ml,None)
        sl = (slice(None,None,None), subsl )
        act_pri[acti]   =  np.vstack( [  act_pri[acti][sl] ,  act_lfp_pri[acti][sl] ] )
        mob_pri[acti]   =  np.vstack( [  mob_pri[acti][sl] ,  mob_lfp_pri[acti][sl] ] )
        compl_pri[acti] =  np.vstack( [compl_pri[acti][sl], compl_lfp_pri[acti][sl] ] )

        wbd_H_pri[acti]           = w1[sl]
        if use_LFP_HFO:
            wbd_H_lfp_hires_pri[acti] = w2[sl]

    #act_lfp   = np.hstack(act_lfp  )
    #mob_lfp   = np.hstack(mob_lfp  )
    #compl_lfp = np.hstack(compl_lfp)

    # show mean Hjorth
    if show_plots and do_plot_Hjorth:
        n_channels_new = len(newchns)  # newchns were set up during TFR

        fig=plt.figure()
        ax = plt.gca()
        ax.plot (  np.min(  act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='min')
        ax.plot (  np.max(  act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='max')
        ax.plot (  np.mean( act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='mean')
        ax.set_xlabel('channel')
        ax.legend()
        ax.set_xticks(range(n_channels_new))
        #ax.set_xticklabels(subfeature_order,rotation=90)
        ax.set_xticklabels(newchns,rotation=90)
        fig.suptitle('min_max of Hjorth params')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #################################  Plot Hjorth

        fig,axs = plt.subplots(nrows=1,ncols=3, figsize = (15,4))
        axs = axs.reshape((1,3))

        for i in range(n_channels_new):
            axs[0,0].plot(times_pri[0],act[i], label= subfeature_order[i])
            axs[0,0].set_title('activity')
            axs[0,1].plot(times_pri[0],mob[i], label= subfeature_order[i])
            axs[0,1].set_title('mobility')
            axs[0,2].plot(times_pri[0],compl[i], label= subfeature_order[i])
            axs[0,2].set_title('complexity')

        for ax in axs.reshape(axs.size):
            ax.legend(loc='upper right')

        pdf.savefig()


    #act = act[:,::skip]
    #mob = mob[:,::skip]
    #compl = compl[:,::skip]
    gc.collect()

if gv.DEBUG_MODE:
    frameinfo = getframeinfo(currentframe())
    debug_plot_ctr += 1
    plt.figure()
    plt.plot( dat_pri[0][0])
    plt.title(f'Line {frameinfo.lineno} plot N={debug_plot_ctr}')


if 'con' in features_to_use:
    for rawind in range(len(dat_pri)):
        a,b = tfrres_wbd_pri[rawind], wbd_H_pri[rawind]
        if a.shape != b.shape:
            print('Warning, tfr and Hjorth have different shapes {},{}, two last {}, {}'.
                    format ( a.shape,b.shape, a[1][-2:],b[1][-2:] ) )
        else:
            assert np.max (np.abs (a-b) ) < 1e-10

if exit_after == 'Hjorth':
    print(f'exit_after={exit_after}, exiting!')
    if show_plots:
        pdf.close()
    sys.exit(0)

#Xtimes_full = raw_srconly.times[nedgeBins:-nedgeBins]
#####################################################

if ('rbcorr' in features_to_use and not load_rbcorr) or ('bpcorr' in features_to_use and not load_bpcorr):
    # we have done it before as well
    print('Filtering and Hilbert')

    sfreqs = [sfreq, sfreq_hires]
    skips = [skip, skip_hires]
    dat_pri_persfreq = [dat_pri, dat_lfp_hires_pri]


    # note that we can have different channel names for different raws
    #raw_perband_flt_pri_persfreq = []
    #raw_perband_bp_pri_persfreq = []

    smoothen_bandpow = 0

    # band filter computed separately for different datasets
    raw_perband_flt_pri, raw_perband_bp_pri, chnames_perband_flt_pri, chnames_perband_bp_pri  = \
        ugf.bandFilter(rawnames, times_pri, main_sides_pri, side_switched_pri,
              sfreqs, skips, dat_pri_persfreq, fband_names_inc_HFO, gv.fband_names_HFO_all,
              fbands, n_jobs_flt, allow_CUDA, subfeature_order, subfeature_order_lfp_hires,
              smoothen_bandpow, ann_MEGartif_prefix_to_use, anndict_per_intcat_per_rawn= anndict_per_intcat_per_rawn)

    if gv.DEBUG_MODE:
        frameinfo = getframeinfo(currentframe())
        debug_plot_ctr += 1
        plt.figure()
        plt.plot( dat_pri[0][0])
        plt.title(f'Line {frameinfo.lineno} plot N={debug_plot_ctr}')

    #raws_flt_pri_perband_ = {}

    if recalc_stats_multi_band:
        #means_perband_flt_pri_,_ = \
        indsets, means_perband_flt_pri_, stds_perband_flt_pri_, stats_per_indset_per_band_flt =  \
        ugf.gatherMultiBandStats(rawnames,raw_perband_flt_pri, times_pri,
                                chnames_perband_flt_pri, side_switched_pri, sfreq,
                                baseline_int, scale_data_combine_type,
                                feat_stats_artif_handling,
                                require_intervals_present = [baseline_int],
                                 bindict_per_rawn=bindict_per_rawn)
        #means_perband_bp_pri_,_ = \
        indsets, means_perband_bp_pri_, stds_perband_bp_pri_, stats_per_indset_per_band_bp =  \
        ugf.gatherMultiBandStats(rawnames,raw_perband_bp_pri, times_pri,
                                chnames_perband_bp_pri, side_switched_pri, sfreq,
                                baseline_int, scale_data_combine_type,
                                feat_stats_artif_handling,
                                require_intervals_present = [baseline_int],
                                 bindict_per_rawn=bindict_per_rawn)

        #stats_multiband_flt = stats_per_indset_per_band_flt
        #stats_multiband_bp  = stats_per_indset_per_band_bp

        #stats_multiband_flt = means_perband_flt_pri_
        #stats_multiband_bp  = means_perband_bp_pri_


        curstatinfo = {'indsets':indsets, 'rawnames':rawnames,
                       'stats_per_indset':stats_per_indset_per_band_flt }
        curstatinfo['means' ]  = means_perband_flt_pri_
        curstatinfo['stds' ]   = stds_perband_flt_pri_
        stats_multiband_flt = curstatinfo

        curstatinfo = {'indsets':indsets, 'rawnames':rawnames,
                       'stats_per_indset':stats_per_indset_per_band_bp }
        curstatinfo['means' ]  = means_perband_bp_pri_
        curstatinfo['stds' ]   = stds_perband_bp_pri_
        stats_multiband_bp = curstatinfo
    else:
        assert not prescale_data, 'If we prescale data, loaded multi band stats is not valid!'
        # first arg should be None so that I can specify explicitly the prefix
        # it is necessary because I run this scripts not with the same set of
        # rawnames compared to what was used for stats gathering
        fname_stats_multi_band = utils.genStatsMultiBandFn(None, new_main_body_side, data_modalities,
                                        use_main_LFP_chan, src_file_grouping_ind,
                                        src_grouping, bands_only, brain_side_to_use, stats_fn_prefix )

        fname_stats_full = pjoin( gv.data_dir, input_subdir, fname_stats_multi_band)
        f = np.load(fname_stats_full, allow_pickle=True)
        rawnames_stats =  f['rawnames']
        assert set(rawnames).issubset(rawnames_stats)
        #means_perband_flt_pri_ =  f['stats_multiband_flt_per_ct'][()][scale_data_combine_type]['means']
        #means_perband_bp_pri_  =  f['stats_multiband_bp_per_ct'][()][scale_data_combine_type]['means']

        stats_multiband_flt =  f['stats_multiband_flt_per_ct'][()][scale_data_combine_type]
        stats_multiband_bp  =  f['stats_multiband_bp_per_ct'][()][scale_data_combine_type]

        indsets = stats_multiband_bp['indsets']
        #stats_per_indset_per_band_flt

    ################# extract stats for the given baseline int

    rawi_mask,allinds = upre.getIndsetMask(indsets, allow_repeating=False, allow_holes = False)

    means_perband_flt_pri = len(rawnames) * [dict()]
    means_perband_bp_pri = len(rawnames) * [dict()]

    statdicts =[stats_multiband_flt, stats_multiband_bp ]
    mdicts =[means_perband_flt_pri, means_perband_bp_pri ]
    for mdi,md in enumerate(mdicts):
        for rawind in range(len(rawnames)):
            indseti = rawi_mask[rawind]
            for bandname in statdicts[mdi]['means']:
                a = statdicts[mdi]['means'][bandname][indseti][baseline_int]
                #print(a)
                mdicts[mdi][rawind][bandname] = a
            #if recalc_stats_multi_band:
            #    for bandname in statdicts[mdi]:
            #        a = statdicts[mdi][bandname][indseti][baseline_int]
            #        #print(a)
            #        mdicts[mdi][rawind][bandname] = a
            #else:
            #    for bandname in statdicts[mdi]['means']:
            #        a = statdicts[mdi]['means'][bandname][indseti][baseline_int]
            #        #print(a)
            #        mdicts[mdi][rawind][bandname] = a

                #means_perband_flt_pri[rawind][bandname] = \
                #    means_perband_flt_pri_[rawind][bandname][baseline_int]


    #means_perband_flt_pri = len(rawnames) * [dict()]
    #for rawind in range(len(rawnames)):
    #    for bandname in means_perband_flt_pri_[rawind]:
    #        means_perband_flt_pri[rawind][bandname] = \
    #            means_perband_flt_pri_[rawind][bandname][baseline_int]

    #means_perband_bp_pri = len(rawnames) * [dict()]
    #for rawind in range(len(rawnames)):
    #    for bandname in means_perband_bp_pri_[rawind]:
    #        means_perband_bp_pri[rawind][bandname] = \
    #            means_perband_bp_pri_[rawind][bandname][baseline_int]

if gv.DEBUG_MODE:
    frameinfo = getframeinfo(currentframe())
    debug_plot_ctr += 1
    plt.figure()
    plt.plot( dat_pri[0][0])
    plt.title(f'Line {frameinfo.lineno} plot N={debug_plot_ctr}')

if 'rbcorr' in features_to_use:  #raw band corr
    if bands_only == 'fine':
        bandPairs = [('tremor','tremor', 'corr'),
                    ('low_beta','low_beta', 'corr'), ('high_beta','high_beta', 'corr'),
                    ('low_gamma','low_gamma', 'corr') , ('high_gamma','high_gamma', 'corr') ]
    else:
        bandPairs = [('tremor','tremor', 'corr'), ('beta','beta', 'corr'), ('gamma','gamma', 'corr') ]

    rbcorrs_pri     = [0] * len(rawnames)
    wbd_rbcorr_pri  = [0] * len(rawnames)
    rbcor_names_pri = [0] * len(rawnames)
    for rawind in range(len(dat_pri)):
        fn = pre_rbcorr_fname.format(rawnames[rawind], n_channels_pri[rawind] )
        fn = pjoin(gv.data_dir, output_subdir, fn)
        if load_rbcorr and os.path.exists(fn):
            print('Load rbcorr from {}'.format(fn) )
            f = np.load(fn, allow_pickle=True)
            bpcorrs      =f['rbcorrs'    ]
            bpcor_names  =f['rbcor_names']
            wbd_bpcorr   =f['wbd_rbcorr' ]
        else:
            print('Starting rbcorr for datset {}'.format(rawind) )
            raw_perband =  raw_perband_flt_pri[rawind]
            chnames_per_band =  chnames_perband_flt_pri[rawind]
            if rbcorr_use_local_means:
                means_perband = None
                assert not rbcorr_use_zero_mean
            else:
                means_perband = means_perband_flt_pri[rawind]

            if rbcorr_use_zero_mean:
                assert not rbcorr_use_local_means
                means_perband = 0.

            rbcorrs = []
            rbcor_names = []
            for bp in bandPairs:
                print('band pair ',bp)
                rbcorrs_curbp,rbcor_names_curbp,dct_nums, wbd_rbcorr = \
                computeCorr(raw_perband, chnames_per_band=chnames_per_band,
                                        defnames=chnames_tfr,
                                        parcel_couplings=parcel_couplings,
                                        LFP2parcel_couplings=LFP2parcel_couplings,
                                        LFP2LFP_couplings=LFP2LFP_couplings,
                                        res_group_id=newchn_grouping_ind,
                                        skip=skip, windowsz = windowsz,
                                        band_pairs = [bp], n_jobs=n_jobs,
                                        positive=0, templ='{}_.*',
                                        roi_labels=roi_labels,
                                        sort_keys=srcgrouping_names_sorted,
                                        means=means_perband,
                                        local_means=rbcorr_use_local_means)
                assert len(rbcor_names_curbp) > 0
                gc.collect()
                rbcorrs     += rbcorrs_curbp
                rbcor_names += rbcor_names_curbp

            rbcorrs = np.vstack(rbcorrs)

            for feati in range(len(rbcor_names) ):
                rbcor_names[feati] = 'rb' + rbcor_names[feati]

            if save_rbcorr:
                print('Saving rbcorr ',fn)
                #fn = pjoin(gv.data_dir, output_subdir,'{}_rbcorr.npz'.format(rawnames[rawind] ) )
                np.savez( fn , rbcorrs=rbcorrs, rbcor_names=rbcor_names, wbd_rbcorr=wbd_rbcorr)
                gc.collect()

        rbcor_names_pri[rawind] = rbcor_names
        rbcorrs_pri    [rawind] = rbcorrs
        wbd_rbcorr_pri [rawind] = wbd_rbcorr
    if do_cleanup:
        del  raw_perband_flt_pri
        del  rbcorrs
    gc.collect()


else:
    rbcorrs_pri = None
    rbcor_names = None
    wbd_rbcorr_pri = None

if gv.DEBUG_MODE:
    frameinfo = getframeinfo(currentframe())
    debug_plot_ctr += 1
    plt.figure()
    plt.plot( dat_pri[0][0])
    plt.title(f'Line {frameinfo.lineno} plot N={debug_plot_ctr}')

if 'bpcorr' in features_to_use:
    if bands_only == 'fine':
        # no need to dubplicate reversing order
        bandPairs = [('tremor','low_beta', 'corr'), ('tremor','high_beta', 'corr'),
                    ('tremor','low_gamma', 'corr'), ('tremor','high_gamma', 'corr'),
                    ('low_beta','low_gamma', 'corr') , ('low_beta','high_gamma', 'corr') ,
                    ('high_beta','low_gamma', 'corr') , ('high_beta','high_gamma', 'corr') ]
    else:
        bandPairs = [('tremor','beta', 'corr'), ('tremor','gamma', 'corr'), ('beta','gamma', 'corr') ]
    if use_LFP_HFO:
        # HFO should be always on the second place unless we use HFO-HFO
        # coupling in LFP!
        bandPairs += [ ('tremor','HFO', 'corr'), ('beta','HFO', 'corr'), ('gamma','HFO', 'corr') ]

        if bands_only == 'fine':
            bandPairs += [ ('HFO1','HFO2', 'div'), ('HFO1','HFO3', 'div'), ('HFO2','HFO3', 'div') ]

    bpcorrs_pri     = [0] * len(rawnames)
    wbd_bpcorr_pri  = [0] * len(rawnames)
    bpcor_names_pri = [0] * len(rawnames)
    for rawind in range(len(dat_pri)):
        fn = pre_bpcorr_fname.format(rawnames[rawind], n_channels_pri[rawind] )
        fn = pjoin(gv.data_dir, output_subdir, fn)
        if load_bpcorr and os.path.exists(fn):
            print('Load bpcorr from {}'.format(fn) )
            f = np.load(fn, allow_pickle=True)
            bpcorrs      =f['bpcorrs'    ]
            bpcor_names  =f['bpcor_names']
            wbd_bpcorr   =f['wbd_bpcorr' ]
        else:
            print('Starting bpcorr for datset {}'.format(rawind) )

            raw_perband =  raw_perband_bp_pri[rawind]
            chnames_per_band =  chnames_perband_bp_pri[rawind]
            means_perband = means_perband_bp_pri[rawind]
            bpcorrs = []
            bpcor_names = []
            for bp in bandPairs:
                print('band pair ',bp)
                bpcorrs_curbp,bpcor_names_curbp,dct_nums,wbd_bpcorr =\
                computeCorr(raw_perband, chnames_per_band=chnames_per_band,
                                        defnames=chnames_tfr,
                                        parcel_couplings=parcel_couplings,
                                        LFP2parcel_couplings=LFP2parcel_couplings,
                                        LFP2LFP_couplings=LFP2LFP_couplings,
                                        res_group_id=newchn_grouping_ind,
                                        skip=skip, windowsz = windowsz,
                                        band_pairs = [bp], n_jobs=n_jobs,
                                        positive=1, templ='{}_.*',
                                        sort_keys=srcgrouping_names_sorted,
                                        means=means_perband)
                gc.collect()
                assert len(bpcor_names_curbp) > 0
                bpcorrs += bpcorrs_curbp
                bpcor_names += bpcor_names_curbp

            bpcorrs = np.vstack(bpcorrs)

            for feati in range(len(bpcor_names) ):
                bpcor_names[feati] = 'bp' + bpcor_names[feati]


            if save_bpcorr:
                print('Saving bpcorr ',fn)
                np.savez( fn , bpcorrs=bpcorrs, bpcor_names=bpcor_names, wbd_bpcorr=wbd_bpcorr)
                gc.collect()

        bpcor_names_pri [rawind] = bpcor_names
        bpcorrs_pri     [rawind] = bpcorrs
        wbd_bpcorr_pri  [rawind] = wbd_bpcorr

    if do_cleanup:
        del  bpcorrs
        del  raw_perband_bp_pri
    gc.collect()

else:
    bpcorrs_pri = None
    bpcor_names = None
    wbd_bpcorr_pri = None


if gv.DEBUG_MODE:
    frameinfo = getframeinfo(currentframe())
    debug_plot_ctr += 1
    plt.figure()
    plt.plot( dat_pri[0][0])
    plt.title(f'Line {frameinfo.lineno} plot N={debug_plot_ctr}')

# at these stage numbers of channels and channel names should be consistent
# across datasets

######################################################
Xtimes_pri = []
X_pri = []
feature_names_all_pri = [0]*len(dat_pri)

defpct = (percentileOffset,100-percentileOffset)
log_was_applied = spec_uselog or log_during_csd or log_before_bandaver
center_spec_feats = log_was_applied
if log_was_applied:
    con_scale = defpct
else:
    con_scale = (0,100-percentileOffset)

if logH:
    fH = lambda x: np.log( np.maximum(x,1e-12)  )
    H_scale = defpct
else:
    fH = lambda x: x
    H_scale = (0,100-percentileOffset)

for rawind in range(len(dat_pri) ):
    feat_dict = {}
    if 'con' in features_to_use:
        feat_dict['con'] = {'data': None, 'pct':con_scale, 'centering':center_spec_feats,
                                'names':None, 'wbd':tfrres_wbd_pri[rawind] }
        feat_dict['con']['centering'] = True
    if 'Hjorth' in features_to_use or 'H_act' in features_to_use:
        feat_dict['H_act']   = {'data': fH(act_pri[rawind] ),   'pct':H_scale,
                                'names':newchns, 'wbd':wbd_H_pri[rawind]}
    if 'Hjorth' in features_to_use or 'H_mob' in features_to_use:
        feat_dict['H_mob']   = {'data': fH(mob_pri[rawind] ),   'pct':H_scale,
                                'names':newchns, 'wbd':wbd_H_pri[rawind]}
    if 'Hjorth' in features_to_use or 'H_compl' in features_to_use:
        feat_dict['H_compl'] = {'data': fH(compl_pri[rawind] ), 'pct':H_scale,
                                'names':newchns, 'wbd':wbd_H_pri[rawind]}
    if 'bpcorr' in features_to_use:
        feat_dict['bpcorr'] = {'data':bpcorrs_pri[rawind], 'pct':defpct,
                                'names':bpcor_names_pri[rawind],
                                'wbd':wbd_bpcorr_pri[rawind]}
    if 'rbcorr' in features_to_use:
        feat_dict['rbcorr'] = {'data':rbcorrs_pri[rawind], 'pct':defpct,
                                'names':rbcor_names_pri[rawind],
                                'wbd':wbd_rbcorr_pri[rawind]}
    #'tfr':{'data':None, 'pct':con_scale, 'centering':center_spec_feats },


    f = lambda x: x
    if spec_uselog:
        f = lambda x: np.log(x)

    if 'con' in feat_dict:
        ntimebins = ntimebins_pri[rawind]
        if bands_only == 'no':
            raise ValueError('Not implemented')
            tfres_ = tfrres_pri[0].reshape( tfrres_pri[0].size//ntimebins , ntimebins )
            #feat_dict['tfr']['data'] = f( np.abs( tfres_) )
            #feat_dict['con']['data'] = con.reshape( con.size//ntimebins , ntimebins )
            feat_dict['con']['data'] = f( np.abs( csd_pri[rawind].reshape(
                csd_pri[rawind].size//ntimebins , ntimebins ) ) )
            if use_imag_coh:
                feat_dict['con']['data'] = csd_pri[rawind].reshape( csd_pri[rawind].size//ntimebins , ntimebins ).imag
        else:
            bpow_abscsd = bpow_abscsd_pri[rawind]
            bpow_abscsd_LFP_HFO = bpow_abscsd_LFP_HFO_pri[rawind]
            csdord_strs = csdord_strs_pri[rawind]

            #feat_dict['tfr']['data'] = f( bpows.reshape( bpows.size//ntimebins , ntimebins ) )
            if bpow_abscsd.ndim == 3:
                ncsds,nfreqs,ntimebins_ = bpow_abscsd.shape
                bpow_abscsd_reshaped = bpow_abscsd.reshape( ncsds*nfreqs, ntimebins_ )
                assert bpow_abscsd_reshaped.shape[0] == len(csdord_strs)
            else:
                bpow_abscsd_reshaped = bpow_abscsd

            feat_dict['con']['names'] = csdord_strs[:]
            if use_LFP_HFO:
                if bpow_abscsd_LFP_HFO.ndim == 3:
                    bpow_abscsd_LFP_HFO_reshaped = bpow_abscsd_LFP_HFO.reshape(
                        bpow_abscsd_LFP_HFO.size//ntimebins_ , ntimebins_ )
                else:
                    bpow_abscsd_LFP_HFO_reshaped = bpow_abscsd_LFP_HFO
                # add HFO to low freq
                bpow_abscsd_all_reshaped = np.vstack( [bpow_abscsd_reshaped, bpow_abscsd_LFP_HFO_reshaped])
                #TODO: note that csdord_strs by that moment already contains LFP HFO
                #names (see when csdord_strs i generated)
                feat_dict['con']['names'] += csdord_strs_HFO_pri[rawind]
            else:
                bpow_abscsd_all_reshaped = bpow_abscsd_reshaped

            assert len(bpow_abscsd_all_reshaped) == len(feat_dict['con']['names'] )

            if not use_LFP_to_LFP:
                templ_same_LFP = r'.*:\s(LFP.*),\1'
                inds_same_LFP = utsne.selFeatsRegexInds(csdord_strs, [templ_same_LFP], unique=1)
                templ_all_LFP = r'.*:\s(LFP.*),(LFP.*)'
                inds_all_LFP = utsne.selFeatsRegexInds(csdord_strs, [templ_all_LFP], unique=1)
                assert len(inds_all_LFP) == len(inds_same_LFP)

                if len(inds_all_LFP) > len(inds_same_LFP):
                    print('Removing cross LFP', inds_same_LFP)
                    inds_notsame_LFP = np.setdiff1d( inds_all_LFP, inds_same_LFP)
                    gi = np.setdiff1d( np.arange(len(csdord_strs) ) , inds_notsame_LFP)
                    bpow_abscsd_all_reshaped = bpow_abscsd_all_reshaped[gi]

                    feat_dict['con']['names'] = np.array(feat_dict['con']['names'])[gi]

            feat_dict['con']['data'] = f( bpow_abscsd_all_reshaped )
        #if use_imag_coh:
        #    feat_dict['con']['data'] = f( tmp )

        #tmp_ord
        #csdord1 = csdord_bandwise.reshape( (bpow_abscsd_reshaped.shape[0], 3) )
        #if use_LFP_HFO:
        #    csdord2 = csdord_bandwise_LFP_HFO.reshape( (bpow_abscsd_LFP_HFO_reshaped.shape[0], 3) )
        #csdords = [csdord1, csdord2  ]
        #csdord = np.vstack(csdords  )


    ##########
#         for feat_name in feat_dict:
#             if feat_name in ['rbcorr', 'bpcorr']:
#                 continue
#             curfeat = feat_dict[feat_name]
#             curfeat['data'] = curfeat['data'][:,nedgeBins//skip:-nedgeBins//skip]
#             print(feat_name, curfeat['data'].shape)


    #Xtimes = (nedgeBins / sfreq) + np.arange(compl[:,nedgeBins//skip:-nedgeBins//skip].shape[1] ) * ( skip / sfreq )
    #main_side_before_change = main_sides_pri[rawind]  # side of body
    #opsidelet = utils.getOppositeSideStr(main_side_before_change[0].upper() ) # side of brain
    #wrong_brain_sidelet = main_side_before_change[0].upper()


    artif_mod_str = [ '_ann_LFPartif', ann_MEGartif_prefix_to_use ]
    #anns_artif, anns_artif_pri, times2, dataset_bounds = utsne.concatArtif(rawnames[rawind],times_pri[rawind])


    anns_artif, anns_artif_pri, times2, dataset_bounds_ = \
        utsne.concatAnns(rawnames[rawind],times_pri[rawind], artif_mod_str,crop=(crop_start,crop_end),
                    allow_short_intervals=True,
                            side_rev_pri = aux_info_perraw[rawnames[rawind]]['side_switched'],
                            wbd_pri = wbd_H_pri[rawind], sfreq=sfreq)

    # here we use new side instead of old because we have done the reversal in  concatAnns
    # this feature rescaling will work wrong if I have new_main_body_side =
    # 'both', but we will do rescaling later anyway. So it might affect only
    # intermediate plots
    wrong_brain_sidelet = new_main_body_side.upper()
    anns_artif = utils.removeAnnsByDescr(anns_artif, ['artif_LFP{}'.format(wrong_brain_sidelet) ])

    ivalis_artif = utils.ann2ivalDict(anns_artif)
    #ivalis_artif_tb, ivalis_artif_tb_indarrays = utsne.getAnnBins(ivalis_artif, Xtimes,
    #                                                            (nedgeBins / sfreq),
    #                                                            sfreq, skip, windowsz, dataset_bounds)
    #ivalis_artif_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)


    #################################  Scale features
    if rescale_feats:
        for feat_name in feat_dict:
            curfeat = feat_dict[feat_name]
            inp = curfeat['data']
            sfo = curfeat['names']
            wbd = curfeat['wbd']
            if feat_name.find('corr') >= 0:
                subfeats = sfo
            else:
                subfeats = ['{}_{}'.format(feat_name,sf) for sf in sfo]

            ivalis_artif_tb_indarrays_merged = \
            utils.getWindowIndicesFromIntervals(wbd,ivalis_artif,
                                                sfreq,ret_type='bins_contig',
                                                wbd_type='contig',
                                                ret_indices_type =
                                                'window_inds', nbins_total=dat_pri[rawind].shape[1] )


            feat_dat_artif_nan  = \
                utils.setArtifNaN(inp.T, ivalis_artif_tb_indarrays_merged, subfeats,
                ignore_shape_warning=test_mode).T

            centering = curfeat.get('centering', True)
            pct = curfeat['pct']
            if pct is not None:
                # TODO: fit to out-of-artifacts, apply to all
                scaler = RobustScaler(quantile_range=pct, with_centering=centering)
                scaler.fit(feat_dat_artif_nan.T)
                outd = scaler.transform(inp.T)
                curfeat['data'] = outd.T
                cnt = scaler.center_
                scl = scaler.scale_
                if feat_name in ['tfr', 'con', 'rbcorr', 'bpcorr']:
                    if cnt is not None:
                        cnt = np.min(cnt), np.max(cnt), np.mean(cnt), np.std(cnt)
                    if scl is not None:
                        scl = np.min(scl), np.max(scl), np.mean(scl), np.std(scl)
                #print(feat_name, curfeat['data'].shape, cnt,scl)

    if features_to_use == 'all':
        features_to_use = feat_dict.keys()
    if len(features_to_use) < len(feat_dict.keys() ):
        print('  !!! Using only {} features out of {}'.format( len(features_to_use) , len(feat_dict.keys() ) ))


    ##########################  Construct full feature vector

    # make sure windows align (sometimes one can a bit longer, than we
    # truncate it makeing sure the remaining ones align perfectly)
    minlen = np.inf
    for feat_name in features_to_use:
        curfeat = feat_dict[feat_name]
        wbd = curfeat['wbd']
        minlen = min( minlen, wbd.shape[1] )
        cd = curfeat['data']
        assert cd.shape[1] == wbd.shape[1], (feat_name, cd.shape, wbd.shape)

    # check that all wbds are are the same
    wbd_l = [feat_dict[feat_name]['wbd'][:,:minlen] for feat_name in features_to_use]
    wbdd = np.vstack(wbd_l)

    v0 = np.var(np.diff(wbdd[::2,:],axis=0))
    v1 = np.var(np.diff(wbdd[1::2,:],axis=0))
    assert max(v0,v1) < 1e-10, wbdd

    #last_windows = []
    #for feat_name in features_to_use:
    #    curfeat = feat_dict[feat_name]
    #    last_windows += [ wbd[:,minlen-1] ]
    ## variance on window starts per feature (windows need to be algned to
    ## var should be zer)
    #assert np.max(  np.var( np.array( last_windows ), axis =0 ) ) < 1e-10, last_windows

    X = []
    feat_order = []
    feature_names_all = []
    for feat_name in features_to_use:
        curfeat = feat_dict[feat_name]
        cd = curfeat['data'][:,:minlen]
        print('feat {} shape {}'.format(feat_name, cd.shape) )
        sfo = curfeat['names']
        if feat_name.find('corr') >= 0:
            subfeats = sfo
        else:
            subfeats = ['{}_{}'.format(feat_name,sf) for sf in sfo]
        feature_names_all += subfeats

        assert len(subfeats) == len(cd)
        X += [ cd]
        feat_order += [feat_name]
    X = np.vstack(X).T
    print('feat order {} and shape {}'.format(feat_order, X.shape) )

    feature_names_all_ = []
    for feat_name in feature_names_all:
        tmp = feat_name.replace('_allf','')
        feature_names_all_ += [tmp]
    feature_names_all = feature_names_all_

    feature_names_all_pri[rawind] = feature_names_all

    # vec_features can have some NaNs since pandas fills invalid indices with them
    assert not np.any( np.isnan ( X ) )
    assert X.dtype == np.dtype('float64')

    X_pri += [X]

    # here times of raws after first one are shfited
    #Xtimes = (nedgeBins / sfreq) + np.arange(X.shape[0]) * ( skip / sfreq )
    Xtimes = wbd[0,:minlen] / sfreq
    Xtimes_pri += [ Xtimes ]  # window starts in sec

    #assert abs( Xtimes[-1] -  times_pri[rawind][-1] ) < windowsz / sfreq

    wbd_H_pri[rawind] = wbd_H_pri[rawind][:,:minlen]  # because it will be used for saving

    print('Skip = {},  Xtimes number = {}'.format(skip, Xtimes.shape[0  ] ) )
#exec( open( '_run_featprep.py').read(), globals() )

# these may be NOT unique, they are just for correspondance
from featlist import replaceMEGsrcChnamesParams
subfeature_order_newsrcgrp_pri = [0] * len(rawnames)
for rawi in range(len(rawnames)):
    subfeature_order_newsrcgrp_pri[rawi] = \
        replaceMEGsrcChnamesParams(subfeature_order_pri[rawi], 0,9, '.*', 0)

nbins_estim_Kalman = 15
nbins_estim_Wiener = 5 # must be odd

X_pri_smooth = len(X_pri) * [ None ]
if do_Kalman:
    for rawind in range(len(X_pri)):
        X = X_pri[rawind]

        estim = X[:nbins_estim_Kalman,:].T


        X_smooth = ugf.smoothData(X.T,Tp_Kalman,estim, n_jobs=n_jobs).T
        X_pri_smooth[rawind] = X_smooth

if do_Wiener:
    for rawind in range(len(X_pri)):
        X = X_pri[rawind]
        #noise_est_shape = (nbins_estim_Wiener,1)
        noise_est_shape = nbins_estim_Wiener

        from scipy.signal import wiener
        X_smooth = wiener(X, mysize=noise_est_shape)
        #X_smooth = ugf.smoothData(X.T,Tp_Kalman,estim, n_jobs=n_jobs).T
        X_pri_smooth[rawind] = X_smooth


if save_feat:
    ind_shift = 0
    for rawind in range(len(rawnames) ):
        rawname_ = rawnames[rawind]
        #len_skipped = len( times_pri[rawind] ) // skip
        #start = ind_shift
        #end = ind_shift + len_skipped
        ##if rawind == 0:
        #start += nedgeBins // skip
        #end   -= nedgeBins // skip
        #feat_inds_cur = slice(start, end )
        #ind_shift += len_skipped

        #ts = times_pri_shifted[rawind]
        #feat_inds_cur = np.where( (Xtimes >= ts[0] + nedgeBins/sfreq ) * (Xtimes <= ts[-1] - nedgeBins/sfreq ) )[0]

        #X_cur = X[feat_inds_cur]

        #rawtimes_cur = times_pri[rawind]
        #Xtimes_cur = rawtimes_cur[nedgeBins:-nedgeBins  :skip]
        #assert len(Xtimes_cur ) == len(X_cur)

        X_cur = X_pri[rawind]
        Xtimes_cur = Xtimes_pri[rawind]

        if sources_type == 'HirschPt2011':
            st = ''
        else:
            st = sources_type

        #crp_str = ''
        #if crop_end is not None:
        #    crp_str = '_crop{}-{}'.format(int(crop_start),int(crop_end) )
        #a = '{}_feats_{}_{}chs_nfeats{}_skip{}_wsz{}_grp{}-{}{}.npz'.\
        #    format(rawname_,st,, X.shape[1], skip, windowsz,
        #          src_file_grouping_ind, src_grouping, crp_str)

        fname_feat  = utils.genFeatFn(rawname_,st, n_channels_pri[rawind],
                                      X.shape[1], skip, windowsz,
                                      src_file_grouping_ind, src_grouping,
                                      new_main_body_side,
                                      crop_start=None,crop_end=None)

        fname_feat_full = pjoin(gv.data_dir, output_subdir, fname_feat)


        # this contains not heavy things
        info = {}
        info['rawnames'] = rawnames  # maybe important if  I do common rescaling
        info['rawind'] = rawind

        info['bands_only'] = bands_only
        info['use_LFP_HFO'] = use_LFP_HFO
        info['use_lfp_HFO'] = use_LFP_HFO  # old ver, for compat
        #info['use_main_moveside'] = use_main_moveside
        info['brain_side_to_use'] = brain_side_to_use
        info['new_main_body_side'] = new_main_body_side
        info['use_main_LFP_chan'] = use_main_LFP_chan
        info['main_side_before_switch'] = aux_info_perraw[rn]['main_body_side']
        info['side_switched']           = aux_info_perraw[rn]['side_switched']
        info['data_modalities'] = data_modalities
        info['features_to_use'] = features_to_use
        info['msrc_inds'] = msrc_inds
        info['use_LFP_to_LFP'] = use_LFP_to_LFP
        info['spec_uselog'] = spec_uselog
        info['log_before_bandaver'] = log_before_bandaver
        info['log_during_csd'] = log_during_csd
        info['normalize_TFR'] = normalize_TFR
        info['freqs'] = freqs
        info['n_cycles'] = n_cycles
        info['freqs_inc_HFO'] = freqs_inc_HFO
        info['n_cycles_inc_HFO'] = n_cycles_inc_HFO
        info['nedgeBins'] = nedgeBins
        info['nedgeBins_highres'] = nedgeBins_hires
        info['percentileOffset'] = percentileOffset
        info['fbands'] = fbands
        info['skip_highres'] = skip_hires
        info['sfreq'] = sfreq
        info['src_grouping'] = src_grouping
        info['crop'] = crop_start,crop_end
        info['cross_types'] = cross_types
        info['src_type_to_use'] = src_type_to_use
        info['do_Kalman']  = do_Kalman
        info['do_Wiener']  = do_Wiener
        info['Tp_Kalman'] = Tp_Kalman
        info['rbcorr_use_local_means'] = rbcorr_use_local_means
        info['baseline_int'] = baseline_int

        info['chnames_newsrcgrp_pri'] = subfeature_order_newsrcgrp_pri
        info['chnames_pri'] = subfeature_order_pri
        info['fname_dat_full_pri'] = fname_dat_full_pri

        #r = raws_permod_both_sides[rawname_]
        #using r['src'].ch_names   would be WRONG !
        _,chnames_src = utsne.selFeatsRegex(None, newchns, ['msrc.*'])
        _,chnames_LFP = utsne.selFeatsRegex(None, newchns, ['LFP.*'])

        # I don't do anything with edgeBins all indices should be valid now
        # TODO: maybe I need to get wbd from somewhere else because I might
        # have changed something else
        # TODO: save dict(enumerate( _pri ))  so that numpy does not complain
        np.savez(fname_feat_full,Xtimes=Xtimes_cur,X=X_cur,
                 X_smooth=X_pri_smooth[rawind],
                 rawname_=rawname_,skip=skip,
                 wbd = wbd_H_pri[rawind],
                feature_names_all = feature_names_all, sfreq=sfreq,
                windowsz=windowsz, nedgeBins=nedgeBins, n_channels=n_channels_pri[rawind],
                rawtimes = times_pri[rawind], freqs=freqs, chnames_LFP=chnames_LFP,
                 chnames_src=chnames_src, feat_info = info,
                  pars=pars,
                 anndict_per_intcat=anndict_per_intcat_per_rawn[rawname_],
                 cmd=np.array([opts,args],dtype=object ),
                 rec_info = rec_info_pri[rawind],
                 cmd_opts=opts,cmd_args=args)
        #ip = feat_inds_cur[0],feat_inds_cur[-1]
        print('{} Features shape {} saved to\n  {}'.format(rawind,X_cur.shape,fname_feat_full) )

if show_plots and do_plot_feat_stat_scatter:
    int_types_to_stat = [it + '_{}'.format(main_side_let) for it in gp.int_types_basic]
    upre.plotFeatStatsScatter(rawnames,X_pri, int_types_to_stat,
                        feature_names_all,sfreq,
                        times_pri,side_switched_pri, wbd_pri=wbd_H_pri,
                                save_fig=False, separate_by = 'feat_type' )
    plt.suptitle('Feat stats (not rescaled)')
    pdf.savefig()
    plt.close()
    gc.collect()


if show_plots and do_plot_feat_stats:
    print('Starting plotting stats of features' )
    #  Plots stats for subsampled data
    utsne.plotBasicStatsMultiCh(X.T, feature_names_all, printMeans = 0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()



# Plot evolutions for subsampled data
if show_plots and do_plot_feat_timecourse:
    extdat = np.hstack(extdat_pri)
    extdat_resampled = [ np.convolve(extdat[i], np.ones(skip) , mode='same') for i in range(extdat.shape[0]) ]
    extdat_resampled = [ed[nedgeBins:-nedgeBins:skip] for ed in extdat_resampled]
    extdat_resampled = np.vstack(extdat_resampled)
    s = np.std(extdat_resampled, axis=1)[:,None]
    s = np.maximum(1e-10, s)
    extdat_resampled /= s

    print('Starting plotting timecourse of subsampled features' )

    for int_name in int_names:
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv

            # it does not make sense to plot ecg for subsampled
            tt = utsne.plotIntervalData(X.T,feature_names_all,iv,
                                        times = Xtimes,
                                        plot_types=['timecourse'],
                                        dat_ext = extdat_resampled[1:],
                                        extend=extend)

            ax_list = plt.gcf().axes
            for ax in ax_list:
                ax.set_rasterization_zorder(0)
            pdf.savefig(dpi=dpi)
            plt.close()

            #for DEBUG
            #pdf.close()
            #sys.exit(1)



if show_plots:
    pdf.close()
