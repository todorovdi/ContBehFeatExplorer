import os,sys
import mne
import utils  #my code
import json
import matplotlib.pyplot as plt
import numpy as np
import h5py
import multiprocessing as mpr
import matplotlib as mpl
import time
import gc;
import scipy.signal as sig
import getopt

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import utils_tSNE as utsne

from matplotlib.backends.backend_pdf import PdfPages
import utils_preproc as upre
import globvars as gv
from globvars import gp

if sys.argv[0].find('ipykernel_launcher') < 0:
    mpl.use('Agg')

############################

data_dir = gv.data_dir

############################

#rawname_ = 'S01_on_hold'  # no trem
#rawname_ = 'S01_on_move'
rawname_ = 'S01_off_hold'
rawname_ = 'S01_off_move'
#
#rawname_ = 'S02_off_hold'
#rawname_ = 'S02_on_hold'


rawnames = [rawname_]
#############################################################
#######  Main params

use_lfp_HFO = 1
use_main_tremorside = 1  # 0 both , -1 opposite
use_main_LFP_chan = False
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

feat_types_all = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr']
#features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
#features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr']
#features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl']
features_to_use = feat_types_all
#features_to_use = [ 'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
#assert 'bandcorrel' not in features_to_use, 'Not implemented yet'
#cross_types = [ ('LFP.*','.?src.*'), ('.*src_Cerebellum.*' , '.*motor-related.*' ) ]  # only couplings of LFP to sources
#cross_types = [ ('LFP.*','.?src.*'), ('.?src_.*' , '.?src.*' ) ]  # only couplings of LFP to sources
# LFP to sources and Cerebellum vs everything else


coupling_types = ['self', 'motorlike_vs_motorlike', 'LFP_vs_all']
cross_types =  []
if 'self' in coupling_types:
    cross_types += [ ('msrc_self','msrc_self'), ('LFP_self','LFP_self') ]
if 'LFP_vs_all' in coupling_types:
    cross_types += [ ('LFP.*','.?src.*') ]
if 'CB_vs_all' in coupling_types:
    cross_types += [ ('.?src_((?!Cerebellum).)+', '.?src_Cerebellum.*') ]
    exclude_CB_interactions = 1  # makes sense if we already included CB vs all
else:
    exclude_CB_interactions = 0

if 'motorlike_vs_motorlike' in coupling_types:
    for lbli in range(len(gp.areas_list_aal_my_guess)):
        for lblj in range(lbli+1, len(gp.areas_list_aal_my_guess)):
            lab1 = gp.areas_list_aal_my_guess[lbli]
            lab2 = gp.areas_list_aal_my_guess[lblj]
            if exclude_CB_interactions and (lab1.find('Cerebellum') >= 0\
                    or lab2.find('Cerebellum') >= 0):
                continue
            #side = ?
            templ1 = '.?src_{}.*'.format(lab1)
            templ2 = '.?src_{}.*'.format(lab2)
            cross_types += [  (templ1 ,templ2 )  ]
            #for sides

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
exit_after_TFR               = 0
#
load_TFR                     = 2
load_CSD                     = 2
save_TFR                     = 1 #gp.hostname != gp.hostname_home
save_CSD                     = 1 #gp.hostname != gp.hostname_home
use_existing_TFR             = 1  # for DEBUG in jupyter only
load_feat                    = 0
save_feat                    = 1
do_Kalman                    = 1


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
extend = 3  # for plotting, in seconds

fmax_raw_psd = 45
dpi = 200

#skip_div_TFR = 2
skip_div_TFR = 1

log_before_bandaver = False
spec_uselog = False
log_during_csd = True
normalize_TFR = True   # needed to avoid too small numbers

if normalize_TFR:
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

do_rescale_raws = False

##############################

print('sys.argv is ',sys.argv)
effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

helpstr = 'Usage example\nrun_genfeats.py --rawname <rawname_naked> '
opts, args = getopt.getopt(effargv,"hr:",
        ["mods=","msrc_inds=","feat_types=","bands=","rawname=",
            "show_plots=", "load_TFR=", "side=", "LFPchan=", "useHFO=",
         "plot_types=", "plot_only", "sources_type=", "crop=" ,
         "src_grouping=", "src_grouping_fn=",
         "load_TFR=", "save_TFR=", "save_CSD=", "load_CSD=", "use_existing_TFR=",
         "Kalman_smooth=", "save_bpcorr=", "save_rbcorr=", "rescale_raws=" ])
print('opts is ',opts)
print('args is ',args)

save_bpcorr = 0
save_rbcorr = 0

for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == "--msrc_inds":
        msrc_inds = [ int(indstr) for indstr in arg.split(',') ]
        msrc_inds = np.array(msrc_inds, dtype=int)
    elif opt == "--crop":
        if len(arg) > 2:
            cr =  arg.split(',')
            crop_start = float(cr[0])
            crop_end = float(cr[1] )
    elif opt == "--src_grouping":
        src_grouping = int(arg)
    elif opt == "--src_grouping_fn":
        src_file_grouping_ind = int(arg)
    elif opt == "--mods":
        data_modalities = arg.split(',')   #lfp of msrc
    elif opt == "--feat_types":
        features_to_use = arg.split(',')
        for ftu in features_to_use:
            assert ftu in feat_types_all, ftu
    elif opt == "--bands":
        bands_only = arg  #crude of fine
        assert bands_only in ['fine', 'crude']
    elif opt == "--sources_type":
        if len(arg):
            sources_type = arg
    elif opt in ('-r','--rawname'):
        if len(arg) < 5:
            print('Empty raw name provided, exiting')
            sys.exit(1)
        rawnames = arg.split(',')  #lfp of msrc
        for rn in rawnames:
            assert len(rn) > 3
        if len(rawnames) > 1:
            print('Using {} datasets at once'.format(len(rawnames) ) )
        #rawname_ = arg
    elif opt == '--show_plots':
        show_plots = int(arg)
    elif opt == '--load_TFR':
        load_TFR = int(arg)
    elif opt == '--side':
        if arg == 'both':
            use_main_tremorside == 0
        elif arg == 'main':
            use_main_tremorside == 1
        elif arg == 'other':
            use_main_tremorside == -1
        elif arg in ['left','right']:
            raise ValueError('to be implemented')
    elif opt == '--LFPchan':
        if arg == 'main':
            use_main_LFP_chan = 1
        elif arg == 'other':
            use_main_LFP_chan = 0
        elif arg.find('LFP') >= 0:   # maybe I want to specify expliclity channel name
            raise ValueError('to be implemented')
    elif opt == '--useHFO':
        use_lfp_HFO = int(arg)
    elif opt == '--plot_only':
        load_feat = 1
        save_feat = 0
        show_plots = 1
    elif opt == '--plot_types':
        plot_types = arg.split(',')  #lfp of msrc
        if 'raw_stats' in plot_types:
            do_plot_raw_stats = 1
        if 'raw_timecourse' in plot_types:
            do_plot_raw_timecourse = 1
        if 'raw_psd' in plot_types:
            do_plot_raw_psd = 1
        if 'csd' in plot_types:
            do_plot_CSD = 1
        if 'feat_stats' in plot_types:
            do_plot_feat_stats = 1
        if 'feat_timecourse' in plot_types:
            do_plot_feat_timecourse = 1
    elif opt == '--rescale_raws':    # makes sense to do within subject (where LFP channels are consistent)
        do_rescale_raws = int(arg)
    elif opt == '--load_TFR':
        load_TFR = int(arg)
    elif opt == '--load_CSD':
        load_CSD = int(arg)
    elif opt == '--save_TFR':
        save_TFR = int(arg)
    elif opt == '--save_CSD':
        save_CSD = int(arg)
    elif opt == '--save_rbcorr':
        save_rbcorr = int(arg)
    elif opt == '--save_bpcorr':
        save_bpcorr = int(arg)
    elif opt == '--Kalman_smooth':
        do_Kalman = int(arg)
    elif opt == '--use_existing_TFR':
        use_existing_TFR = int(arg)
    else:
        print('Unknown option ',opt,arg)


#if src_file_grouping_ind == 9:
#    raise ValueError("AAA")
##############################
test_mode = int(rawnames[0][1:3]) > 50
crop_to_integer_second = False

mods_to_load = ['LFP', 'src', 'EMG']
#mods_to_load = ['LFP', 'src', 'EMG', 'SSS','resample', 'FTraw']
#mods_to_load = ['LFP', 'src', 'EMG', 'resample', 'afterICA']
#mods_to_load = ['src', 'FTraw']
if use_lfp_HFO:
    mods_to_load += ['LFP_hires']

raws_permod_both_sides = upre.loadRaws(rawnames,mods_to_load, sources_type, src_type_to_use,
             src_file_grouping_ind)

sfreqs = [ int(raws_permod_both_sides[rn]['LFP'].info['sfreq']) for rn in rawnames]
assert len(set(sfreqs)) == 1
sfreq = sfreqs[0]

if use_lfp_HFO:
    sfreqs_hires = [ int(raws_permod_both_sides[rn]['LFP_hires'].info['sfreq']) for rn in rawnames]
    assert len(set(sfreqs_hires)) == 1
    sfreq_hires = sfreqs_hires[0]

if crop_to_integer_second:
    dt = 1/sfreq
    for rn in raws_permod_both_sides:
        for mod in raws_permod_both_sides[rn]:
            endt = raws_permod_both_sides[rn][mod].times[-1]
            if abs( endt - (int(endt) - dt) ) > dt/2:
                newend = int(endt) - dt
                print('crop to integer from {} to {}'.format(endt,newend ) )
                raws_permod_both_sides[rn][mod].crop(0,newend )

if crop_end is not None:
    for rn in raws_permod_both_sides:
        for mod in raws_permod_both_sides[rn]:
            raws_permod_both_sides[rn][mod].crop(crop_start,crop_end)



################
src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rawname_,sources_type,src_file_grouping_ind)
src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)

roi_labels = rec_info['label_groups_dict'][()]      # dict of (orderd) lists with keys from srcgroups_key_order
srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]  # this is usually just one item

# it does not really make sense if I don't work with them together anyway
# probably rescaling features in the very end makes more sense
if do_rescale_raws:
    upre.rescaleRaws(raws_permod_both_sides, mod='LFP', combine_within_medcond=True)  # rescales LFP in place

    upre.rescaleRaws(raws_permod_both_sides, mod='src',combine_within_medcond=True,
                    roi_labels=roi_labels, srcgrouping_names_sorted=srcgrouping_names_sorted)
    gc.collect()

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
    pdf= PdfPages( os.path.join(gv.dir_fig,  '{}_feat_plots{}_side{}_LFP{}_{}_nmod{}_nfeattp{}.pdf'.format(
        rawname_,show_plots, use_main_tremorside, use_main_LFP_chan, bands_only,
        len(data_modalities),len(features_to_use)   )) )

#############################################################



# get info about bad MEG channels (from separate file)
gen_subj_info = gv.gen_subj_info

subj,medcond,task  = utils.getParamsFromRawname(rawname_)
# these are just to play and plot because the side can change from patient to
# patient
maintremside = gen_subj_info[subj]['tremor_side']
mainmoveside = gen_subj_info[subj].get('move_side',None)
mainLFPchan = gen_subj_info[subj]['lfpchan_used_in_paper']
tremfreq_Jan = gen_subj_info[subj]['tremfreq']
print('main trem side and LFP ',maintremside, mainLFPchan)


move_sides = []
tremor_sides = []
for rn in rawnames:
    subj_cur,medcond_cur,task_cur  = utils.getParamsFromRawname(rn)
    mainmoveside_cur = gen_subj_info[subj_cur].get('move_side',None)
    maintremside_cur = gen_subj_info[subj_cur].get('tremor_side',None)
    mainLFPchan_cur =      gen_subj_info[subj_cur]['lfpchan_used_in_paper']
    tremfreq_Jan_cur =     gen_subj_info[subj_cur]['tremfreq']

    move_sides  += [ mainmoveside_cur ]
    tremor_sides+= [ maintremside_cur ]


def_main_side = 'left'
side_to_use = 'move_side'

if side_to_use == 'move_side':
    if len(set(move_sides) ) == 1 and move_sides[0] is not None:
        main_side = move_sides[0]
        new_main_side = main_side
    else:
        main_side = 'undef'
        print('Setting new main side to be ',def_main_side)
        new_main_side = def_main_side
elif side_to_use == 'tremor_side':
    if len(set(tremor_sides) ) == 1:
        main_side = tremor_sides[0]
        new_main_side = main_side
    else:
        main_side = 'undef'
        new_main_side = def_main_side
        print('Setting new main side to be ',def_main_side)
elif side_to_use in ['left','right']:  # forcibly
    main_side = side_to_use
    new_main_side = main_side
else:
    main_side = 'undef'
    raise ValueError('wrong side_to_use')

#mts_letter = maintremside[0].upper()

ms_letter = main_side[0].upper()
int_names = ['{}_{}'.format(task,ms_letter), 'trem_{}'.format(ms_letter), 'notrem_{}'.format(ms_letter)]

###########################################################

fbands = gv.fbands
#{'tremor': [3,10], 'low_beta':[11,22], 'high_beta':[22,30],
#           'low_gamma':[30,60], 'high_gamma':[60,90],
#          'HFO1':[91,200], 'HFO2':[200,300], 'HFO3':[300,400],
#          'beta':[15,30],   'gamma':[30,100], 'HFO':[91,400]}


fband_names_crude = ['tremor', 'beta', 'gamma']
fband_names_fine = ['tremor', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma' ]
fband_names_HFO_crude = ['HFO']
fband_names_HFO_fine =  ['HFO1', 'HFO2', 'HFO3']
fband_names_HFO_all = fband_names_HFO_crude + fband_names_HFO_fine
fband_names_crude_inc_HFO = fband_names_crude + fband_names_HFO_crude
fband_names_fine_inc_HFO = fband_names_fine + fband_names_HFO_fine

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


# the output is dat only from the selected hemisphere
dat_pri, dat_lfp_hires_pri, extdat_pri, anns_pri, ivalis_pri, rec_info_pri, times_pri,\
times_hires_pri, subfeature_order_pri, subfeature_order_lfp_hires_pri, aux_info_perraw = \
    utils.collectDataFromMultiRaws(rawnames, raws_permod_both_sides, sources_type,
                             src_file_grouping_ind, src_grouping, use_main_LFP_chan,
                             side_to_use, new_main_side, data_modalities,
                             only_load_data,crop_start,crop_end,msrc_inds)

main_sides_pri = [ aux_info_perraw[rn]['main_body_side'] for rn in rawnames]
side_switched_pri = [ aux_info_perraw[rn]['side_switched'] for rn in rawnames]

subfeature_order = subfeature_order_pri[0]
subfeature_order_lfp_hires = subfeature_order_lfp_hires_pri[0]

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
srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]  # order of grouping names
assert set(roi_labels.keys() ) == set(srcgrouping_names_sorted )
assert len(roi_labels) == 1, 'several groupings in single run -- not implmemented'
# assuming we have only one grouping present
#roi_labels_cur = roi_labels[srcgrouping_names_sorted[src_grouping ]  ]

########### merge interval dictionaries

times_pri_shifted = [times_pri[0] ]
ivalis = ivalis_pri[0]
tshift = 0
for rawind in range(1, len(rawnames) ):
    ivalis_cur = ivalis_pri[rawind]
    if rawind > 0:
        tshift += len(times_pri[rawind-1] )/ sfreq
    times_pri_shifted += [ tshift + times_pri[rawind]  ]
    for it in ivalis_cur:
        upd = [ (s+tshift,e+tshift,it) for s,e,it in ivalis_cur[it]  ]
        if it in ivalis:
            ivalis[it] += upd
        else:
            ivalis[it] = upd

times = np.hstack(times_pri_shifted)
dft = np.diff(times)
dftmn,dftmx = np.min(dft),np.max(dft)
assert ( abs(dftmn - 1/sfreq) < 1e-10 ) and ( abs(dftmx - 1/sfreq) < 1e-10 )

print('Total raw data length for {} datasets is {} bins (={}s)'.format(len(rawnames), len(times),
                                                                       len(times) // sfreq ) )

if only_load_data:
    print(' only_load_data, exiting! ')
    sys.exit(1)
############# scale raw data

scale_data_before_featgen = True

if scale_data_before_featgen:
    print('Start data prescaling')
    # scale channel data just so that we don't have numerical problems later
    for rawind in range(len(dat_pri)):
        datcur = dat_pri[rawind]
        #dat_scaled = np.zeros(datcur.shape)
        for i in range( datcur.shape[0] ):
            #inp = dat_artif_nan[i]
            #inp = inp[ np.logical_not(np.isnan(inp ) ) ] [:,None]
            inp_transform = datcur[i][:,None]
            inp = inp_transform
            # TODO: fit to data without aftifacts, apply to whole
            scaler = RobustScaler(quantile_range=(percentileOffset,100-percentileOffset), with_centering=1)
            scaler.fit(inp)
            outd = scaler.transform(inp_transform)
            #dat_scaled[i,:] = outd[:,0]
            datcur[i,:] = outd[:,0]

        assert not np.any( np.isnan ( datcur ) )

        if use_lfp_HFO:
            dat_lfp_hires = dat_lfp_hires_pri[rawind]
            #dat_lfp_hires_scaled = np.zeros(dat_lfp_hires.shape)
            for i in range( dat_lfp_hires.shape[0] ):
                #inp = dat_lfp_hires_artif_nan[i]
                #inp = inp[ np.logical_not(np.isnan(inp ) ) ][:,None]
                inp_transform = dat_lfp_hires[i][:,None]
                inp = inp_transform

                scaler = RobustScaler(quantile_range=(percentileOffset,100-percentileOffset), with_centering=1)
                scaler.fit(inp)
                outd = scaler.transform(inp_transform)
                #dat_lfp_hires_scaled[i,:] = outd[:,0]
                dat_lfp_hires[i,:] = outd[:,0]


###############  plot interval data

if do_plot_raw_psd:
    print('Starting plottign raw psd')
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv
            #utsne.plotIntervalData(dat_scaled,subfeature_order,iv, times=times,
            #                    plot_types = ['psd'], fmax=fmax_raw_psd )
            utsne.plotIntervalData(dat_pri[0],subfeature_order,iv, times=times,
                                plot_types = ['psd'], fmax=fmax_raw_psd )

            for ax in plt.gcf().get_axes():
                ax.axvline(x=tremfreq_Jan, ls=':')
            pdf.savefig()
            plt.close()


############## plot raw stats
if do_plot_raw_stats:
    #utsne.plotBasicStatsMultiCh(dat_scaled, subfeature_order, printMeans = 0)
    utsne.plotBasicStatsMultiCh(dat_pri[0], subfeature_order, printMeans = 0)
    plt.tight_layout()
    pdf.savefig()

    if use_lfp_HFO:
        #utsne.plotBasicStatsMultiCh(dat_lfp_hires_scaled, subfeature_order_lfp_hires,
        #                            printMeans = 0)
        utsne.plotBasicStatsMultiCh(dat_lfp_hires, subfeature_order_lfp_hires,
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
            raw_lfponly = raws_permod_both_sides[rawnames[0]]['LFP']
            #tt = utsne.plotIntervalData(dat_scaled,subfeature_order,iv,
            #                            raw=raw_lfponly, plot_types=['timecourse'], dat_ext = extdat,
            #                            extend=extend)
            tt = utsne.plotIntervalData(dat_pri[0],subfeature_order,iv,
                                        raw=raw_lfponly, plot_types=['timecourse'], dat_ext = extdat,
                                        extend=extend)
            ax_list = plt.gcf().axes
            for ax in ax_list:
                ax.set_rasterization_zorder(0)
            pdf.savefig(dpi=dpi)
            plt.close()

##################

##################

rawnstr = ','.join(rawnames)
a = '{}_tfr_{}chs.npz'.format(rawnstr,n_channels_str)
fname_tfr_full = os.path.join( data_dir,a)

a = '{}_csd_{}chs.npz'.format(rawnstr,n_channels_str)
fname_csd_full = os.path.join( data_dir,a)


a = '{}_feats_{}chs_skip{}_wsz{}.npz'.\
    format(rawname_,n_channels_str, skip, windowsz)
fname_feat_full = os.path.join( data_dir,a)
######################

# later I would add features that are too long to compute for every time point,
# so I don't want to save X itself

do_cleanup = not test_mode

if load_feat and os.path.exists(fname_feat_full):
    print('Loading feats from ',fname_feat_full)
    f = np.load(fname_feat_full)
    X =  f['X']
    Xtimes = f ['Xtimes']
    skip =f['skip']
    feature_names_all = f['feature_names_all']

else:
    '''
    if I compute all source couplings first and then do something with them
    it takes a lot of memory perhaps
    '''
    #exec( open( '_run_featprep.py').read(), globals() )

    # compute diagonal terms
    have_TFR = False
    try:
        print('tfr (existing) num channels =', tfrres_pri[0][0].shape[0] )
    except NameError as e:
        have_TFR = False
    else:
        have_TFR = True #and (tfrres.shape[0] == n_channels)

    #if not have_TFR:
    #    print('OOO')
    #    sys.exit(0)
    #def computeTFR():
    #    return

    cond_load_CSD = (load_CSD > 0) and os.path.exists( fname_csd_full )
    cond2_CSD = ( (load_CSD == 1) or \
                            (load_CSD == 2 and upre.getFileAge(fname_csd_full) < load_CSD_max_age_h) )

    if not (use_existing_TFR and have_TFR):
        cond_load_TFR = (load_TFR > 0) and os.path.exists( fname_tfr_full )
        cond2 = ( (load_TFR == 1) or \
                              (load_TFR == 2 and upre.getFileAge(fname_tfr_full) < load_TFR_max_age_h) )

        # if we will load CSD, we don't need TFR
        if cond_load_TFR and cond2  and not ( cond_load_CSD and cond2_CSD ):
            print('Start loading TFR from {}'.format(fname_tfr_full) )
            tfrf = np.load(fname_tfr_full, allow_pickle=True)
            tfrres_pri_dict = tfrf['tfrres_pri'][()]
            tfrres_wbd_pri_dict = tfrf['tfrres_wbd_pri'][()]
            tfrres_wbd_HFO_pri_dict = tfrf['tfrres_wbd_HFO_pri'][()]
            tfrres_LFP_HFO_pri_dict = tfrf['tfrres_LFP_HFO_pri'][()]
            tfrres_pri = [0] *len(tfrres_pri_dict)
            tfrres_wbd_pri = [0] *len(tfrres_pri_dict)
            tfrres_wbd_HFO_pri = [0] *len(tfrres_pri_dict)
            tfrres_LFP_HFO_pri = [0] *len(tfrres_pri_dict)
            for i in range(len(tfrres_pri)):
                tfrres_pri[i] = tfrres_pri_dict[i]
                tfrres_wbd_pri[i] = tfrres_wbd_pri_dict[i]
                tfrres_wbd_HFO_pri[i] = tfrres_wbd_HFO_pri_dict[i]
                tfrres_LFP_HFO_pri[i] = tfrres_LFP_HFO_pri_dict[i]
            chnames_tfr = tfrf['chnames_tfr'][()]
            #chnames_tfr = list(names_src)  + subfeature_order_lfp_hires
        elif not ( cond_load_CSD and cond2_CSD ):
            tfrres_pri = []
            tfrres_wbd_pri = []
            tfrres_wbd_HFO_pri = []
            #tfrres_LFP_LFO_pri = []
            tfrres_LFP_HFO_pri = []

            for rawind,dat_cur in enumerate(dat_pri):
                #print('Starting TFR for data with shape ',dat_scaled.shape)

                #dat_scaled_src,names_src = utsne.selFeatsRegex(dat_scaled, subfeature_order, ['msrc.*'])
                #dat_scaled_lfp,names_lfp = utsne.selFeatsRegex(dat_scaled, subfeature_order, ['LFP.*'])
                #dat_src,names_src = utsne.selFeatsRegex(dat_cur, subfeature_order, ['msrc.*'])
                #dat_lfp,names_lfp = utsne.selFeatsRegex(dat_cur, subfeature_order, ['LFP.*'])
                dat_cur_src,names_src = utsne.selFeatsRegex(dat_cur, subfeature_order, ['msrc.*'])
                dat_cur_lfp,names_lfp = utsne.selFeatsRegex(dat_cur, subfeature_order, ['LFP.*'])
                # if we use hires LFP raw, better to do entire TFR on LFP
                if use_lfp_HFO:
                    #dat_for_tfr = dat_scaled_src
                    dat_for_tfr = dat_cur_src
                    chnames_tfr = names_src
                else:
                    #dat_for_tfr = dat_scaled
                    dat_for_tfr = dat_cur
                    chnames_tfr = subfeature_order

                # perhaps we want to ensure that wavelets intersect well. Then it's
                # better to use smaller skip and then downsample
                print('Starting TFR for data #{} with shape {}'.format(rawind,dat_for_tfr.shape) )
                assert ( skip - (skip // skip_div_TFR)  * skip_div_TFR ) < 1e-10

                tfrres_,wbd = utils.tfr(dat_for_tfr, sfreq, freqs, n_cycles, windowsz, decim = skip // skip_div_TFR,
                                    n_jobs=n_jobs)
                if skip_div_TFR > 1:
                    raise ValueError('wbd not debugged for that')
                    tfrres = utsne.downsample(tfrres_, skip_div_TFR, axis=-1)
                else:
                    tfrres = tfrres_

                if use_lfp_HFO:

                    #dat_for_tfr = dat_lfp_hires_scaled
                    #dat_for_tfr = dat_lfp_hires
                    dat_for_tfr = dat_lfp_hires_pri[rawind]

                    print('Starting TFR for LFP HFO data #{} with shape {}'.format(rawind,dat_for_tfr.shape) )
                    tfrres_LFP_,wbd_HFO = utils.tfr(dat_for_tfr, sfreq_hires, freqs_inc_HFO, n_cycles_inc_HFO,
                                        windowsz_hires, decim = skip_hires // skip_div_TFR, n_jobs=n_jobs)
                    if skip_div_TFR > 1:
                        raise ValueError('wbd not debugged for that')
                        tfrres_LFP = utsne.downsample(tfrres_LFP_, skip_div_TFR, axis=-1)
                    else:
                        tfrres_LFP = tfrres_LFP_


                    tfrres_LFP_LFO = tfrres_LFP[:,:len(freqs),:]
                    tfrres_LFP_HFO = tfrres_LFP[:,len(freqs):,:]

                    #tfrres_LFP_LFO_pri += [tfrres_LFP_HFO] # for debug
                    tfrres_LFP_HFO_pri += [tfrres_LFP_HFO]
                    tfrres_wbd_HFO_pri += [wbd_HFO]

                    tfrres = np.concatenate( [tfrres, tfrres_LFP_LFO], axis=0 )
                    chnames_tfr = chnames_tfr.tolist()  + subfeature_order_lfp_hires

                    # no I have to do it later, I cannot vstack because it has differtn
                    # freq count
                    #assert tfrres.shape[-1] = tfrres_LFP.shape[-1]
                    #tfrres = np.vstack( [tfrres, tfrres_LFP] )
                    #subfeature_order = chnames_tfr + subfeature_order_lfp_hires

                tfrres_pri += [ tfrres ]
                tfrres_wbd_pri += [wbd]
                assert not ( np.any( np.isnan ( tfrres ) ) or np.any( np.isinf ( tfrres ) ) )
                gc.collect()

            if save_TFR:
                # savez does not like list of dicts :(
                dct = dict( enumerate(tfrres_pri) )
                dct2 =dict( enumerate(tfrres_LFP_HFO_pri) )
                dct3 =dict(enumerate(tfrres_wbd_pri))
                dct4 =dict(  enumerate(tfrres_wbd_HFO_pri))
                np.savez(fname_tfr_full, tfrres_pri=dct, tfrres_LFP_HFO_pri=dct2,
                            tfrres_wbd_pri = dct3, tfrres_wbd_HFO_pri = dct4,
                         chnames_tfr = chnames_tfr)
                print('TFR saved to ',fname_tfr_full)

            del tfrres

    if cond_load_CSD and cond2_CSD:
        print('Start loading CSD from {}'.format(fname_csd_full) )
        csdf = np.load(fname_csd_full, allow_pickle=True)
        csd_pri_    = csdf['csd_pri'][()]
        csdord_pri_ =csdf['csdord_pri'][()]
        csd_LFP_HFO_pri_    = csdf['csd_LFP_HFO_pri'][()]
        csdord_LFP_HFO_pri_ =csdf['csdord_LFP_HFO_pri'][()]

        tfrres_wbd_pri_dict = csdf['tfrres_wbd_pri'][()]


        csd_pri = [0]    * len( csd_pri_.keys() )
        csdord_pri = [0] * len( csdord_pri_.keys() )
        csd_LFP_HFO_pri    = [0] * len( csd_LFP_HFO_pri_.keys() )
        csdord_LFP_HFO_pri = [0] * len( csdord_LFP_HFO_pri_.keys() )
        tfrres_wbd_pri = [0]    * len( tfrres_wbd_pri_dict.keys() )

        for i in range(len(csdord_pri) ):
            csd_pri[i]    = csd_pri_[i]
            csdord_pri[i] = csdord_pri_[i]
            csd_LFP_HFO_pri   [i] = csd_LFP_HFO_pri_   [i]
            csdord_LFP_HFO_pri[i] = csdord_LFP_HFO_pri_[i]
            tfrres_wbd_pri[i]    = tfrres_wbd_pri_dict[i]


        newchns=list( csdf['newchns'][()] )
        res_couplings=csdf['res_couplings'][()]
        ind_distr, ind_distr_parcels, ind_pairs_parcelsLFP, \
            parcel_couplings, LFP2parcel_couplings, LFP2LFP_couplings = res_couplings

        _,names_src = utsne.selFeatsRegex(None, subfeature_order, ['msrc.*'])
        _,names_lfp = utsne.selFeatsRegex(None, subfeature_order, ['LFP.*'])
        chnames_tfr = list(names_src) + list(names_lfp)
    else:
        chnames_nicened = utils.nicenMEGsrc_chnames(chnames_tfr, roi_labels, srcgrouping_names_sorted,
                                prefix='msrc_')

        # since I have TFR in separate chunks, I implement the computation by hand
        # note that it is different from rescaling of the raws in the beginning
        # -- here I don't need to unify the scales, only to get rid of too
        # small values (because later when computing CSD I will mutiply them
        # and it falls below double precision)
        if normalize_TFR:
            print('Start computing TFR stats for normalization')
            s1,s2,s3 = tfrres_pri[0].shape
            s = np.zeros( (s1,s2 ), dtype=np.complex )
            nb = 0
            for tfrres_cur in tfrres_pri:
                s += np.sum(tfrres_cur, axis=-1)
                nb += tfrres_cur.shape[-1]
            tfr_mean = s / nb

            var = np.zeros( (s1,s2 ), dtype=np.complex )
            for tfrres_cur in tfrres_pri:
                y = tfrres_cur - tfr_mean[:,:,None]
                var += np.sum( y * np.conj(y) , axis=-1)
            tfr_std = np.sqrt( var / nb )


            s1,s2,s3 = tfrres_LFP_HFO_pri[0].shape
            s = np.zeros( (s1,s2 ), dtype=np.complex )
            nb = 0
            for tfrres_cur in tfrres_LFP_HFO_pri:
                s += np.sum(tfrres_cur, axis=-1)
                nb += tfrres_cur.shape[-1]
            tfr_LFP_HFO_mean = s / nb

            var = np.zeros( (s1,s2 ), dtype=np.complex )
            for tfrres_cur in tfrres_LFP_HFO_pri:
                y = tfrres_cur - tfr_LFP_HFO_mean[:,:,None]
                var += np.sum( y * np.conj(y) , axis=-1)
            tfr_LFP_HFO_std = np.sqrt( var / nb )

        LFP2LFP_only_self = True  # that we don't want to compute cross couplings LFP to LFP
        # we DO NOT want only upper diag because cross_types does not
        # does not contain symmetric entries
        res_couplings = utils.selectIndPairs(chnames_nicened, chnames_tfr, cross_types, upper_diag=False,
                                LFP2LFP_only_self=LFP2LFP_only_self, cross_within_parcel=False)

        ind_distr, ind_distr_parcels, ind_pairs_parcelsLFP, \
            parcel_couplings, LFP2parcel_couplings, LFP2LFP_couplings = res_couplings
        import itertools
        ind_distr_merged = list(itertools.chain.from_iterable(ind_distr))
        print('Starting CSD for {} pairs'.format( len(ind_distr_merged) ) )

        # chreate new chnames
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
        newchns = aa + np.array(chnames_tfr)[lfpinds].tolist()

        csd_pri = [];
        csdord_pri = []
        for tfri,tfrres_cur in enumerate(tfrres_pri):
            tfrres_cur_ = tfrres_cur
            if normalize_TFR:
                tfrres_cur_ =  (tfrres_cur - tfr_mean[:,:,None] ) / tfr_std[:,:,None]
            csd_cur, csdord = utils.tfr2csd(tfrres_cur_, sfreq, returnOrder=1,
                                            ind_pairs=None,
                                            parcel_couplings=parcel_couplings,
                                            LFP2LFP_couplings=LFP2LFP_couplings,
                                            LFP2parcel_couplings=LFP2parcel_couplings,
                                            oldchns=chnames_tfr,
                                            newchns=newchns,
                                            res_group_id=newchn_grouping_ind,
                                            log=log_during_csd)
            # csdord.shape = (2, csdsize)
            csdord_pri += [csdord]
            csd_pri += [csd_cur]
            gc.collect()

        csdord_LFP_HFO = None
        csd_LFP_HFO_pri = []
        csdord_LFP_HFO_pri = []
        for tfri,tfrres_LFP_HFO_cur in enumerate(tfrres_LFP_HFO_pri):
            tfrres_LFP_HFO_cur_ = tfrres_LFP_HFO_cur
            if normalize_TFR:
                tfrres_LFP_HFO_cur_ =  (tfrres_LFP_HFO_cur - tfr_LFP_HFO_mean[:,:,None] ) / tfr_LFP_HFO_std[:,:,None]
            # I don't really need HFO csd across LFP contacts
            #csd_LFP, csdord_LFP = utils.tfr2csd(tfrres_LFP, sfreq_hires, returnOrder=1)  # csdord.shape = (2, csdsize)
            csd_LFP_HFO_cur = tfrres_LFP_HFO_cur_ * np.conj(tfrres_LFP_HFO_cur_)
            tmp = np.arange( tfrres_LFP_HFO_cur.shape[0] )  # n_LFP_channels
            csdord_LFP_HFO = np.vstack([tmp,tmp] ) # same to same index, so just i->i

            csdord_LFP_HFO_pri += [csdord_LFP_HFO]
            csd_LFP_HFO_pri += [csd_LFP_HFO_cur]
        gc.collect()


        # remember that we have rescaled raws before so concatenating should be ok
        #csd = np.concatenate(csd_pri,axis=-1)
        #csd_LFP_HFO = np.concatenate(csd_LFP_HFO_pri,axis=-1)

        if save_CSD:
            #dct = {}
            print('Saving CSD to {}'.format(fname_csd_full) )
            dct1 = dict(    enumerate(csd_pri)  )
            dct2 = dict(    enumerate(csdord_pri)  )
            dct3 = dict(    enumerate(csd_LFP_HFO_pri)  )
            dct4 = dict(    enumerate(csdord_LFP_HFO_pri)  )
            dct5 =dict(enumerate(tfrres_wbd_pri))
            np.savez(fname_csd_full, csd_pri=dct1, csdord_pri=dct2,
                     newchns=newchns, res_couplings=res_couplings,
                     csd_LFP_HFO_pri = dct3, csdord_LFP_HFO_pri=dct4,
                     tfrres_wbd_pri=dct5)

        if do_cleanup:
            del tfrres_pri

    for csdi in range(len(csd_pri) ):
        assert not ( np.any( np.isnan ( csd_pri[csdi] ) )    or np.any( np.isinf ( csd_pri[csdi] ) )    )
        assert not ( np.any( np.isnan ( csd_LFP_HFO_pri[csdi] ) )  or np.any( np.isinf ( csd_LFP_HFO_pri[csdi] ) )    )
    gc.collect()

    ntimebins = csd_pri[0].shape[-1]

    if exit_after_TFR:
        print( ' exit_after_TFR, exiting')
        sys.exit(1)

    n_channels_new = len(newchns)

    if bands_only == 'fine':
        fband_names = fband_names_fine
    else:
        fband_names = fband_names_crude

    if bands_only == 'fine':
        fband_names_inc_HFO = fband_names_fine_inc_HFO
    else:
        fband_names_inc_HFO = fband_names_crude_inc_HFO
        fband_names_HFO = fband_names_inc_HFO[len(fband_names):]  # that HFO names go after


    print('Averaging over freqs within bands')
    #csdord_strs is longer than csdord because it
    #currently (Jan 5, 2021) bpow_imagcsd is useless
    # here I assume that channel counts are the same across datasets
    bpow_abscsd_pri, bpow_imagcsd, csdord_strs_pri, csdord_strs_HFO_pri,bpow_abscsd_LFP_HFO_pri  = \
        utils.bandAverage( freqs,freqs_inc_HFO,csd_pri,csdord_pri,csdord_LFP_HFO_pri,
               csd_LFP_HFO_pri, fbands,fband_names, fband_names_inc_HFO,
               newchns, subfeature_order_lfp_hires, log_before_bandaver= log_before_bandaver )
    if do_cleanup:
        del csd_pri
        del csdord_LFP_HFO_pri
    gc.collect()

    #################################  Plot CSD at some time point
    if do_plot_CSD:
        for int_name in int_names:
            #nt_name = 'trem_{}'.format(maintremside)
            #intervals = ivalis[int_name]
            intervals = ivalis.get(int_name,[])
            for iv in intervals:
                start,end,_itp = iv

                raw_srconly = raws_permod_both_sides[rawnames[0]]['src']
                ts,r, int_names = utsne.getIntervalSurround(start,end,extend, raw_srconly)

                #timebins = raw_lfponly.time_as_index
                utsne.plotCSD(csd_pri[0], fband_names, chnames_tfr, list(r) , sfreq=sfreq, intervalMode=1,
                            int_names=int_names)
                pdf.savefig()
                plt.close()


    ############################## Hjorth

    if 'H_act' in features_to_use or 'H_mob' in features_to_use or 'H_compl' in features_to_use:
        print('Computing Hjorth')

        #dat_total = np.hstack( dat_pri )   # it should be already scaled since I did rescale raws
        #dat_src,names_src = utsne.selFeatsRegex(dat_total, subfeature_order, ['msrc.*'])
        #dat_lfp,names_lfp = utsne.selFeatsRegex(dat_total, subfeature_order, ['LFP.*'])
        dat_src_pri,names_src = utsne.selFeatsRegex(dat_pri, subfeature_order, ['msrc.*'])

        #TODO maybe it would be better to make a simple coupling between old names and new names
        #act,mob,compl  = utils.Hjorth(dat_scaled_src, 1/sfreq, windowsz=windowsz)
        #n_newchns_srconly = len(newchns) - dat_scaled_lfp.shape[0]
        act_pri,mob_pri,compl_pri,wbd_H_pri  = utils.Hjorth(dat_src_pri, 1/sfreq,
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

        #if use_lfp_HFO:
        #    dat_for_H = dat_scaled_src
        #else:
        #    dat_for_H = dat_scaled
        #act,mob,compl  = utils.Hjorth(dat_for_H, 1/sfreq, windowsz=windowsz)


        # if we have LFP data we better obtain Hjorth parameter from hires data
        if use_lfp_HFO:
            #dat_lfp_hires_total = np.hstack( dat_lfp_hires_pri )
            #act_lfp,mob_lfp,compl_lfp  = utils.Hjorth(dat_lfp_hires_scaled, 1/sfreq_hires,
            #                            windowsz=int( (windowsz/sfreq)*sfreq_hires ) )
            act_lfp_pri,mob_lfp_pri,compl_lfp_pri,wbd_H_lfp_hires_pri  =\
                utils.Hjorth(dat_lfp_hires_pri, 1/sfreq_hires, windowsz=int(
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
                utils.Hjorth(dat_lfp_pri, 1/sfreq, windowsz=windowsz, skip=skip,
                            remove_invalid=True )

        for acti in range(len(act_pri) ):
            w1 = wbd_H_pri[acti]
            w2 = wbd_H_lfp_hires_pri[acti]
            assert w1[0][0] == w2[0][0]
            ml = min( w1.shape[1], w2.shape[1] )
            assert w1[1][ml-1] == int( w2[1][ml-1] / sfreq_hires * sfreq)
            subsl = slice(0,ml,None)
            sl = (slice(None,None,None), subsl )
            act_pri[acti]   = np.vstack( [  act_pri[acti][sl] ,  act_lfp_pri[acti][sl] ] )
            mob_pri[acti]   = np.vstack( [  mob_pri[acti][sl] ,  mob_lfp_pri[acti][sl] ] )
            compl_pri[acti] = np.vstack( [compl_pri[acti][sl], compl_lfp_pri[acti][sl] ] )

            wbd_H_pri[acti]           = w1[sl]
            wbd_H_lfp_hires_pri[acti] = w2 [sl]

        #act_lfp   = np.hstack(act_lfp  )
        #mob_lfp   = np.hstack(mob_lfp  )
        #compl_lfp = np.hstack(compl_lfp)

        if show_plots and do_plot_Hjorth:
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
                axs[0,0].plot(times,act[i], label= subfeature_order[i])
                axs[0,0].set_title('activity')
                axs[0,1].plot(times,mob[i], label= subfeature_order[i])
                axs[0,1].set_title('mobility')
                axs[0,2].plot(times,compl[i], label= subfeature_order[i])
                axs[0,2].set_title('complexity')

            for ax in axs.reshape(axs.size):
                ax.legend(loc='upper right')

            pdf.savefig()


        #act = act[:,::skip]
        #mob = mob[:,::skip]
        #compl = compl[:,::skip]
        gc.collect()

    for rawind in range(len(dat_pri)):
        a,b = tfrres_wbd_pri[rawind], wbd_H_pri[rawind]
        if a.shape != b.shape:
            print('Warning, tfr and Hjorth have different shapes {},{}, two last {}, {}'.
                  format ( a.shape,b.shape, a[1][-2:],b[1][-2:] ) )
        else:
            assert np.max (np.abs (a-b) ) < 1e-10

    #Xtimes_full = raw_srconly.times[nedgeBins:-nedgeBins]
    #####################################################

    if 'rbcorr' in features_to_use or 'bpcorr' in features_to_use:
        # we have done it before as well
        if bands_only == 'fine':
            fband_names = fband_names_fine
        else:
            fband_names = fband_names_crude

        print('Filtering and Hilbert')

        sfreqs = [sfreq, sfreq_hires]
        skips = [skip, skip_hires]
        dat_pri_persfreq = [dat_pri, dat_lfp_hires_pri]

        n_jobs_flt = n_jobs

        # note that we can have different channel names for different raws
        #raw_perband_flt_pri_persfreq = []
        #raw_perband_bp_pri_persfreq = []
        # TODO: set artifacts before band filtering
        smoothen_bandpow = 0
        raw_perband_flt_pri = []
        raw_perband_bp_pri  = []
        chnames_perband_flt_pri = []
        chnames_perband_bp_pri = []
        means_perband_flt_pri = []
        means_perband_bp_pri  = []
        for rawind in range(len(dat_pri)):
            raw_perband_flt = {}
            raw_perband_bp  = {}
            chnames_perband_flt = {}
            chnames_perband_bp  = {}
            means_perband_flt = {}
            means_perband_bp  = {}

            rn = rawnames[rawind]
            main_side_before_change = main_sides_pri[rawind]  # side of body
            opsidelet = utils.getOppositeSideStr(main_side_before_change[0].upper() ) # side of brain

            fname_full_LFPartif = os.path.join(gv.data_dir, '{}_ann_LFPartif.txt'.format(rn) )
            fname_full_MEGartif = os.path.join(gv.data_dir, '{}_ann_MEGartif.txt'.format(rn) )
            anns_LFPartif = mne.read_annotations(fname_full_LFPartif)
            anns_MEGartif = mne.read_annotations(fname_full_MEGartif)

            main_side_before_change = main_sides_pri[rawind]  # side of body
            opsidelet = utils.getOppositeSideStr(main_side_before_change[0].upper() ) # side of brain
            wrong_brain_sidelet = main_side_before_change[0].upper()
            anns_LFPartif = utils.removeAnnsByDescr(anns_LFPartif, ['artif_LFP{}'.format(wrong_brain_sidelet) ])

            for bandi,bandname in enumerate(fband_names_inc_HFO):
                means_perchan_flt = {}
                means_perchan_bp = {}
                for si,dat_pri_cur in enumerate(dat_pri_persfreq):
                    sfreq_cur = sfreqs[si]
                    # for hires we will only process HFO bands
                    if sfreq_cur > sfreq + 1e-10 and (bandname not in fband_names_HFO_all):
                        continue
                    if abs(sfreq-sfreq_cur) < 1e-10 and (bandname in fband_names_HFO_all):
                        continue
                    print(si,bandname)
                    bpow_mav_wsz = skips[si] // 2
                    wnd_mav = np.ones( bpow_mav_wsz )


                    dat_cur = dat_pri_cur[rawind]
                    low,high = fbands[bandname]
                    r = utils.makeSimpleRaw(dat_cur, ch_names=None, sfreq=sfreq_cur)
                    if bandname.find('HFO') < 0:
                        r.set_annotations(anns_MEGartif)
                        r.set_annotations(anns_LFPartif)
                    else: # we have only LFP channels then
                        r.set_annotations(anns_LFPartif)
                    r.filter(l_freq=low,h_freq=high, n_jobs=n_jobs_flt,
                            skip_by_annotation='BAD_{}'.format(opsidelet), pad='symmetric')#, filter_length="1s" )

                    if sfreq_cur <= sfreq + 1e-10:
                        rbp = r.copy()
                        chnames_perband_flt [bandname] =chnames_tfr
                        chnames_perband_bp [bandname]  =chnames_tfr
                    else:
                        rbp = r  # we won't need filtered itself for hires
                        chnames_perband_bp [bandname] =subfeature_order_lfp_hires
                    rbp.apply_hilbert()
                    # zeroth is convolution with 1 elemnt. After strideing it
                    # should be like in tfr/Hjorth
                    if smoothen_bandpow:
                        # or maybe for hires I can do it after resampling with
                        # a smaller window
                        fn = lambda x: np.convolve(wnd_mav, np.abs(x), mode='full' )[:dat_cur.shape[-1] ]
                    else:
                        fn = np.abs
                    rbp.apply_function(fn, n_jobs=n_jobs_flt, dtype=np.float)

                    #raw_perband_flt += [ r  ]
                    #raw_perband_bp  += [ rbp]
                    if sfreq_cur <= sfreq + 1e-10:
                        assert bandname not in raw_perband_flt
                        raw_perband_flt[bandname] =  r
                    else:
                        rbp.resample(sfreq=sfreq,n_jobs=n_jobs_flt)

                    assert bandname not in raw_perband_bp
                    raw_perband_bp [bandname] =  rbp

                    q_to_include_mean_comp = 0.05
                    # rejects bad.*
                    mflt = utsne.robustMean(r.get_data(reject_by_annotation='omit'),
                                     q=q_to_include_mean_comp, per_dim=True, axis=-1)
                    mrb = utsne.robustMeanPos(rbp.get_data(reject_by_annotation='omit'),
                                        q=q_to_include_mean_comp, per_dim=True,axis=-1)
                    means_perchan_flt = mflt
                    means_perchan_bp = mrb
                means_perband_flt[bandname] = means_perchan_flt
                means_perband_bp[bandname] = means_perchan_bp

            raw_perband_bp_pri  += [raw_perband_bp ]
            raw_perband_flt_pri += [raw_perband_flt]

            chnames_perband_flt_pri += [chnames_perband_flt]
            chnames_perband_bp_pri  += [chnames_perband_bp]

            means_perband_flt_pri += [means_perband_flt]
            means_perband_bp_pri  += [means_perband_bp]

    if 'rbcorr' in features_to_use:  #raw band corr
        if bands_only == 'fine':
            bandPairs = [('tremor','tremor', 'corr'),
                        ('low_beta','low_beta', 'corr'), ('high_beta','high_beta', 'corr'),
                        ('low_gamma','low_gamma', 'corr') , ('high_gamma','high_gamma', 'corr') ]
        else:
            bandPairs = [('tremor','tremor', 'corr'), ('beta','beta', 'corr'), ('gamma','gamma', 'corr') ]



        rbcorrs_pri = []
        wbd_rbcorr_pri = []
        rbcor_names_pri = []
        for rawind in range(len(dat_pri)):
            print('Starting rbcorr for datset {}'.format(rawind) )
            raw_perband =  raw_perband_flt_pri[rawind]
            chnames_perband =  chnames_perband_flt_pri[rawind]
            means_perband = means_perband_flt_pri[rawind]

            rbcorrs = []
            rbcor_names = []
            for bp in bandPairs:
                print('band pair ',bp)
                rbcorrs_curbp,rbcor_names_curbp,dct_nums, wbd_rbcorr = utsne.computeFeatOrd3(raw_perband, names=chnames_perband,
                                                        defnames=chnames_tfr,
                                                        parcel_couplings=parcel_couplings,
                                                        LFP2parcel_couplings=LFP2parcel_couplings,
                                                        LFP2LFP_couplings=LFP2LFP_couplings,
                                                        res_group_id=newchn_grouping_ind,
                                                        skip=skip, windowsz = windowsz, band_pairs = [bp],
                                                        n_jobs=n_jobs, positive=0, templ='{}_.*',
                                                        roi_labels=roi_labels, sort_keys=srcgrouping_names_sorted,
                                                                means=means_perband)
                assert len(rbcor_names_curbp) > 0
                gc.collect()
                rbcorrs     += rbcorrs_curbp
                rbcor_names += rbcor_names_curbp

            rbcorrs = np.vstack(rbcorrs)
            rbcorrs_pri += [rbcorrs]
            wbd_rbcorr_pri += [wbd_rbcorr]

            for feati in range(len(rbcor_names) ):
                rbcor_names[feati] = 'rb' + rbcor_names[feati]

            rbcor_names_pri += [rbcor_names]


            if save_rbcorr:
                fn = os.path.join(data_dir,'{}_rbcorr.npz'.format(rawnames[rawind] ) )
                print('Saving rbcorr ',fn)
                np.savez( fn , rbcorrs=rbcorrs)
                gc.collect()

        del  raw_perband_flt_pri
        del  rbcorrs
        gc.collect()


    else:
        rbcorrs_pri = None
        rbcor_names = None
        wbd_rbcorr_pri = None

    if 'bpcorr' in features_to_use:
        if bands_only == 'fine':
            # no need to dubplicate reversing order
            bandPairs = [('tremor','low_beta', 'corr'), ('tremor','high_beta', 'corr'),
                        ('tremor','low_gamma', 'corr'), ('tremor','high_gamma', 'corr'),
                        ('low_beta','low_gamma', 'corr') , ('low_beta','high_gamma', 'corr') ,
                        ('high_beta','low_gamma', 'corr') , ('high_beta','high_gamma', 'corr') ]
        else:
            bandPairs = [('tremor','beta', 'corr'), ('tremor','gamma', 'corr'), ('beta','gamma', 'corr') ]
        if use_lfp_HFO:
            # HFO should be always on the second place unless we use HFO-HFO
            # coupling in LFP!
            bandPairs += [ ('tremor','HFO', 'corr'), ('beta','HFO', 'corr'), ('gamma','HFO', 'corr') ]

            if bands_only == 'fine':
                bandPairs += [ ('HFO1','HFO2', 'div'), ('HFO1','HFO3', 'div'), ('HFO2','HFO3', 'div') ]

        bpcorrs_pri = []
        wbd_bpcorr_pri = []
        bpcor_names_pri = []
        for rawind in range(len(dat_pri)):
            print('Starting bpcorr for datset {}'.format(rawind) )

            raw_perband =  raw_perband_bp_pri[rawind]
            chnames_perband =  chnames_perband_bp_pri[rawind]
            means_perband = means_perband_bp_pri[rawind]
            bpcorrs = []
            bpcor_names = []
            for bp in bandPairs:
                print('band pair ',bp)
                bpcorrs_curbp,bpcor_names_curbp,dct_nums,wbd_bpcorr =\
                utsne.computeFeatOrd3(raw_perband, names=chnames_perband,
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
            bpcorrs_pri += [bpcorrs]
            wbd_bpcorr_pri += [wbd_bpcorr]

            for feati in range(len(rbcor_names) ):
                bpcor_names[feati] = 'bp' + rbcor_names[feati]

            bpcor_names_pri += [bpcor_names]

            if save_bpcorr:
                fn = os.path.join(data_dir,'{}_bpcorr.npz'.format(rawnames[rawind] ) )
                print('Saving bpcorr ',fn)
                np.savez( fn , bpcorrs=bpcorrs)
                gc.collect()

        del  bpcorrs
        del  raw_perband_bp_pri
        gc.collect()

    else:
        bpcorrs_pri = None
        bpcor_names = None
        wbd_bpcorr_pri = None


    # at these stage numbers of channels and channel names should be consistent
    # across datasets

    ######################################################
    Xtimes_pri = []
    X_pri = []

    defpct = (percentileOffset,100-percentileOffset)
    log_was_applied = spec_uselog or log_during_csd or log_before_bandaver
    center_spec_feats = log_was_applied
    if log_was_applied:
        con_scale = defpct
    else:
        con_scale = (0,100-percentileOffset)

    for rawind in range(len(dat_pri) ):
        feat_dict = { 'con':{'data': None, 'pct':con_scale, 'centering':center_spec_feats,
                             'names':None, 'wbd':tfrres_wbd_pri[rawind] },
                    'H_act':{'data':act_pri[rawind], 'pct':defpct, 'names':newchns, 'wbd':wbd_H_pri[rawind]},
                    'H_mob':{'data':mob_pri[rawind], 'pct':defpct, 'names':newchns, 'wbd':wbd_H_pri[rawind]},
                    'H_compl':{'data':compl_pri[rawind], 'pct':defpct, 'names':newchns, 'wbd':wbd_H_pri[rawind]} }
        if 'bpcorr' in features_to_use:
            feat_dict['bpcorr'] = {'data':bpcorrs_pri[rawind], 'pct':defpct,
                                   'names':bpcor_names_pri[rawind],
                                   'wbd':wbd_bpcorr_pri[rawind]}
        if 'rbcorr' in features_to_use:
            feat_dict['rbcorr'] = {'data':rbcorrs_pri[rawind], 'pct':defpct,
                                   'names':rbcor_names_pri[rawind],
                                   'wbd':wbd_rbcorr_pri[rawind]}
        #'tfr':{'data':None, 'pct':con_scale, 'centering':center_spec_feats },

        feat_dict['con']['centering'] = True

        f = lambda x: x
        if spec_uselog:
            f = lambda x: np.log(x)

        if bands_only == 'no':
            tfres_ = tfrres.reshape( tfrres.size//ntimebins , ntimebins )
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
            ncsds,nfreqs,ntimebins_ = bpow_abscsd.shape
            bpow_abscds_reshaped = bpow_abscsd.reshape( ncsds*nfreqs, ntimebins_ )
            assert bpow_abscds_reshaped.shape[0] == len(csdord_strs)
            feat_dict['con']['names'] = csdord_strs[:]
            if use_lfp_HFO:
                bpow_abscds_LFP_HFO_reshaped = bpow_abscsd_LFP_HFO.reshape(
                    bpow_abscsd_LFP_HFO.size//ntimebins_ , ntimebins_ )
                # add HFO to low freq
                bpow_abscds_all_reshaped = np.vstack( [bpow_abscds_reshaped, bpow_abscds_LFP_HFO_reshaped])
                #TODO: note that csdord_strs by that moment already contains LFP HFO
                #names (see when csdord_strs i generated)
                feat_dict['con']['names'] += csdord_strs_HFO_pri[rawind]
            else:
                bpow_abscds_all_reshaped = bpow_abscds_reshaped

            assert len(bpow_abscds_all_reshaped) == len(feat_dict['con']['names'] )

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
                    bpow_abscds_all_reshaped = bpow_abscds_all_reshaped[gi]

                    feat_dict['con']['names'] = np.array(feat_dict['con']['names'])[gi]

            feat_dict['con']['data'] = f( bpow_abscds_all_reshaped )
            #if use_imag_coh:
            #    feat_dict['con']['data'] = f( tmp )

            #tmp_ord
            #csdord1 = csdord_bandwise.reshape( (bpow_abscds_reshaped.shape[0], 3) )
            #if use_lfp_HFO:
            #    csdord2 = csdord_bandwise_LFP_HFO.reshape( (bpow_abscds_LFP_HFO_reshaped.shape[0], 3) )
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


        artif_mod_str = [ '_ann_LFPartif', '_ann_MEGartif' ]
        #anns_artif, anns_artif_pri, times2, dataset_bounds = utsne.concatArtif(rawnames[rawind],times_pri[rawind])


        anns_artif, anns_artif_pri, times2, dataset_bounds_ = \
            utsne.concatAnns(rawnames[rawind],times_pri[rawind], artif_mod_str,crop=(crop_start,crop_end),
                        allow_short_intervals=True,
                             side_rev_pri = aux_info_perraw[rawnames[rawind]]['side_switched'],
                             wbd_pri = wbd_H_pri[rawind], sfreq=sfreq)

        # here we use new side instead of old because we have done the reversal in  concatAnns
        wrong_brain_sidelet = new_main_side.upper()
        anns_artif = utils.removeAnnsByDescr(anns_artif, ['artif_LFP{}'.format(wrong_brain_sidelet) ])

        ivalis_artif = utils.ann2ivalDict(anns_artif)
        #ivalis_artif_tb, ivalis_artif_tb_indarrays = utsne.getAnnBins(ivalis_artif, Xtimes,
        #                                                            (nedgeBins / sfreq),
        #                                                            sfreq, skip, windowsz, dataset_bounds)
        #ivalis_artif_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)


        #################################  Scale features
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
            assert cd.shape[1] == wbd.shape[1]

        last_windows = []
        for feat_name in features_to_use:
            curfeat = feat_dict[feat_name]
            last_windows += [ wbd[:,minlen-1] ]
        # variance on window starts per feature (windows need to be algned to
        # var should be zer)
        assert np.max(  np.var( np.array( last_windows ), axis =0 ) ) < 1e-10, last_windows

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

        # vec_features can have some NaNs since pandas fills invalid indices with them
        assert not np.any( np.isnan ( X ) )
        assert X.dtype == np.dtype('float64')

        X_pri += [X]

        # here times of raws after first one are shfited
        #Xtimes = (nedgeBins / sfreq) + np.arange(X.shape[0]) * ( skip / sfreq )
        Xtimes = wbd[0,:minlen] / sfreq
        Xtimes_pri += [ Xtimes ]  # window starts in sec

        wbd_H_pri[rawind] = wbd_H_pri[rawind][:,:minlen]  # because it will be used for saving

        print('Skip = {},  Xtimes number = {}'.format(skip, Xtimes.shape[0  ] ) )
    #exec( open( '_run_featprep.py').read(), globals() )


    if do_Kalman:
        for rawind in range(len(X_pri)):
            X = X_pri[rawind]
            nbins_estim_Kalman = 15
            estim = X[:nbins_estim_Kalman,:].T


            X_nonsmooth = X
            X_smooth = utils.smoothData(X.T,Tp_Kalman,estim, n_jobs=n_jobs).T
            X_pri[rawind] = X_smooth


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

        crp_str = ''
        if crop_end is not None:
            crp_str = '_crop{}-{}'.format(int(crop_start),int(crop_end) )
        a = '{}_feats_{}_{}chs_nfeats{}_skip{}_wsz{}_grp{}-{}{}.npz'.\
            format(rawname_,st,n_channels_pri[rawind], X.shape[1], skip, windowsz,
                  src_file_grouping_ind, src_grouping, crp_str)
        fname_feat_full = os.path.join( data_dir,a)

        # this contains not heavy things
        info = {}
        info['rawnames'] = rawnames  # maybe important if  I do common rescaling
        info['rawind'] = rawind
        info['rescale_raws'] = do_rescale_raws

        info['bands_only'] = bands_only
        info['use_lfp_HFO'] = use_lfp_HFO
        info['use_main_tremorside'] = use_main_tremorside
        info['side_to_use'] = side_to_use
        info['new_main_side'] = new_main_side
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
        info['fband_names_crude'] = fband_names_crude
        info['fband_names_fine'] = fband_names_fine
        info['fband_names_crude_inc_HFO'] = fband_names_crude_inc_HFO
        info['fband_names_fine_inc_HFO'] = fband_names_fine_inc_HFO
        info['skip_highres'] = skip_hires
        info['sfreq'] = sfreq
        info['src_grouping'] = src_grouping
        info['crop'] = crop_start,crop_end
        info['cross_types'] = cross_types
        info['src_type_to_use'] = src_type_to_use
        info['do_Kalman']  = do_Kalman
        info['Tp_Kalman'] = Tp_Kalman

        #r = raws_permod_both_sides[rawname_]
        #using r['src'].ch_names   would be WRONG !
        _,chnames_src = utsne.selFeatsRegex(None, newchns, ['msrc.*'])
        _,chnames_LFP = utsne.selFeatsRegex(None, newchns, ['LFP.*'])

        # I don't do anything with edgeBins all indices should be valid now
        # TODO: maybe I need to get wbd from somewhere else because I might
        # have changed something else
        np.savez(fname_feat_full,Xtimes=Xtimes_cur,X=X_cur,rawname_=rawname_,skip=skip,
                 wbd = wbd_H_pri[rawind],
                feature_names_all = feature_names_all, sfreq=sfreq,
                windowsz=windowsz, nedgeBins=nedgeBins, n_channels=n_channels_pri[rawind],
                rawtimes = times_pri[rawind], freqs=freqs, chnames_LFP=chnames_LFP,
                 chnames_src=chnames_src, feat_info = info)
        #ip = feat_inds_cur[0],feat_inds_cur[-1]
        print('{} Features shape {} saved to\n  {}'.format(rawind,X_cur.shape,fname_feat_full) )


if do_plot_feat_stats:
    print('Starting plotting stats of features' )
    #  Plots stats for subsampled data
    utsne.plotBasicStatsMultiCh(X.T, feature_names_all, printMeans = 0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()



# Plot evolutions for subsampled data
if do_plot_feat_timecourse:
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
