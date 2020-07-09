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

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import utils_tSNE as utsne

from matplotlib.backends.backend_pdf import PdfPages
import utils_preproc as upre

mpl.use('Agg')

############################

if os.environ.get('DATA_DUSS') is not None:
    data_dir = os.path.expandvars('$DATA_DUSS')
else:
    data_dir = '/home/demitau/data'

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
bands_only = 'crude'  #'fine'  or 'no'
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
features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr']
#features_to_use = [ 'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
#assert 'bandcorrel' not in features_to_use, 'Not implemented yet'
msrc_inds = np.arange(8,dtype=int)  #indices appearing in channel (sources) names, not channnel indices

use_LFP_to_LFP = 0

corr_time_lagged = []
corr_time_lagged = [0.5, 1] # in window size fractions

extend = 3  # for plotting


##########################  ploting params

load_TFR                     = 0
save_TFR                     = 0  # maybe better no to waste space, since I change params often
use_existing_TFR             = 1
load_feat                    = 0
save_feat                    = 1

show_plots                   = 0
#
do_plot_raw_stats            = 1 * show_plots
do_plot_raw_psd              = 1 * show_plots
do_plot_raw_timecourse       = 1 * show_plots
#do_plot_feat_timecourse_full = 0 * show_plots
#do_plot_feat_stats_full      = 0 * show_plots
do_plot_feat_timecourse      = 0 * show_plots
do_plot_feat_stats           = 1 * show_plots
do_plot_CSD                  = 0 * show_plots

fmax_raw_psd = 45

##############################
import sys, getopt

print('sys.argv is ',sys.argv)
effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

helpstr = 'Usage example\nrun_genfeats.py --rawname <rawname_naked> '
opts, args = getopt.getopt(effargv,"hr:",
        ["mods=","msrc_inds=","feat_types=","bands=","rawname=",
            "show_plots=", "load_TFR=", "side=", "LFPchan=", "useHFO=",
         "plot_types=", "plot_only" ])
print(sys.argv, opts, args)

for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == "--msrc_inds":
        msrc_inds = [ int(indstr) for indstr in arg.split(',') ]
        msrc_inds = np.array(msrc_inds, dtype=int)
    elif opt == "--mods":
        data_modalities = arg.split(',')   #lfp of msrc
    elif opt == "--feat_types":
        features_to_use = arg.split(',')
        for ftu in features_to_use:
            assert ftu in feat_types_all, ftu
    elif opt == "--bands":
        bands_only = arg  #crude of fine
        assert bands_only in ['fine', 'crude']
    elif opt in ('-r','--rawname'):
        rawnames = arg.split(',')  #lfp of msrc
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


##############################

raws_lfponly = []
raws_lfp_highres = []
raws_srconly = []
raws_emg_rectconv = []

for rawname_ in rawnames:
    print('!!!!!! current rawname --- ',rawname_)

    upre.saveRectConv(rawname_, skip_if_exist = 1)

    src_fname_noext = 'srcd_{}_HirschPt2011'.format(rawname_)

    rawname_LFPonly = rawname_ + '_LFPonly'+ '.fif'
    rawname_LFPonly_full = os.path.join( data_dir, rawname_LFPonly )
    raw_lfponly = mne.io.read_raw_fif(rawname_LFPonly_full, None)
    if use_lfp_HFO:
        raw_lfp_highres = upre.saveLFP_nonresampled(rawname_, skip_if_exist = 1)
        if raw_lfp_highres is None:
            lfp_fname_full = os.path.join(data_dir, '{}_LFP_1kHz.fif'.format(rawname_) )
            raw_lfp_highres = mne.io.read_raw_fif(lfp_fname_full)

        sfreq_highres = raw_lfp_highres.info['sfreq']
        sfreq_highres = int(sfreq_highres)

    newsrc_fname_full = os.path.join( data_dir, 'av_' + src_fname_noext + '.fif' )
    raw_srconly =  mne.io.read_raw_fif(newsrc_fname_full, None)

    rectconv_fname_full = os.path.join(data_dir, '{}_emg_rectconv.fif'.format(rawname_) )
    raw_emg_rectconv = mne.io.read_raw_fif(rectconv_fname_full)

    raws_lfponly       += [ raw_lfponly ]
    raws_lfp_highres   += [ raw_lfp_highres   ]
    raws_srconly       += [ raw_srconly   ]
    raws_emg_rectconv  += [ raw_emg_rectconv   ]

sfreq = int(raw_srconly.info['sfreq'])

####################  data processing params

spec_uselog = 1

min_freq = 3
#freq_step = 1
#max_freq = 100
#freqs = np.arange(min_freq,max_freq,freq_step)
freqs,n_cycles = utils.prepFreqs(min_freq = min_freq, max_freq = 90)
freqs_inc_HFO,n_cycles_inc_HFO = utils.prepFreqs(min_freq = min_freq, max_freq = 400)

windowsz  =  1 * sfreq
windowsz_highres  =  1 * sfreq_highres
# because my definition of wavelets for low freq has larger window
nedgeBins = int( windowsz * 1.5 )
nedgeBins_highres = int( windowsz_highres * 1.5 )
# (longest window width (right - left) = 0.93s, shortest = 0.233, ~ 1/4 of a sec
# so sampling at 1/8 of a sec is safe

# I want 1 sec window sz
#cf =  windowsz/ ( 5/(2*np.pi) * sfreq  )
#freq2cycles_mult = cf  # 1.2566370614359172
#print('cf= ',cf)

percentileOffset = 25

import globvars as gv
################################

if show_plots:
    pdf= PdfPages( os.path.join(gv.dir_fig,  '{}_feat_plots{}_side{}_LFP{}_{}_nmod{}_nfeattp{}.pdf'.format(
        rawname_,show_plots, use_main_tremorside, use_main_LFP_chan, bands_only,
        len(data_modalities),len(features_to_use)   )) )

#############################################################



# get info about bad MEG channels (from separate file)
with open('subj_info.json') as info_json:
    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)

subj,medcond,task  = utils.getParamsFromRawname(rawname_)
maintremside = gen_subj_info[subj]['tremor_side']
mainLFPchan = gen_subj_info[subj]['lfpchan_used_in_paper']
tremfreq_Jan = gen_subj_info[subj]['tremfreq']
print('main trem side and LFP ',maintremside, mainLFPchan)


mts_letter = maintremside[0].upper()
int_names = ['{}_{}'.format(task,mts_letter), 'trem_{}'.format(mts_letter), 'notrem_{}'.format(mts_letter)]

###########################################################

fbands = {'tremor': [3,10], 'low_beta':[11,22], 'high_beta':[22,30],
           'low_gamma':[30,60], 'high_gamma':[60,90],
          'HFO1':[91,200], 'HFO2':[200,300], 'HFO3':[300,400],
          'beta':[15,30],   'gamma':[30,100], 'HFO':[91,400]}



fband_names_crude = ['tremor', 'beta', 'gamma']
fband_names_fine = ['tremor', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma' ]
fband_names_crude_inc_HFO = fband_names_crude + ['HFO']
fband_names_fine_inc_HFO = fband_names_fine + ['HFO1', 'HFO2', 'HFO3']

#######################
if subsample_type == 'half':
    skip = (1 * sfreq) // 2
elif subsample_type == 'desiredNpts':
    ntimebins = None
    skip = ntimebins // desiredNpts
elif subsample_type == 'prescribedSkip':
    skip = prescribedSkip
skip_highres = sfreq_highres //   (sfreq  // skip)

#############################################################

dat_pri = []
times_pri = []
times_highres_pri = []
dats_lfp_highres_pri = []
ivalis_pri = []

extdat_pri = []

for rawind in range(len(rawnames) ):
    rawname_ = rawnames[rawind]


    anns_fn = rawname_ + '_anns.txt'
    anns_fn_full = os.path.join(data_dir, anns_fn)
    anns = mne.read_annotations(anns_fn_full)

    ivalis_pri += [ utils.ann2ivalDict(anns) ]

    #############################################################



    raw_lfponly       = raws_lfponly[rawind]
    raw_srconly       = raws_srconly[rawind]
    raw_lfp_highres   = raws_lfp_highres[rawind]
    raw_emg_rectconv  = raws_emg_rectconv[rawind]

    raw_lfponly.load_data()
    raw_srconly.load_data()
    raw_lfp_highres.load_data()

    sides = ['L', 'R'] # brain sides   # this is just for construction of data, we will restrict later
    hand_sides_all = ['L', 'R']  # hand sides

    raws_lfp_perside = {'L': raw_lfponly.copy(), 'R': raw_lfponly.copy() }
    raws_lfp_highres_perside = {'L': raw_lfp_highres.copy(), 'R': raw_lfp_highres.copy() }
    raws_srconly_perside = {'L': raw_srconly.copy(), 'R': raw_srconly.copy() }
    for side in sides:
        chis = mne.pick_channels_regexp(raw_lfponly.ch_names, 'LFP{}.*'.format(side))
        chnames_lfp = [raw_lfponly.ch_names[chi] for chi in chis]
        if use_main_LFP_chan and mainLFPchan in chnames_lfp:
            raws_lfp_perside[side].pick_channels(   [mainLFPchan]  )
            if use_lfp_HFO:
                raws_lfp_highres_perside[side].pick_channels(   [mainLFPchan]  )
        else:
            raws_lfp_perside[side].pick_channels(   chnames_lfp  )
            if use_lfp_HFO:
                raws_lfp_highres_perside[side].pick_channels(   chnames_lfp  )
        print(raws_lfp_perside[side].ch_names)


        chis =  mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_all_*'.format(side)  )
        chnames_src = [raw_srconly.ch_names[chi] for chi in chis]
        raws_srconly_perside[side].pick_channels(   chnames_src  )


    #TODO

    ####################  Load emg


    EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
    if use_main_tremorside:
        chnames_emg = EMG_per_hand[maintremside]
    else:
        chnames_emg = raw_emg_rectconv.ch_names
    print(rawname_,chnames_emg)

    rectconv_emg, ts_ = raw_emg_rectconv[chnames_emg]
    chnames_emg = [chn+'_rectconv' for chn in chnames_emg]


    ############# Concat
    raws_permod = {'LFP' : raws_lfp_perside, 'msrc' : raws_srconly_perside }
    if use_main_tremorside:
        hand_sides = [mts_letter ]
    else:
        hand_sides = hand_sides_all
    print('maintremside {}, hand_sides to construct features '.format(mts_letter) ,hand_sides)

    allowd_srcis_subregex = '[{}]'.format( ','.join( map(str, msrc_inds ) ))
    subfeature_order = []
    dats = []
    for side_hand in hand_sides:
        for mod in data_modalities:
            sd = hand_sides_all[1-hand_sides.index(side_hand) ]  #
            #if mod in ['src','msrc']:  No! They are both in the brain, so both contralat!

            curraw = raws_permod[mod][sd]

            if mod == 'msrc':
                chns = curraw.ch_names
                inds = mne.pick_channels_regexp(  chns  , 'msrc.*_{}'.format(allowd_srcis_subregex) )
                chns_selected = list( np.array(chns)[inds]  )
                curdat, times = curraw[chns_selected]
                #msrc_inds
                chnames_added = chns_selected
            else:
                curdat = curraw.get_data()
                chnames_added = curraw.ch_names
            dats += [ curdat ]

            subfeature_order += chnames_added
            print(mod,sd)
    #dats = {'lfp': dat_lfp, 'msrc':dat_src}
    dat = np.vstack(dats)
    n_channels = dat.shape[0];
    times = raw_srconly.times

    dat_pri += [dat]
    times_pri += [times]
    times_highres_pri += [raw_lfp_highres.times]

    if use_lfp_HFO:
        dats_lfp_highres = []
        subfeature_order_lfp_highres = []
        for side_hand in hand_sides:
            sd = hand_sides_all[1-hand_sides.index(side_hand) ]  #
            curraw = raws_lfp_highres_perside[sd]
            curdat  = curraw.get_data()
            chnames_added = curraw.ch_names
            subfeature_order_lfp_highres += chnames_added
            dats_lfp_highres += [ curdat]

        dat_lfp_highres = np.vstack(dats_lfp_highres)

    dats_lfp_highres_pri += [dat_lfp_highres]


    f = np.load( os.path.join(data_dir, '{}_ica_ecg.npz'.format(rawname_) ) )
    ecg = f['ecg']
    #ecg_normalized = (ecg - np.min(ecg) )/( np.max(ecg) - np.min(ecg) )
    ecg_normalized = (ecg - np.mean(ecg) )/( np.quantile(ecg,0.93) - np.quantile(ecg,0.01) )

    extdat_pri += [ np.vstack( [ecg, ecg_normalized] ) ]

extnames = ['ecg'] + chnames_emg
extdat = np.hstack( extdat_pri )

dat = np.hstack( dat_pri )
dat_lfp_highres = np.hstack( dats_lfp_highres_pri )


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

print('Total raw data length for {} datasets is {} bins (={}s)'.format(len(rawnames), len(times),
                                                                       len(times) // sfreq ) )

fname_full_LFPartif = os.path.join(gv.data_dir, '{}_ann_LFPartif.txt'.format(rawname_) )
fname_full_MEGartif = os.path.join(gv.data_dir, '{}_ann_MEGartif.txt'.format(rawname_) )

################

anns_artif, anns_artif_pri, times2, dataset_bounds = utsne.concatArtif(rawnames,times_pri)
ivalis_artif = utils.ann2ivalDict(anns_artif)
ivalis_artif_tb, ivalis_artif_tb_indarrays = utsne.getAnnBins(ivalis_artif, times,
                                                            nedgeBins, sfreq, skip, 1, dataset_bounds)
ivalis_artif_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)

dat_artif_nan  = utils.setArtifNaN(dat.T, ivalis_artif_tb_indarrays_merged, subfeature_order).T

##

anns_artif, anns_artif_pri, times2_highres, dataset_bounds_highres = utsne.concatArtif(rawnames,times_highres_pri)
ivalis_artif = utils.ann2ivalDict(anns_artif)
ivalis_artif_highres_tb, ivalis_artif_highres_tb_indarrays = utsne.getAnnBins(ivalis_artif, times2_highres,
                                                            nedgeBins_highres, sfreq_highres, skip_highres,
                                                                              1, dataset_bounds_highres)
ivalis_artif_highres_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_highres_tb_indarrays)

dat_lfp_highres_artif_nan  = \
    utils.setArtifNaN(dat_lfp_highres.T, ivalis_artif_highres_tb_indarrays_merged,
                      subfeature_order_lfp_highres).T

############# scale raw data


# scale channel data separately
dat_scaled = np.zeros(dat.shape)
for i in range( dat.shape[0] ):
    #inp = dat[i][:,None]

    inp = dat_artif_nan[i]
    inp = inp[ np.logical_not(np.isnan(inp ) ) ] [:,None]
    inp_transform = dat[i][:,None]
    # TODO: fit to data without aftifacts, apply to whole
    scaler = RobustScaler(quantile_range=(percentileOffset,100-percentileOffset), with_centering=1)
    scaler.fit(inp)
    outd = scaler.transform(inp_transform)
    dat_scaled[i,:] = outd[:,0]

if use_lfp_HFO:
    dat_lfp_highres_scaled = np.zeros(dat_lfp_highres.shape)
    for i in range( dat_lfp_highres.shape[0] ):
        #inp = dat_lfp_highres[i][:,None]

        inp = dat_lfp_highres_artif_nan[i]
        inp = inp[ np.logical_not(np.isnan(inp ) ) ][:,None]
        inp_transform = dat_lfp_highres[i][:,None]

        scaler = RobustScaler(quantile_range=(percentileOffset,100-percentileOffset), with_centering=1)
        scaler.fit(inp)
        outd = scaler.transform(inp_transform)
        dat_lfp_highres_scaled[i,:] = outd[:,0]


###############  plot interval data

if do_plot_raw_psd:
    print('Starting plottign raw psd')
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv
            utsne.plotIntervalData(dat_scaled,subfeature_order,iv, times=times,
                                plot_types = ['psd'], fmax=fmax_raw_psd )

            for ax in plt.gcf().get_axes():
                ax.axvline(x=tremfreq_Jan, ls=':')
            pdf.savefig()
            plt.close()


############## plot raw stats
if do_plot_raw_stats:
    utsne.plotBasicStatsMultiCh(dat_scaled, subfeature_order, printMeans = 0)
    plt.tight_layout()
    pdf.savefig()

    if use_lfp_HFO:
        utsne.plotBasicStatsMultiCh(dat_lfp_highres_scaled, subfeature_order_lfp_highres,
                                    printMeans = 0)
        plt.tight_layout()
        pdf.savefig()

############## plot raw



if do_plot_raw_timecourse:
    print('Starting plotting timecourse of scaled data' )
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv
            tt = utsne.plotIntervalData(dat_scaled,subfeature_order,iv,
                                        raw=raw_lfponly, plot_types=['timecourse'], dat_ext = extdat,
                                        extend=extend)
            pdf.savefig()
            plt.close()

##################


##################

a = '{}_tfr_{}chs.npz'.format(rawname_,n_channels)
fname_tfr_full = os.path.join( data_dir,a)


a = '{}_feats_{}chs_skip{}_wsz{}.npz'.\
    format(rawname_,n_channels, skip, windowsz)
fname_feat_full = os.path.join( data_dir,a)
######################

# later I would add features that are too long to compute for every time point,
# so I don't want to save X itself

if load_feat and os.path.exists(fname_feat_full):
    print('Loading feats from ',fname_feat_full)
    f = np.load(fname_feat_full)
    X =  f['X']
    Xtimes = f ['Xtimes']
    skip =f['skip']
    feature_names_all = f['feature_names_all']

else:
    exec( open( '_run_featprep.py').read(), globals() )

    #Xtimes = raw_srconly.times[nedgeBins:-nedgeBins  :skip]
    # here times of raws after first one are shfited
    Xtimes = (nedgeBins / sfreq) + np.arange(X.shape[0]) * ( skip / sfreq )

#exec( open( '_run_featprep.py').read(), globals() )



print('Skip = {},  Xtimes number = {}'.format(skip, Xtimes.shape[0  ] ) )

#Xtimes = np.where( np.arange( len(X) ) * skip / sfreq  )  +
if save_feat:
    ind_shift = 0
    for rawind in range(len(rawnames) ):
        rawname_ = rawnames[rawind]
        len_skipped = len( times_pri[rawind] ) // skip
        start = ind_shift
        end = ind_shift + len_skipped
        #if rawind == 0:
        start += nedgeBins // skip
        end   -= nedgeBins // skip
        feat_inds_cur = slice(start, end )
        ind_shift += len_skipped

        ts = times_pri_shifted[rawind]
        feat_inds_cur = np.where( (Xtimes >= ts[0] + nedgeBins/sfreq ) * (Xtimes <= ts[-1] - nedgeBins/sfreq ) )[0]

        X_cur = X[feat_inds_cur]

        rawtimes_cur = times_pri[rawind]
        Xtimes_cur = rawtimes_cur[nedgeBins:-nedgeBins  :skip]

        assert len(Xtimes_cur ) == len(X_cur)

        a = '{}_feats_{}chs_nfeats{}_skip{}_wsz{}.npz'.\
            format(rawname_,n_channels, X.shape[1], skip, windowsz)
        fname_feat_full = os.path.join( data_dir,a)

        info = {}
        info['bands_only'] = bands_only
        info['use_lfp_HFO'] = use_lfp_HFO
        info['use_main_tremorside'] = use_main_tremorside
        info['mts_letter'] = mts_letter
        info['use_main_LFP_chan'] = use_main_LFP_chan
        info['data_modalities'] = data_modalities
        info['features_to_use'] = features_to_use
        info['msrc_inds'] = msrc_inds
        info['use_LFP_to_LFP'] = use_LFP_to_LFP
        info['spec_uselog'] = spec_uselog
        info['freqs'] = freqs
        info['n_cycles'] = n_cycles
        info['freqs_inc_HFO'] = freqs_inc_HFO
        info['n_cycles_inc_HFO'] = n_cycles_inc_HFO
        info['nedgeBins'] = nedgeBins
        info['nedgeBins_highres'] = nedgeBins_highres
        info['percentileOffset'] = percentileOffset
        info['fbands'] = fbands
        info['fband_names_crude'] = fband_names_crude
        info['fband_names_fine'] = fband_names_fine
        info['fband_names_crude_inc_HFO'] = fband_names_crude_inc_HFO
        info['fband_names_fine_inc_HFO'] = fband_names_fine_inc_HFO
        info['skip_highres'] = skip_highres
        info['sfreq'] = sfreq

        np.savez(fname_feat_full,Xtimes=Xtimes_cur,X=X_cur,rawname_=rawname_,skip=skip,
                feature_names_all = feature_names_all, sfreq=sfreq,
                windowsz=windowsz, nedgeBins=nedgeBins, n_channels=n_channels,
                rawtimes = rawtimes_cur, freqs=freqs, chnames_LFP=raw_lfponly.ch_names,
                 chnames_src=raw_srconly.ch_names, feat_info = info)
        ip = feat_inds_cur[0],feat_inds_cur[-1]
        print('{}: Features shape {} saved to {}'.format(ip,X_cur.shape,fname_feat_full) )


if do_plot_feat_stats:
    print('Starting plotting stats of features' )
    #  Plots stats for subsampled data
    utsne.plotBasicStatsMultiCh(X.T, feature_names_all, printMeans = 0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

extdat_resampled = [ np.convolve(extdat[i], np.ones(skip) , mode='same') for i in range(extdat.shape[0]) ]
extdat_resampled = [ed[nedgeBins:-nedgeBins:skip] for ed in extdat_resampled]
extdat_resampled = np.vstack(extdat_resampled)
extdat_resampled /= np.std(extdat_resampled, axis=1)[:,None]


# Plot evolutions for subsampled data
if do_plot_feat_timecourse:
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

            pdf.savefig()
            plt.close()



if show_plots:
    pdf.close()
