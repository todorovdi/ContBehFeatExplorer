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

rawname_ = 'S01_on_hold'  # no trem
#rawname_ = 'S01_off_hold'
#rawname_ = 'S01_off_move'
#rawname_ = 'S01_on_move'
#
#rawname_ = 'S02_off_hold'
#rawname_ = 'S02_on_hold'

#upre.saveRectConv(rawname_, skip_if_exist = 1)
print('!!!!!! current rawname --- ',rawname_)

src_fname_noext = 'srcd_{}_HirschPt2011'.format(rawname_)
rawname_LFPonly = rawname_ + '_LFPonly'+ '.fif'
rawname_LFPonly_full = os.path.join( data_dir, rawname_LFPonly )
raw_lfponly = mne.io.read_raw_fif(rawname_LFPonly_full, None)

newsrc_fname_full = os.path.join( data_dir, 'av_' + src_fname_noext + '.fif' )
raw_srconly =  mne.io.read_raw_fif(newsrc_fname_full, None)

rectconv_fname_full = os.path.join(data_dir, '{}_emg_rectconv.fif'.format(rawname_) )
emg_rectconv = mne.io.read_raw_fif(rectconv_fname_full)

sfreq = int(raw_lfponly.info['sfreq'])

#############################################################
#######  Main params

use_main_tremorside = 1  # 0 both , -1 opposite
use_main_LFP_chan = False
bands_only = 'crude'  #'fine'  or 'no'
#subsample_type = 'half'
subsample_type = 'prescribedSkip'  # or 'desiredNpts'
#desiredNpts = 7000
#desiredNpts = 4000
desiredNpts = 5000
prescribedSkip = 128  # 35, 80 , 128  only used if subsample_type is such
data_modalities = ['lfp', 'msrc']
#data_modalities = ['lfp']
#data_modalities = ['msrc']

features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
#features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bandcorrel']
#features_to_use = [ 'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
assert 'bandcorrel' not in features_to_use, 'Not implemented yet'
msrc_inds = np.arange(8,dtype=int)  #indices appearing in channel (sources) names, not channnel indices

extend = 3

#######   tSNE params
#nPCA_comp = 0.95
nPCA_comp = 0.99

perplex_values = [5, 10, 30, 40, 50]
seeds = range(5)
lrate = 200.

#lrate = 100.
#seeds = range(5)
perplex_values = [10, 30, 50, 65]
seeds = range(0,2)

#######################

##########################  ploting params

load_TFR                     = 0
save_TFR                     = 0
use_existing_TFR             = 1
load_tSNE                    = 0
save_tSNE                    = 1
use_existing_tSNE            = 0
load_feat                    = 0
save_feat                    = 1
use_existing_feat            = 1

show_plots                   = 0
do_tSNE                      = 1
#
do_plot_raw_stats            = 1 * show_plots
do_plot_raw_psd              = 1 * show_plots
do_plot_raw_timecourse       = 1 * show_plots
do_plot_feat_timecourse_full = 0 * show_plots
do_plot_feat_stats_full      = 0 * show_plots
do_plot_feat_timecourse      = 0 * show_plots
do_plot_feat_stats           = 1 * show_plots
do_plot_CSD                  = 0 * show_plots

fmax_raw_psd = 45
nPCAcomponents_to_plot = 5

####################  data processing params

spec_uselog = 1

min_freq = 3
freq_step = 1
max_freq = 100
freqs = np.arange(min_freq,max_freq,freq_step)

windowsz  =  1 * sfreq
nedgeBins = windowsz

# I want 1 sec window sz
cf =  windowsz/ ( 5/(2*np.pi) * sfreq  )
freq2cycles_mult = cf  # 1.2566370614359172
print('cf= ',cf)

percentileOffset = 25

################################

pdf= PdfPages(  '{}_plots{}_tSNE_side{}_LFP{}_{}_nmod{}_nfeattp{}.pdf'.format(
    rawname_,show_plots, use_main_tremorside, use_main_LFP_chan, bands_only,
    len(data_modalities),len(features_to_use)   ))

#############################################################

anns_fn = rawname_ + '_anns.txt'
anns_fn_full = os.path.join(data_dir, anns_fn)
anns = mne.read_annotations(anns_fn_full)
#raw.set_annotations(anns)

ivalis = utils.ann2ivalDict(anns)

#############################################################


 # get info about bad MEG channels (from separate file)
with open('subj_info.json') as info_json:
        #raise TypeError

    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)

subj,medcond,task  = utils.getParamsFromRawname(rawname_)
 # for current raw
maintremside = gen_subj_info[subj]['tremor_side']
mainLFPchan = gen_subj_info[subj]['lfpchan_used_in_paper']
tremfreq_Jan = gen_subj_info[subj]['tremfreq']
print('main trem side and LFP ',maintremside, mainLFPchan)


mts_letter = maintremside[0].upper()
int_names = ['{}_{}'.format(task,mts_letter), 'trem_{}'.format(mts_letter), 'notrem_{}'.format(mts_letter)]

#############################################################
raw_lfponly.load_data()
raw_srconly.load_data()

sides = ['L', 'R']    # this is just for construction of data, we will restrict later
raws_lfp_perside = {'L': raw_lfponly.copy(), 'R': raw_lfponly.copy() }
raws_srconly_perside = {'L': raw_srconly.copy(), 'R': raw_srconly.copy() }
for side in sides:
    chis = mne.pick_channels_regexp(raw_lfponly.ch_names, 'LFP{}.*'.format(side))
    chnames_lfp = [raw_lfponly.ch_names[chi] for chi in chis]
    if use_main_LFP_chan and mainLFPchan in chnames_lfp:
        raws_lfp_perside[side].pick_channels(   [mainLFPchan]  )
    else:
        raws_lfp_perside[side].pick_channels(   chnames_lfp  )
    print(raws_lfp_perside[side].ch_names)


    chis =  mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_all_*'.format(side)  )
    chnames_src = [raw_srconly.ch_names[chi] for chi in chis]
    raws_srconly_perside[side].pick_channels(   chnames_src  )


# fbands = {'tremor': [4,10],   'beta':[15,30],   'gamma':[30,100] }
# fbands_fine = {'tremor': [4,10], 'low_beta':[15,22], 'high_beta':[22,30],
#           'low_gamma':[30,60], 'high_gamma':[60,90],}
fbands = {'tremor': [3,10], 'low_beta':[15,22], 'high_beta':[22,30],
           'low_gamma':[30,60], 'high_gamma':[60,90], 'beta':[15,30],   'gamma':[30,100]}



fband_names_crude = ['tremor', 'beta', 'gamma']
fband_names_fine = ['tremor', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']

#freq2cycles_mult = 0.75

####################  Load emg


EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
if use_main_tremorside:
    chnames_emg = EMG_per_hand[maintremside]
else:
    chnames_emg = emg_rectconv.ch_names
print(chnames_emg)

rectconv_emg, ts_ = emg_rectconv[chnames_emg]
chnames_emg = [chn+'_rectconv' for chn in chnames_emg]


############# Concat
raws_permod = {'lfp' : raws_lfp_perside, 'msrc' : raws_srconly_perside }
hand_sides_all = ['L', 'R']  # hand sides
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
times = raw_lfponly.times

############# scale raw

# scale channel data separately
dat_scaled = np.zeros(dat.shape)
for i in range( dat.shape[0] ):
    inp = dat[i][:,None]
    scaler = RobustScaler(quantile_range=(percentileOffset,100-percentileOffset), with_centering=1)
    scaler.fit(inp)
    outd = scaler.transform(inp)
    dat_scaled[i,:] = outd[:,0]


###############  plot interval data

if do_plot_raw_psd:
    print('Starting plottign raw psd')
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv
            utsne.plotIntervalData(dat_scaled,subfeature_order,iv, raw=raw_lfponly,
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

############## plot raw
int_type = '{}_{}'.format(task,mts_letter)

f = np.load( os.path.join(data_dir, '{}_ica_ecg.npz'.format(rawname_) ) )
ecg = f['ecg']
#ecg_normalized = (ecg - np.min(ecg) )/( np.max(ecg) - np.min(ecg) )
ecg_normalized = (ecg - np.mean(ecg) )/( np.quantile(ecg,0.93) - np.quantile(ecg,0.01) )
dat_ecg = np.concatenate( (dat_scaled,ecg_normalized), axis=0)

extdat = np.vstack( [ecg, rectconv_emg] )
extnames = ['ecg'] + chnames_emg

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

#utsne.plotEvolutionMultiCh(dat_ecg, times, subfeature_order + ['ecg'], interval=ivalis[int_type][0],
#                           rawname=rawname_ , prefix='raw_scaled')

#utsne.plotEvolutionMultiCh(dat_scaled, times, subfeature_order, interval=ivalis[int_type][0],
#                     rawname=rawname_, prefix='raw_scaled')

##################


##################

a = '{}_tfr_{}chs_{},{},{}.npz'.format(rawname_,n_channels, min_freq,max_freq,freq_step)
fname_tfr_full = os.path.join( data_dir,a)


######################



have_feats = False
try:
    len(X)
except NameError as e:
    have_feats = False
else:
    have_feats = True

feats_were_loaded = False
if not (have_feats and use_existing_feat):
    if load_feat and os.path.extsts(fname_feat_full):
        print('Loading feats from ',fname_feat_full)
        f = np.load(fname_feat_full)
        X =  f['X']
        Xtimes = f ['Xtimes']
        skip =f['skip']
    else:
        exec( open( '_run_featprep.py').read(), globals() )
else:
    print('Using existing features')

if subsample_type == 'half':
    skip = 256 // 2
elif subsample_type == 'desiredNpts':
    skip = ntimebins // desiredNpts
elif subsample_type == 'prescribedSkip':
    skip = prescribedSkip


X = Xfull[::skip]
Xtimes = raw_lfponly.times[nedgeBins:-nedgeBins:skip]


a = '{}_feats_{}chs_skip{}_wsz{}_{},{},{}.npz'.\
    format(rawname_,n_channels, skip, windowsz, min_freq,max_freq,freq_step)
fname_feat_full = os.path.join( data_dir,a)
if save_feat and not feats_were_loaded:
    np.savez(fname_feat_full,Xtimes=Xtimes,X=X,rawname_=rawname_,skip=skip,
             feature_names_all = feature_names_all, sfreq=sfreq,
             windowsz=windowsz, nedgeBins=nedgeBins)

print('Skip = {},  Xtimes number = {}'.format(skip, Xtimes.shape[0  ] ) )
