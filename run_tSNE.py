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

rawname_ = 'S01_on_hold'
rawname_ = 'S01_off_hold'
rawname_ = 'S01_off_move'
rawname_ = 'S01_on_move'

rawname_ = 'S02_off_hold'
#rawname_ = 'S02_on_hold'

upre.saveRectConv(rawname_, skip_if_exist = 1)


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
subsample_type = 'desiredNpts'
#desiredNpts = 7000
#desiredNpts = 4000
desiredNpts = 5000
data_modalities = ['lfp', 'msrc']
#data_modalities = ['lfp']
#data_modalities = ['msrc']

features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
#features_to_use = [ 'H_act', 'H_mob', 'H_compl']  #csd includes tfr in my implementation
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

load_TFR = 0
save_TFR = 0
use_existing_TFR             = 1
load_tSNE                    = 0
save_tSNE                    = 1
use_existing_tSNE             = 1

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
a = '{}_tsne_{}chs_{},{},{}.npz'.format(rawname_,n_channels, min_freq,max_freq,freq_step)
fname_tsne_full = os.path.join( data_dir,a)

# compute diagonal terms
have_TFR = False
try:
    print('tfr (existing) num channels =', tfrres.shape[0] )
except NameError as e:
    have_TFR = False
else:
    have_TFR = True and (tfrres.shape[0] == n_channels)

#if not have_TFR:
#    print('OOO')
#    sys.exit(0)

if not (use_existing_TFR and have_TFR):
    if load_TFR and os.path.exists( fname_tfr_full ):
        tfrres = np.load(fname_tfr_full)['tfrres']
    else:
        print('Starting TFR ')
        tfrres = mne.time_frequency.tfr_array_morlet(dat_scaled[None,:], sfreq,
                                                    freqs, freqs *
                                                    freq2cycles_mult, n_jobs=10)
        tfrres = tfrres[0]
        if save_TFR:
            np.savez(fname_tfr_full, tfrres=tfrres)
            print('TFR saved to ',fname_tfr_full)

    csd, csdord = utils.tfr2csd(tfrres, sfreq, returnOrder=1)  # csdord.shape = (2, csdsize)

assert tfrres.shape[0] == n_channels


gc.collect()

############################# CSD


if bands_only in ['fine', 'crude']:
    if bands_only == 'fine':
        fband_names = fband_names_fine
    else:
        fband_names = fband_names_crude

    bpow_abscsd = []
    csdord_bandwise = []
    for bandi,bandname in enumerate(fband_names):
        low,high = fbands[bandname]
        freqis = np.where( (freqs >= low) * (freqs <= high) )[0]
        bandpow = np.mean( np.abs(  csd[:,freqis,:])  , axis=1 )
        bpow_abscsd += [bandpow[:,None,:]]

        #csdord_bandwise += [ np.vstack( [csdord,  np.ones(28, dtype=int)[None,:]*bandi  ] )[ :,:,None] ]

    bpow_abscsd = np.concatenate(bpow_abscsd, axis=1)

csdord_bandwise = []
for bandi,bandname in enumerate(fband_names):
    csdord_bandwise += [ np.concatenate( [csdord.T,  np.ones(csd.shape[0], dtype=int)[:,None]*bandi  ] , axis=-1) [:,None,:] ]

csdord_bandwise = np.concatenate(csdord_bandwise,axis=1)
csdord_bandwise.shape

################################  Plot CSD at some time point
if do_plot_CSD:
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        #intervals = ivalis[int_name]
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv
            ts,r, int_names = utsne.getIntervalSurround(start,end,extend, raw_lfponly)

            #timebins = raw_lfponly.time_as_index
            utsne.plotCSD(csd, fband_names, subfeature_order, list(r) , sfreq=sfreq, intervalMode=1,
                        int_names=int_names)
            pdf.savefig()
            plt.close()


############################## Hjorth

fig=plt.figure()
act,mob,compl  = utils.Hjorth(dat_scaled, 1/sfreq, windowsz=windowsz)

ax = plt.gca()
ax.plot (  np.min( act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='min')
ax.plot (  np.max( act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='max')
ax.plot (  np.mean( act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='mean')
ax.set_xlabel('channel')
ax.legend()
ax.set_xticks(range(n_channels))
ax.set_xticklabels(subfeature_order,rotation=90)
fig.suptitle('min_max of Hjorth params')

plt.tight_layout()
pdf.savefig()
plt.close()

#################################  Plot Hjorth

fig,axs = plt.subplots(nrows=1,ncols=3, figsize = (15,4))
axs = axs.reshape((1,3))

for i in range(n_channels):
    axs[0,0].plot(times,act[i], label= subfeature_order[i])
    axs[0,0].set_title('activity')
    axs[0,1].plot(times,mob[i], label= subfeature_order[i])
    axs[0,1].set_title('mobility')
    axs[0,2].plot(times,compl[i], label= subfeature_order[i])
    axs[0,2].set_title('complexity')

for ax in axs.reshape(axs.size):
    ax.legend(loc='upper right')

pdf.savefig()

#####################################################

ntimebins = tfrres.shape[-1]
ntimebins


rectconv_emg_rightsize = rectconv_emg[:,nedgeBins:-nedgeBins]


bpows = []
for bandi in range(len(fband_names)):
    bpow = np.vstack( [ utils.getCsdVals(bpow_abscsd[:,bandi,:],i,i, n_channels) for i in range (n_channels) ] )
    bpows += [bpow[:,None,:]]
bpows = np.concatenate( bpows, axis=1)
bpows.shape



defpct = (percentileOffset,100-percentileOffset)
center_spec_feats = spec_uselog
if spec_uselog:
    con_scale = defpct
else:
    con_scale = (0,100-percentileOffset)

feat_dict = {'tfr':{'data':None, 'pct':con_scale, 'centering':center_spec_feats },
             'con':{'data':None, 'pct':con_scale, 'centering':center_spec_feats },
             'H_act':{'data':act, 'pct':defpct},
             'H_mob':{'data':mob, 'pct':defpct}, 'H_compl':{'data':compl, 'pct':defpct} }

feat_dict['con']['centering'] = True

f = lambda x: x
if spec_uselog:
    f = lambda x: np.log(x)

if bands_only == 'no':
    tfres_ = tfrres.reshape( tfrres.size//ntimebins , ntimebins )
    feat_dict['tfr']['data'] = f( np.abs( tfres_) )
    #feat_dict['con']['data'] = con.reshape( con.size//ntimebins , ntimebins )
    feat_dict['con']['data'] = f( np.abs( csd.reshape( csd.size//ntimebins , ntimebins ) ) )
else:
    feat_dict['tfr']['data'] = f( bpows.reshape( bpows.size//ntimebins , ntimebins ) )
    feat_dict['con']['data'] = f( bpow_abscsd.reshape( bpow_abscsd.size//ntimebins , ntimebins ) )

    csdord_bandwise.reshape( (feat_dict['con']['data'].shape[0], 3) )


##########
for feat_name in feat_dict:
    curfeat = feat_dict[feat_name]
    curfeat['data'] = curfeat['data'][:,nedgeBins:-nedgeBins]
    print(feat_name, curfeat['data'].shape)




#################################  Scale features
for feat_name in feat_dict:
    curfeat = feat_dict[feat_name]
    inp = curfeat['data']
    centering = curfeat.get('centering', True)
    pct = curfeat['pct']
    if pct is not None:
        scaler = RobustScaler(quantile_range=pct, with_centering=centering)
        scaler.fit(inp.T)
        outd = scaler.transform(inp.T)
        curfeat['data'] = outd.T
        cnt = scaler.center_
        scl = scaler.scale_
        if feat_name in ['tfr', 'con']:
            if cnt is not None:
                cnt = np.min(cnt), np.max(cnt), np.mean(cnt), np.std(cnt)
            if scl is not None:
                scl = np.min(scl), np.max(scl), np.mean(scl), np.std(scl)
        print(feat_name, curfeat['data'].shape, cnt,scl)



#csdord_strs = []
#for i in range(csdord.shape[1]):
#    k1,k2 = csdord[:,i]
#    k1 = int(k1); k2=int(k2)
#    s = '{},{}'.format( subfeature_order[k1] , subfeature_order[k2] )
#    csdord_strs += [s]
#print(csdord_strs)

csdord_strs = []
for bandi in range( csdord_bandwise.shape[1] ):
    for i in range(csdord.shape[1]):
        k1,k2 = csdord[:,i]
        k1 = int(k1); k2=int(k2)
        s = '{}: {},{}'.format( fband_names[bandi], subfeature_order[k1] , subfeature_order[k2] )
        csdord_strs += [s]
csdord_strs



if features_to_use == 'all':
    features_to_use = feat_dict.keys()
#features_to_use = [ 'tfr',  'H_act', 'H_mob', 'H_compl']
if len(features_to_use) < len(feat_dict.keys() ):
    print('  !!! Using only {} features out of {}'.format( len(features_to_use) , len(feat_dict.keys() ) ))


##########################  Construct full feature vector

Xfull = []
feat_order = []
feature_names_all = []
for feat_name in features_to_use:
    curfeat = feat_dict[feat_name]
    cd = curfeat['data']
    print(feat_name, cd.shape)
    if feat_name == 'con':
        sfo = csdord_strs
    else:
        sfo = subfeature_order
    subfeats = ['{}_{}'.format(feat_name,sf) for sf in sfo]
    feature_names_all += subfeats
    Xfull += [ cd]
    feat_order += [feat_name]
Xfull = np.vstack(Xfull).T
print(feat_order, Xfull.shape)

Xtimes_full = raw_lfponly.times[nedgeBins:-nedgeBins]

######################################

if do_plot_feat_stats_full:
    # Plot feature stats
    utsne.plotBasicStatsMultiCh(Xfull.T, feature_names_all, printMeans = 0)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

if do_plot_feat_timecourse_full:
    print('Starting plotting timecourse of features' )
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        #intervals = ivalis[int_name]
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv

            tt = utsne.plotIntervalData(Xfull.T,feature_names_all,iv,
                                        times = Xtimes_full,
                                        plot_types=['timecourse'],
                                        dat_ext = extdat[:,nedgeBins:-nedgeBins],
                                        extend=extend)

            pdf.savefig()
            plt.close()


# vec_features can have some NaNs since pandas fills invalid indices with them
assert np.any( np.isnan ( Xfull ) ) == False
assert Xfull.dtype == np.dtype('float64')



if subsample_type == 'half':
    skip = 256 // 2
elif subsample_type == 'desiredNpts':
    skip = ntimebins // desiredNpts

X = Xfull[::skip]

Xtimes = raw_lfponly.times[nedgeBins:-nedgeBins:skip]
print('Skip = {},  Xtimes number = {}'.format(skip, Xtimes.shape[0  ] ) )


if do_plot_feat_stats:
    print('Starting plotting stats of features' )
    #  Plots stats for subsampled data
    utsne.plotBasicStatsMultiCh(X.T, feature_names_all, printMeans = 0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# Plot evolutions for subsampled data
if do_plot_feat_timecourse:
    print('Starting plotting timecourse of subsampled features' )

    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        #intervals = ivalis[int_name]
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv

            tt = utsne.plotIntervalData(X.T,feature_names_all,iv,
                                        times = Xtimes,
                                        plot_types=['timecourse'],
                                        dat_ext = extdat[:,nedgeBins:-nedgeBins:skip],
                                        extend=extend)

            pdf.savefig()
            plt.close()


################  Prep colors

tremcolor = 'r'
nontremcolor = 'g'
mvtcolor = 'b'  #c,y
neutcolor = 'grey'

hsfc = mts_letter
#hsfc = 'L'; print('Using not hand side (perhabs) for coloring')
annot_colors = { 'trem_{}'.format(hsfc): tremcolor  }
annot_colors[ 'notrem_{}'.format(hsfc)   ] = nontremcolor
annot_colors[ '{}_{}'.format(task, hsfc) ] = mvtcolor

colors =  np.array(  [neutcolor] * len(Xtimes) )

markers = np.array( ['o'] * len(Xtimes) )
mrk = ['<','>','o','^','v']

for an in anns:
    for descr in annot_colors:
        if an['description'] == descr:
            col = annot_colors[descr]

            start = an['onset']
            end = start + an['duration']

            timesBnds, indsBnd, sliceNames = utsne.getIntervalSurround( start,end, extend,
                                                                 times=Xtimes)
            #print('indBnds in color prep ',indsBnd)
            for ii in range(len(indsBnd)-1 ):
                #inds2 = np.where((Xtimes >= timesBnds[ii])* (Xtimes <= timesBnds[ii+1])  )[0]
                #

                # window size correction
                bnd0 = min(len(Xtimes)-1, indsBnd[ii]   + windowsz -1   )
                bnd1 = min(len(Xtimes)-1, indsBnd[ii+1] + windowsz -1   )
                inds2 = slice( bnd0, bnd1 )

                inds2 = slice( indsBnd[ii], indsBnd[ii+1] )
                markers[inds2] = mrk[ii]

                colors[inds2] = [col]
                #print(len(inds2))

############################### Run PCA


pca = PCA(n_components=nPCA_comp)
pca.fit(Xfull)   # fit to all data, apply to subsampled

pcapts = pca.transform(X)
XX = pcapts

print('total explained variance proportion {} with {} components'.
      format(np.sum(pca.explained_variance_ratio_), pcapts.shape[1] ) )
print(pca.explained_variance_ratio_[:10])

##################  Plot PCA
nc = nPCAcomponents_to_plot;
nr = 1; ww = 5; hh = 4
fig,axs = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*ww,nr*hh))
ii = 0
while ii < nc:
    indx = 0
    indy = ii+1
    ax = axs[ii];  ii+=1
    utsne.plotMultiMarker(ax, pcapts[:,indx], pcapts[:,indy], c=colors, m = markers, alpha=0.5);
    ax.set_xlabel('PCA comp {}'.format(indx) )
    ax.set_ylabel('PCA comp {}'.format(indy) )


#mrk = ['<','>','o','^','v']
mrknames = ['_pres','_posts','','_pree','_poste']
legend_elements = []
for m,mn in zip(mrk,mrknames):
    legel_trem = mpl.lines.Line2D([0], [0], marker=m, color='w', label='trem'+mn,
                                markerfacecolor=tremcolor, markersize=8)
    legel_notrem = mpl.lines.Line2D([0], [0], marker=m, color='w', label='notrem'+mn,
                                markerfacecolor=nontremcolor, markersize=8)
    legel_mvt = mpl.lines.Line2D([0], [0], marker=m, color='w', label='mvt'+mn,
                                markerfacecolor=mvtcolor, markersize=8)

    legend_elements += [legel_trem, legel_notrem, legel_mvt]

legel_unlab = mpl.lines.Line2D([0], [0], marker='o', color='w', label='unlab'+mn,
                            markerfacecolor=neutcolor, markersize=8)
legend_elements += [legel_unlab]

plt.legend(handles=legend_elements)
plt.suptitle('PCA')
#plt.show()
pdf.savefig()
plt.close()

######################### Plot PCA components structure

nr = nPCAcomponents_to_plot
nc = 1
hh=2
ww = max(14 , min(30, len(feature_names_all)/3 ) )
fig,axs = plt.subplots(nrows=nr, ncols=nc, figsize=(ww*nc, hh*nr), sharex='col')
for i in range(nr):
    ax = axs[i]
    dd = np.abs(pca.components_[i] )
    ax.plot( dd )
    ax.set_title('(abs of) PCA component {}, expl {:.2f} of variance (ratio)'.format(i, pca.explained_variance_ratio_[i]))

    ax.grid()

dd = np.abs(pca.components_[0] )
strongInds = np.where( dd  > np.quantile(dd,0.75) ) [0]
strongestInd = np.argmax(dd)

ax.set_xticks(np.arange(pca.components_.shape[1]))
ax.set_xticklabels(feature_names_all, rotation=90)

tls = ax.get_xticklabels()
for i in strongInds:
    tls[i].set_color("red")
tls[strongestInd].set_color("purple")

plt.tight_layout()
#plt.suptitle('PCA first components info')
#plt.savefig('PCA_info.png')
pdf.savefig()
plt.close()


###############################  tSNE

if do_tSNE:
    print('Starting tSNE')
    # function to be run in parallel
    def run_tsne(p):
        t0 = time.time()
        pi,si, XX, seed, perplex_cur, lrate = p
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplex_cur, learning_rate=lrate)
        X_embedded = tsne.fit_transform(XX)

        dur = time.time() - t0
        print('computed in {:.3f}s: perplexity = {};  lrate = {}; seed = {}'.
            format(dur,perplex_cur, lrate, seed))

        return pi,si,X_embedded, seed, perplex_cur, lrate


    #perplex_values = [30]
    #seeds = [0]


    res = []
    args = []
    for pi,perplex_cur in enumerate(perplex_values):
        subres = []
        for si,seed in enumerate(seeds):

            args += [ (pi,si, XX.copy(), seed, perplex_cur, lrate)]

    have_tSNE = False
    try:
        len(tSNE_result)
    except NameError as e:
        have_tSNE = False
    else:
        have_tSNE = True

    if not (have_tSNE and use_existing_tSNE):
        if load_tSNE and os.path.extsts(fname_tsne_full):
            print('Loading tSNE from ',fname_tsne_full)
            tSNE_result = np.load(fname_tsne_full) ['tSNE_result']
        else:
            ncores = min(len(args) , mpr.cpu_count()-2)
            if ncores > 1:
                pool = mpr.Pool(ncores)
                print('tSNE:  Starting {} workers on {} cores'.format(len(args), ncores))
                tSNE_result = pool.map(run_tsne, args)

                pool.close()
                pool.join()
            else:
                tSNE_result = [run_tsne(args[0])]

            if save_tSNE:
                np.savez(fname_tsne_full, tSNE_result=tSNE_result)
    else:
        assert len(tSNE_result) == len(args)
        print('Using exisitng tSNE')


    #cols = [colors, colors2, colors3]
    cols = [colors]

    colind = 0
    nr = len(seeds)
    nc = len(perplex_values)
    ww = 8; hh=8
    fig,axs = plt.subplots(ncols =nc, nrows=nr, figsize = (nc*ww, nr*hh))
    if not isinstance(axs,np.ndarray):
        axs = np.array([[axs]] )
    # for pi,pv in enumerate(perplex_values):
    #     for si,sv in enumerate(seeds):
    for tpl in tSNE_result:
        pi,si,X_embedded, seed, perplex_cur, lrate = tpl
        ax = axs[si,pi]
        #X_embedded = res[si][pi]
        #ax.scatter(X_embedded[:,0], X_embedded[:,1], c = cols[colind], s=5)

        utsne.plotMultiMarker(ax,X_embedded[:,0], X_embedded[:,1], c = cols[colind], s=5,
                              m=markers)
        ax.set_title('perplexity = {};  lrate = {}; seed = {}'.format(perplex_cur, lrate, seed))

    axs[0,0].legend(handles=legend_elements)
    figname = 'tSNE_inpdim{}_PCAdim{}_Npts{}_hside{}_trem_minFreq={}_lrate{}.pdf'.\
        format(X.shape[-1], pcapts.shape[-1], X.shape[0], mts_letter, min_freq, lrate)
    plt.suptitle(figname)

    plt.savefig(figname)

    pdf.savefig()
pdf.close()

#plt.savefig(figname)
