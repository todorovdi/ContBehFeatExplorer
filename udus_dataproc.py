import mne
import multiprocessing as mpr
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import re
from lspopt.lsp import spectrogram_lspopt
import scipy
import scipy.signal as sig
import h5py
import pywt

import utils
import globvars as gv


def readSrcrec(fname):
    f = loadmat(fname)
    srcdynall = f['source_data']['avg']['mom']
    srcdynone = srcdynall[i][0,:]


def prepareSpec(pp):
    '''
    # proxy to be used in parallel computation
    '''
    rawname,chn, i,chtimes,chdata,sampleFreq,NFFT,noverlap = pp
    #specgram_scaling = 'density'   # divide over freq


    par = {'x':chdata, 'fs':sampleFreq, 'nperseg':NFFT, 'scaling':specgram_scaling, 
                                          'noverlap':noverlap }
    if spec_specgramtype == 'lspopt':
        specfun = spectrogram_lspopt
        par['c_parameter' ] = c_parameter
    elif spec_specgramtype == 'scaleogram':
        #par = {'data':chdata, 'scales':spec_cwtscales, 'wavelet':spec_wavelet, 'sampling_period':1/sampleFreq}
        Sxx, freqs = pywt.cwt(chdata, spec_cwtscales , spec_wavelet, sampling_period = 1/sampleFreq)
        freqs = freqs[::-1]
        Sxx = Sxx[::-1,:]
        bins = chtimes

        print('min freq {} max {}'.format( min(freqs), max(freqs) ) )
    elif spec_specgramtype == 'mne.wavelet':
        freqs = spec_MNEwletFreqs
        Sxx = mne.time_frequency.tfr_array_morlet(chdata[None,None,:], sfreq=int(sampleFreq),
                                         freqs=freqs, 
                                         n_cycles=spec_MNEwletFreqs * spec_freqs2wletcyclesCoef, 
                                         output='avg_power')
        Sxx  = Sxx[0,:]
        bins = chtimes
    elif spec_specgramtype == 'scipy':
        specfun = sig.spectrogram
    else:
        raise ValueError('Unknown spec_specgramtype')

    if spec_specgramtype in ['lspopt','scipy']:
        freqs, bins, Sxx = specfun(**par) 

    SxxC = None

    prepareAutocorrel = 0
    if prepareAutocorrel:
        autocorrel = np.correlate( chdata, chdata, 'same' ) 
        par['x'] = autocorrel
        par['nperseg'] = len(chdata)
        del par['noverlap'] 
        freqsC, binsC, SxxC = specfun(**par ) 

    return rawname,chn,i,freqs,bins,Sxx,SxxC

#def getAllSpecgrams(chdata,sampleFreq,NFFT,specgramoverlap):
#
#    '''
#    get spectrograms of the given chdata
#    '''
#    #tetmp = min(te+NFFT,int(maxtime*sampleFreq) )
#    pars = [(None,None,i,chdata[i,:],sampleFreq,NFFT,int(NFFT*specgramoverlap)) for i in range(chdata.shape[0])]
#
#    p = mpr.Pool( min(chdata.shape[0], mpr.cpu_count() ) )
#    r = p.map(prepareSpec, pars)
#    p.close()
#    p.join()
#
#    specgramsComputed = [0]*len(r)
#    autocorrelComputed = [0]*len(r)
#    for pp in r:
#        rawname,chn,i,freqs,bins,Sxx,SxxC = pp
#        specgramsComputed[i] = (freqs,bins,Sxx)
#        autocorrelComputed[i] = freqs,SxxC
#        
#    return specgramsComputed,autocorrelComputed


def precomputeSpecgrams(raws,ks=None,NFFT=256, specgramoverlap=0.75,forceRecalc=True, 
       modalities = None ):
    '''
    packs the computed spectrograms in a structure
    '''
    if modalities is None:
        modalities = ['LFP', 'EMG', 'EOG', 'MEGsrc']

    if ks is None:
        ks = raws.keys()
        
    print('Starting spectrograms computation ',raws.keys() )
    specgrams = {}
    pars = []
    for k in ks:
        raw = raws[k]
        
        #if (not forceRecalc) and ('specgram' in f.keys()):
        #    continue
        #sampleFreq = int( raw.info['sfreq'] )  # use global instead now

        #assert NFFT >= sampleFreq 

        chnames = raw.info['ch_names']
        sind_str,medcond,task = getParamsFromRawname(k)
        orderEMG, chinds_tuples, chnames_tuples = utils.getTuples(sind_str)

        specdict = {}
        for channel_pair_ind in range(len(orderEMG)):
            #inds = chinds_tuples[channel_pair_ind]['LFP'] + chinds_tuples[channel_pair_ind]['EMG']
            #ch_toplot = [chnames[i] for i in inds]
            
            chns = []
            for modality in modalities:
                chns += chnames_tuples[channel_pair_ind][modality]
            #chns = chnames_tuples[channel_pair_ind]['LFP'] + chnames_tuples[channel_pair_ind]['EMG']
            #chis = mne.pick_channels(chnames,include=chns, ordered=True )
            #chdata, chtimes = raw[chis,:]

            ts,te = raws[k].time_as_index([spec_time_start, spec_time_end])
            chdata, chtimes = utils.getData(k,chns,ts,te)
            #specgramsComputed,autocrrelComputed = getAllSpecgrams(chdata,sampleFreq,NFFT,specgramoverlap)
            for i,chn in enumerate(chns):
                pars += [ (k,chn,i,chtimes,chdata[i,:],sampleFreq,NFFT,int(NFFT*specgramoverlap)) ]
        
            #for i,chn in enumerate(chns):
            #    specdict[chn] = specgramsComputed[i]
        #raw['specgram'] = specdict
        #specgrams[k] = specdict

    p = mpr.Pool(len(pars) )
    r = p.map(prepareSpec, pars)
    p.close()
    p.join()

    #specgramsComputed = [0]*len(r)
    #autocorrelComputed = [0]*len(r)
    autocorrel = {}
    for pp in r:
        rawname,chn,i,freqs,bins,Sxx,SxxC = pp
        tmpd = { chn: (freqs,bins,Sxx) }
        if rawname in specgrams:
            specgrams[rawname].update( tmpd )
            autocorrel[rawname].update( tmpd )
        else:
            specgrams[rawname]  = tmpd  
            autocorrel[rawname] = tmpd 

        #specgramsComputed[i] = (freqs,bins,Sxx)
        #autocorrelComputed[i] = freqs,SxxC

        
    print('Spectrograms computation finished')
    return specgrams

#def getStatPerChan_singleThread(argdict):
#    return getStatPerChan(**argdict)
#
#def getStatPerChan_singleRaw(argdict):
#    return getStatPerChan(**argdict)

########  Gen stats across conditions
def getStatPerChan(time_start,time_end, glob_stats = None, singleRaw = None, mergeTasks = False, modalities=None):
    '''
    returns task independent
    stats = glob_stats[sind_str][medcond][chname]
    '''
    if modalities is None:
        modalities = ['LFP', 'EMG', 'EOG', 'MEGsrc']

    print('Starting computing glob stats for modalities {}, singleRaw {}, ({},{})'.
            format(modalities, singleRaw,time_start,time_end) )

    if singleRaw is not None:
        sind_str,medcond, task = getParamsFromRawname(singleRaw)
        subjs = [sind_str]
        medconds = [medcond]
        tasks = [task]
    else:
        subjs = gv.subjs_analyzed.keys()

    res = {}
    for subj in subjs:   # over subjects 
        stat_persubj = {}
        if singleRaw is None:
            medconds = gv.subjs_analyzed[subj]['medconds']
        for medcond in medconds:
            if mergeTasks:
                stat_leaflevel = {}
            else:
                stat_pertask = {}

            if singleRaw is None:
                raws_from_subj = gv.subjs_analyzed[subj][medcond]
            else:
                raws_from_subj = tasks 

            for task in raws_from_subj:   # over raws
                k = getRawname(subj,medcond,task)
                if not mergeTasks:
                    stat_leaflevel = {}

                sp = gv.specgrams[k]  # spectrogram from a given raw file

                raw = raws[k]
                #chnames = list( sp.keys() )
                #chnames = raw.info['ch_names'] 

                cht = gv.gen_subj_info[ subj ]['chantuples']

                #orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)
                #chnames2 = []
                for side in cht:
                    #chnames2 = chnames_tuples[side_ind]['LFP'] + chnames_tuples[side_ind]['EMG']

                    chnames2 = []
                    for modality in modalities:
                        chnames2 += cht[side]['nametuples'][modality]
                    #chnamesEMG = chnames_tuples[side_ind]['EMG']

                    #chis = mne.pick_channels(chnames,include=chnames2, ordered=True )
                    #ts,te = raw.time_as_index([time_start, time_end])
                    #chdata, chtimes = raw[chis,ts:te]
                    chdata, chtimes = utils.getData(k, chnames2 )

                    specsTremorEMGcurSide = []
                    for chii,chn in enumerate(chnames2):
                        f,b,spdata = sp[chn]

                        if isinstance( spdata[0,0] , complex):
                            spdata = np.abs(spdata)

                        tremDetTimeEnd = nonTaskTimeEnd
                        #if k.find('rest'):
                        bb,be = gv.freqBands['tremor']
                        ft,bt,spdatat = utils.getSubspec(f,b,spdata,bb, be, 
                                time_start, time_end)
                        assert spdatat.shape[0] > 0

                        bandpows = {}
                        fbnames = ['slow', 'tremor','beta', 'lowgamma']
                        for fbname in fbnames:
                            r = utils.getBandpow(k,chn,fbname, time_start, time_end) 
                            bins_b, bandpower = r
                            bandpows[fbname] = bandpower 



                        # 3-48, 60-80

                        st = {}

                        if chn.find('EMG') >= 0: 
                            specsTremorEMGcurSide += [ (chn, ft,bt, spdatat)  ]
                         
                            if tremrDet_clusterMultiFreq:
                                cnt = utils.getDataClusters( spdatat ) 
                                st['tremorfreq_clusters'] = cnt
                            else:
                                thrs = [0] * len(ft)
                                for freqi in range( len(ft) ):  # separately per frequency subband
                                    #thr = calcSpecThr( spdatat[freqi,:]  )
                                    cnt = utils.getDataClusters( spdatat[freqi,:] ) 
                                    thr = np.mean(cnt)

                                    m = tremorThrMult.get(subj,1.)  # maybe we have customized it a bit
                                    thr *= m
                                    thrs[freqi] = thr

                                st['thrPerFreq_trem'] = thrs

                        # get subspdata
                        st['max_spec'] = 0
                        st['min_spec'] = 1e8
                        st['mean_spec'] = 0

                        chdat = chdata[chii,:]
                        st['max']  = -1e8
                        st['min']  = 1e8
                        st['mean'] = 0
                        if chn in stat_leaflevel:
                            st['max_spec'] = max( st['max_spec'], np.max(spdata))
                            st['min_spec'] = min( st['min_spec'], np.min(spdata))
                            st['mean_spec'] += np.mean(spdata,axis=1) / len(raws_from_subj)

                            for fbname in fbnames:
                                s0 = '{}_bandpow_'.format(fbname)
                                s1max = '{}{}'.    format(s0, 'max')
                                s1min = '{}{}'.    format(s0, 'min')
                                s1max_distr = '{}{}'.    format(s0, 'max_distr')
                                s1min_distr = '{}{}'.    format(s0, 'min_distr')
                                s1normL05 = '{}{}'.format(s0, 'L05')
                                s1normL1 = '{}{}'. format(s0, 'L1' )
                                s1normL2 = '{}{}'. format(s0, 'L2' )
                                #s1mean = '{}_{}'.format('mean',s0)
                                if s1max in st:
                                    st[s1max  ] = max( st[s1max], np.max(bandpows[fbname] ))
                                    st[s1min] = min( st[s1min], np.min(bandpows[fbname] ))

                                    st[s1max_distr  ] = max( st[s1max_distr], np.max(bandpows[fbname] ))
                                    st[s1min_distr] = min( st[s1min_distr], np.min(bandpows[fbname] ))
                                else:
                                    st[s1max] = np.max(absval)
                                    st[s1min] = np.min(absval)
                                    if singleRaw is None:
                                        n_splits = 25
                                    else:
                                        n_splits = 15
                                    q = utils.getDataClusters_fromDistr(absval, n_splits=n_splits, 
                                            clusterType='outmostPeaks', lbl = '{} {}'.format(chn,fbname))
                                    st[s1max_distr] = q[1]
                                    st[s1min_distr] = q[0]


                                if glob_stats is not None:
                                    mx = glob_stats[sind_str][medcond][task][chn][s1max]
                                    mn = glob_stats[sind_str][medcond][task][chn][s1min]
                                    bpc = (absval - mn)   /  (mx - mn)
                                    duration = (time_end - time_start) 
                                    st[s1normL05] = np.sum( np.sqrt(bpc)  ) / duration 
                                    st[s1normL1] = np.sum( bpc   ) / duration
                                    st[s1normL2] = np.sum( np.power(bpc,2)  ) / duration
                                #st[s1mean] += np.mean(spdatat) / len(raws_from_subj)

                            st['max'] = max( st['max'], np.max(chdat))
                            st['min'] = min( st['min'], np.min(chdat))
                            st['mean'] += np.mean(chdat) / len(raws_from_subj)
                        else:
                            st['max_spec'] = np.max( spdata)
                            st['min_spec'] = np.min( spdata)
                            me = np.mean(spdata, axis=1) / len(raws_from_subj) 
                            st['mean_spec'] = me

                            spdata_mc =  ( spdata -  me[:,None] ) / me[:,None]    # divide by mean
                            st['max_spec_mc'] = np.max( spdata_mc)
                            st['min_spec_mc'] = np.min( spdata_mc)

                            minfreq = 3
                            maxfreq = plot_maxFreqInSpec
                            freqinds = np.where( np.logical_and(f >= minfreq,f <= plot_maxFreqInSpec) )[0]
                            sptmp = spdata_mc[freqinds,:]
                            st['max_spec_mc_plot'] = np.max( sptmp)
                            st['min_spec_mc_plot'] = np.min( sptmp)

                            if singleRaw is None:
                                n_splits = 25
                            else:
                                n_splits = 15
                            #q = utils.getDataClusters_fromDistr(sptmp, n_splits=n_splits, 
                            #        clusterType='outmostPeaks', lbl = '{} spec'.format(chn) )
                            m1,m2 = utils.getSpecEffMax(sptmp) 

                            st['max_distr_spec_mc_plot'] = m2
                            st['min_distr_spec_mc_plot'] = m1

                            for fbname in fbnames:
                                s0 = '{}_bandpow_'.format(fbname)
                                s1max = '{}{}'.    format(s0, 'max')
                                s1min = '{}{}'.    format(s0, 'min')
                                s1normL05 = '{}{}'.format(s0, 'L05')
                                s1normL1 = '{}{}'. format(s0, 'L1' )
                                s1normL2 = '{}{}'. format(s0, 'L2' )
                                s1max_distr = '{}{}'.    format(s0, 'max_distr')
                                s1min_distr = '{}{}'.    format(s0, 'min_distr')
                                #s1mean = '{}_{}'.format('mean',s0)
                                bpc = bandpows[fbname]
                                absval = np.abs(bpc)
                                st[s1max] = np.max(absval)
                                st[s1min] = np.min(absval)

                                if singleRaw is None:
                                    n_splits = 25
                                else:
                                    n_splits = 15
                                q = utils.getDataClusters_fromDistr(absval, n_splits=n_splits, 
                                        clusterType='outmostPeaks')
                                st[s1max_distr] = q[1]
                                st[s1min_distr] = q[0]


                                if glob_stats is not None:
                                    mx = glob_stats[sind_str][medcond][task][chn][s1max]
                                    mn = glob_stats[sind_str][medcond][task][chn][s1min]
                                    bpc = (absval - mn)   /  (mx - mn)
                                    duration = (time_end - time_start) 
                                    st[s1normL05] = np.sum( np.sqrt(bpc)  ) / duration 
                                    st[s1normL1] = np.sum( bpc   ) / duration
                                    st[s1normL2] = np.sum( np.power(bpc,2)  ) / duration


                                #st[s1mean] = np.mean(spdatat) / len(raws_from_subj)

                            # of raw signal
                            st['max']  = np.max( chdat)
                            st['min']  = np.min( chdat)
                            st['mean'] = np.mean(chdat) / len(raws_from_subj)

                            #if chn == 'EMG063_old':
                            #    import pdb; pdb.set_trace()

                        stat_leaflevel[chn] = st

                
                    # want to do per side
                    chns = [ specsTremorEMGcurSide[i][0] for i  in range(len(specsTremorEMGcurSide) ) ]  
                    ft = specsTremorEMGcurSide[0][1] 
                    bt = specsTremorEMGcurSide[0][2]   
                    jointspec = [ specsTremorEMGcurSide[i][3] for i in range(len(specsTremorEMGcurSide) ) ]
                    spdatat = np.vstack( jointspec)  # increase data dimension by using info from both EMG channels
                    if len(ft) > 1 and specgram_scaling == 'psd':
                        freqres = ft[1] - ft[0]
                    else:
                        freqres = 1.
                    assert spdatat.shape[0] == 2 * len(ft), spdatat.shape
                    # for each raw, for each side, for each channel there can be multiple freqs 
                    if tremrDet_clusterMultiMEG:
                        # sum over frequencies
                        rr = np.zeros( (len(specsTremorEMGcurSide), len(bt) ) )
                        for ii in range( len(specsTremorEMGcurSide) ):
                            rr[ii,:] = np.sum( spdatat[ ( ii * len(ft) ) : ( (ii+1) * len(ft) ), :  ] 
                                    , axis = 0) * freqres
                        #cnt = getDataClusters( spdatat ) 
                        # rr is NchansPerSide x numbins, power in tremor band versus time 
                        cnt = utils.getDataClusters( rr )   #cnt is numcluster x dim
                        ythr = {}
                        yclust = {}
                        yclust_sep = {}
                        thrPerMEGch = cnt.mean(axis=1)   # take mean over clusters (keeping channel dim)
                        thrPerMEGch_sep = np.zeros( len(chns) )
                        assert len(thrPerMEGch) == len(chns)
                        ythr_sep = {}
                        for i in range(len(chns) ):
                            ythr[chns[i] ] = thrPerMEGch[i]  
                            yclust[chns[i] ] = cnt[:,i]
                            cntcur = utils.getDataClusters( rr[i,:] )  
                            assert cntcur.ndim == 1
                            yclust_sep[chns[i] ] = cntcur
                            ythr_sep[chns[i] ] = np.mean(cntcur)

                        do_plotClust = 0
                        if do_plotClust:
                            ww = 8
                            hh = 8
                            nc = 2
                            nr = 2
                            fig,axs = plt.subplots(nrows=nc,ncols=nc, figsize= (ww*nc,ww*nr)  )
                            #import pdb; pdb.set_trace()
                            ax = axs[1,1]  # x
                            ax.hist( rr[0,:], log=True )
                            ax.set_xlim( min( rr[0,:] ), max( rr[0,:] )  )

                            ax = axs[0,0]  # y
                            ax.hist( rr[1,:], orientation = 'horizontal' , log=True)
                            ax.set_ylim( min( rr[1,:] ), max(rr[1,:] )  )

                            ax = axs[0,1]
                            ax.scatter( rr[0,:], rr[1,:] )
                            ax.scatter( cnt[0,0], cnt[1,0] ,s = 80)
                            ax.scatter( cnt[0,1], cnt[1,1] ,s = 80)
                            ax.set_xlim( min( rr[0,:] ), max( rr[0,:] )  )
                            ax.set_ylim( min( rr[1,:] ), max(rr[1,:] )  )
                            ax.set_xlabel(chns[0])
                            ax.set_ylabel(chns[1])
                            ax.set_xscale('log')
                            ax.set_yscale('log')

                            plt.savefig('Scatters_{}_{}.pdf'.format( k,side))
                            plt.close()

                        ythr = ythr_sep
                        yclust = yclust_sep
                        if 'thrPerCh_trem_allEMG' in stat_leaflevel:
                            stat_leaflevel['thrPerCh_trem_allEMG'].update(  ythr  )
                        else:
                            stat_leaflevel['thrPerCh_trem_allEMG'] =   ythr   

                        if 'tremorfreq_clusters_allEMG' in stat_leaflevel:
                            stat_leaflevel['tremorfreq_clusters_allEMG'].update(  yclust )
                        else:
                            stat_leaflevel['tremorfreq_clusters_allEMG'] = yclust 
                # end of cycle over channels

                if not mergeTasks:
                    stat_pertask[task] = stat_leaflevel

            # end of cycle over rawnames with the same medication state
            if not mergeTasks:
                stat_persubj[medcond] = stat_pertask
            else:
                stat_persubj[medcond] = stat_leaflevel
                    
        res[subj] = stat_persubj
        #glob_stats[subj] = stat_perchan

    print('Glob stats computation finished')
    return res
 
def getStatsFromTremIntervals(intervalDict):
    '''
    time_start, time_end, intervalType = intervals [rawname][interval index]
    intervalType from  [ incPre,incPost,incBoth,middle, no_tremor ]
    '''
    intervalTypes =   [ 'pre', 'post', 'incPre', 'incPost', 'incBoth', 'middle', 'middle_full', 'no_tremor', 'entire' ]  # ,'pre', 'post' -- some are needed for barplot, some for naive vis
    #intervalTypes =   [ 'pre', 'post', 'middle_full', 'no_tremor' ]  # ,'pre', 'post'
    stats = {}

    for rawname in intervalDict:
        if rawname not in raws:
            continue
        intervals = intervalDict[rawname]
        statlist = [0]*len(intervals)
        for intind,interval in enumerate(intervals):
            t1,t2,itype = interval
            if itype not in intervalTypes:
                continue

            sind_str,medcond, task = getParamsFromRawname(rawname)
            st = getStatPerChan(t1,t2,singleRaw = rawname, mergeTasks=False, glob_stats=glob_stats )
            statlist[intind] = st[sind_str][ medcond][ task]
        stats[rawname] = statlist

    return stats

#def getBandpowWavelet(chdata):
                    
# merge across two EMG channels

## Define left / right corresdpondance (left LFP to right EMG)
def genChanTuples(rawname, chnames ):
    emg_inds =  [i for i,s in enumerate(chnames) if 0 <= s.find('EMG') ]
    emgold_inds = [i for i,s in enumerate(chnames) if 0 <= s.find('_old') ]
    emgkil_inds = [i for i,s in enumerate(chnames) if 0 <= s.find('_kil') ]
    lfp_inds =  [i for i,s in enumerate(chnames) if 0 <= s.find('LFP') ]
    lfpl_inds = [i for i,s in enumerate(chnames) if 0 <= s.find('LFPL') ]  # left STN LFP
    lfpr_inds = [i for i,s in enumerate(chnames) if 0 <= s.find('LFPR') ]  # right STN LFP
    eye_vert_name = 'EOG127'
    eye_hor_name = 'EOG128'
    eog_inds = [i for i,s in enumerate(chnames) if 0 <= s.find('EOG') ]
    eog_names = [eye_vert_name, eye_hor_name]


    right_edc = 'EMG061'  # right normal anatomically
    right_fds = 'EMG062'
    left_edc = 'EMG063'
    left_fds = 'EMG064'
    EMGnames = {}
    EMGnames['left'] = [left_edc, left_fds]
    EMGnames['right'] = [right_edc, right_fds]
    for k in EMGnames:
        for i in range(len(EMGnames[k] ) ):
            if useKilEMG:
                EMGnames[k][i] += '_kil'
            else:
                EMGnames[k][i] += '_old'

    EMGinds = {}
    for k in EMGnames:
        EMGinds[k] = [ chnames.index(chname) for chname in EMGnames[k] ]

    order_by_EMGside = ['left','right']

    MEGsrc_inds = { 'left':[], 'right':[] } 
    MEGsrc_names = { 'left':[], 'right':[] } 
    MEGsrc_names_unfilt = { 'left':[], 'right':[] } 

    indshift = 0
    for roistr in sorted(MEGsrc_roi):  # very important to use sorted!
        sind_str,medcond,task = getParamsFromRawname(rawname)
        srcname = 'srcd_{}_{}_{}_{}'.format(sind_str,medcond,task,roistr)
        if srcname not in srcs:
            print('GetChantuples Warning: {} not in srcs'.format(srcname) )
            continue
        f = srcs[srcname]
        src = f['source_data'] 
        nsrc = src['avg']['mom'].shape[1]
        curinds = {}
        curinds['left'] = np.where( f['source_data']['pos'][0,:] < 0  )[0]  # 3 x Nsrc
        curinds['right']= np.where( f['source_data']['pos'][0,:] >= 0 )[0] 

        for side in order_by_EMGside:
            for ind in curinds[side]:
                chn = 'MEGsrc_{}_{}'.format(roistr,ind)
                MEGsrc_names_unfilt[side] += [chn]
                if MEGsrc_names_toshow is None or chn in MEGsrc_names_toshow:
                    MEGsrc_names[side] += [chn]

            MEGsrc_inds[side] += list( np.array( curinds[side] ) + indshift  )
        indshift += nsrc

    if len(roistr) > 0:
        assert len(MEGsrc_names) > 0
        #assert isinstance(MEGsrc_names[order_by_EMGside[0] ][0], str)

    LFPnames = {}
    LFPnames['left']  = [s for s in chnames if 0 <= s.find('LFPL') ]
    LFPnames['right'] = [s for s in chnames if 0 <= s.find('LFPR') ]

    LFPinds = {}
    LFPinds['left']  = lfpl_inds
    LFPinds['right'] = lfpr_inds


    #chnames_tuples = []
    #chinds_tuples = []
    chnames_tuples = {}
    chinds_tuples  = {}
    for sidei in range(len(order_by_EMGside)):
        side = order_by_EMGside[sidei]
        revside = order_by_EMGside[1-sidei]
        #chnames_tuples += [{'EMG':EMGnames[side], 
        #    'LFP':LFPnames[revside], 'MEGsrc':MEGsrc_names[revside], 'EOG':eog_names }]
        #chinds_tuples += [{'EMG':EMGinds[side],  
        #    'LFP':LFPinds[revside], 'MEGsrc':MEGsrc_inds[revside], 'EOG':eog_inds }]
        chnames_tuples[side] = {'EMG':EMGnames[side], 
            'LFP':LFPnames[revside], 'MEGsrc':MEGsrc_names[revside], 'EOG':eog_names }
        chinds_tuples[side] = {'EMG':EMGinds[side],  
            'LFP':LFPinds[revside], 'MEGsrc':MEGsrc_inds[revside], 'EOG':eog_inds }
        # both EOGs goe to both sides
    
#     chnames_tuples += [{'EMG':EMGnames['right'], 'LFP':LFPnames['left'] }]
#     chinds_tuples += [{'EMG':EMGinds['right'], 'LFP':LFPinds['left'] }]
    
    r = {}
    r['order'] = order_by_EMGside
    r['indtuples'] = chinds_tuples
    r['nametuples'] = chnames_tuples
    r['LFPnames'] = LFPnames
    r['MEGsrcnames'] = MEGsrc_names
    r['EMGnames'] = EMGnames
    r['MEGsrcnames_all'] = MEGsrc_names_unfilt
    #return order_by_EMGside, chinds_tuples, chnames_tuples, LFPnames, EMGnames, MEGsrc_names
    return r




def plotChannelBand(ax,k,chn,fbname,time_start,time_end,logscale=False, bandPow=True,
        color = None, skipPlot = False, normalization = ('whole','channel_distr') ):
    '''
    bandPow -- if True, plot band power, if False, plot band passed freq
    '''
    if bandPow:
        #if spec_specgramtype == 'scaleogram':  #  [to be implemented] -- need to design a wavelet
        #    bandpower = np.sum(Sxx_b,axis=0) * freqres
        normrange,normtype = normalization

        
        r = utils.getBandpow(k,chn,fbname, time_start, time_end)
        if r is None:
            return None

        bins_b, bandpower = r
        if normtype == 'channel' and normrange == 'whole':
            bandpower /= np.max(bandpower) 
        elif normtype == 'channel_distr' and normrange == 'whole':
            sind_str,medcond,task = utils.getParamsFromRawname(k)
            r = glob_stats[sind_str][medcond][task][chn] 

            s0 = '{}_bandpow_'.format(fbname)
            s1max_distr = '{}{}'.    format(s0, 'max_distr')
            s1min_distr = '{}{}'.    format(s0, 'min_distr')

            if s1min_distr not in r:
                print( "{} not in glob stats".format(s1max_distr) )
                bandpower /= np.max(bandpower) 
            else:
                mn = r[s1min_distr]
                mx = r[s1max_distr]
                span = mx-mn
                bandpower = (bandpower - mn) / span
        else:
            raise ValueError('Other normalization types are not implemented')


        #specgramsComputed = specgrams[k]
        #freqs, bins, Sxx = specgramsComputed[chn]
        #fbs,fbe = gv.freqBands[fbname]
        #r = getSubspec(freqs,bins,Sxx, fbs,fbe, 
        #        time_start,time_end)

        #if r is not None:
        #    freqs_b, bins_b, Sxx_b = r


        #    if specgram_scaling == 'psd':
        #        freqres = freqs[1]-freqs[0]
        #    else:
        #        freqres = 1.

        #    if isinstance(Sxx_b[0,0], complex):
        #        Sxx_b = np.abs(Sxx_b)
        #    bandpower = np.sum(Sxx_b,axis=0) * freqres

        #else:
        #    return None
    else:
        bandpassFltorder = 10
        bandpassFreq =  gv.freqBands[fbname] 

        lowfreq = bandpassFreq[0]
        if lowfreq < 1e-6:
            sos = sig.butter(bandpassFltorder, bandpassFreq[1], 
                    'lowpass', fs=sampleFreq, output='sos')
        else:
            sos = sig.butter(bandpassFltorder, bandpassFreq, 
                    'bandpass', fs=sampleFreq, output='sos')

        bandpower = sig.sosfilt(sos, ys)


    if not skipPlot:
        #print('--- plotting {} {} max {}'.format(chn, fbname, np.max(bandpower) ) )
        chlbl = chn
        if chn.find('MEGsrc') >= 0:
            chlbl = utils.getMEGsrc_chname_nice(chn)

        if gv.gen_subj_info[sind_str]['lfpchan_used_in_paper'] == chn:
            chlbl = '* ' + chn

        lbl = '{}, {}'.format(chlbl,fbname)
        ax.plot(bins_b,bandpower,
                label=lbl, 
                ls = plot_freqBandsLineStyle[fbname], alpha=0.5, c=color )
        if logscale:
            ax.set_yscale('log')
            ax.set_ylabel('logscale')

    return bins_b, bandpower

def plotSpectralData(plt,time_start = 0,time_end = 400, 
        chanTypes_toshow = None, onlyTremor = False ):

    mainSideColor = 'w'; otherSideColor = 'lightgrey'
    normType='uniform' # or 'log'
    #normType='log' # or 'log'

    ndatasets = len(gv.raws)
    #nffts = [512,1024]

    #tremor_intervals_use_merged = 0
    tremor_intervals_use_merged = 1
    tremor_intervals_use_Jan = 1

    plotEMG_shiftMean = 1

    ymaxEMG = 0.00035
    ymaxLFP = 0.00012
    ymaxLFP = 0.00007

    #plot_maxFreqInSpec = 60

    #try: 
    #    specstat_per_subj
    #except NameError:
    #    specstat_per_subj = getStatPerChan(time_start,time_end,0,plot_maxFreqInSpec)
    #specstat_per_subj = getStatPerChan(time_start,time_end,0,plot_maxFreqInSpec)

    
    ymaxs = {'EMG':ymaxEMG, 'LFP':ymaxLFP}

    pair_inds = [0,1]

    nspecAxs_perModality = {}
    #nLFPchans = 7 # max, per side
    if gv.gparams['plot_LFP_onlyFavoriteChannels']:
        nspecAxs_perModality['LFP'] = 1
        nspec_LFPchans = 1
    else:
        nLFPs = 3
        if 'S08' in gv.subjs_analyzed or 'S09' in gv.subjs_analyzed:
            nLFPs = 7 
        nspecAxs_perModality['LFP'] = nLFPs
        nspec_LFPchans = nLFPs # max, per side

    nspecAxs_perModality['EMG'] = 2
    nspecAxs_perModality['EOG'] = 2

    # MEGsrcnames = list( gv.gen_subj_info.values() )[0 ] ['MEGsrcnames_perSide']
    # assume that already taken into account MEGsrc_inds_toshow
    s0 = list( gv.subjs_analyzed.keys() )[0]
    MEGsrcnames_subj0 = gv.subjs_analyzed[s0] ['MEGsrcnames_perSide']
    n = max( len( MEGsrcnames_subj0['left'] ), len( MEGsrcnames_subj0['right'] )  )  # since we can have different sides for different subjects
    nspecAxs_perModality['MEGsrc'] = min( n , plot_numBestMEGsrc)
    
    #nr_bandpow = 2

    nspect_per_pair = 0
    pt = chanTypes_toshow['spectrogram']
    for modality in pt['chantypes']:   
        nspect_per_pair += nspecAxs_perModality[modality]  # 1 row per spectrogram

    chanTypes_toshow['spectrogram']['nplots'] = nspect_per_pair 

    # band pow
    #chanTypes_toshow['bandpow']['nplots'] = len(chanTypes_toshow['bandpow']['chantypes'] ) # several bands per modality
    n = 0
    #from IPython.core.debugger import Tracer; debug_here = Tracer(); debug_here()
    for mod in chanTypes_toshow['bandpow']['chantypes']:
        n += len( plot_freqBandNames_perModality[mod] )
    chanTypes_toshow['bandpow']['nplots'] = n

    if 'MEGsrc' in chanTypes_toshow['bandpow']['chantypes']:
        #if plot_MEGsrc_rowPerBand:
        #    chanTypes_toshow['bandpow']['nplots'] += (-1) + len( plot_freqBandNames_perModality['MEGsrc'] )
        if plot_MEGsrc_separateRoi and len(MEGsrc_roi) > 1:
            chanTypes_toshow['bandpow']['nplots'] += (len(MEGsrc_roi) - 1) * len( plot_freqBandNames_perModality['MEGsrc'] ) 

    #if 'LFP' in chanTypes_toshow['bandpow']['chantypes'] and plot_LFP_rowPerBand:
    #    chanTypes_toshow['bandpow']['nplots'] += (-1) + len( plot_freqBandNames_perModality['LFP'] )

    chanTypes_toshow['EMGband_corr']['nplots'] = chanTypes_toshow['bandpow']['nplots'] * show_EMG_band_corr
    chanTypes_toshow['timecourse']['nplots'] = len(chanTypes_toshow['timecourse']['chantypes'] )  # one axis per modality
    chanTypes_toshow['tremcvl']['nplots'] = 1 * show_tremconv

    chanTypes_toshow['stats_barplot']['nplots'] = 1 * show_stats_barplot

    print(  chanTypes_toshow )
    nplots_per_side = sum( chanTypes_toshow[pt]['nplots'] for pt in chanTypes_toshow )

    raw_int_pairs = []
    for ki,k in enumerate(ks):
        if plot_colPerInterval:
            l = len( plot_timeIntervalPerRaw.get( k, [0]  ) ) 
        else:
            l = 1
        raw_int_pairs +=  zip( [ki] * l, range(l) )
    nc = len(raw_int_pairs)

    nr = nplots_per_side
    if not plot_onlyMainSide:
        nr *= len(pair_inds) 

    if plot_onlyMainSide:
        print('----- plotting only main tremor side')
    
    #if show_timecourse:
    #    nr += len(chanTypes_toshow) * len(pair_inds)
    #if show_spec:
    #    nr += nspect_per_pair * len(pair_inds)
    #if show_bandpow:
    #    nr += nr_bandpow * len(pair_inds)

    ftc_instead_psd = 1   # if we want to plot freq bands power dynamics or PSD
        
    if tremor_intervals_use_Jan:
        tremIntervalMerged = tremIntervalJan  
    else:
        tremIntervalMerged = tremIntervalMergedPerRaw 

    ncformal = nc
    gridspec_kw = None
    wweff = ww
    if nc == 1:
        ncformal = 2   # don'a make sinlge row because it would ruin adressing
        gridspec_kw={'width_ratios': [1, 0.01]}
        wweff *= 10

    fig, axs = plt.subplots(ncols = ncformal, nrows=nr, figsize= (wweff*nc,hh*nr), sharey='none',
            gridspec_kw= gridspec_kw)
    plt.subplots_adjust(top=0.97, bottom=0.02, right=0.95, left=0.05, hspace=0.65)
    colind = 0
    maxBandpowPerRaw = {}
    for colind in range(nc):
        ki,intind = raw_int_pairs[colind]
        k = ks[ki]
        
        sind_str,medcond,task = getParamsFromRawname(k)
        deftimeint =  [(time_start,time_end,'no_tremor') ]

        allintervals = plot_timeIntervalPerRaw.get( k, deftimeint  )
        desind = intind
        #for i,p in enumerate(allintervals):  # find first interval that starts not at zero (thus potentially we can what was before the tremor start)
        #    if p[2] == 'incPre':
        #        desind = i
        #        break
        time_start, time_end, intervalType =    allintervals [desind]   # 0 means first interval, potentially may have several

        #chnames = raws[k].info['ch_names']
        orderEMG, chinds_tuples, chnames_tuples = utils.getTuples(sind_str)
        
        if k not in maxBandpowPerRaw:
            maxBandpowPerRaw[k] = {}

        if tremor_intervals_use_merged:
            tremorIntervals = tremIntervalMerged[k]
        else:
            tremorIntervals = tremIntervalPerRaw[k]
 
        if plot_onlyMainSide:
            pair_inds = [orderEMG.index( gv.gen_subj_info[sind_str]['tremor_side'] ) ]
        for channel_pair_ind in pair_inds:
            if plot_onlyMainSide:
                channel_pair_ind_forAx = 0
            else:
                channel_pair_ind_forAx = channel_pair_ind
            side_body = orderEMG[channel_pair_ind]

            if channel_pair_ind not in maxBandpowPerRaw[k]:
                maxBandpowPerRaw[k][channel_pair_ind] = {}

            #inds = chinds_tuples[channel_pair_ind]['LFP'] + chinds_tuples[channel_pair_ind]['EMG']
            #ch_toplot = [chnames[i] for i in inds]
            #lfpnames = chnames_tuples[channel_pair_ind]['LFP']
            #ch_toplot_timecourse = []
            #if 'LFP' in chanTypes_toshow['timecourse']['chantypes']:
            #    if plot_LFP_onlyFavoriteChannels:
            #       # get intersection
            #       lfpnames = list( set(lfpnames) & set( gv.gen_subj_info[subj]['favoriteLFPch'] ) )
            #       ch_toplot_timecourse += lfpnames
            #if 'EMG' in chanTypes_toshow['timecourse']['chantypes']:
            #    ch_toplot_timecourse += chnames_tuples[channel_pair_ind]['EMG']
            spcht = chanTypes_toshow['timecourse']['chantypes']

            ch_toplot_timecourse = []
            for modality in spcht:
                ch_toplot_timecourse += chnames_tuples[channel_pair_ind][modality]
            ch_toplot_timecourse = utils.filterFavChnames( ch_toplot_timecourse, sind_str )
            

            ts,te = gv.raws[k].time_as_index([time_start, time_end])

            #chs = mne.pick_channels(chnames,include=ch_toplot_timecourse, ordered=True )
            #chdata, chtimes = raws[k][chs,ts:te]
            
            chdata, chtimes = utils.getData(k, ch_toplot_timecourse, ts,te )

            mintime = min(chtimes)
            maxtime = max(chtimes)
            
            tremorIntervalsCurSide = tremorIntervals[side_body]

            pairs = tremorIntervalsCurSide
            #import pdb; pdb.set_trace()
            if isinstance(pairs,dict) and len(pairs):
                pairs = pairs['tremor']

            ################## plot timecourse
            for modalityi,modality in enumerate(chanTypes_toshow['timecourse']['chantypes'] ):
                ax = axs[nplots_per_side*channel_pair_ind_forAx + modalityi,colind]   # one row per modality

                #addstr = ''
                curModMin = 1e20
                curModMax = -1e20
                for i in range(chdata.shape[0] ):
                    #addstr = ''
                    #chn = chnames[chs[i]]
                    chn = ch_toplot_timecourse[i]
                    if chn.find(modality) < 0:
                        continue
                    mn = np.mean( chdata[i,:] )

                    st = glob_stats[sind_str][medcond][task][chn]
                    pars = plot_paramsPerModality[modality]

                    ys = chdata[i,:]
                    if pars.get('shiftMean',True):
                        ys -= mn

                        #addstr += ', meanshited'
                    # either bandpass or low/highpass
                    cutlowfreq = False
                    filtered = False
                    if 'bandpass_freq' in pars:
                        bandpassFltorder =  pars['bandpass_order'] 
                        bandpassFreq = pars['bandpass_freq']
                        sos = sig.butter(bandpassFltorder, bandpassFreq, 
                                'bandpass', fs=sampleFreq, output='sos')
                        ys = sig.sosfilt(sos, ys)

                        if sampleFreq[0] > 0.5:
                            cutlowfreq = True
                        #addstr += ', bandpass {}Hz'.format(bandpassFreq)
                        filtered = True
                    else:
                        if 'highpass_freq' in pars:
                            highpassFltorder =  pars['highpass_order'] 
                            highpassFreq = pars['highpass_freq']
                            sos = sig.butter(highpassFltorder, highpassFreq, 
                                    'highpass', fs=sampleFreq, output='sos')
                            ys = sig.sosfilt(sos, ys)

                            #addstr += ', highpass {}Hz'.format(highpassFreq)
                            cutlowfreq = True
                            filtered = True
                        if 'lowpass_freq' in pars:
                            lowpassFltorder =  pars['lowpass_order'] 
                            lowpassFreq = pars['lowpass_freq']
                            sos = sig.butter(lowpassFltorder, lowpassFreq, 
                                    'lowpass', fs=sampleFreq, output='sos')
                            ys = sig.sosfilt(sos, ys)
                            filtered = True
                            #addstr += ', lowpass {}Hz'.format(lowpassFreq)

                    axLims = pars.get('axLims', {} )
                    if not filtered:
                        ymin, ymax = axLims.get(sind_str, (st['min'],st['max'] ) )
                    else:
                        ymin, ymax = np.min(ys), np.max(ys)
                        curModMax = max(ymax,curModMax)
                        curModMin = min(ymin,curModMin)
                    if pars.get('shiftMean',True):
                        axLims_meanshifted = pars.get('axLims_meanshifted',{} )
                        if sind_str in axLims_meanshifted: 
                            ymin, ymax = axLims_meanshifted.get(sind_str )
                        elif not cutlowfreq: 
                            ymin -= mn
                            ymax -= mn
                    #else: # because if we filter, we remove highest components
                    #    ymin, ymax = -ymaxs[modality],ymaxs[modality]

                    if pars.get('rectify',False):
                        ys = np.max(ys,0)
                        ymin = max(0,ymin)

                    ax.plot(chtimes, ys, label = '{}'.format(chn), alpha=0.7 )
                    ax.set_ylim(ymin,ymax)
                    ax.set_xlim(mintime,maxtime)
                    
                    if isinstance(tremorIntervalsCurSide, dict) and chn in tremorIntervalsCurSide:
                        pairs = tremorIntervalsCurSide[chn]
                        for pa in pairs:
                            ax.fill_between( list(pa) , ymin, ymax, facecolor='red', alpha=0.2)

                # don't want overlay multiple times
                #if not (modality == 'EMG' and isinstance(tremorIntervalsCurSide, dict) ):
                #    pairs = tremIntervalMerged[k][side_body]
                #    for pa in pairs:
                #        ax.fill_between( list(pa) , ymin, ymax, facecolor='red', alpha=0.2)


                #print( 'PAIRSSSSSSSSSSS ',k,len(pairs), pairs )
                for pa in pairs:
                    ax.fill_between( list(pa) , ymin, ymax, facecolor='red', alpha=0.2)

                addstr = ''
                if pars['shiftMean']:
                    addstr += ', meanshited'
                # either bandpass or low/highpass
                if 'bandpass_freq' in pars:
                    bandpassFreq = pars['bandpass_freq']
                    addstr += ', bandpass {}Hz'.format(bandpassFreq)
                else:
                    if 'highpass_freq' in pars:
                        highpassFreq = pars['highpass_freq']
                        addstr += ', highpass {}Hz'.format(highpassFreq)
                    if 'lowpass_freq' in pars:
                        lowpassFreq = pars['lowpass_freq']
                        addstr += ', lowpass {}Hz'.format(lowpassFreq)
                    

                ax.legend(loc=legendloc,framealpha = legalpha)
                l = len( plot_timeIntervalPerRaw.get( k, [0]  ) ) 
                ax.set_title('{}, {} hand: {}{} _interval_{}/{}_{}'.format(k,side_body,modality,addstr,
                    intind,l,intervalType) )
                ax.set_xlabel('Time, [s]')
                #ax.set_ylabel('{} Voltage, mean subtracted{}'.format(modality,addstr))
                ax.set_ylabel('{} Voltage'.format(modality))
                
                if side_body == gv.gen_subj_info[sind_str]['tremor_side']:
                    ax.patch.set_facecolor(mainSideColor)
                else:
                    ax.patch.set_facecolor(otherSideColor)
            #ax.axvline(300,c='k',ls='--')
            #ax.axvline(365,c='k',ls='--')

            if extend_plot_time:
                #chdata, chtimes = raws[k][chs,ts:te]
                chdata = chdata[:,ts:te]
                chtimes = chtimes[ts:te]
                chdataForPSD = chdata
            else:
                #chdata, chtimes = raws[k][chs,:]
                chdataForPSD = chdata[ts:te]

            rowind_shift = channel_pair_ind_forAx * nplots_per_side + chanTypes_toshow[ 'timecourse']['nplots']


            ##################### Tremor convolutions
            if show_tremconv:
                modality = 'EMG'
                spcht = chanTypes_toshow['timecourse']['chantypes']
                ch_EMG = chnames_tuples[channel_pair_ind][modality] 
                ax = axs[rowind_shift,colind]
                for chn in ch_EMG:
                    ys = tremCvlPerRaw[k].get(chn,None)
                    if ys is not None:
                        if gv.gparams['tremDet_useTremorBand']:
                            specgramsComputed = gv.specgrams[k]
                            freqs, bins, Sxx = specgramsComputed[chn]
                        else:
                            ts_,te_ = gv.raws[k].time_as_index([gv.gparams['tremDet_timeStart'], 
                    gv.gparams['tremDet_timeEnd']])
                            chdata,chtimes = utils.getData(k,[chn],ts_,te_)
                            bins = chtimes
                        #from IPython.core.debugger import Tracer; debug_here = Tracer(); debug_here()

                        print(chn,'plot cvl max', np.max(ys),'argmax ',np.argmax(ys) )
                        ax.plot(bins, ys, label=chn)
                
                ymin = 0
                ymax = plot_tremConvAxMax
                ax.set_ylim(ymin,ymax)
                ax.set_xlim(np.min(bins),np.max(bins))
                ax.axhline(y=tremIntDef_convThr, label='conv thr',ls=':',c='k')
                ax.legend(loc=legendloc,framealpha = legalpha)
                ax.set_title('Tremor band convolution')

                #pairs = tremIntervalMerged[k][side_body]
                for pa in pairs:
                    ax.fill_between( list(pa) , ymin, ymax, facecolor='red', alpha=0.2)
                if side_body == gv.gen_subj_info[sind_str]['tremor_side']:
                    ax.patch.set_facecolor(mainSideColor)

                ax.set_xlim(mintime,maxtime)
            rowind_shift += chanTypes_toshow['tremcvl']['nplots']



            axpsd = axs[rowind_shift,colind] 
            ################## plot PSD
            if show_bandpow and not ftc_instead_psd:
                wnd = np.hamming(NFFT)
                #ax = axs[rowind_shift,colind]
                ax = axpsd
                for ii,chind in enumerate(chs):
                    ax.psd(chdataForPSD[ii,:], Fs=sampleFreq, window=wnd, label=chnames[chind],scale_by_freq=0) # here always subset of times
                    ax.set_xlim(0,plot_maxFreqInSpec)
                ax.legend(loc=legendloc)
                ax.set_title('PSD whole signal')
            if side_body == gv.gen_subj_info[sind_str]['tremor_side']:
                ax.patch.set_facecolor(mainSideColor)
            else:
                ax.patch.set_facecolor(otherSideColor)

            tetmp = min(te+NFFT,int(maxtime*sampleFreq) )

            ################  plot freq bands ################################
            #exec(open('bandpow_plot_code.py').read(), globals(), locals() )
            if show_bandpow and ftc_instead_psd:
                EMGbandpow = {}

                print('Starting plotting frequency bands')
                specgramsComputed = gv.specgrams[k]
                spcht = chanTypes_toshow['bandpow']['chantypes']
                rowshift2 = 0
                maxpow = 0
                for modalityi,modality in enumerate(spcht):
                    if modality not in maxBandpowPerRaw[k][channel_pair_ind]:
                        maxBandpowPerRaw[k][channel_pair_ind ][modality] = {}
                    #ch_toplot_bandpow = []
                    #for modality in spcht:
                    #    ch_toplot_bandpow += chnames_tuples[channel_pair_ind][modality]
                    ch_toplot_bandpow = chnames_tuples[channel_pair_ind][modality] 
                    
                    ch_toplot_bandpow = utils.filterFavChnames( ch_toplot_bandpow, sind_str ) 
                    pars = plot_paramsPerModality[modality]

                    colorEMGind = 0

                    #if chn.find(modality) < 0:
                    #    continue
                    # by default we show all interesting frequency bands
                    freqBands_names = plot_freqBandNames_perModality[modality]
                    # although for EMG only tremor band
                    for fbi,fbname in enumerate(freqBands_names):
                        if fbname not in maxBandpowPerRaw[k][channel_pair_ind ][modality]:
                            maxBandpowPerRaw[k][channel_pair_ind ][modality][fbname] = [],0

                        maxpow_band = 0
                        axcoordy = rowshift2
                        title = '{}, {}, Freq bands powers'.format(k, modality)
                        #if modality.find( 'MEGsrc') >= 0 or modality.find('LFP') >= 0 :
                        #    if plot_MEGsrc_rowPerBand or plot_LFP_rowPerBand:   # assume MEGsrc are always the last to plot 
                        #        axcoordy += fbi
                        #        if plot_MEGsrc_rowPerBand and plot_LFP_rowPerBand:  # assume MEGsrc goes after LFP
                        #            axcoordy += len( plot_freqBandNames_perModality['LFP'] ) 
                        #        title = '{}, {}, {} band power'.format(k, modality,fbname)
                        #    #getSrcname
                        #    if plot_MEGsrc_separateRoi:
                        #        curroi,srci = parseMEGsrcChname(chn)
                        #        axcoordy += MEGsrc_roi.index(curroi)     # assume MEGsrc are always the last to plot

                        #print('{} ,  {}'.format(chn, axcoordy ) )
                        ax = axs[rowind_shift + rowshift2, colind]


                        for chn in ch_toplot_bandpow:
                            freqs, bins, Sxx = specgramsComputed[chn]
                            if specgram_scaling == 'psd':
                                freqres = freqs[1]-freqs[0]
                            else:
                                freqres = 1.
                            assert chn.find(modality) >= 0 

                            color = None
                            if chn.find('EMG') >= 0:      # use fixed colors for EMG
                                color = plot_colorsEMG[colorEMGind]
                                colorEMGind += 1
                            #
                            #bandpower = plotChannelBand(ax,k,chn,fbname,time_start,tetmp/sampleFreq,logscale=0,color=color)
                            bandinfo = plotChannelBand(ax,k,chn,fbname,mintime,maxtime,
                                    logscale=0,color=color, normalization = ('whole','channel_distr'))
                            if bandinfo is not None:
                                bins, bandpower = bandinfo

                                if modality == 'EMG' and fbname == 'tremor':
                                    EMGbandpow[chn] = bandpower

                                if spec_specgramtype == 'scaleogram' and bins[0] < 1 or plot_time_end - bins[-1] < 1:
                                    bininds_ = np.where( 
                                            np.logical_and(bins > 1, bins < plot_time_end - 1) )[0]
                                    maxpowcur = np.max(bandpower[bininds_] )
                                else:
                                    maxpowcur = np.max(bandpower)

                                axstmp,mbppr = maxBandpowPerRaw[k][channel_pair_ind][modality][fbname]  
                                curmax = max(mbppr, maxpowcur)
                                maxBandpowPerRaw[k][channel_pair_ind][modality][fbname] =  (axstmp + [ax] )  ,curmax
                                #ymax = glob_stats_perint[k][chn][intind ][ s0 + 'max' ]

                                maxpow = max(maxpow, maxpowcur )
                                maxpow_band = max(maxpow_band, maxpowcur)
                                ax.legend(loc=legendloc,framealpha = legalpha)

                                ymax = curmax
                                s0 = '{}_bandpow_'.format(fbname)
                                ymax = 1.
                                ax.set_ylim(0,ymax)
                                ax.set_xlim(mintime,maxtime)
                                #print('fsdddddddddddddddddddddddddddddddddddd')

                            if modality == 'MEGsrc':
                                chlbl = utils.getMEGsrc_chname_nice(chn)
                                #print('{}__{}__{} maxpow {:6E}'.format(k,chlbl,fbname, maxpowcur) )
                                #import pdb; pdb.set_trace()

                            ax.set_title(title )

                            #if chn.find('LFP') >= 0:
                            #    print('--------DEBUG',fbname, a, aa, ymin,ymax)


                            if side_body == gv.gen_subj_info[sind_str]['tremor_side']:
                                ax.patch.set_facecolor(mainSideColor)
                            else:
                                ax.patch.set_facecolor(otherSideColor)


                        # end of cycle over freq bands
                        if modality.find('EMG') >= 0:
                            # computed using k-means 
                            clust = glob_stats[sind_str][medcond][task]['tremorfreq_clusters_allEMG'][chn]
                            if tremrDet_clusterMultiMEG:
                                for clusti in range(len(clust) ):
                                    ax.axhline(y = clust[clusti], 
                                            label = '{} clust{}'.format(chn,clusti) ,
                                            #ls = plot_freqBandsLineStyle['tremor' ], 
                                            ls='--', 
                                            c=plot_colorsEMG[colorEMGind-1] )
                            else:
                                thrs = glob_stats[sind_str][medcond][task][chn]['thrPerFreq_trem'] 
                                ax.axhline(y = freqres * np.sum(thrs), 
                                        label = '{} tremor thr'.format(chn) ,ls = ltype_tremorThr)

                            if tremorDetectUseCustomThr:
                                try:
                                    thr = gv.gen_subj_info[sind_str]['tremorDetect_customThr'][medcond][side_body][chn] 
                                except KeyError:
                                    thr = glob_stats[sind_str][medcond][task]['thrPerCh_trem_allEMG'][chn]
                                ax.axhline(y=thr, ls = ltype_tremorThr,lw=2, c= plot_colorsEMG[colorEMGind-1], 
                                        label = '{} tremor thr'.format(chn) )

                        #deflims = (0, maxpow_band)
                        deflims = (-1, 1)
                        a = pars.get('axLims_bandPow',{}).get(sind_str, deflims ) 
                        if isinstance(a,dict):
                            if medcond in a:
                                suba = a[medcond]
                                if side_body in a:
                                    suba2 = suba[side_body]
                                    aa = suba2.get(fbname, suba2['default'] )
                                else:
                                    aa = suba.get(fbname, suba['default'] )
                            elif side_body in a:
                                suba2 = a[side_body]
                                aa = suba2.get(fbname, suba2['default'] )
                            else:
                                aa = a.get(fbname, a['default'] )
                            ymin,ymax = aa
                        else:
                            ymin,ymax = a

                        ax.set_ylim(ymin,ymax)

                        #pairs = tremIntervalMerged[k][side_body]
                        for pa in pairs:
                            ax.fill_between( list(pa) , ymin, ymax, facecolor='red', alpha=0.2)

                        rowshift2 += 1
                    # end of cycle over freq bands

                # end of cycle over modality
            rowind_shift += chanTypes_toshow['bandpow']['nplots']


            if show_EMG_band_corr:
                print('Starting plotting frequency bands')
                specgramsComputed = gv.specgrams[k]
                spcht = chanTypes_toshow['EMGband_corr']['chantypes']
                rowshift2 = 0
                for modalityi,modality in enumerate(spcht):
                    ch_toplot_bandpow = chnames_tuples[channel_pair_ind][modality] 
                    ch_toplot_bandpow = utils.filterFavChnames( ch_toplot_bandpow, sind_str ) 
                    pars = plot_paramsPerModality[modality]
                    colorEMGind = 0
                    freqBands_names = plot_freqBandNames_perModality[modality]
                    # although for EMG only tremor band
                    for fbi,fbname in enumerate(freqBands_names):
                        maxpow_band = 0
                        axcoordy = rowshift2
                        title = '{} Correl with EMG tremor, {}, Freq band {} power'.format(k, modality, fbname)

                        #print('{} ,  {}'.format(chn, axcoordy ) )
                        ax = axs[rowind_shift + rowshift2, colind]

                        for chn in ch_toplot_bandpow:
                            freqs, bins, Sxx = specgramsComputed[chn]
                            if specgram_scaling == 'psd':
                                freqres = freqs[1]-freqs[0]
                            else:
                                freqres = 1.
                            assert chn.find(modality) >= 0 


                            color = None
                            if chn.find('EMG') >= 0:      # use fixed colors for EMG
                                color = plot_colorsEMG[colorEMGind]
                                colorEMGind += 1
                            #
                            #bandpower = plotChannelBand(ax,k,chn,fbname,time_start,tetmp/sampleFreq,logscale=0,color=color)
                            bins, bandpower = plotChannelBand(ax,k,chn,fbname,mintime,maxtime,
                                    logscale=0,color=color,skipPlot = 1)
                            for EMGchn in EMGbandpow:
                                #freqs,coher = sig.coherence( , ,fs = sampleFreq, nperseg=NFFT 
                                correl = np.correlate( bandpower /np.max(bandpower), 
                                        EMGbandpow[EMGchn] / np.max(EMGbandpow[EMGchn] ), 'same' )
                                tm = bins - np.min(bins) - ( np.max(bins) - np.min(bins) )/2
                                ax.plot(tm, correl, label = '{} -- {}'.format(chn,EMGchn ) )
                                mi = np.min(tm)
                                ma = np.max(tm)

                                correlTimeMax = 25  # in sec
                                ma = min( ma, correlTimeMax )
                                mi = max( mi, -correlTimeMax )
                                ax.set_xlim(mi,ma)

                                maxpowcur = np.max(correl)
                                maxpow_band = max(maxpow_band, maxpowcur)

                            ax.legend(loc=legendloc,framealpha = legalpha)

                            ax.set_title(title )

                            #if chn.find('LFP') >= 0:
                            #    print('--------DEBUG',fbname, a, aa, ymin,ymax)


                            if side_body == gv.gen_subj_info[sind_str]['tremor_side']:
                                ax.patch.set_facecolor(mainSideColor)
                            else:
                                ax.patch.set_facecolor(otherSideColor)

                        deflims = (0, maxpow_band)
                        ymin,ymax = deflims
                        ax.set_ylim(ymin,ymax)
                        if side_body == gv.gen_subj_info[sind_str]['tremor_side']:
                            ax.patch.set_facecolor(mainSideColor)

                        rowshift2 += 1
                    # end of cycle over freq bands
                # end of cycle over modality
                rowind_shift += chanTypes_toshow['EMGband_corr']['nplots']


            if show_stats_barplot:
                ax = axs[rowind_shift, colind]
                spcht = chanTypes_toshow['stats_barplot']['chantypes']
                ch_toplot_barplot = []
                for modality in spcht:
                    ch_toplot_barplot += chnames_tuples[channel_pair_ind][modality] 
                    for rowind,chname in enumerate(ch_toplot_barplot):
                        makeBarPlot(ax, k, chname)
                        break
                    break
                    
                rowind_shift += chanTypes_toshow['stats_barplot']['nplots']


            if show_spec:
                print('Starting plotting spectrum')
                spcht = chanTypes_toshow['spectrogram']['chantypes']
                ch_toplot_spec = []
                for modality in spcht:
                    ch_toplot_spec += chnames_tuples[channel_pair_ind][modality] 
                ch_toplot_spec = utils.filterFavChnames( ch_toplot_spec, sind_str )
                #chs = mne.pick_channels(chnames,include=ch_toplot_spec, ordered=True )
                
                #import pdb; pdb.set_trace()
                specgramsComputed = gv.specgrams[k]

                for rowind,chname in enumerate(ch_toplot_spec):
                    freqs, bins, Sxx = gv.specgrams[k][chname]
                
                    minfreq = 0
                    if chname.find('LFP') >= 0:
                        minfreq = plot_minFreqInSpec
                    freqs, bins, Sxx = utils.getSubspec(freqs,bins,Sxx,
                                                  minfreq,plot_maxFreqInSpec,
                                                  time_start,tetmp/sampleFreq)
                    
                    ax = axs[rowind + rowind_shift,colind]
                    stats = glob_stats[sind_str][medcond][task][chname]
                    mx = stats['max_spec']; mn = stats['min_spec']; me = stats['mean_spec']                
                    #mx_mc = stats['max_spec_mc_plot']; mn_mc = stats['min_spec_mc_plot']; 
                    mx_mc = stats['max_distr_spec_mc_plot']; 
                    mn_mc = stats['min_distr_spec_mc_plot']; 

                    me = stats['mean_spec']                
                    #while (mx - me) > (mx - mn)*0.9:  # catch short highrise
                    #    mx *= 0.9
                    #print(mx,mn,me)

                    #mn_mc = -1
                    #mx_mc = 10
                    
                    if normType == 'uniform':
                        #norm = mpl.colors.Normalize(vmin=0.,vmax=mx_mc);
                        norm = mpl.colors.Normalize(vmin=mn_mc,vmax=mx_mc);
                    elif normType == 'log':
                        norm = mpl.colors.LogNorm(vmin=mn_mc,vmax=mx_mc);
                    #print(chname,Sxx.shape,len(freqs),mx, mn)
                    if chname.find('MEGsrc') >= 0:
                        modality = 'MEGsrc'
                    else:
                        modality = 'LFP'

                    if isinstance(Sxx[0,0], complex):
                        Sxx = np.abs(Sxx)

                    freqinds = np.where( np.logical_and(freqs >= minfreq,freqs <= plot_maxFreqInSpec) )[0]

                    Sxx =  ( Sxx -  me[freqinds,None] ) / me[freqinds,None]    # divide by mean
                    im = ax.pcolormesh(bins, freqs, Sxx, 
                            cmap=plot_specgramCmapPerModality[modality], norm=norm)

                    #fig.colorbar(im, ax=ax) #for debugging
                    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.specgram.html

                    if chname.find('MEGsrc') >= 0:
                        chlbl = utils.getMEGsrc_chname_nice(chname)
                    else:
                        chlbl = chname
                        if gv.gen_subj_info[sind_str]['lfpchan_used_in_paper'] == chname:
                            chlbl = '* ' + chname

                    specparstr = ''
                    if spec_specgramtype == 'scipy':
                        specparstr =  'NFFT={} overlap={:d}'.format(NFFT, int(NFFT*specgramoverlap) )
                    elif spec_specgramtype == 'lspopt':
                        specparstr =  'NFFT={} overlap={:d} c={}'.format(NFFT, int(NFFT*specgramoverlap), c_parameter, )
                    elif spec_specgramtype == 'mne.wavelet':
                        specparstr = 'cycc={}'.format(spec_freqs2wletcyclesCoef )

                    ax.set_title('Spec -:min {}: {}, {}\n min={:.6E}, max={:.6E}'.
                            format(k, chlbl, specparstr,  mn_mc,mx_mc) )
                    ax.set_xlabel('Time, [s]')
                    ax.set_ylabel('Freq, [Hz] '+normType)
                    ax.set_xlim(mintime,maxtime)
#                     if chname.find('LFP') >= 0:
#                         ax.set_ylim(plot_minFreqInSpec,plot_maxFreqInSpec)
#                     else:
#                         ax.set_ylim(0,plot_maxFreqInSpec)
                        
                    if side_body == gv.gen_subj_info[sind_str]['tremor_side']:
                        ax.patch.set_facecolor(mainSideColor)
                    else:
                        ax.patch.set_facecolor(otherSideColor)
        print('Plotting {} finished'.format(k))

        if intervalType == 'entire' and nc == 1:
            for ri in range(nr):
                ax = axs[ri, colind]
                ax.set_xticks( np.arange(plot_time_start, plot_time_end, 2 ) )

    if plot_setBandlims_naively:  # just take max over what is plotted
        # second pass to set right limits
        for colind in range(nc):
            ki,intind = raw_int_pairs[colind]
            k = ks[ki]
            for channel_pair_ind in pair_inds:
                spcht = chanTypes_toshow['bandpow']['chantypes']
                for modalityi,modality in enumerate(spcht):
                    freqBands_names = plot_freqBandNames_perModality[modality]
                    for fbi,fbname in enumerate(freqBands_names):
                        axstmp,mbppr = maxBandpowPerRaw[k][channel_pair_ind][modality][fbname]  
                        for ax in axstmp:
                            ax.set_ylim(0,mbppr)


    ss = []
    subjstr = ''
    for k in ks:
        sind_str,medcond,task = getParamsFromRawname(k)
        if sind_str in ss:
            continue
        ss += [sind_str]
        subjstr += '{},'.format( sind_str )

    #plt.tight_layout()
    if savefig:
        #figname = 'Spec_{}_pairno{}_{} nr{}, {:.2f},{:.2f}, spec{} timecourse{} c{}.{}'. \
        #            format(subjstr, channel_pair_ind, data_type, nr, float(time_start), \
        #                   float(time_end),show_spec,show_timecourse,c_parameter,ext)
        if spec_specgramtype == 'lspopt':
            sp = 'lspopt{}'.format(c_parameter)
        elif spec_specgramtype == 'scaleogram':
            sp = 'pywt'
        else:
            sp = spec_specgramtype

        if nc == 1:
            prename = 'single{}_{}_{}'.format(subjstr,medconds[0], tasks[0] )
        else:
            prename = 'Spec_{}'.format(subjstr  )

        figname = '{}_pairno{}_{} nr{}, nsrc{} _{}.{}'. \
                    format(prename, channel_pair_ind, data_type, nr, len(srcs), sp,ext)
        plt.savefig( os.path.join(plot_output_dir, figname ) )
        print('Figure saved to {}'.format(figname) )
    else:
        print('Skipping saving fig')

    if not showfig:
        plt.close()
    else:
        plt.show()
        
    print('Plotting all finished')




def makeBarPlot( ax, rawname, chname ):
    intTypes = ['pre', 'post', 'middle_full', 'no_tremor' ]
    statTypes = ['max', 'L1', 'L2', 'L05' ]

    intType2col =  {'pre':'blue', 'post':'yellow', 'middle_full':'red', 'no_tremor':'green' }
    intervals = timeIntervalPerRaw_processed[rawname]
    intervalStats = glob_stats_perint[rawname]

    ivalis = {}
    for itype in intTypes:
        ivit = []
        for i,interval in enumerate(intervals):
            t1,t2,it = interval
            if it == itype:
                ivit += [i]
        ivalis[itype] = ivit


    binvalsDict  = {}
    binnamesDict = {}
    bincoordsDict = {}

    binwidth = 1

    modality = utils.chname2modality(chname)
    for iti,itype in enumerate(intTypes):
        binvals  = []
        binerrs  = []
        binnames = []

        inds = ivalis[itype]  
        if len(inds) == 0:
            continue

        nbinsPerStatType = len(intTypes) + 1  # +1 to leave space between
        nbinsPerFB = nbinsPerStatType * len(statTypes) + 2   # +2 to leave space between 
        xs = []
        xcur = 0
        for fbname in plot_freqBandNames_perModality[modality]:
            for statType in statTypes:
                #gv.freqBands[modality]
                s0 = '{}_bandpow_{}'.format(fbname, statType)
                pp = intervalStats[i][chname]
                if s0 not in pp:
                    print('{} is not in {}{}{}'.format(s0,chname,fbname,itype ) )
                    continue
                vals = [ pp[s0] for i in inds]
                mn = np.mean(vals)
                std = np.std(vals)
                binvals += [ mn ]
                binerrs += [ std ]

                binnames += [ '{}_{}'.format(fbname,statType) ]
                xs += [ xcur  ]
                xcur +=  len(intTypes)

        binvalsDict[itype]  = binvals
        binnamesDict[itype] = binnames

        #tot = len(binvals) * binwidth * len(intTypes)
        xs = np.array(xs) * binwidth + iti * binwidth 
        bincoordsDict[itype] = xs

        ax.bar( xs, binvals, binwidth, 
                yerr=binerrs, label = itype, color = intType2col[itype] )

    #import pdb; pdb.set_trace()

    #ax.set_xticks
    ticklab = [0] * len(intTypes) * len(binnames)
    for iti,itype in enumerate(intTypes):
        for bni, binname in enumerate(binnames):
            ticklab[ len(intTypes) * bni + iti ] = binname
    ax.set_xticklabels ( ticklab ,rotation=45)

    ax.legend(loc=legendloc,framealpha = legalpha)
    ax.set_title('{} {}'.format(rawname, chname) )
            

    #nbins = len(binvals)

    # groups of gv.freqBands, within them groups of stats, within -- different int types

    #hspace_perfb = 1
    #shift_fblevel = 0
    #for fbname in plot_freqBandNames_perModality:
    #    #shift_stattype_level
    #    for itype in intTypes:
    #        for statType in statTypes:
    #            ects1 = ax.bar(x - binwidth/2, men_means, width, label='Men')
    #    shift_fblevel += hspace_perfb



    # plot max and norms -- their stats over intervals within all raws
    # better do in in jupyter
    # merge EMG channels somehow


        

import globvars as gv

####################################
####################################
####################################
####################################

if __name__ == '__main__':
    from utils import getParamsFromRawname
    from utils import getRawname


    if os.environ.get('DATA_DUSS') is not None:
        data_dir = os.path.expandvars('$DATA_DUSS')
    else:
        data_dir = '/home/demitau/data'


    ########################################## Data selection
    MEGsrc_roi = ['Brodmann area 4']
    MEGsrc_roi = ['Brodmann area 4', 'Brodmann area 6']
    #MEGsrc_roi = ['Brodmann area 6']
    MEGsrc_roi = ['HirschPt2011']
    #MEGsrc_roi = ['HirschPt2011,2013direct']

    try:
        subjinds
        print('Using existing subjinds list')
    except NameError:
        subjinds = [1,2,3,4,5,6,7,8,9,10]
        subjinds = [1]

    try:
        tasks
        print('Using existing tasks list')
    except NameError:
        tasks = ['hold', 'move', 'rest']
        tasks = ['move']

    try:
        medconds
        print('Using existing medconds list')
    except NameError:
        medconds = ['off', 'on']
        medconds = ['on']

    #subjinds = [1,2]
    #tasks = ['hold','move']

    #subjinds = [1]
    #tasks = ['hold','move']

    #tasks = ['hold', 'rest']

    #subjinds = [4]
    ###tasks = ['hold']
    #tasks = ['rest', 'move']
    #medconds = ['off', 'on']

    #subjinds = [3,4,5,6]
    ##tasks = ['hold']
    #tasks = ['rest', 'move', 'hold']
    #medconds = ['off', 'on']

    import sys, getopt
    helpstr = 'Usage example\nudus_dataproc.py -i <comma sep list> -t <comma sep list> -m <comma sep list>'
    try:
        effargv = sys.argv[1:]  # to skip first
        opts, args = getopt.getopt(effargv,"hi:t:m:r:",["subjinds=","tasks=","medconds=","MEG_ROI="]) 
        print(sys.argv, opts, args)

        for opt, arg in opts:
          if opt == '-h':
              print (helpstr) 
              sys.exit(0)
          elif opt in ("-i", "--subjinds"):
              subjinds = [ int(indstr) for indstr in arg.split(',') ]
          elif opt in ("-t", "--tasks"):
              tasks = arg.split(',')
          elif opt in ("-m", "--medconds"):
              medconds = arg.split(',')
          elif opt in ("-r", "--MEG_ROI"):
              MEGsrc_roi = arg.split(',')
    except getopt.GetoptError: 
        print('Error in argument parsing!', helpstr) 
        print('Putting hardcoded vals')


    print('Subjinds:{}, tasks:{}, medconds:{}'.format(subjinds, tasks, medconds) )

    with open(os.path.join(data_dir,'trem_times.json') ) as jf:
        trem_times_Jan = json.load(jf)   
        # trem_times_Jan[medcond][task]['tremor'] is a list of times that one needs to couple to get begin/end
        # there is also "no_tremor"

    with open(os.path.join('.','coord_labels.json') ) as jf:
        coord_labels = json.load(jf)
        gv.gparams['coord_labels'] = coord_labels



    # used on the level of chanTuples generation first, so for unselected nothing will be computed 
    MEGsrc_names_toshow = ['MEGsrc_Brodmann area 4_0', 'MEGsrc_Brodmann area 4_10', 'MEGsrc_Brodmann area 4_15'  ]  # right
    MEGsrc_names_toshow += [ 'MEGsrc_Brodmann area 4_3', 'MEGsrc_Brodmann area 4_30', 'MEGsrc_Brodmann area 4_60' ]  # left
    MEGsrc_names_toshow += [ 'MEGsrc_Brodmann area 6_3', 'MEGsrc_Brodmann area 6_60', 'MEGsrc_Brodmann area 4_299' ]  # right
    MEGsrc_names_toshow += [ 'MEGsrc_Brodmann area 6_0', 'MEGsrc_Brodmann area 6_72', 'MEGsrc_Brodmann area 4_340', 'MEGsrc_Brodmann area 4_75'  ]  # left
    #MEGsrc_names_toshow = []
    MEGsrc_names_toshow = None    # more accurate but requires computation of spectrograms for all channels, which can be long
    
    # left indices in Brodmann 4 --  [3, 4, 7, 8, 9, 13, 14, 16, 18, 22, 23, 24, 26, 30, 31, 32, 34, 36, 37, 40, 41, 43, 45, 46, 47, 51, 52, 58, 59, 60, 63, 64, 67, 68, 72, 73, 74]
    # left indiced in Brodmann 6 -- '
    '''
    0, 2, 3, 4, 5, 7, 9, 11, 12, 14, 15, 19, 20, 21, 22, 23, 29, 30, 31, 32,
    34, 35, 36, 41, 42, 47, 48, 49, 50, 53, 54, 55, 56, 62, 63, 64, 69, 70, 71,
    74, 75, 76, 77, 79, 80, 81, 86, 87, 88, 92, 93, 96, 97, 101, 102, 105, 106,
    107, 108, 111, 112, 113, 116, 117, 118, 121, 123, 124, 125, 126, 129, 131,
    132, 135, 136, 137, 138, 139, 145, 146, 147, 148, 149, 156, 157, 161, 162,
    163, 164, 165, 172, 173, 174, 175, 176, 177, 183, 186, 188, 189, 190, 191,
    195, 196, 197, 198, 199, 203, 204, 205, 208, 209, 210, 214, 215, 216, 217,
    218, 223, 224, 225, 226, 227, 233, 234, 235, 236, 240, 241, 242, 243, 247,
    248, 249, 253, 254, 257, 258, 259, 261, 262, 263, 264, 267, 268, 269, 270,
    271, 272, 277, 278, 279, 284, 285, 288, 289, 290, 294, 295, 296, 297, 301,
    302, 303, 307, 308, 309, 310, 313, 314, 315, 318, 319, 322, 324, 326, 327,
    328, 329, 333, 334, 335, 336, 339, 340, 341, 342, 343, 349, 350, 353, 355,
    356, 359, 360, 361, 365, 367, 368, 369] 
    '''
                
    gv.raws = {}
    gv.srcs = {}

    from globvars import raws
    from globvars import srcs

    #########################  Plot params
    plot_output_dir = 'output'

    specgramoverlap = 0.75
    c_parameter = 20.0  # for the spectrogram calc
    #c_parameter = 3
    #NFFT = 256  # for spectrograms
    NFFT = 256  # for spectrograms
    nonTaskTimeEnd = 300
    specgram_scaling = 'spectrum'  # power, not normalized by freq band (i.e. not denisty)
    gv.gparams['specgram_scaling'] = specgram_scaling
 
    # which time range use to compute spectrograms
    spec_time_start=0 
    spec_time_end=nonTaskTimeEnd
    spec_wavelet = "cmor1.5-1.5"     # first is width of the Gaussian, second is number of cycles
    spec_wavelet = "cmor1-1.5"    
    spec_FToverlap = 0.75
    spec_specgramtype = 'lspopt'
    spec_specgramtype = 'scaleogram'
    spec_specgramtype = 'mne.wavelet'
    spec_freqs2wletcyclesCoef = 0.75  # larger coef -- wider gaussian 
    #spec_specgramtype = 'scipy'
    spec_cwtscales = np.arange(2, 120, 4)  # lower scale = a means that highest freq is 2/dt = 2*sampleFreq
    base = 5
    spec_cwtscales = 2 + (np.logspace(0.0, 1, 25,base=base) - 1 ) * 400/base;

    maxscale = 14400   # for large scales (>10k) it takes too long
    maxscale = 800  
    #maxscale = 400  
    minscale = 7  # ~ 54 Hz  (smaller minscale means higher max freq, minscale 1 means max freq = sampling freq)
    N = 25
    pw = 3.5  # higher values -- larger density of higher freq
    xs = np.arange(1, N + 1) / N
    s = np.exp(xs) 
    s = np.power(xs, pw)
    s -= np.min(s)
    smax = np.max(s)
    spec_cwtscales = minscale + ( s/smax ) * maxscale/base; ttl = 'log {} spacing of scales'.format(base);
    #
    #spec_MNEwletFreqs = np.arange(1,50,2)
    spec_MNEwletFreqs = np.arange(1,50,1)
    spec_MNEwletFreqs = np.arange(1,90,1)

    # max freq 128, min scale 2, max scale 120, want concentr higher closer to two
    # 2 + 120 * (

    #tremorThrMult = {'S05': 20. }
    tremorThrMult = { }
    updateJSON = False
    useKilEMG = False
    plot_highPassLFP = True

    tremrDet_clusterMultiFreq = True
    tremrDet_clusterMultiMEG = True
    gv.gparams['tremDet_useTremorBand'] = False
    gv.gparams['tremDet_timeStart'] = 0
    gv.gparams['tremDet_timeEnd'] = nonTaskTimeEnd

    gv.gparams['plot_LFP_onlyFavoriteChannels'] = 0
    plot_EMG_spectrogram          = False
    plot_MEGsrc_spectrogram       = True
    plot_onlyMainSide             = True
    #plot_onlyMainSide             = False
    #show_EMG_band_corr            = True
    show_EMG_band_corr            = False
    show_stats_barplot            = True
    # I have selected them myself, not the ones chosen by Jan
    # those with more apparent contrast between background and beta and tremor


    plot_timeIntervalPerRaw = { 'S01_off_hold':[(200,208),  (180,220) ], 'S01_on_hold':[ (0,30) ], 
            'S01_off_move':[ (100,150) ],
            'S02_off_hold':[ (210,215), (20,50) ], 'S02_on_hold':[ (100,110), (20,60) ], 
            'S02_off_move':[ (90,120) ],'S02_on_move':[ (105,153) ], 
            'S03_off_hold':[ (55,70), (100,130) ],
            'S04_off_hold':[ (220,250), (180,210)  ],
            'S04_on_move':[ (103,110)  ],
            'S04_on_hold':[ (100,110)  ],
            'S08_off_rest': [(237,257)],
            'S08_on_rest':[(180,250) ], 'S09_off_rest':[(0,100 ) ] }
    plot_timeIntervalPerRaw = {}  # to plot all intervals

    obsbetalims = [ (0, 20) ]
    #plot_timeIntervalPerRaw = { 'S01_off_hold':obsbetalims, 'S01_on_hold':obsbetalims, 
    #        'S01_off_move':obsbetalims, 'S01_on_move': obsbetalims}

    favoriteLFPch_perSubj = {'S01': ['LFPR23', 'LFPL01' ], 
            'S02': ['LFPR23', 'LFPL12'], 'S03': ['LFPR12', 'LFPL12'], 'S09':['LFPL78', 'LFPR67'], 
            'S08':['LFPR12' , 'LFPL56' ], 'S04':['LFPL01', 'LFPR01'] } 
    plot_time_start = 0
    plot_time_end = 300

    plot_minFreqInSpec = 2.5  # to get rid of heart rate
    plot_minFreqInBandpow = 2.5  # to get rid of heart rate
    #plot_maxFreqInSpec = 50
    #plot_maxFreqInSpec = 80
    #plot_maxFreqInSpec = 35
    plot_maxFreqInSpec = 35 # max beta
    #plot_maxFreqInSpec = 100
    plot_MEGsrc_sortBand = 'tremor'  # if we have many MEG sources we don't want to show all, only those having some feature being highest
    plot_numBestMEGsrc = 3
    plot_numBestMEGsrc = 10
    if MEGsrc_names_toshow is not None:
        plot_numBestMEGsrc = len( MEGsrc_names_toshow ) // 2





    EMGlimsBySubj =  { 'S01':(0,0.001) }  
    #EMGlimsBySubj_meanshifted =  { 'S01':(-0.0001,0.0001),  'S02':(-0.0002,0.0002)}   # without highpassing
    EMGlimsBySubj_meanshifted =  { 'S01':(-0.00005,0.00005),  'S02':(-0.0002,0.0002), 
            'S03':(-0.0002,0.0002), 'S04':(-0.0001,0.0001), 
            'S05':(-0.0004,0.0004), 'S06':(-0.0002,0.0002), 
            'S07':(-0.0002,0.0002), 'S10':(-0.0002,0.0002), 
            'S08':(-0.0002,0.0002), 'S09':(-0.0002,0.0002)   }  
    EMGlimsBandPowBySubj = {}
    if spec_specgramtype in ['scipy', 'lspopt']:
        EMGlimsBandPowBySubj =  { 'S01':(0,15),  'S02':(0,200), 'S03':(0,200),  'S04':(0,200), 
                'S05':(0,200), 'S08':(0,700), 'S09':(0,700) }  
      
    LFPlimsBySubj_meanshifted =  { 'S01':(-0.000015,0.000015), 'S02':(-0.000015,0.000015) }  

    LFPlimsBySubj = {}
    LFPlimsBandPowBySubj =  {}; #{ 'S01':(0,1e-11), 'S02':(0,1e-11) }  
    MEGsrclimsBandPowBySubj =  {  }
    if spec_specgramtype in ['scipy', 'lspopt']:
        #LFPlimsBySubj =  { 'S01':(0,0.001) }  
        LFPlimsBandPowBySubj = { 'S01':(0,1e-11), 'S02':(0,1e-11) }  
        bandlims = {'default':(0,0.2), 'tremor':(0,2.6 ), 'beta':(0,0.5), 'gamma_motor':(0,0.23) }
        LFPlimsBandPowBySubj = { 'S01': bandlims,   'S02':(0,0.2) }  
        bandlims = {'default':(0,0.2), 'tremor':(0,2 ), 'beta':(0,0.3), 'gamma_motor':(0,0.07) }
        LFPlimsBandPowBySubj[ 'S02' ]  = bandlims
        bandlims = {'default':(0,0.2), 'tremor':(0,8 ), 'beta':(0,1), 'gamma_motor':(0,0.2) }
        LFPlimsBandPowBySubj[ 'S03' ]  = bandlims
        bandlims = {'off': {'default':(0,0.2), 'tremor':(0,100 ), 'beta':(0,1), 'gamma_motor':(0,0.2) } }
        bandlims['on'] = {'default':(0,0.2), 'tremor':(0,200 ), 'beta':(0,8), 'gamma_motor':(0,0.6) } 
        LFPlimsBandPowBySubj[ 'S04' ]  = bandlims
        bandlims = {'default':(0,8), 'tremor':(0,5 ), 'beta':(0,6), 'gamma_motor':(0,0.3) }
        LFPlimsBandPowBySubj[ 'S05' ]  = bandlims

        #MEGsrclimsBandPowBySubj =  { 'S01':(0,400),  'S02':(0,600), 'S03':(0,700)  }
        #MEGsrclimsBandPowBySubj =  { 'S01':(0,2.5e8),  'S02':(0,3e8), 'S03':(0,2e9)  }
        bandlims = {'default':(0,4e9), 'tremor':(0,8e8 ), 'beta':(0,4e8 ), 'gamma_motor':(0,1e8) }
        MEGsrclimsBandPowBySubj['S02'] = bandlims
        bandlims = {'default':(0,4e9), 'tremor':(0,6e9 ), 'beta':(0,3e8 ), 'gamma_motor':(0,1e8) }
        MEGsrclimsBandPowBySubj['S03'] = bandlims
    
    EMGlimsBySubj = {}
    EMGlimsBySubj_meanshifted = {}

    EMGplotPar = {'shiftMean':True, 
            'axLims':EMGlimsBySubj, 'axLims_meanshifted':EMGlimsBySubj_meanshifted,
            'axLims_bandPow':EMGlimsBandPowBySubj  }
    #EMGplotPar.update( {'lowpass_freq':15, 'lowpass_order':10} )

    #EMGplotPar.update( {'bandpass_freq':(plot_minFreqInBandpow,15), 'bandpass_order':10} )
    EMGplotPar.update( {'highpass_freq':10, 'highpass_order':10, 'rectify':False} )

    #if 'bandpass_freq' in EMGplotPar:
    #    if EMGplotPar['bandpass_freq'][0] < 1e-10:
    #        EMGplotPar['lowpass_freq'] = EMGplotPar['bandpass_freq'][1]
    #        del EMGplotPar['bandpass_freq']
    #        EMGplotPar['lowpass_order'] = EMGplotPar['bandpass_order']
    #        del EMGplotPar['bandpass_order']
    #elif 'highpassFreq' not in EMGplotPar:  # if we look at raw signal at low freq:
    #    for s in EMGlimsBySubj_meanshifted:
    #        lims = EMGlimsBySubj_meanshifted[s]
    #        EMGlimsBySubj_meanshifted[s] = ( lims[0]*2, lims[1]*2 )

    LFPplotPar = {'shiftMean':True, 'highpass_freq': plot_minFreqInBandpow, 'highpass_order':10, 
            'axLims':LFPlimsBySubj, 'axLims_meanshifted':LFPlimsBySubj_meanshifted,
            'axLims_bandPow':LFPlimsBandPowBySubj  }

    EOGplotPar = {'shiftMean':True, 
            'axLims': {}, 'axLims_meanshifted':{} } #'axLims_meanshifted':LFPlimsBySubj_meanshifted,
            #'axLims_bandPow':LFPlimsBandPowBySubj,
            # 'highpass_freq': 1.5, 'highpass_order':10 , }

    MEGsrcplotPar = {'shiftMean':True, 'highpass_freq': plot_minFreqInBandpow, 
            'highpass_order':10, 'axLims': {},
            'axLims_meanshifted':{}, 'axLims_bandPow': MEGsrclimsBandPowBySubj }
            #'axLims':LFPlimsBySubj, 'axLims_meanshifted':LFPlimsBySubj_meanshifted,
            #'axLims_bandPow':LFPlimsBandPowBySubj  }


    plot_paramsPerModality = {}
    plot_paramsPerModality = {'EMG':EMGplotPar, 'LFP':LFPplotPar, 
            'EOG':EOGplotPar, 'MEGsrc':MEGsrcplotPar }

    plot_MEGsrc_rowPerBand = True
    #plot_LFP_rowPerBand = True  # now default


    tremorDetectUseCustomThr = 1
    tremorDetect_customThr = {}
    ## left
    #_off = { 'EMG063_old': 0.65e-10  , 'EMG064_old': 0.4e-10   }
    #_on  = { 'EMG063_old': 0.65e-10  , 'EMG064_old': 0.4e-10   }
    ## right
    #_off.update( { 'EMG061_old': 5e-11  , 'EMG062_old': 4e-11   } )
    #_on.update(  { 'EMG061_old': 5e-11  , 'EMG062_old': 4e-11   } )
    #tremorDetect_customThr['S01'] = {'off':_off, 'on':_on }

    # left
    thrLeft = 200
    _off = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    _on  = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    thrRight = 200
    _off.update( {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    _on.update(  {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    tremorDetect_customThr['S08'] = {'off':_off, 'on':_on }
    tremorDetect_customThr['S09'] = {'off':_off, 'on':_on }

    # left
    thrLeft = 4.5
    _off = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    _on  = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    thrRight = 4.5
    _off.update( {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    _on.update(  {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    tremorDetect_customThr['S01'] = {'off':_off, 'on':_on }

    # left
    thrLeft = 100
    _off = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    _on  = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    # right
    thrRight = 80
    _off.update( {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    _on.update(  {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    tremorDetect_customThr['S02'] = {'off':_off, 'on':_on }

    # left
    thrLeft = 100
    _off = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    _on  = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    # right
    thrRight = 70
    _off.update( {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    _on.update(  {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    tremorDetect_customThr['S03'] = {'off':_off, 'on':_on }

    # left
    thrLeft = 100
    _off = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    _on  = {'left': { 'EMG063_old': thrLeft  , 'EMG064_old': thrLeft   } }
    # right
    thrRight = 200
    _off.update( {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    _on.update(  {'right': { 'EMG061_old': thrRight  , 'EMG062_old': thrRight   } } )
    tremorDetect_customThr['S04'] = {'off':_off, 'on':_on }

    # recall that we have 1s window, sliding by 1s * overlap
    #tremIntDef_convWidth = 5
    #tremIntDef_convWidth = 1.1 # in secs
    tremIntDef_convWidth = 0.5 # in secs
    tremIntDef_convThr   = 0.3
    tremIntDef_incNbins   = 1
    tremIntDef_percentthr=0.1
    tremIntDef_minlen=1
    tremIntDef_extFactorL=1.5
    tremIntDef_extFactorR=0.12


    ################### Set plot params
    show_timecourse = 1
    show_tremconv = 1
    show_spec = 1
    show_bandpow = 1
    extend_plot_time = 1
    ext = 'pdf'
    ext = 'png'
    showfig = 0
    try: 
        savefig
    except NameError:
        savefig = 1
    if not showfig and savefig:
        mpl.use('Agg')
    hh = 3       # individual axis height
    ww = 15      # individual axis width 
    legendloc = 'lower right'
    legalpha = 0.3
    mpl.rcParams.update({'font.size': 14})
    plot_tremConvAxMax = 0.07


    #############  Generate filenames ############
    fnames_noext = []
    for subjind in subjinds:
        for medcond in medconds:
            for task in tasks:
                #fn = 'S{:02d}_{}_{}'.format(subjind,medcond,task)
                fn = getRawname(subjind,medcond,task)
                if fn in fnames_noext:
                    continue
                fnames_noext = fnames_noext + [fn]

    #fnames_noext = ['S01_off_hold', 'S01_off_move', 'S02_on_move', 'S03_off_hold', 'S04_off_hold' ]
    #fnames_noext = ['S01_off_hold', 'S01_on_hold',  'S01_off_move', 'S01_on_move' ]
    print('Filenames to be read ',fnames_noext)

    srcPerRawname = {}
    fnames_src_noext = []
    for raw_fname in fnames_noext:
        subjstr,medcond,task = getParamsFromRawname(raw_fname)
        subjind = int( subjstr[1:] )
    #for subjind in subjinds:
    #    for medcond in medconds:
    #        for task in tasks:
        rawname = getRawname(subjind,medcond,task)
        srcPerRawname[rawname] = []
        for curroi in MEGsrc_roi:
            if curroi.find('_') >= 0:
                raise ValueError("Src roi contains underscore, we'll have poblems with parsing")

            #fn = 'srcd_S{:02d}_{}_{}_{}'.format(subjind,medcond,task,curroi)
            fn = utils.getSrcname(subjind,medcond,task,curroi)
            srcPerRawname[rawname] += [fn]
            if fn in fnames_src_noext:
                continue
            fnames_src_noext = fnames_src_noext + [fn]
    print('Filenames src to be read ',fnames_noext)

    ###########  Read raw data ###################
    sfreqs = []
    for fname_noext in fnames_noext:
        fname = fname_noext + '_noMEG_resample_raw.fif'
        print(fname)
        fname_full = os.path.join(data_dir,fname)
        if not os.path.exists(fname_full):
            print('Warning: path does not exist! {}'.format(fname_full))
            continue

        if fname_noext not in raws:
            f = mne.io.read_raw_fif(fname_full, None)
            raws[fname_noext] = f 

            sfreqs += [ int(f.info['sfreq']) ]

    for fname_noext in fnames_src_noext:
        fname = fname_noext + '.mat'
        print(fname)
        fname_full = os.path.join(data_dir,fname)
        if not os.path.exists(fname_full):
            print('Warning: path does not exist! {}'.format(fname_full))
            continue

        if fname_noext not in srcs:
            f = h5py.File(fname_full,'r')
            #f = h5py.File(fname_full)
            srcs[fname_noext] = f
            #nsrc = f['source_data']['avg']['mom'].shape[1]

            #src = f['source_data'] 
            #ref = src['avg']['mom']
            #srcval = f[ref[0,srci] ][:,0]   # 1D array with reconstructed source activity
    #np.sum( np.unique(nums_src) )

    ##########  check freq consistency
    sampleFreq = np.unique(sfreqs)
    if len(sampleFreq) > 1:
        raise ValueError('Different sample frequencies found',sampleFreq)
    else:
        sampleFreq = sampleFreq[0]
        print('Sample freq is ',sampleFreq)
    gv.gparams['sampleFreq']=sampleFreq

    fr = pywt.scale2frequency(spec_wavelet,spec_cwtscales) * sampleFreq
    print('Wavelet freqs ',fr)
        
    ###########  Determine data type (condition type) ###################
    data_type = ''
    move_found = 0
    hold_found = 0
    rest_found = 0
    for k in raws.keys():
        if k.find('move') >= 0:
            move_found = 1
        elif k.find('hold') >= 0:
            hold_found = 1
        elif k.find('rest') >= 0:
            rest_found = 1
        
    if move_found + hold_found + rest_found > 1:
        data_type = 'mix'
    else:
        if move_found:
            data_type = 'mix'
        elif hold_found:
            data_type = 'hold'
        elif rest_found:
            data_type = 'rest'
        else:
            data_type = 'unk'

    print('Data type overall is {}'.format(data_type) )

    ###########################################

    ks = list( raws.keys() )


    #####################   Obtain some superficial info about loaded raws
    gv.subjs_analyzed = {}  # keys -- subjs,  vals -- arrays of keys in raws
    '''
    subj
        datasets
        medconds
        tasks
        medcond1 [on or off]
            task 1    --> rawname
            ...
            task n_1  --> rawname
        medcond2 [on or off]
            task 1    --> rawname 
            ...
            task n_2  --> rawname 
    '''
    for k in raws:
        f = raws[k]
        sind_str,medcond,task = getParamsFromRawname(k)

        cursubj = {}
        if sind_str in gv.subjs_analyzed:
            cursubj = gv.subjs_analyzed[sind_str]

        dat = cursubj.get('datasets',{} )
        dat[k] = srcPerRawname[k] 
        if  'datasets' in cursubj:
            cursubj['datasets'].update( dat )
        else:
            cursubj['datasets'] = dat 

        if 'medconds' in cursubj: 
            if medcond not in cursubj['medconds']:
                cursubj['medconds'] += [medcond]
        else:
            cursubj['medconds'] = [medcond]

        if 'tasks' in cursubj:
            if task not in cursubj['tasks']:
                cursubj['tasks'] += [task]
        else:
            cursubj['tasks'] = [task]

        if medcond in cursubj:
            m = cursubj[medcond]
            if task in m:
                raise ValueError('Duplicate raw key!')
            else:
                m[task] = k
        else:
            cursubj[medcond] = { task: k}

        gv.subjs_analyzed[sind_str] =  cursubj
    print(gv.subjs_analyzed)


    #####################   Read subject information, provided by Jan
    with open(os.path.join(data_dir,'Info.json') ) as info_json:
        gv.gen_subj_info = json.load(info_json)
            
    ####################    generate some supplementary info, save to a multi-level dictionary 
    for k in raws:
        sind_str,medcond,task = getParamsFromRawname(k)
        subjinfo = gv.gen_subj_info[sind_str]
        if 'chantuples' not in subjinfo: # assume human has fixed set of channels 
            chnames = raws[k].info['ch_names'] 
            rr = genChanTuples(k, chnames) 
            #orderEMG, ind_tuples, name_tuples, LFPnames_perSide, EMGnames_perSide, MEGsrcnames_perSide = rr 
            chantuples = {}
            
            subjinfo['LFPnames_perSide'] = rr['LFPnames'] 
            subjinfo['EMGnames_perSide'] = rr['EMGnames']   

            #gv.subjs_analyzed[sind_str]['MEGsrcnames_perSide_all'] = rr['MEGsrcnames_all'] 
            if len(srcs) > 0:
                subjinfo['MEGsrcnames_perSide_all'] = rr['MEGsrcnames_all'] 
                gv.subjs_analyzed[sind_str]['MEGsrcnames_perSide'] = rr['MEGsrcnames'] 
                #subjinfo['MEGsrcnames_perSide'] = rr['MEGsrcnames'] 
            for side in rr['order'] :
                yy = {}
                yy['indtuples'] = rr['indtuples'][side]
                yy['nametuples'] = rr['nametuples'][side]     
                chantuples[ side ] = yy

            subjinfo['chantuples' ] = chantuples
    #['S01']['chantuples']['left']['nametuples']['EMG']

    #import pdb; pdb.set_trace()

    if MEGsrc_names_toshow is None:
        n = 0
        MEGsrc_names_toshow = []
        k0 = list(gv.subjs_analyzed.keys() )[0]
        MEGsrcnames = gv.subjs_analyzed[k0 ]  ['MEGsrcnames_perSide']
        for side in MEGsrcnames:
            n+= len( MEGsrcnames[side] )
            MEGsrc_names_toshow += MEGsrcnames[side]
        #MEGsrc_inds_toshow = np.arange(n)  # differnt roi's will have repeating indices? No because the way I handle it in getChantuples!
    #print( 'MEGsrc_inds_toshow = ', MEGsrc_inds_toshow )
    print( 'MEGsrc_names_toshow = ', MEGsrc_names_toshow )


    for subj in gv.gen_subj_info:
        if subj in tremorDetect_customThr:
            gv.gen_subj_info[subj]['tremorDetect_customThr'] = tremorDetect_customThr[subj]
        if subj in favoriteLFPch_perSubj:
            gv.gen_subj_info[subj]['favoriteLFPch'] = favoriteLFPch_perSubj[subj]

        #bc = gv.gen_subj_info[subj]['bad_channels']
        #for medcond in bc:
        #    bc2 = bc[medcond] 
        #    for task in bc2:
        #        #for tremor 

    # if we want to save the updated information back to the file
    if updateJSON:
        with open(os.path.join(data_dir,'Info.json'), 'w' ) as info_json:
            json.dump(gv.gen_subj_info, info_json)

    # compute spectrograms for all channels (can be time consuming, so do it only if we don't find them in the memory)
    #loadSpecgrams = False
    singleSpecgramFile = False
    loadSpecgrams = True  # if exists
    #loadSpecgrams = False

    saveSpecgrams = True
    #saveSpecgrams_skipExist = False
    saveSpecgrams_skipExist = True

    stats_fname = os.path.join( data_dir, 'last_data.npz')
    save_stats = 1
    save_stats_onlyIfNotExists = 1
    load_stats = 0

    try: 
        gv.specgrams.keys()
        if len(gv.specgrams) == 0:
            raise NameError('empty specgrams')
    except (NameError, AttributeError):
        if singleSpecgramFile:
            specgramFname = 'nraws{}_nsrcs{}_specgrams.npz'.format( len(gv.raws), len(srcs) )
            #specgramFname = 'specgrams_1,2,3.npz'
            specgramFname = os.path.join(data_dir, specgramFname)
            print('Loading specgrams from ',specgramFname)
            if loadSpecgrams and os.path.exists(specgramFname) :
                gv.specgrams = np.load(specgramFname, allow_pickle=True)['arr_0'][()]
            else:
                gv.specgrams = precomputeSpecgrams(raws,NFFT=NFFT,specgramoverlap=spec_FToverlap)
                if saveSpecgrams:
                    if not (saveSpecgrams_skipExist and os.path.exists(specgramFname) ):
                        np.savez(specgramFname, gv.specgrams)
        else:
            gv.specgrams = {}
            for k in raws:
                sind_str,medcond,task = getParamsFromRawname(k)
                nsources = len( gv.subjs_analyzed[sind_str] ['MEGsrcnames_perSide'] ) * 2
                nroi = len(MEGsrc_roi)
                specgramFname = '{}_nroi{}_nMEGsrcInds{}_spec_{}.npz'.format(k, nroi, 
                        len(MEGsrc_names_toshow  ), spec_specgramtype  )
                #specgramFname = 'specgrams_1,2,3.npz'
                specgramFname = os.path.join(data_dir, specgramFname)
                if loadSpecgrams and os.path.exists(specgramFname) :
                    print('Loading specgrams from ',specgramFname)
                    specgramscur = np.load(specgramFname, allow_pickle=True)['arr_0'][()]
                    gv.specgrams.update( specgramscur   )
                else:
                    tmp = { k: raws[k] }
                    specur  = precomputeSpecgrams(tmp,NFFT=NFFT,specgramoverlap=spec_FToverlap)
                    gv.specgrams.update( specur   )
                    if saveSpecgrams:
                        if not (saveSpecgrams_skipExist and os.path.exists(specgramFname) ):
                            print('Saving specgrams to ',specgramFname)
                            np.savez(specgramFname, specur)
    else:
        print('----- Using previously precomuted spectrograms in memory')


    ############################################################
    #time_start, time_end = 0,1000
    time_start_forstats, time_end_forstats = 0,300
    #freq_min_forstats, freq_max_forstats = 0, NFFT//2   #NFFT//2

    # check what Jan considered as tremor frequency 
    tfreqs = []
    for subj in gv.subjs_analyzed:
        tfreq = gv.gen_subj_info[subj]['tremfreq']
        tside = gv.gen_subj_info[subj]['tremor_side']
        print('{} has tremor at {} side with freq {}'.format(subj,tside,tfreq) )
        tfreqs += [ tfreq]

        if subj not in favoriteLFPch_perSubj and gv.gparams['plot_LFP_onlyFavoriteChannels']:
            raise ValueError('Want to plot only favorite LFP they are not set for all subjects!' )

    # define tremor band with some additional margin
    safety_freq_shift = 0.5
    tremorBandStart = min(tfreqs) - safety_freq_shift #3.8 
    tremorBandEnd   = max(tfreqs) + safety_freq_shift #6.8
    if  tremorBandEnd - tremorBandStart < 1.5:
        tremorBandStart -= 0.8
        tremorBandEnd += 0.8
    print('Tremor freq band to be used: from {} to {}'.format(tremorBandStart,tremorBandEnd) )

    notchFreqThr = 5 # in Hz
    # define other bands that will be used
    betaBand = 13,30
    lowGammaBand = 30.1, 50 - notchFreqThr
    highammaBand = 50 + notchFreqThr,100
    motorGammaBand = 30.1,100
    #motorGammaBand = 60,90
    motorGammaBand = 60,90
    slowBand = 0,3

    gv.freqBands = {'tremor':(tremorBandStart,tremorBandEnd), 'beta':betaBand,
            'lowgamma':lowGammaBand, 'highgamma':lowGammaBand,
            'gamma_motor':motorGammaBand, 'slow':slowBand }
    #plot_freqBandNames_perModality = {'EMG': ['tremor' ], 'LFP': ['tremor', 'beta','gamma_motor'], 
    #        'MEGsrc':['tremor','beta', 'gamma_motor' ], 'EOG':['lowgamma'] }
    plot_freqBandNames_perModality = {'EMG': ['tremor' ,'slow' ], 'LFP': ['tremor', 'beta', 'gamma_motor'], 
            'MEGsrc':['tremor','beta', 'gamma_motor' ], 'EOG':['lowgamma'] }
    #plot_freqBandsLineStyle = {'tremor':'-', 'beta':'--', 'lowgamma':':', 'gamma_motor':':', 'slow':'-'  }
    plot_freqBandsLineStyle = {'tremor':'-', 'beta':'-', 'lowgamma':'-', 'gamma_motor':'-', 'slow':'-'  }

    ############## plotting params 
    plot_colorsEMG = [ 'black', 'green', 'black', 'green'  ] 
    ltype_Clust = '--'
    ltype_tremorThr = '-.'
    #plot_colorsEMGClust = [ 'blue', 'red' ] 

    plot_specgramCmapPerModality = {'LFP': 'inferno', 'MEGsrc':'viridis' }
    plot_MEGsrc_separateRoi = False
    plot_colPerInterval = True
    plot_setBandlims_naively = True
    #plot_setBandlims_naively = False

    #cmap = 'hot'
    #cmap = 'gist_rainbow'
    #cmap = 'viridis'

    ############# # update in place gen_
    for subj in gv.subjs_analyzed:
        for modality in ['MEGsrc', 'LFP' ]:
            for fbname in ['tremor']:
                utils.sortChans(subj, modality, fbname, replace=True, numKeep = plot_numBestMEGsrc)

    ##############   compute some statisitics, like max, min, mean, cluster in subbands, etc -- can be time consuming


    try:
        glob_stats.keys()
    except (NameError, AttributeError):
        if load_stats:
            f = np.load(stats_fname, allow_pickle=1)
            glob_stats = f['glob_stats'][()]
            glob_stats_perint = f['glob_stats_perint'][()]
        else:
            glob_stats = getStatPerChan(time_start_forstats,
                    time_end_forstats)  # subj,channame, keys [! NO rawname!]
    else:
        print('----- Using previously computed stats!')

    ##############   find tremor intercals in all subjects and raws 
    reuseExistingTremorInterval = False
    recalcTrem = 0
    try: 
        tremIntervalPerRaw.keys()
    except (NameError, AttributeError):
        recalcTrem = 1
    else:
        if not reuseExistingTremorInterval:
            recalcTrem = 1
        else:
            print('----- Using previously tremor intervals!')

    if recalcTrem:
        print('Recomputing tremor intervals')
        tremIntervalPerRaw,tremCvlPerRaw = utils.findAllTremor(width=tremIntDef_convWidth, inc=tremIntDef_incNbins, 
                thr=tremIntDef_convThr, minlen=tremIntDef_minlen, 
                extFactorL=tremIntDef_extFactorL,
                extFactorR=tremIntDef_extFactorR,
                percentthr=tremIntDef_percentthr)


    # merge intervals within side
    mergeMode = 'union' 
    #mergeMode = 'intersect' 
    tremIntervalMergedPerRaw = utils.mergeTremorIntervals(tremIntervalPerRaw, mode=mergeMode)
 
    # unpack tremor intervals sent by Jan
    tremIntervalJan = {}
    for subjstr in trem_times_Jan:
        s = trem_times_Jan[subjstr]
        for medcond in s:
            ss = s[medcond]
            for task in ss:
                sss = ss[task]
                rawname = getRawname(subjstr,medcond,task )
                tremdat = {}
                for kt in sss:
                    s4 = sss[kt]
                    #s4p = copy.deepcopy(s4) 
                    s4p = s4
                    # or list of lists 
                    # or list with one el which is list of lists
                    cond1 = isinstance(s4,list) and len(s4) > 0
                    condshared = isinstance(s4[0],list) and len(s4[0]) > 0 and isinstance(s4[0][0],list)
                    cond2 = len(s4) == 1 and condshared
                    # sometimes we have two lists on the first level but the second is empty
                    cond3 = len(s4) == 2 and len(s4[1]) == 0 and condshared
                    # sometimes it is not empty but the second one has indices very large
                    cond4 = len(s4) == 2 and len(s4[1]) > 0 and condshared
                    if cond1 and (cond2 or cond3 or cond4):
                        s4p = s4[0]
                        if len(s4) > 1:
                            s4p += s4[1]
                            
                    if len( s4p[-1] ) == 0:
                        del s4p[-1]
    #                     if cond4:
    #                         s4p = s4[0] + s4[1]
    #                         #del s4p[1]
    #                     else:
    #                         s4p = s4[0]
    #                     if cond3:
    #                         del s4p[1]

                    if len(s4p) == 1 and len(s4p[0]) == 0: 
                        del s4p[0]

                    tremdat[kt] = s4p  # array of 2el lists
                    
                tremIntervalJan[rawname] = { 'left':[], 'right':[] }
                maintremside = gv.gen_subj_info[subjstr]['tremor_side']
                tremIntervalJan[rawname][maintremside] =  tremdat 
                    #print(subjstr,medcond,task,kt,len(s4), s4)
                    #print(subjstr,medcond,task,kt,len(s4p), s4p)



    #timeIntervalsFromTremor = True
    timeIntervalsFromTremor = False

    plotTremNegOffset = 2.
    plotTremPosOffset = 2.
    maxPlotLen = 6
    longToLeft = True
    if timeIntervalsFromTremor:
        mvtTypes = ['tremor', 'no_tremor']
        timeIntervalPerRaw_processed = utils.processJanIntervals( tremIntervalJan, 
                maxPlotLen, plotTremNegOffset, plotTremPosOffset, plot_time_end, mvtTypes=mvtTypes)
        # [rawname][interval index]
    else:
        timeIntervalPerRaw_processed = {}
        for k in raws:
            timeIntervalPerRaw_processed[k] = [ (plot_time_start, plot_time_end, 'entire') ]

    try:
        glob_stats_perint
    except (NameError, AttributeError):
        glob_stats_perint = getStatsFromTremIntervals(timeIntervalPerRaw_processed)
    else:
        print('----- Using previously computed interval stats!')
    # [raw] -- list of 3-tuples

    # I want to make a copy where pre and post are not present (for plotting), but gather stats for all intervals
    plot_timeIntervalPerRaw = {}
    if timeIntervalsFromTremor:
        itypesExcludeFromPlotting =  ['pre', 'post', 'middle_full', 'entire']
        for k in timeIntervalPerRaw_processed:
            intervals = timeIntervalPerRaw_processed[k]
            res = []
            for interval in intervals:
                t1,t2,intType = interval
                if intType in itypesExcludeFromPlotting:
                    continue
                res += [interval]
            plot_timeIntervalPerRaw[k] = res
    else:
        for k in raws:
            plot_timeIntervalPerRaw[k] = [ (plot_time_start, plot_time_end, 'entire') ]

    if save_stats:
        if not save_stats_onlyIfNotExists or  (not os.path.exists(stats_fname) ):
            np.savez(stats_fname , glob_stats_perint=glob_stats_perint, 
                    glob_stats=glob_stats, timeIntervalPerRaw_processed=timeIntervalPerRaw_processed)

    # [rawname][interval type][chname]

    # Set matplotlib params
    #inferno1 = mpl.cm.get_cmap('viridis', 256)
    #inferno2 = mpl.colors.ListedColormap(inferno1(np.linspace(0, 1, 128)))


    #chanTypes_toshow = {'timecourse': ['EMG', 'LFP'], 'spectrogram': ['LFP'], 'bandpow': ['EMG', 'LFP'] }
    chanTypes_toshow = {} 
    if show_timecourse:
        #chanTypes_toshow[ 'timecourse'] = {'chantypes':['EMG', 'LFP', 'MEGsrc', 'EOG'] }
        chanTypes_toshow[ 'timecourse'] = {'chantypes':['EMG'] }
    if show_bandpow:
        chanTypes_toshow[ 'bandpow']    = {'chantypes': ['EMG', 'LFP', 'MEGsrc' ] }

    if show_tremconv:
        chanTypes_toshow[ 'tremcvl'] = {'chantypes':['EMG'] }
    else:
        chanTypes_toshow[ 'tremcvl'] = {}

    if show_spec:
        chanTypes_toshow[ 'spectrogram'] = {'chantypes': ['LFP']    }
        if plot_EMG_spectrogram:
            chanTypes_toshow[ 'spectrogram']['chantypes'] += ['EMG']   
        if plot_MEGsrc_spectrogram:
            chanTypes_toshow[ 'spectrogram']['chantypes'] += ['MEGsrc']   
    #if show_EMG_corr:
    #    chanTypes_toshow[ 'EMGcorr'] = {'chantypes':['EMG', 'LFP', 'MEGsrc', 'EOG'] }
    if show_EMG_band_corr:
        chanTypes_toshow[ 'EMGband_corr'] = {'chantypes':['EMG', 'LFP', 'MEGsrc'] }
    else:
        chanTypes_toshow[ 'EMGband_corr'] = {'chantypes':[] }

    chanTypes_toshow['stats_barplot'] = {'chantypes':['LFP','MEGsrc'] }

    show_timecourse = 1
    show_tremconv = 1
    show_spec = 1

    ######################################################
    ######################################################
    ######################################################

    ppp = 1
    try:
        ppp = not skipPlot
    except NameError as e:
        print('Skipping plot',str(e))
    #plotSpectralData(plt,time_start=100,time_end=150, chanTypes_toshow=['EMG' ,'LFP'] ) 
    except Exception as e:
        raise
            
    if ppp:
        print('Starting to plot!')
        #plotSpectralData(plt,time_start=0,time_end=10 ) 
        #plotSpectralData(plt,time_start=180,time_end=220 ) 
        plotSpectralData(plt,time_start=plot_time_start,time_end=plot_time_end, 
                chanTypes_toshow = chanTypes_toshow ) 
