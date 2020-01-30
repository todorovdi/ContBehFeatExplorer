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

########  Gen stats across conditions
def getStatPerChan(time_start,time_end, glob_stats = None, singleRaw = None, mergeTasks = False, modalities=None, datPrep = None, specgramPrep = None):
    '''
    returns task independent
    stats = glob_stats[sind_str][medcond][chname]
    '''
    if modalities is None:
        modalities = ['LFP', 'EMG', 'EOG', 'MEGsrc']


    if singleRaw is not None:
        sind_str,medcond, task = getParamsFromRawname(singleRaw)
        subjs = [sind_str]
        medconds = [medcond]
        tasks = [task]
    else:
        print('Starting computing glob stats for modalities {}, singleRaw {}, ({},{})'.
                format(modalities, singleRaw,time_start,time_end) )
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

                if specgramPrep is None:
                    sp = gv.specgrams[k]  # spectrogram from a given raw file
                else:
                    sp = specgramPrep

                #raw = raws[k]
                #chnames = list( sp.keys() )
                #chnames = raw.info['ch_names'] 

                cht = gv.gen_subj_info[ subj ]['chantuples']

                #orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)
                #chnames2 = []
                for side in cht:
                    #chnames2 = chnames_tuples[side_ind]['LFP'] + chnames_tuples[side_ind]['EMG']

                    if datPrep is not None:
                        #if glob_stats is not None:
                        #    import ipdb; ipdb.set_trace()

                        if singleRaw not in datPrep:
                            raise ValueError("wrong datPrep")
                        if side not in datPrep[singleRaw]:
                            continue
                        chnames2, chdata = datPrep[singleRaw][side]
                    else:
                        chnames2 = []
                        for modality in modalities:
                            chnames2 += cht[side]['nametuples'][modality]
                        #chnamesEMG = chnames_tuples[side_ind]['EMG']

                        #chis = mne.pick_channels(chnames,include=chnames2, ordered=True )
                        #ts,te = raw.time_as_index([time_start, time_end])
                        #chdata, chtimes = raw[chis,ts:te]

                        chdata, chtimes = utils.getData(k, chnames2 )

                    specsTremorEMGcurSide = []

                    # to avoid problems of scaleogram boundary values

                    if singleRaw is None:
                        n_splits = 25
                    else:
                        n_splits = 15

                    
                    #args_outer += [ (k, f,b,spdata, chdat, chn, stat_leaflevel, time_start,time_end,
                    #        len(raws_from_subj), n_splits, gsarg ) ]

                    args = []
                    for chii,chn in enumerate(chnames2):
                        f,b,spdata = sp[chn]     # entire spectrum, for all times
                        chdat = chdata[chii,:]

                        time_start = max(time_start, b[0] + spec_thrBadScaleo) 
                        time_end = min(time_end, b[-1] - spec_thrBadScaleo)


                        bb,be = gv.freqBands['tremor']
                        ft,bt,spdatat = utils.getSubspec(f,b,spdata,bb, be, 
                                time_start, time_end)
                        if chn.find('EMG') >= 0: 
                            specsTremorEMGcurSide += [ (chn, ft,bt, spdatat)  ]


                        gsarg = None
                        if glob_stats is not None:
                            gsarg = glob_stats[chn]

                        args += [ (k, f,b,spdata, chdat, chn, stat_leaflevel, time_start,time_end,
                                len(raws_from_subj), n_splits, gsarg ) ]
                        #st = stat_ch(k, f,b,spdata, chdat, chn, stat_leaflevel, time_start,time_end,
                        #        len(raws_from_subj), n_splits, gsarg )

                    if glob_stats is None and len(raws) == 1 and not forceOneCore_globStats:
                        ncores = mpr.cpu_count()
                        p = mpr.Pool( min(len(args), ncores)  )
                        pr = p.map(stat_ch_proxy, args)
                        p.close()
                        p.join()

                        for k,chn,st in pr:
                            stat_leaflevel[chn] = st
                    else:
                        for arg in args:
                            k,chn,st = stat_ch_proxy(arg)
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

    if singleRaw is None:
        print('Glob stats computation finished')
    return res


def stat_ch_proxy( arg ):

    rawname, f,b,spdata, chdat, chn, stat_leaflevel, time_start,time_end, nraws_from_subj , n_splits, glob_stats  = arg

    st = stat_ch(rawname, f,b,spdata, chdat, chn, stat_leaflevel, 
        time_start,time_end, nraws_from_subj , n_splits,
        glob_stats)

    return k,chn,st


def stat_ch(rawname, f,b,spdata, chdat, chn, stat_leaflevel, 
        time_start,time_end, nraws_from_subj , n_splits,
        glob_stats):
    if isinstance( spdata[0,0] , complex):
        spdata = np.abs(spdata)

    nout_thr_spec = 1e-2  # several dozen bins for 256 Hz discr freq and 300 sec 
    nout_thr_bandpow = 1e-3  # several dozen bins for 256 Hz discr freq and 300 sec 

    bandpows = {}
    fbnames = ['slow', 'tremor','beta', 'gamma_motor']
    for fbname in fbnames:
        r = utils.getBandpow(k,chn,fbname, time_start, time_end, spdat=(f,b,spdata) ) 
        if r is not None:
            bins_b, bandpower = r
            if bandpower.size == 0:
                continue
            assert not isinstance( bandpower[0], complex) 
            bandpows[fbname] = bins_b, bandpower 

    # 3-48, 60-80
    st = {}
    if chn.find('EMG') >= 0 and useEMG_cluster: 
        bb,be = gv.freqBands['tremor']
        ft,bt,spdatat = utils.getSubspec(f,b,spdata,bb, be, 
                time_start, time_end)
        assert spdatat.shape[0] > 0

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

    st['max']  = -1e8
    st['min']  = 1e8
    st['mean'] = 0

    def mifNeeded(optype, key, val):
        if key is st:
            if optype == 'max':
                st[key] = max( st[key], val )
            elif optype == 'min':
                st[key] = min( st[key], val )
            elif optype == 'sum':
                st[key] += val 
        else:
            st[key] = val

    mifNeeded('max', 'max_spec', np.max(spdata, axis=1))
    mifNeeded('min', 'min_spec', np.min(spdata, axis=1))
    mifNeeded('max', 'max', np.max(chdat))
    mifNeeded('min', 'min', np.min(chdat))

    me = np.mean(spdata, axis=1) / nraws_from_subj 
    mifNeeded('sum', 'mean_spec',  me )
    mifNeeded('sum', 'mean',  np.mean(chdat) / nraws_from_subj         )


    validbins_bool = utils.getBinsInside(b, max(time_start,spec_thrBadScaleo), 
            min(nonTaskTimeEnd-spec_thrBadScaleo, time_end), retBool = True) 
    if np.sum(validbins_bool ) == 0:
        print ( 'WARNING stat_ch: No valid bins! {}:{} {}-{}'.format(rawname,chn, time_start,time_end ))
        return { 'error':'no_valid_bins' }

    validbins_bool2 = utils.filterArtifacts(k,chn,b)
    validbins_bool = np.logical_and( validbins_bool, validbins_bool2)

    if np.sum(validbins_bool ) == 0:
        print ( 'WARNING stat_ch: No valid bins! {}:{} {}-{}'.format(rawname,chn, time_start,time_end ))
        return { 'error':'no_valid_bins' }

    mn_nout, mx_nout, me_nout = utils.calcNoutMMM_specgram(spdata[:,validbins_bool], thr=nout_thr_spec )
    mifNeeded('sum', 'mean_spec_nout_full',  me_nout  / nraws_from_subj           )

    goodfreqs = f > spec_minfreq
    goodfreqs = np.logical_and( goodfreqs , 
            np.logical_or(f < spec_DCfreq - spec_DCoffs, f > spec_DCfreq + spec_DCoffs) )

    #goodinds_bool  = utils.filterRareOutliers_specgram(spdata,retBool = True, thr=4e-3)
    #goodinds_bool = np.logical_and(validbins_bool, goodinds_bool)
    #spdata_good = spdata[:,goodinds_bool]
    #me_nout = np.mean(spdata_good, axis=1)
    def gett(gf, vi):
        spdata_valid = spdata[gf,:] [:, vi]
        mn_nout, mx_nout, me_nout = utils.calcNoutMMM_specgram(spdata_valid, thr=nout_thr_spec )
        spdata_mc =  ( spdata_valid -  me_nout[:,None] ) / me_nout[:,None]    # divide by mean
        return mn_nout, mx_nout, me_nout, spdata_mc

    mn_nout, mx_nout, me_nout, spdata_mc  = gett(goodfreqs, validbins_bool)

    #spdata_valid = spdata[goodfreqs,:] [:, validbins_bool]
    #mn_nout, mx_nout, me_nout = utils.calcNoutMMM_specgram(spdata_valid, thr=nout_thr_spec )
    mifNeeded('sum', 'mean_spec_nout',  me_nout  / nraws_from_subj           )
    mifNeeded('max', 'min_spec_nout',  mx_nout         )
    mifNeeded('min', 'max_spec_nout',  mn_nout         )
        
    #spdata_mc =  ( spdata_valid -  me_nout[:,None] ) / me_nout[:,None]    # divide by mean
    #mifNeeded('max', 'max_spec_mc', np.max( spdata_mc) )
    #mifNeeded('min', 'min_spec_mc', np.min( spdata_mc) )

    #sp_notslow = spdata_mc[goodfreqs,:]
    #maxfreq = plot_maxFreqInSpec
    #freqinds = np.where( np.logical_and(f >= minfreq,f <= plot_maxFreqInSpec) )[0]

    #mifNeeded('max', 'max_spec_mc', np.max( sp_notslow) )
    #mifNeeded('min', 'min_spec_mc', np.min( sp_notslow) )

    #q = utils.getDataClusters_fromDistr(sp_notslow, n_splits=n_splits, 
    #        clusterType='outmostPeaks', lbl = '{} spec'.format(chn) )

    #m1,m2 = utils.getSpecEffMax(sp_notslow, thr=nout_thr) 
    mxn = ( mx_nout - me_nout) / me_nout
    mnn = ( mn_nout - me_nout) / me_nout
    mifNeeded('max', 'max_spec_mc_nout', mxn )
    mifNeeded('min', 'min_spec_mc_nout', mnn )

    mnn, mxn, men = utils.calcNoutMMM_specgram(spdata_mc, thr=nout_thr_spec )
    mifNeeded('max', 'max2_spec_mc_nout', mxn )
    mifNeeded('min', 'min2_spec_mc_nout', mnn )

    m1,m2 = utils.getSpecEffMax(spdata_mc, thr=nout_thr_spec) 
    mifNeeded('min', 'min3_spec_mc_nout', m1 )
    mifNeeded('max', 'max3_spec_mc_nout', m2 )

    #goodfreqs30 = np.logical_and(goodfreqs, f < 30) 
    #goodfreqs100 = np.logical_and(goodfreqs, f < 100)

    #mn_nout30, mx_nout30, me_nout30, spdata_mc30  = gett(goodfreqs30, validbins_bool)
    #mn_nout100, mx_nout100, me_nout100, spdata_mc100  = gett(goodfreqs100, validbins_bool)

    #mifNeeded('sum', 'mean_spec_nout30',  me_nout30  / nraws_from_subj           )
    #mifNeeded('max', 'min_spec_nout30',  mx_nout30         )
    #mifNeeded('min', 'max_spec_nout30',  mn_nout30         )
    #mxn = ( mx_nout30 - me_nout30) / me_nout30
    #mnn = ( mn_nout30 - me_nout30) / me_nout30
    #mifNeeded('max', 'max_spec_mc_nout30', mxn )
    #mifNeeded('min', 'min_spec_mc_nout30', mnn )

    #mnn, mxn, men = utils.calcNoutMMM_specgram(spdata_mc30, thr=nout_thr_spec )
    #mifNeeded('max', 'max2_spec_mc_nout30', mxn )
    #mifNeeded('min', 'min2_spec_mc_nout30', mnn )

    #m1,m2 = utils.getSpecEffMax(spdata_mc30, thr=nout_thr_spec) 
    #mifNeeded('min', 'min3_spec_mc_nout30', m1 )
    #mifNeeded('max', 'max3_spec_mc_nout30', m2 )

    #mifNeeded('sum', 'mean_spec_nout100',  me_nout100  / nraws_from_subj           )
    #mifNeeded('max', 'min_spec_nout100',  mx_nout100         )
    #mifNeeded('min', 'max_spec_nout100',  mn_nout100         )
    #mxn = ( mx_nout100 - me_nout100) / me_nout100
    #mnn = ( mn_nout100 - me_nout100) / me_nout100
    #mifNeeded('max', 'max_spec_mc_nout100', mxn )
    #mifNeeded('min', 'min_spec_mc_nout100', mnn )

    #mnn, mxn, men = utils.calcNoutMMM_specgram(spdata_mc100, thr=nout_thr_spec )
    #mifNeeded('max', 'max2_spec_mc_nout100', mxn )
    #mifNeeded('min', 'min2_spec_mc_nout100', mnn )

    #m1,m2 = utils.getSpecEffMax(spdata_mc100, thr=nout_thr_spec) 
    #mifNeeded('min', 'min3_spec_mc_nout100', m1 )
    #mifNeeded('max', 'max3_spec_mc_nout100', m2 )

    def di(k,add=''):
        if chn!= 'LFPL12':
            return
        t = st[k]  
        if isinstance(t,np.ndarray) and len(t) > 1:
            t = max(t)
        print('{:12}: {:20}={:20}'.format(add, k,t )  )

    printLog = 0
    if printLog:
        di('mean_spec_nout',    'good freq only' )
        di('max_spec_nout',     'good freq only')
        di('max_spec_mc_nout',  'good freq only, arithm on orig max')
        di('max2_spec_mc_nout', 'good freq only, calcNoutMMM on mc')
        di('max3_spec_mc_nout', 'good freq only, EffMax on mc')

        di('mean_spec_nout30',    'good freq only' )
        di('max_spec_nout30',     'good freq only')
        di('max_spec_mc_nout30',  'good freq only, arithm on orig max')
        di('max2_spec_mc_nout30', 'good freq only, calcNoutMMM on mc')
        di('max3_spec_mc_nout30', 'good freq only, EffMax on mc')

        di('mean_spec_nout100',    'good freq only' )
        di('max_spec_nout100',     'good freq only')
        di('max_spec_mc_nout100',  'good freq only, arithm on orig max')
        di('max2_spec_mc_nout100', 'good freq only, calcNoutMMM on mc')
        di('max3_spec_mc_nout100', 'good freq only, EffMax on mc')



    for fbname in bandpows:
        s0 = '{}_bandpow_'.format(fbname)
        s1max = '{}{}'.    format(s0, 'max')
        s1min = '{}{}'.    format(s0, 'min')
        s1max_nout = '{}{}'.    format(s0, 'max_nout') # no outliers
        s1min_nout = '{}{}'.    format(s0, 'min_nout') # no outliers 
        s1mc_max_nout = '{}{}'.    format(s0, 'mc_max_nout') # no outliers
        s1mc_min_nout = '{}{}'.    format(s0, 'mc_min_nout') # no outliers 
        s1mean_nout = '{}{}'.    format(s0, 'mean_nout')
        s1max_pct = '{}{}'.    format(s0, 'max_pct')
        s1max_distr = '{}{}'.    format(s0, 'max_distr')
        s1min_distr = '{}{}'.    format(s0, 'min_distr')
        s1max_distr_pct = '{}{}'.    format(s0, 'max_distr_pct')
        s1max_nout_pct = '{}{}'.    format(s0, 'max_nout_pct')
        s1normL05 = '{}{}'.format(s0, 'L05')
        s1normL1 = '{}{}'. format(s0, 'L1' )
        s1normL2 = '{}{}'. format(s0, 'L2' )
        s1normMeanDiv_L05 = '{}{}'.format(s0, 'meanDiv_L05')
        s1normMeanDiv_L1 = '{}{}'. format(s0, 'meanDiv_L1' )
        s1normMeanDiv_L2 = '{}{}'. format(s0, 'meanDiv_L2' )

        bins_cbp, absval = bandpows[fbname]          # alredy only valid inds
        #s1mean = '{}_{}'.format('mean',s0)
        if s1max in st:
            st[s1max  ] = max( st[s1max], np.max(bpc ))
            st[s1min] = min( st[s1min], np.min(bpc ))
        else:
            st[s1max] = np.max(absval)
            st[s1min] = np.min(absval)
        

        if stat_computeDistrMax:
            q = utils.getDataClusters_fromDistr(absval, n_splits=n_splits, 
                    clusterType='outmostPeaks', lbl = '{} {}'.format(chn,fbname))
            if s1max_distr not in st:
                st[s1max_distr] = q[1]
                st[s1min_distr] = q[0]
            else:
                st[s1max_distr  ] = max( st[s1max_distr], q[1] )
                st[s1min_distr] = min( st[s1min_distr], q[0] )

        mn,mx = utils.getSpecEffMax( absval , thr=nout_thr_bandpow)
        #binBool0 = absval <= mx 
        #binBool = np.logical_and( binBool0 , validbins_bool)
        binBool = absval <= mx 

        bininds = np.where( binBool )[0]
        me = np.mean( absval[bininds] )
        st[s1mean_nout] = me
        st[s1min_nout] = mn
        st[s1max_nout] = mx

        #absval_mc =  ( absval -  me ) / me    # divide by mean
        mn_mc = (mn - me) / me
        mx_mc = (mx - me) / me
        st[s1mc_min_nout] = mn_mc
        st[s1mc_max_nout] = mx_mc

        # we don't want to check binBool, since artifacts can be long
        duration = ( bins_b[-1] - bins_b[0] ) * ( np.sum(binBool) / len(bins_b)  ) # take into account thrown away bins
        if glob_stats is None:
            assert duration > (time_end - time_start) * 0.9


        if glob_stats is not None:
            mx = glob_stats[s1max_nout]
            mn = glob_stats[s1min_nout]
            bpc = (absval[bininds] - mn)   /  (mx - mn) # this is NOT mean correction, because min and max are used -- this is to avoid negativity and make sqrt computation work
            #duration = (time_end - time_start) 
            st[s1normL05] = np.sum( np.sqrt(bpc)  ) / duration 
            st[s1normL1] = np.sum( bpc   ) / duration
            st[s1normL2] = np.sum( np.power(bpc,2)  ) / duration

            bpc2 = (absval[bininds] - mn)   /  np.abs(me) # this is NOT mean correction, because min and max are used -- this is to avoid negativity and make sqrt computation work
            st[s1normMeanDiv_L05] = np.sum( np.sqrt(bpc2)  ) / duration 
            st[s1normMeanDiv_L1] = np.sum( bpc2   ) / duration
            st[s1normMeanDiv_L2] = np.sum( np.power(bpc2,2)  ) / duration


            if stat_computeDistrMax:
                st[s1max_distr_pct  ] = st[s1max_distr] / glob_stats[s1max_distr] 
            st[s1max_pct  ]      = st[s1max] / glob_stats[s1max]
            st[s1max_nout_pct  ] = st[s1max_nout] / glob_stats[s1max_nout]

    return st

def stat_proxy( arg ):

    t1,t2,rawname, gs,intind, itype, datPrep, intside, specgramPrep = arg

    print('{}: starting {} No {} stats computation'.format(rawname, itype, intind) )
    st = getStatPerChan(t1,t2,singleRaw = rawname, mergeTasks=False, glob_stats=gs, datPrep = datPrep,
            specgramPrep = specgramPrep)
    return  rawname, intside, intind, st
 
def getStatsFromTremIntervals(intervalDict):
    '''
    time_start, time_end, intervalType = intervals [rawname][interval index]
    intervalType from  [ incPre,incPost,incBoth,middle, no_tremor ]
    '''
    #intervalTypes =   [ 'pre', 'post', 'initseg', 'endseg', 'incBoth',
    #         'middle', 'middle_full', 'no_tremor', 'unk_activity_full' ]  # ,'pre', 'post' -- some are needed for barplot, some for naive vis
    intervalTypes = gv.gparams['intTypes']

    # 'incPre', 'incPost',  'entire'
    #intervalTypes =   [ 'pre', 'post', 'middle_full', 'no_tremor' ]  # ,'pre', 'post'
    statsMainSide = {}
    statsOtherSide = {}

    args = []
    for rawname in intervalDict:
        if rawname not in raws:
            continue

        sind_str,medcond, task = getParamsFromRawname(rawname)
        for intside in ['left', 'right']:
            intervals = intervalDict[rawname][intside]
            statlist = [0]*len(intervals)
            for intind,interval in enumerate(intervals):
                t1,t2,itype = interval
                if itype not in intervalTypes:
                    #print('{} not in intervalTypes!'.format(itype) )
                    continue

                isect =  utils.getIntervalIntersection(t1,t2,time_start_forstats,time_end_forstats)
                if len(isect) == 0 or isect[1] - isect[0] < 1:
                    continue

                t1eff,t2eff = isect

                # extract data to put it to subprocess
                modalities = ['LFP', 'EMG', 'EOG', 'MEGsrc']
                dtcr = {}
                cht = gv.gen_subj_info[ sind_str ]['chantuples']

                #orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)
                #chnames2 = []
                #for side in cht:
                for side in [intside]:
                    chnames2 = []
                    for modality in modalities:
                        chnames2 += cht[side]['nametuples'][modality]
                    chdata, chtimes = utils.getData(rawname, chnames2 )

                    dtcr[side] = chnames2, chdata 

                datPrep = {}
                datPrep[rawname] = dtcr
                #import ipdb; ipdb.set_trace()

                specgramPrep = gv.specgrams[rawname]

                gs = gv.glob_stats[sind_str][medcond][task]
                # for interval stats computation
                args += [ (t1eff,t2eff,rawname, gs, intind, itype, datPrep, intside, specgramPrep) ]

                #print('len args ',len(args) )

                #st = getStatPerChan(t1,t2,singleRaw = rawname, mergeTasks=False, glob_stats=gv.glob_stats )
                #statlist[intind] = st[sind_str][ medcond][ task]
            if intside == gv.gen_subj_info[sind_str]['tremor_side']:
                statsMainSide[rawname]  = statlist
            else:
                statsOtherSide[rawname] = statlist

    #from IPython import embed; embed()

    print("Will compute stats for {} intervals".format(len(args) ) )

    if len(args) > 1 and (not statsInterval_forceOneCore):
        ncores = mpr.cpu_count()
        p = mpr.Pool( min(len(args), ncores)  )
        pr = p.map(stat_proxy, args)
        p.close()
        p.join()

        for k,intside, intind,st in pr:
            if intside == gv.gen_subj_info[sind_str]['tremor_side']:
                statsMainSide[k][intind] = st 
            else:
                statsOtherSide[k][intind] = st 
            #stats[k][intind] = st
    else:
        for arg in args:
            #import pdb; pdb.set_trace()

            #import ipdb; ipdb.set_trace()
            k,intside,intind,st = stat_proxy(arg)
            sind_str,medcond, task = getParamsFromRawname(rawname)
            if intside == gv.gen_subj_info[sind_str]['tremor_side']:
                statsMainSide[k][intind] = st 
            else:
                statsOtherSide[k][intind] = st 

    return statsMainSide, statsOtherSide

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

    # positive X coord (first coord) implies right side of the brain (as seen from the back)
    # look at coords at source data file
    # here we collect indices of channels only
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
        color = None, skipPlot = False, normalization = ('whole','channel_nout'),
        mean_corr=False):
    '''
    bandPow -- if True, plot band power, if False, plot band passed freq
    '''
    if bandPow:
        #if spec_specgramtype == 'scaleogram':  #  [to be implemented] -- need to design a wavelet
        #    bandpower = np.sum(Sxx_b,axis=0) * freqres
        normrange,normtype = normalization

        
        r = utils.getBandpow(k,chn,fbname, time_start, time_end, mean_corr=mean_corr)
        if r is None:
            return None

        bins_b, bandpower = r
        if normtype == 'channel' and normrange == 'whole':
            bandpower /= np.max(bandpower) 
        elif normtype == 'channel_distr' and normrange == 'whole':
            sind_str,medcond,task = utils.getParamsFromRawname(k)
            r = gv.glob_stats[sind_str][medcond][task][chn] 

            s0 = '{}_bandpow_'.format(fbname)
            if mean_corr:
                raise ValueError('Not implemented')
            else:
                s1max_some = '{}{}'.    format(s0, 'max_distr')
                s1min_some = '{}{}'.    format(s0, 'min_distr')

            if s1min_some not in r:
                print( "WARNING plotChannelBand: {} not in glob stats".format(s1max_some) )
                bandpower /= np.max(bandpower) 
            else:
                mn = r[s1min_some]
                mx = r[s1max_some]
                span = mx-mn
                bandpower = (bandpower - mn) / span
        elif normtype == 'channel_nout' and normrange == 'whole':
            sind_str,medcond,task = utils.getParamsFromRawname(k)
            r = gv.glob_stats[sind_str][medcond][task][chn] 

            s0 = '{}_bandpow_'.format(fbname)
            if mean_corr:
                s1max_some = '{}{}'.    format(s0, 'mc_max_nout')
                s1min_some = '{}{}'.    format(s0, 'mc_min_nout')
            else:
                s1max_some = '{}{}'.    format(s0, 'max_nout')
                s1min_some = '{}{}'.    format(s0, 'min_nout')

            if s1min_some not in r:
                print( "WARNING plotChannelBand: {} not in glob stats".format(s1max_some) )
                bandpower /= np.max(bandpower) 
            else:
                mn = r[s1min_some]
                mx = r[s1max_some]
                span = mx-mn
                bandpower = (bandpower - mn) / span
        elif normtype == 'none':
            print('plotChannelBand: no normalization')
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

def plotSpectralData(plt,time_start = 0,time_end = 400, chanTypes_toshow = None, onlyTremor = False ):

    mainSideColor = 'w'; 
    otherSideColor = 'lightgrey'; otherSideColor = 'gainsboro'
    #normType='uniform' # or 'log'
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
        #import pdb; pdb.set_trace()
        l = {}
        if plot_colPerInterval:
            if k not in plot_timeIntervalPerRaw:
                for side in ['left','right']:
                    l[side] = 1
            else:
                for side in ['left','right']:
                    l[side] = len( plot_timeIntervalPerRaw[k].get(side, [0] ) )
        else:
            for side in ['left','right']:
                l[side] = 1

        for side in ['left','right']:
            raw_int_pairs +=  zip( [ki] * l[side], np.arange(l[side]), l[side] * [side] )

    if singleRawMode:
        nc = 1
    else:
        nc = len(raw_int_pairs)

    nr = nplots_per_side
    if plot_onlyMainSide == 0:
        nr *= len(pair_inds) 

    if plot_onlyMainSide == 1:
        print('----- plotting only main tremor side')
    elif plot_onlyMainSide == -1:
        print('----- plotting only other side')
    
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
    hspace = 0.65
    wspace = 0.2
    leftspace = 0.05
    if nc == 1:
        ncformal = 2   # don'a make sinlge row because it would ruin adressing
        gridspec_kw={'width_ratios': [1, 0.001]}
        wweff *= 15
        wweff *= (plot_time_end / 300)
        wspace = 0.01
        leftspace = 0.005

    fig, axs = plt.subplots(ncols = ncformal, nrows=nr, figsize= (wweff*nc,hh*nr), sharey='none',
            gridspec_kw= gridspec_kw)
    plt.subplots_adjust(top=0.98, bottom=0.01, right=0.999, left=leftspace, hspace=hspace,
            wspace = wspace)
    colind = 0
    maxBandpowPerRaw = {}
    for colind in range(nc):
        ki,intind,sideint = raw_int_pairs[colind]
        k = ks[ki]
        
        sind_str,medcond,task = getParamsFromRawname(k)
        deftimeint =  [(time_start,time_end,'no_tremor') ]

        if k not in plot_timeIntervalPerRaw:
            allintervals = deftimeint
        else:
            allintervals = plot_timeIntervalPerRaw[k][ gv.gen_subj_info[sind_str]['tremor_side'] ]
        #allintervals = plot_timeIntervalPerRaw.get( k, deftimeint  )
        #for i,p in enumerate(allintervals):  # find first interval that starts not at zero (thus potentially we can what was before the tremor start)
        #    if p[2] == 'incPre':
        #        desind = i
        #        break
        time_start, time_end, intervalType =    allintervals [intind]   # 0 means first interval, potentially may have several

        #chnames = raws[k].info['ch_names']
        orderEMG, chinds_tuples, chnames_tuples = utils.getTuples(sind_str)
        
        if k not in maxBandpowPerRaw:
            maxBandpowPerRaw[k] = {}

        if tremor_intervals_use_merged:
            tremorIntervals = tremIntervalMerged.get(k, 
                    { 'left': [ (plot_time_start,plot_time_end,'entire') ],
                    'right': [ (plot_time_start,plot_time_end,'entire') ], } )
        else:
            tremorIntervals = tremIntervalPerRaw[k]
 
        tremSideCur = gv.gen_subj_info[sind_str]['tremor_side']
        if plot_onlyMainSide == 1:
            pair_inds = [orderEMG.index( tremSideCur ) ]
        elif plot_onlyMainSide == -1:
            opside = utils.getOppositeSideStr( tremSideCur)
            opsideind =  orderEMG.index( opside )
            pair_inds = [ opsideind   ]
        for channel_pair_ind in pair_inds:
            if plot_onlyMainSide != 0:
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
            chdata, chtimes = utils.getData(k, ch_toplot_timecourse, ts,te )

            mintime = min(chtimes)
            maxtime = max(chtimes)
            
            #tremorIntervalsCurSide = tremorIntervals[side_body]

            special_intervals = tremorIntervals[side_body] 
            #import pdb; pdb.set_trace()
            #if isinstance(special_intervals,dict) and len(special_intervals):
            #    special_intervals = special_intervals['tremor']

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

                    st = gv.glob_stats[sind_str][medcond][task][chn]
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
                    
                    #if isinstance(tremorIntervalsCurSide, dict) and chn in tremorIntervalsCurSide:
                    #    special_intervals = tremorIntervalsCurSide[chn]
                    #    for pa in special_intervals:
                    #        clrfb = plot_intervalColors[intervalType]
                    #        ax.fill_between( list(pa) , ymin, ymax, facecolor=clrfb, alpha=0.2)

                # don't want overlay multiple times
                #if not (modality == 'EMG' and isinstance(tremorIntervalsCurSide, dict) ):
                #    special_intervals = tremIntervalMerged[k][side_body]
                #    for pa in special_intervals:
                #        ax.fill_between( list(pa) , ymin, ymax, facecolor='red', alpha=0.2)


                #print( 'PAIRSSSSSSSSSSS ',k,len(special_intervals), special_intervals )
                for intervalType in plot_intervalColors:
                    clrfb = plot_intervalColors[intervalType]
                    #if intervalType not in special_intervals:
                    #    continue
                    #for pa in special_intervals[intervalType]:
                    #    ax.fill_between( list(pa) , ymin, ymax, facecolor=clrfb, alpha=plot_intFillAlpha)
                    for pa in timeIntervalPerRaw_processed[k][side_body]:
                        t1,t2,ity = pa
                        if ity != intervalType:
                            continue
                        ax.fill_between( [t1,t2] , ymin, ymax, facecolor=clrfb, alpha=plot_intFillAlpha)

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

                        #print(chn,'plot cvl max', np.max(ys),'argmax ',np.argmax(ys) )
                        ax.plot(bins, ys, label=chn)
                
                ymin = 0
                ymax = plot_tremConvAxMax
                ax.set_ylim(ymin,ymax)
                ax.set_xlim(np.min(bins),np.max(bins))
                ax.axhline(y=tremIntDef_convThr, label='conv thr',ls=':',c='k')
                ax.legend(loc=legendloc,framealpha = legalpha)
                ax.set_title('Tremor band convolution')

                #special_intervals = tremIntervalMerged[k][side_body]
                for intervalType in plot_intervalColors:
                    #if intervalType not in special_intervals:
                    #    continue
                    clrfb = plot_intervalColors[intervalType]
                    #for pa in special_intervals[intervalType]:
                    #    ax.fill_between( list(pa) , ymin, ymax, facecolor=clrfb, alpha=plot_intFillAlpha)

                    for pa in timeIntervalPerRaw_processed[k][side_body]:
                        t1,t2,ity = pa
                        if ity != intervalType:
                            continue
                        ax.fill_between( [t1,t2] , ymin, ymax, facecolor=clrfb, alpha=plot_intFillAlpha)

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
                        title = '{}, {}, Freq bands powers: {}'.format(k, modality, fbname)
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
                                    logscale=0,color=color, 
                                    normalization = ('whole','channel_nout'),
                                    mean_corr = True)
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
                        if modality.find('EMG') >= 0 and useEMG_cluster:
                            # computed using k-means 
                            clust = gv.glob_stats[sind_str][medcond][task]['tremorfreq_clusters_allEMG'][chn]
                            if useEMG_cluster:
                                if tremrDet_clusterMultiMEG:
                                    for clusti in range(len(clust) ):
                                        ax.axhline(y = clust[clusti], 
                                                label = '{} clust{}'.format(chn,clusti) ,
                                                #ls = plot_freqBandsLineStyle['tremor' ], 
                                                ls='--', 
                                                c=plot_colorsEMG[colorEMGind-1] )
                                else:
                                    thrs = gv.glob_stats[sind_str][medcond][task][chn]['thrPerFreq_trem'] 
                                    ax.axhline(y = freqres * np.sum(thrs), 
                                            label = '{} tremor thr'.format(chn) ,ls = ltype_tremorThr)

                            if tremorDetectUseCustomThr:
                                try:
                                    thr = gv.gen_subj_info[sind_str]['tremorDetect_customThr'][medcond][side_body][chn] 
                                except KeyError:
                                    thr = gv.glob_stats[sind_str][medcond][task]['thrPerCh_trem_allEMG'][chn]
                                ax.axhline(y=thr, ls = ltype_tremorThr,lw=2, c= plot_colorsEMG[colorEMGind-1], 
                                        label = '{} tremor thr'.format(chn) )

                        #deflims = (0, maxpow_band)
                        deflims = (-0.15, 1)
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

                        #special_intervals = tremIntervalMerged[k][side_body]
                        for intervalType in plot_intervalColors:
                            clrfb = plot_intervalColors[intervalType]
                            #if intervalType not in special_intervals:
                            #    continue
                            #for pa in special_intervals[intervalType]:
                            #    ax.fill_between( list(pa) , ymin, ymax, facecolor=clrfb, alpha=plot_intFillAlpha)
                            for pa in timeIntervalPerRaw_processed[k][side_body]:
                                t1,t2,ity = pa
                                if ity != intervalType:
                                    continue
                                ax.fill_between( [t1,t2] , ymin, ymax, facecolor=clrfb, alpha=plot_intFillAlpha)

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
                                    logscale=0,color=color,skipPlot = 1,
                                    mean_corr = True)
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

                for rowind,chname in enumerate(ch_toplot_spec):
                    ax = axs[rowind + rowind_shift,colind]
                    plotSpecMeanCorr(ax, k, chname, time_start, tetmp/sampleFreq)
                        
                    if side_body == gv.gen_subj_info[sind_str]['tremor_side']:
                        ax.patch.set_facecolor(mainSideColor)
                    else:
                        ax.patch.set_facecolor(otherSideColor)
        print('Plotting {}, col {} / {} finished'.format(k,colind,nc))

        if nc == 1:
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

        figname = '{}_pairno{}_{} nr{}, nsrc{} _{}_side{}.{}'. \
                    format(prename, channel_pair_ind, data_type, nr, len(srcs), sp, plot_onlyMainSide, ext)
        plt.savefig( os.path.join(plot_output_dir, figname ) )
        print('Figure saved to {}'.format(figname) )
    else:
        print('Skipping saving fig')

    if not showfig:
        plt.close()
    else:
        plt.show()
        
    print('Plotting all finished')


def plotSpecMeanCorr(ax, k, chname, time_start, time_end, normType='uniform'):
    freqs, bins, Sxx = gv.specgrams[k][chname]
    minfreq = 0
    if chname.find('LFP') >= 0:
        minfreq = plot_minFreqInSpec
    freqs, bins, Sxx = utils.getSubspec(freqs,bins,Sxx,
                                  minfreq,plot_maxFreqInSpec,
                                  time_start,time_end)
    
    stats = gv.glob_stats[sind_str][medcond][task][chname]
    #mx = stats['max_spec']; mn = stats['min_spec']; 

    #mx_mc = stats['max_spec_mc_plot']; mn_mc = stats['min_spec_mc_plot']; 
    me = stats.get('mean_spec_nout', None)                
    if me is None:
        return None
    mx_mc = stats['max_spec_mc_nout']; 
    mn_mc = stats['min_spec_mc_nout']; 

    #mn = stats['min_spec']; 
    #mx = stats['max_spec']; 
    #me = stats['mean_spec']                

    mx_mc = stats['max3_spec_mc_nout']; 
    mn_mc = stats['min3_spec_mc_nout']; 

    goodfreqs = freqs > spec_minfreq
    goodfreqs = np.logical_and( goodfreqs , 
            np.logical_or(freqs < spec_DCfreq - spec_DCoffs, 
                freqs > spec_DCfreq + spec_DCoffs) )

    if isinstance(Sxx[0,0], complex):
        Sxx = np.abs( Sxx )
    Sxx = Sxx[goodfreqs,:]

    #me = me[goodfreqs]
    freqinds = np.where( np.logical_and(freqs[goodfreqs] >= minfreq,
        freqs[goodfreqs] <= plot_maxFreqInSpec) )[0]
    Sxx =  ( Sxx -  me[freqinds,None] ) / me[freqinds,None]    # divide by mean

    #mn_mc = np.min(Sxx)
    #mx_mc = np.max(Sxx)

    #mn_mc = (mn - me) / me
    #mx_mc = (mx - me) / me
    #mn_mc =  mn_mc[freqinds]
    #mx_mc =  mx_mc[freqinds]

    if normType == 'uniform':
        #norm = mpl.colors.Normalize(vmin=0.,vmax=mx_mc);
        norm = mpl.colors.Normalize(vmin=np.min(mn_mc),vmax= np.max(mx_mc) );
    elif normType == 'log':
        norm = mpl.colors.LogNorm(vmin=np.min(mn_mc),vmax= np.max(mx_mc) );
    #print(chname,Sxx.shape,len(freqs),mx, mn)
    if chname.find('MEGsrc') >= 0:
        modality = 'MEGsrc'
    else:
        modality = 'LFP'


    im = ax.pcolormesh(bins, freqs[goodfreqs], Sxx, 
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
            format(k, chlbl, specparstr,  np.min(mn_mc),np.max(mx_mc) ) )
    ax.set_xlabel('Time, [s]')
    ax.set_ylabel('Freq, [Hz] '+normType)
    #ax.set_xlim(mintime,maxtime)
    ax.set_xlim(time_start,time_end)

def makeBarPlot( ax, rawname, chname, legloc = None, printLog = False, xoffset = 0, axtop=None,
        spaceBetweenGroups = 2, skipPlot = False, binwidth = 1, mainSide = 1):
    #statTypes = ['max_nout_pct', 'L1', 'L2', 'L05' ]
    statTypes = ['mc_max_nout', 'meanDiv_L1', 'meanDiv_L2', 'meanDiv_L05' ]
    incCoef = {'mc_max_nout': 70 }  # to look normal on the plot

    assert mainSide in [-1,1]

    sind_str,medcond,task = getParamsFromRawname(rawname)
    tremSideCur = gv.gen_subj_info[sind_str]['tremor_side']
    if mainSide == 1:
        side = tremSideCur
        intervalStats = gv.glob_stats_perint[rawname]
    else:
        side = utils.getOppositeSideStr(tremSideCur)
        intervalStats = gv.glob_stats_perint_nms[rawname]

    intervals = timeIntervalPerRaw_processed[rawname] [ side ]
    assert len(intervals) <= len(intervalStats), "{} There are more intervals than stats, perhaps need to recalc stats".format( rawname, chname)

    if len(intervalStats) == 0:
        raise ValueError( 'intervalStats for {} is emptry, perhaps json was misread'.format(rawname) )

    ivalis = {}  # dict of indices of interval
    for itype in gv.gparams['intTypes']:
        ivit = []
        for i,interval in enumerate(intervals):
            t1,t2,it = interval
            
            if it == itype and not isinstance(intervalStats[i ], int) :
                ivit += [i]
        if len(ivit) > 0:
            ivalis[itype] = ivit



    skipLegendMode = False
    if  isinstance(legloc,int) and legloc == -1:
        skipLegendMode = True

    binvalsDict  = {}
    binerrssDict  = {}
    binnamesDict = {}
    bincoordsDict = {}

    totnum = 0

    modality = utils.chname2modality(chname)

    #import pdb; pdb.set_trace()

    #find out how many bars we'll have in each group
    statNum = 0
    for iti,itype in enumerate(gv.gparams['intTypes']):
        if itype not in ivalis:
            continue
        inds = ivalis[itype]  
        if len(inds) == 0:
            continue
        for fbname in plot_freqBandNames_perModality[modality]:
            for statType in statTypes:
                s0 = '{}_bandpow_{}'.format(fbname, statType)
                
                # if at least one of the indices of current type has this statistic
                for ii in inds:
                    intst = intervalStats[ii ]
                    if isinstance(intst,int):
                        continue
                    
                    st = intervalStats[ii ] [sind_str][medcond][task]
                    pp =  st [chname]
                    if s0 not in pp:
                        if printLog:
                            print('{} is not in {}{}{}'.format(s0,chname,fbname,itype ) )
                        continue
                    else:
                        statNum += 1

                    break
        break

    #print('makeBarPlot: ivalis = ',ivalis)
    #print('makeBarPlot: statNum = ',statNum)
    #print('makeBarPlot: intervalStats types = ',[type(kkk) for kkk in intervalStats] )
    #printLog = 1

    iti = 0
    for itype in gv.gparams['intTypes']:
        if itype not in ivalis:
            continue
        inds = ivalis[itype]  
        if len(inds) == 0:
            continue

        binvals  = []
        binerrs  = []
        binnames = []

        #nbinsPerStatType = len(gv.gparams['intTypes']) + 1  # +1 to leave space between
        #nbinsPerFB = nbinsPerStatType * len(statTypes) + 2   # +2 to leave space between 
        xs = []
        xcur = 0
        for fbname in plot_freqBandNames_perModality[modality]:
            for statType in statTypes:
                #gv.freqBands[modality]
                s0 = '{}_bandpow_{}'.format(fbname, statType)
                

                #import pdb; pdb.set_trace()
                leaveLoop = True
                vals = []
                for i in inds:
                    intst  = intervalStats[i] 
                    if isinstance(intst,int):
                        #print('ind {} is int, continue'.format(i) )
                        continue

                    prer = intst[sind_str][medcond][task] [chname]
                    if s0 not in prer:
                        if printLog:
                            print('{} is not in {} itype {}'.format(s0,chname,itype ) )
                        break
                    
                    r = intst[sind_str][medcond][task] [chname][s0]
                    vals += [r]
                    leaveLoop = False

                if leaveLoop:
                    continue

                coef = incCoef.get( statType, 1)
                mn = np.mean(vals) * coef
                std = np.std(vals) * coef
                binvals += [ mn ]
                binerrs += [ std ]

                bn = '{}_{}'.format(fbname,statType)
                if coef > 1:
                    bn += '*{}'.format(coef)
                binnames += [ bn ]
                xs += [ xcur  ]
                xcur +=  len(gv.gparams['intTypes'])
                #print(binnames)

        binvalsDict[itype]  = binvals
        binnamesDict[itype] = binnames
        binerrssDict[itype] = binerrs

        #tot = len(binvals) * binwidth * len(intTypes)
        xs = np.array(xs) * binwidth + iti * binwidth 
        bincoordsDict[itype] = xoffset + xs
        totnum += len(xs)

        iti += 1  # like that to skip interval types that did not appear


    # add intervals
    for iti,itype in enumerate(bincoordsDict):

        xscur = bincoordsDict[itype] 
        if len(xscur) == 0:
            continue

        for sti  in range(statNum):
            xscur[sti] += sti * spaceBetweenGroups 

        bincoordsDict[itype]  = xscur # maybe not necessary

        lbl = None
        if not skipLegendMode:
            lbl = itype
        # note that xscur already have offset in it
        #if ax is not None:
        if not skipPlot:
            ax.bar( xscur, binvalsDict[itype], binwidth, 
                    yerr=binerrssDict[itype], label = lbl, color = gv.gparams['intType2col'][itype] )


    #ticklab = [0] * len(intTypes) * len(binnames)
    #for iti,itype in enumerate(intTypes):
    #    for bni, binname in enumerate(binnames):
    #        ticklab[ len(intTypes) * bni + iti ] = binname
    #ax.set_xticklabels ( ticklab ,rotation=45)

    if len(bincoordsDict ) == 0:
        raise ValueError('bincoordsDict is empty for {}, probably there are no intervals marked. Are you sure it is like that?'.format(rawname) )
        return 0, {}, {}, {}, {}
    xticks = list(bincoordsDict.values() )[0]
    xlabels = binnames


    if xoffset > 0:
        xticks_existing = ax.get_xticks()

        #if xoffset > 0 and rawname == 'S01_off_hold':
        #    print(bincoordsDict)
        #    print(xticks_existing)
        #    print(xticks)
        #    import pdb; pdb.set_trace()


        xticks = np.append( xticks_existing, xticks)

        xlabels_existing = [item.get_text() for item in ax.get_xticklabels()]
        xlabels = xlabels_existing + binnames

    #if ax is not None:
    ax.set_xticks( xticks  )
    ax.set_xticklabels(xlabels, rotation=90)

    if axtop is not None:
        xticksTop = np.array( [xoffset] )

        chlbl = chname
        if gv.gen_subj_info[sind_str]['lfpchan_used_in_paper'] == chname:
            chlbl = '* ' + chname

        xlabelsTop = [  utils.getMEGsrc_chname_nice( chlbl) ]
    
        if xoffset > 0:
            xticksTop_existing = axtop.get_xticks()
            xticksTop = np.append( xticksTop_existing, xticksTop)

            xlabelsTop_existing = [item.get_text() for item in axtop.get_xticklabels()]
            xlabelsTop = xlabelsTop_existing + xlabelsTop

            #print(xticksTop, xlabelsTop)

        if axtop is not None:
            axtop.set_xticks( xticksTop  )
            axtop.set_xticklabels(xlabelsTop, rotation=0)

    #print(binnames)
                        
    if legloc is None:
        legloc = legendloc
    if not skipLegendMode and not skipPlot:
        ax.legend(loc=legloc,framealpha = legalpha)


    if not skipPlot:
        ax.set_title('{} {}'.format(rawname, chname) )

    if len(xticks) == 0:
        totbarwidth = xoffset
    else:
        totbarwidth = np.max( xticks ) - xoffset
    #totbarwidth = ( totnum + statNum  * spaceBetweenGroups ) 

    return totbarwidth, binvalsDict, binnamesDict, binerrssDict, bincoordsDict
            
    #return totbarwidth

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

def plotBarplotTable(modalities, onlyMainSide = 1):
    '''
    plot for several raws, outer function
    '''
    ww = 23
    hh = 10

    maxrawcols = 10
    ndoublerows = int( np.ceil( len(raws) / maxrawcols ) )
    nc = min(   max( len(raws), 2) + 1   , maxrawcols + 1 )  # matplotlib doesn't allow too large images
    nr = max( len(modalities) , 2)

    if ndoublerows > 1:
        nc = maxrawcols
        nr = ndoublerows * 2

    fig,axs = plt.subplots(nrows = nr, ncols = nc, figsize = (nc*ww, nr*hh) )

    pair_inds = None
    
    spaceBetween = 15

    hshift = 1

    def getAxCoord( modi, horind):
        assert modi <= 1
        colind = horind % maxrawcols 
        rowind = 2 * (horind // maxrawcols) + modi  # is number of modalities
        #if rowind == 0:
        #    colind += 1 
         
        return rowind, colind


    chansPerModPerRaw = {}
    chansPerModPerRaw_otherside = {}
    # 1 raw per modality, 1 column per raw, 0th column is average
    for modi, modality in enumerate(modalities ):
        chperraw = {}
        if onlyMainSide == 0:
            chperraw_otherside = {}  # opposite to the main one, if we plot both
        else:
            chperraw_otherside = None

        for rawi, k in enumerate(raws ):

            sind_str,medcond, task  = getParamsFromRawname(k)

            orderEMG, chinds_tuples, chnames_tuples = utils.getTuples(sind_str)
            mts = gv.gen_subj_info[sind_str]['tremor_side']
            pi = orderEMG.index( mts ) 
            opi = orderEMG.index( utils.getOppositeSideStr( mts ) ) 
            if onlyMainSide == -1:
                pi = opi

            chnames = []
            #for pi in pair_inds:
            chnames = chnames_tuples[pi][modality]
            chperraw[ k ] = chnames
            
            if onlyMainSide == 0:
                chnames_otherside = chnames_tuples[opi][modality]
                chperraw_otherside[ k ] = chnames_otherside
            

            #if rawi < maxrawcols: 
            axcoord = getAxCoord( modi, hshift + rawi)
            ax = axs[axcoord]
            makeMutliBarPlot(ax, k, chnames, spaceBetween, mainSide=onlyMainSide )
            ax.legend( loc = (1.01,0) )
            ax.set_title('{}: {} barplot '.format(k,modality) )

        chansPerModPerRaw[modality] = chperraw
        chansPerModPerRaw_otherside[modality] = chperraw_otherside

    ################ now first column
    sind_str,medcond, task  = getParamsFromRawname( list(raws.keys() )[0] )
    for modi, modality in enumerate(modalities ):
        ax = axs[modi, 0]

        #orderEMG, chinds_tuples, chnames_tuples = utils.getTuples(sind_str)
        #if plot_onlyMainSide:
        #    pair_inds = [orderEMG.index( gv.gen_subj_info[sind_str]['tremor_side'] ) ]

        #chnames = []
        #for pi in pair_inds:
        #    chnames += chnames_tuples[pi][modality]

        makeMutliBarPlot(ax, list(raws.keys() ), chansPerModPerRaw[modality], spaceBetween, 
                chnamesOtherSide = chansPerModPerRaw_otherside[modality],
                mainSide=onlyMainSide)
        ax.legend( loc = (1.01,0) )
        ax.set_title('Averge  {} barplot total {} raws'.format(modality, len(raws) ) )


    ##############################
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

        prename = 'barstats_{}'.format(subjstr  )

        figname = '{}_nr{}, nsrc{}_{}_side{}.{}'. \
                    format(prename, nr, len(srcs), sp, onlyMainSide, ext)
        plt.tight_layout()
        plt.savefig( os.path.join(plot_output_dir, figname ) )
        print('Figure saved to {}'.format(figname) )
    else:
        print('Skipping saving fig')

def makeMutliBarPlot(ax, rawname, chnames, spaceBetween=14, binwidth=1, spaceBetweenGroups = 2,
        chnamesOtherSide=None, mainSide = 1):
    '''
    plot several chnames on one horizontal plot
    '''

    if chnamesOtherSide is not None:
        raise ValueError('barplots for non main side are not implemented yet')
    
    chperraw_otherside = chnamesOtherSide

    multirawMode = False
    if isinstance(chnames , dict):
        multirawMode = True
        chperraw = chnames
        rawnames = rawname
    else:
        rawnames = [rawname]
        chperraw = {rawname:chnames}

    bardata = {}
    alliTypes = set()
    rawsAvailPeriType = {}
    numStats = -1
    for rawni,rawn in enumerate(rawnames):
        shift =0
        sind_str,medcond,task = getParamsFromRawname(rawn)

        ax2 = ax.twiny()
        ret = None
        itypes = None
        
        sublist = []   # for channel there are things there
    
        chnames_flt = chperraw[rawn]
        if chnames_flt[0].find('LFP') >= 0 and plot_barplotOnlyFavLFP:
            chnames_flt = utils.filterFavChnames( chperraw[rawn], sind_str ) 

        #if onlyMainSide == 0:
        #    mainSideVals = [-1,1]: 
        #else
        #    mainSideVale = [onlyMainSide]
        #for mainSide in mainSideVals: 
        for chni, chname in enumerate(chnames_flt ):
            if chni == 0 and rawni == 0:
                legloc_ = None
            else:
                legloc_ = -1

            #if chname.find('LFP') >= 0 and plot_barplotOnlyFavLFP:
            #    if not gv.gen_subj_info[sind_str]['lfpchan_used_in_paper'] == chname:
            #        continue
                    
            #print('shift is ',shift)
            ret = makeBarPlot(ax, rawn, chname, xoffset=shift, legloc = legloc_, axtop=ax2,
                    skipPlot = multirawMode , binwidth = binwidth, 
                    spaceBetweenGroups = spaceBetweenGroups, mainSide = mainSide) #legloc= (1.1,0)) 

            totbarwidth, binvalsDict, binnamesDict, binerrssDict, bincoordsDict = ret
            shift += totbarwidth
            shift += spaceBetween
            
            #print(binvalsDict)

            numStats = len( list(binvalsDict.values() )[0] )
            itypes = binvalsDict.keys()
            #print(rawn,itypes)
            
            kk = list(binnamesDict.keys() )[0]
            #print( rawn, chname, len(binnamesDict), len( bincoordsDict[kk] ), bincoordsDict[kk] )

            sublist += [ (chname,totbarwidth, binvalsDict, binnamesDict, binerrssDict, bincoordsDict) ]

        # look at last
        bardata[rawn] = sublist
        #alliTypes += set( binvalsDict.keys() )

        for intType in itypes:
            if intType not in rawsAvailPeriType:
                rawsAvailPeriType[intType] = [ rawn ]
            else:
                rawsAvailPeriType[intType] += [ rawn ]
    
    if multirawMode:
        valsPerItype = {}
        labels = []
        labelset = False

        indsToExclude = {}
        for intType in rawsAvailPeriType:
            indsToExclude[intType] = []

            ra = rawsAvailPeriType[intType]

            ll = len(list(chperraw.values() )[0] )
            vpi =  np.zeros( (len(ra) ,numStats * ll )  )
            means = np.zeros( numStats * ll) 
            errs  = np.zeros( numStats * ll)
            xs = np.zeros( numStats * ll)

            for rawni,rawn in enumerate(ra):
                sublist = bardata[rawn]
                for ii,sdcn in enumerate(sublist):
                    #ii = chperraw[rawn].index(chname)
                    #sdcn = subdict[chname]
                    chname, totbarwidth, binvalsDict, binnamesDict, binerrssDict, bincoordsDict = sdcn
                    #print('sublist',binvalsDict[intType] ) 
                    #print(vpi[rawni,ii * numStats + np.arange( numStats) ])

                    bincoords = bincoordsDict[intType] 
                    if len(bincoords) > 0:
                        vpi[rawni,ii * numStats + np.arange( numStats) ] = binvalsDict[intType]
                        xs[ii * numStats + np.arange( numStats)] = bincoords
                    else:
                        indsToExclude[intType] += [rawni]
                    
                    #print(ii,chname)
                #vals = binvalsDict[intType]
                #vpi = np.vstack( (vpi, vals) )

                #labels = list( binnamesDict[intType] )[0]

            if np.max(xs) < 1e-5:
                raise ValueError('xs left anassigned :(')

            # we need to exclude some inds that corresponds to interval types that were not found in certain raws
            goodrawinds = set(range(vpi.shape[0] ) ) - set(indsToExclude[intType] )
            goodrawinds = list(goodrawinds)
            vpi = vpi[goodrawinds,: ]

            valsPerItype[intType] = vpi
            print('intType {} , average among {}'.format( intType, vpi.shape[0] ) )

            means = np.mean( vpi, axis=0)
            errs = np.std( vpi, axis=0)
            #means +=  np.array(vals)     # binvalsDict[intType] is a list 
            #means /= numStats

            if not labelset:
                lbl = intType
            else:
                lbl = None
            ax.bar( xs, means, binwidth, 
                    yerr=errs, label = lbl, color = gv.gparams['intType2col'][intType] )
            #labelset = True
            #print('xs',xs)

            #ax2 = ax.twiny()
            #xlabels_existing = [item.get_text() for item in ax.get_xticklabels()]

        legloc = legendloc
        ax.legend(loc=legloc,framealpha = legalpha)
        

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

    plot_barplotOnlyFavLFP = 1

    timeIntervalsFromTremor = True
    rawnameList_fromCMDarg = False

    loadSpecgrams = True  # if exists
    saveSpecgrams = True
    saveSpecgrams_skipExist = True

    save_stats = 1
    save_stats_onlyIfNotExists = 1
    load_stats = 1

    plot_onlyMultiBarplot = False
    plot_time_start = 0
    plot_time_end = 300
    plot_onlyMainSide             = 1  # 1 only main,  -1 only opposite,  0 both

    time_start_forstats, time_end_forstats = 0,300
    forceOneCore_globStats     = 0
    statsInterval_forceOneCore = 0

    nonTaskTimeEnd = 300
    spec_time_end=nonTaskTimeEnd
    stat_computeDistrMax = False

    singleRawMode = False
    gv.gparams['intTypes'] = ['pre', 'post', 'initseg', 'endseg', 'middle_full', 'no_tremor', 'unk_activity_full' ]
    gv.gparams['intType2col'] =  {'pre':'blue', 'post':'gold', 'middle_full':'red', 'no_tremor':'green',
            'unk_activity_full': 'cyan', 'initseg': 'teal', 'endseg':'blueviolet' }

    import sys, getopt
    helpstr = 'Usage example\nudus_dataproc.py -i <comma sep list> -t <comma sep list> -m <comma sep list> -s'
    try:
        effargv = sys.argv[1:]  # to skip first
        if sys.argv[0].find('ipykernel_launcher') >= 0:
            effargv = sys.argv[3:]  # to skip first three

        opts, args = getopt.getopt(effargv,"hi:t:m:r:s",
                ["subjinds=","tasks=","medconds=","MEG_ROI=","singleraw","rawname=",
                    "update_spec", "update_stats", "barplot", "no_specload", "skipPlot" ,
                    "plot_time_start=", "plot_time_end=", "time_end_forstats=", 
                    "spec_time_end=", "debug", "plot_other_side"]) 
        print(sys.argv, opts, args)

        for opt, arg in opts:
            print(opt)
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
            elif opt in ("-s", "--singleraw"):
                #timeIntervalsFromTremor = False
                singleRawMode = True
            elif opt == '--rawname':
                fnames_noext = arg.split(',') 
                rawnameList_fromCMDarg = True
                subjstr,medcond,task = getParamsFromRawname(fnames_noext[0])
                assert len(fnames_noext ) == 1
                subjinds = [ int( subjstr[2] ) ]
                medconds = [medcond]
                tasks = [task]
            elif opt == '--update_spec':
                #flag = int(arg)
                #if flag:
                loadSpecgrams = False
                saveSpecgrams = True
                saveSpecgrams_skipExist = False
                #else:
                #    loadSpecgrams = True
                #    saveSpecgrams = True
                #    saveSpecgrams_skipExist = True
            elif opt == '--plot_other_side':
                plot_onlyMainSide = -1
            elif opt == '--plot_both_sides':
                plot_onlyMainSide = 0

            elif opt == '--update_stats':
                #flag = int(arg)
                #if flag:
                save_stats = 1
                save_stats_onlyIfNotExists = False
                load_stats = 0
                #else:
                #    save_stats = 1
                #    save_stats_onlyIfNotExists = 1
                #    load_stats = 1
            elif opt == '--barplot':
                plot_onlyMultiBarplot = 1
                print('Barplot only mode!')
            elif opt == '--no_specload':
                loadSpecgrams = False
            elif opt == '--skipPlot':
                skipPlot = True
            elif opt == '--plot_time_start':
                plot_time_start = float(arg)
            elif opt == '--plot_time_end':
                plot_time_end = float(arg)
            elif opt == '--time_end_forstats': 
                time_end_forstats = float(arg)
            elif opt == '--spec_time_end': 
                spec_time_end = float(arg)
            elif opt == '--debug': 
                forceOneCore_globStats     = 1
                statsInterval_forceOneCore = 1
            else:
                print('Unk option {}, exiting'.format(opt) )
                sys.exit(1)
              
    except (ValueError, getopt.GetoptError) as e: 
        print('Error in argument parsing, exiting!', helpstr,str(e)) 
        #print('Putting hardcoded vals')
        sys.exit(0)


    #if plot_onlyMultiBarplot:
    #    loadSpecgrams = False
    #timeIntervalsFromTremor = False  # needed when observing


    assert isinstance(subjinds[0], int)
    print('Subjinds:{}, tasks:{}, medconds:{}'.format(subjinds, tasks, medconds) )


    trem_times_fn = 'trem_times_tau.json'
    #trem_times_fn = 'trem_times.json'
    #with open(os.path.join(data_dir,trem_times_fn) ) as jf:
    with open(trem_times_fn ) as jf:
        trem_times_byhand = json.load(jf)   
        # trem_times_byhand[medcond][task]['tremor'] is a list of times that one needs to couple to get begin/end
        # there is also "no_tremor"

    trem_times_nms_fn = 'trem_times_tau_nms.json'
    with open(trem_times_nms_fn ) as jf:
        trem_times_nms_byhand = json.load(jf)   

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
    specgram_scaling = 'spectrum'  # power, not normalized by freq band (i.e. not denisty)
    gv.gparams['specgram_scaling'] = specgram_scaling
 
    # which time range use to compute spectrograms
    spec_time_start=0 
    spec_wavelet = "cmor1.5-1.5"     # first is width of the Gaussian, second is number of cycles
    spec_wavelet = "cmor1-1.5"    
    spec_wavelet = "cmor1-2"    
    #
    #spec_wavelet = "cmor1.5-1.5"     # test
    spec_FToverlap = 0.75
    spec_specgramtype = 'lspopt'
    spec_specgramtype = 'scaleogram'
    spec_specgramtype = 'mne.wavelet'
    spec_freqs2wletcyclesCoef = 0.75  # larger coef -- wider gaussian 
    #spec_specgramtype = 'scipy'
    spec_cwtscales = np.arange(2, 120, 4)  # lower scale = a means that highest freq is 2/dt = 2*sampleFreq
    base = 5
    spec_cwtscales = 2 + (np.logspace(0.0, 1, 25,base=base) - 1 ) * 400/base;
    spec_minfreq = 3
    spec_minfreq = 3
    spec_DCoffs = 3
    spec_DCfreq = 50
    spec_thrBadScaleo = 0.8

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
    useEMG_cluster = False
    tremrDet_clusterMultiMEG = useEMG_cluster
    gv.gparams['tremDet_useTremorBand'] = False
    gv.gparams['tremDet_timeStart'] = 0
    gv.gparams['tremDet_timeEnd'] = nonTaskTimeEnd

    gv.gparams['plot_LFP_onlyFavoriteChannels'] = 1
    gv.gparams['plot_LFP_favRandomNonMainSide'] = 1

    plot_EMG_spectrogram          = False
    plot_MEGsrc_spectrogram       = True
    #show_EMG_band_corr            = True
    show_EMG_band_corr            = False
    show_stats_barplot            = True
    show_stats_barplot            = False
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

    useTauFavChannels = False
    favoriteLFPch_perSubj = {'S01': ['LFPR23', 'LFPL01' ], 
            'S02': ['LFPR23', 'LFPL12'], 'S03': ['LFPR12', 'LFPL12'], 'S09':['LFPL78', 'LFPR67'], 
            'S08':['LFPR12' , 'LFPL56' ], 'S04':['LFPL01', 'LFPR01'] } 

    #favoriteLFPch_perSubj_nms = {'S01': ['LFPL01', 'HirschPt2011_1', 'HirschPt2011_3' ], 
    favoriteLFPch_perSubj_nms = {'S01': ['LFPL01' ], 
            'S02': [ 'LFPL23'], 'S03': [ 'LFPL12' ], 'S04':[ 'LFPR12' ], 'S05':[ 'LFPR23'],
            'S06': ['LFPL01'], 'S07':['LFPR01'], 'S08':['LFPL12'], 'S09':['LFPR56'], 'S10':['LFPR12' ] } 

    plot_minFreqInSpec = 2.5  # to get rid of heart rate
    plot_minFreqInBandpow = 2.5  # to get rid of heart rate
    #plot_maxFreqInSpec = 50
    #plot_maxFreqInSpec = 80
    #plot_maxFreqInSpec = 35
    plot_maxFreqInSpec = 100 # max beta
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
    if not rawnameList_fromCMDarg:
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


    if singleRawMode:
        assert len(fnames_noext) == 1, 'more than one raw cannot be used with singleRaw mode'

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

        gv.gen_subj_info[subj]['favoriteLFPch'] = []
        if useTauFavChannels:
            if subj in favoriteLFPch_perSubj:
                gv.gen_subj_info[subj]['favoriteLFPch'] = favoriteLFPch_perSubj[subj]
        else:
            gv.gen_subj_info[subj]['favoriteLFPch'] += [ gv.gen_subj_info[subj]['lfpchan_used_in_paper'] ]
            
            if subj in favoriteLFPch_perSubj_nms:
                gv.gen_subj_info[subj]['favoriteLFPch'] += favoriteLFPch_perSubj_nms[subj] 
            elif subj in gv.subjs_analyzed and gv.gparams['plot_LFP_favRandomNonMainSide']:
                import time
                np.random.seed( int ( time.time() ) )

                tremSideCur = gv.gen_subj_info[sind_str]['tremor_side']
                orderEMG, chinds_tuples, chnames_tuples = utils.getTuples(subj)
                tremSideInd = orderEMG.index( tremSideCur )
                revSideInd = 1- tremSideInd 
                lfpnames = chnames_tuples[  revSideInd ]['LFP']  
                nlfps = len(lfpnames)
                favlfp = np.random.choice( lfpnames)
                assert isinstance(favlfp, str)
                gv.gen_subj_info[subj]['favoriteLFPch'] += [favlfp]
            elif subj in gv.subjs_analyzed:
                raise ValueError('no fav chan for subj {}'.format(subj) )

                # look 


        #bc = gv.gen_subj_info[subj]['bad_channels']
        #for medcond in bc:
        #    bc2 = bc[medcond] 
        #    for task in bc2:
        #        #for tremor 

    #import pdb; pdb.set_trace()

    # if we want to save the updated information back to the file
    if updateJSON:
        with open(os.path.join(data_dir,'Info.json'), 'w' ) as info_json:
            json.dump(gv.gen_subj_info, info_json)

    # compute spectrograms for all channels (can be time consuming, so do it only if we don't find them in the memory)
    #loadSpecgrams = False
    singleSpecgramFile = False


    stats_basefname = 'stats'
    #stats_fname = os.path.join( data_dir, 'last_data.npz')

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
            no_needforspec = (plot_onlyMultiBarplot and load_stats)
            gv.specgrams = {}
            for k in raws:
                sind_str,medcond,task = getParamsFromRawname(k)
                nsources = len( gv.subjs_analyzed[sind_str] ['MEGsrcnames_perSide'] ) * 2
                nroi = len(MEGsrc_roi)
                specgramFname = '{}_nroi{}_nMEGsrcInds{}_spec_{}.npz'.format(k, nroi, 
                        len(MEGsrc_names_toshow  ), spec_specgramtype  )
                #specgramFname = 'specgrams_1,2,3.npz'
                specgramFname = os.path.join(data_dir, specgramFname)
                if loadSpecgrams and (not no_needforspec) and os.path.exists(specgramFname) :
                    print('Loading specgrams from ',specgramFname)
                    specgramscur = np.load(specgramFname, allow_pickle=True)['arr_0'][()]
                    gv.specgrams.update( specgramscur   )
                elif not no_needforspec and not loadSpecgrams:  #if yes, we don't apriory need specgrams, only stats
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
    #freq_min_forstats, freq_max_forstats = 0, NFFT//2   #NFFT//2

    # check what Jan considered as tremor frequency 
    tfreqs = []
    for subj in gv.subjs_analyzed:
        tfreq = gv.gen_subj_info[subj]['tremfreq']
        tside = gv.gen_subj_info[subj]['tremor_side']
        print('{} has tremor at {} side with freq {}'.format(subj,tside,tfreq) )
        tfreqs += [ tfreq]

        #if subj not in gv.gen_subj_info[subj]['favoriteLFPch'] and gv.gparams['plot_LFP_onlyFavoriteChannels']:
        #    raise ValueError('Want to plot only favorite LFP they are not set for all subjects!' )

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
    plot_freqBandNames_perModality = {'EMG': ['tremor' ], 'LFP': ['tremor', 'beta','gamma_motor'], 
            'MEGsrc':['tremor','beta', 'gamma_motor' ], 'EOG':['lowgamma'] }
    #plot_freqBandNames_perModality = {'EMG': ['tremor' ], 'LFP': [ 'beta', 'gamma_motor'], 
    #        'MEGsrc':['tremor','beta', 'gamma_motor' ], 'EOG':['lowgamma'] }

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
    #plot_setBandlims_naively = True
    plot_setBandlims_naively = False  # use stats for bandlims

    plot_intervalColors = { 'middle_full':'red', 'tremor':'red',  'unk_activity':'yellow', 
            'unk_activity_full':'yellow', 'post':'cyan', 'pre':'skyblue', 'no_tremor':'violet' }
    plot_intFillAlpha = 0.14

    #cmap = 'hot'
    #cmap = 'gist_rainbow'
    #cmap = 'viridis'

    ############# # update in place gen_
    if len( gv.specgrams ) > 0 and not plot_onlyMultiBarplot:
        for subj in gv.subjs_analyzed:
            for modality in ['MEGsrc', 'LFP' ]:
                for fbname in ['tremor']:
                    utils.sortChans(subj, modality, fbname, replace=True, numKeep = plot_numBestMEGsrc)

    ##############   compute some statisitics, like max, min, mean, cluster in subbands, etc -- can be time consuming

    gv.artifact_intervals = {}
    def unpackTimeIntervals(trem_times_byhand, mainSide = True):
        # unpack tremor intervals sent by Jan
        tremIntervalJan = {}
        for subjstr in trem_times_byhand:
            maintremside = gv.gen_subj_info[subjstr]['tremor_side']
            if mainSide:
                side = maintremside  
            else: 
                side = utils.getOppositeSideStr(maintremside)

            s = trem_times_byhand[subjstr]
            for medcond in s:
                ss = s[medcond]
                for task in ss:
                    sss = ss[task]
                    rawname = getRawname(subjstr,medcond,task )
                    if rawname not in raws:
                        continue
                    tremdat = {}
                    for intType in sss: # sss has keys -- interval types
                        s4 = sss[intType]
                        #s4p = copy.deepcopy(s4) 
                        s4p = s4
                        # or list of lists 
                        # or list with one el which is list of lists
                        cond1 = isinstance(s4,list) and len(s4) > 0
                        if not cond1:
                            continue
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

                        if intType.find('artifact') >= 0:
                            print(rawname,intType, s4p)
                            r = re.match( "artifact_(.+)", intType ).groups()
                            chn = r[0]
                            if chn == 'MEG':
                                chn += side
                            if rawname in gv.artifact_intervals:
                                gai = gv.artifact_intervals[rawname]
                                if chn not in gai:
                                    gv.artifact_intervals[rawname][chn] = s4p
                                else: 
                                    invalids = []
                                    for a,b in gai[chn]:
                                        for ii,ival in enumerate(s4p):
                                            aa,bb = ival
                                            if abs(a-aa) < 1e-6 and abs(b-bb) < 1e-6:
                                                invalids += [ii]
                                    validinds = list( set(range(len(s4p) ) ) - set(invalids) )
                                    if len(validinds) > 0:
                                        gv.artifact_intervals[rawname][chn] += [s4p[ii] for ii in validinds]
                            else:
                                gv.artifact_intervals[ rawname] ={ chn: s4p } 

                        tremdat[intType] = s4p  # array of 2el lists
                        
                    tremIntervalJan[rawname] = { 'left':[], 'right':[] }
                    tremIntervalJan[rawname][side] =  tremdat 
                        #print(subjstr,medcond,task,kt,len(s4), s4)
                        #print(subjstr,medcond,task,kt,len(s4p), s4p)
        return tremIntervalJan


    tremIntervalJan = unpackTimeIntervals(trem_times_byhand, mainSide = True)
    tremIntervalJan_nms = unpackTimeIntervals(trem_times_nms_byhand, mainSide = False)
    for rawn in tremIntervalJan:
        sind_str,medcond,task = getParamsFromRawname(rawn)
        maintremside = gv.gen_subj_info[sind_str]['tremor_side']
        opside= utils.getOppositeSideStr(maintremside)
        if rawn in tremIntervalJan_nms:
            tremIntervalJan[rawn][opside] = tremIntervalJan_nms[rawn][opside] 

    print('Found artifacts info: ',gv.artifact_intervals)
    #sys.exit(0)

    #import ipdb; ipdb.set_trace()

    def checkIntStatUpdateNeeded(rawname):
        sind_str,medcond,task = getParamsFromRawname(rawname)
        tremSideCur = gv.gen_subj_info[sind_str]['tremor_side']
        for ms in [-1,1]:
            if ms == 1:
                side = tremSideCur
                intervalStats = gv.glob_stats_perint[rawname]
            else:
                side = utils.getOppositeSideStr(tremSideCur)
                intervalStats = gv.glob_stats_perint_nms[rawname]

            intervals = timeIntervalPerRaw_processed[rawname] [ side ]
            if len(intervals) != len(intervalStats):
                return True

        return False

    doStatRecalc = 0   #var for technical use
    rawsNeedStatUpd = []
    try:
        gv.glob_stats.keys()
    except (NameError, AttributeError):
        if load_stats:
            try:
                gv.glob_stats_perint = {}
                gv.glob_stats_perint_nms = {}
                gv.glob_stats = {}
                timeIntervalPerRaw_processed = {}
                for k in raws:
                    sind_str,medcond,task = getParamsFromRawname(k)
                    fn = stats_basefname + '_{}.npz'.format(k)
                    fn = os.path.join(data_dir, fn)
                    f = np.load(fn, allow_pickle=1)

                    assert f['rawname'] == k
                    gv.glob_stats_perint[k] = f['glob_stats_perint'][()]
                    gv.glob_stats_perint_nms[k] = f['glob_stats_perint_nms'][()]
                    timeIntervalPerRaw_processed[k] = f['timeIntervalPerRaw_processed'][()]
                    if sind_str not in gv.glob_stats:
                        gv.glob_stats[sind_str] = {}
                    gs_s = gv.glob_stats[sind_str]
                    if medcond not in gs_s:
                        gs_s[medcond] = {}
                    gs_sm = gs_s[medcond]
                    gs_sm[task] = f['glob_stats'][()]

                    if checkIntStatUpdateNeeded(k):
                        rawsNeedStatUpd += [k]
                        doStatRecalc = 1

                print('----- Existing stats loaded!')
            except FileNotFoundError as e:
                print('FileNotFoundError while trying to load stats, recomputing {}'.format(str(e) ) )
                doStatRecalc = 1
                gv.glob_stats = {}
                gv.glob_stats_perint = None
                gv.glob_stats_perint_nms = None
                del timeIntervalPerRaw_processed

            #f = np.load(stats_fname, allow_pickle=1)
            #glob_stats = f['glob_stats'][()]
            #glob_stats_perint = f['glob_stats_perint'][()]
        else:
            doStatRecalc = 1

    else:
        print('----- Using previously computed stats!')

    if doStatRecalc:
        if len(gv.specgrams) == 0:
            raise ValueError('We need to update stats, but not specgrams were loaded!')
            sys.exit(1)
        print('----- Computing new stats!')
        gv.glob_stats = getStatPerChan(time_start_forstats,
                time_end_forstats)  # subj,channame, keys [! NO rawname!]


    ##############   find tremor intercals in all subjects and raws 
    #if False:
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
 



    plotTremNegOffset = 2.
    plotTremPosOffset = 2.
    maxPlotLen = 6   # for those interval that are made for plotting, not touching intervals for stats
    addIntLenStat = 5
    longToLeft = True
    if timeIntervalsFromTremor:
        mvtTypes = ['tremor', 'no_tremor', 'unk_activity']
        timeIntervalPerRaw_processed = utils.processJanIntervals( tremIntervalJan, 
                maxPlotLen, addIntLenStat, plotTremNegOffset, plotTremPosOffset, plot_time_end, mvtTypes=mvtTypes)
        # [rawname][interval index]
    else:
        timeIntervalPerRaw_processed = {}
        for k in raws:
            ips = {}
            for side in ['left', 'right' ]:
                ips[side] = [ (plot_time_start, plot_time_end, 'entire') ]  
            timeIntervalPerRaw_processed[k] = ips
 
    #print( timeIntervalPerRaw_processed )
    #import pdb; pdb.set_trace()


    try:
        gv.glob_stats_perint.keys()
    except (NameError, AttributeError):
        print('Computing interval stats!')
        for k in raws:
            for side in ['left', 'right' ]:
                ivals0 = timeIntervalPerRaw_processed[k][side]
                ivals = utils.removeBadIntervals(ivals0)
                lendif = len(ivals0) - len(ivals)
                if lendif > 0:
                    print('{}, {}:  {} interavls were removed!'.format(k,side,lendif) )
                timeIntervalPerRaw_processed[ k ][side] =ivals  # maybe not necessary but just in case

        gv.glob_stats_perint, gv.glob_stats_perint_nms = getStatsFromTremIntervals(timeIntervalPerRaw_processed)
    else:
        print('----- Using previously computed interval stats!')
    # [raw] -- list of 3-tuples

    # I want to make a copy where pre and post are not present (for plotting), but gather stats for all intervals
    plot_timeIntervalPerRaw = {}  # dict of dicts of lists of 3-tuples
    if timeIntervalsFromTremor and not singleRawMode:
        itypesExcludeFromPlotting =  ['pre', 'post', 'initseg', 'endseg', 'middle_full', 'entire']
        itypesExcludeFromPlotting =  ['initseg', 'endseg', 'middle_full', 'entire']
        for k in timeIntervalPerRaw_processed:
            plot_timeIntervalPerRaw[k] = {}
            for side in ['left', 'right']:
                intervals = timeIntervalPerRaw_processed[k][side]
                res = []
                for interval in intervals:
                    t1,t2,intType = interval
                    if intType in itypesExcludeFromPlotting:
                        continue
                    res += [interval]
                plot_timeIntervalPerRaw[k][side] = res
    else:
        for k in raws:
            plot_timeIntervalPerRaw[k] = {}
            for side in ['left', 'right']:
                plot_timeIntervalPerRaw[k][side] = [ (plot_time_start, plot_time_end, 'entire') ]

    if save_stats:
        for k in gv.glob_stats_perint:
            sind_str,medcond,task = getParamsFromRawname(k)
            fn = stats_basefname + '_{}.npz'.format(k)
            fn = os.path.join(data_dir, fn)
            if not save_stats_onlyIfNotExists or  (not os.path.exists(fn) ):
                gs = gv.glob_stats[sind_str][medcond][task]
                gspi = gv.glob_stats_perint[k]
                gspi_nms = gv.glob_stats_perint_nms[k]
                tirp = timeIntervalPerRaw_processed[k]

                np.savez(fn , glob_stats_perint=gspi, glob_stats_perint_nms=gspi_nms,
                    glob_stats=gs, timeIntervalPerRaw_processed=tirp, rawname=k)

            #np.savez(stats_fname , glob_stats_perint=glob_stats_perint, 
            #        glob_stats=glob_stats, timeIntervalPerRaw_processed=timeIntervalPerRaw_processed)

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
        if not plot_onlyMultiBarplot:
            plotSpectralData(plt,time_start=plot_time_start,time_end=plot_time_end, 
                    chanTypes_toshow = chanTypes_toshow ) 
        else:
            plot_multibar_modalities = ['LFP', 'MEGsrc' ]
            #cleanIntervals( )
            plotBarplotTable(plot_multibar_modalities, onlyMainSide = plot_onlyMainSide)


