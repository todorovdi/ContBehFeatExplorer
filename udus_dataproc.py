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


def readSrcrec(fname):
    f = loadmat(fname)
    srcdynall = f['source_data']['avg']['mom']
    srcdynone = srcdynall[i][0,:]


def prepareSpec(pp):
    '''
    # proxy to be used in parallel computation
    '''
    i,chdata,sampleFreq,NFFT,noverlap = pp
    #specgram_scaling = 'density'   # divide over freq
    freqs, bins, Sxx = spectrogram_lspopt(chdata, fs=sampleFreq, c_parameter=c_parameter,
                                     nperseg=NFFT, scaling=specgram_scaling, 
                                      noverlap=noverlap) 
    return i,freqs,bins,Sxx

def getAllSpecgrams(chdata,sampleFreq,NFFT,specgramoverlap):

    '''
    get spectrograms of the given chdata
    '''
    #tetmp = min(te+NFFT,int(maxtime*sampleFreq) )
    pars = [(i,chdata[i,:],sampleFreq,NFFT,int(NFFT*specgramoverlap)) for i in range(chdata.shape[0])]

    p = mpr.Pool(chdata.shape[0])
    r = p.map(prepareSpec, pars)
    p.close()
    p.join()

    specgramsComputed = [0]*len(r)
    for pp in r:
        i,freqs,bins,Sxx = pp
        specgramsComputed[i] = (freqs,bins,Sxx)
        
    return specgramsComputed


def precomputeSpecgrams(raws,ks=None,NFFT=256, specgramoverlap=0.75,forceRecalc=True, 
       modalities = None ):
    '''
    packs the computed spectrograms in a structure
    '''
    if modalities is None:
        modalities = ['LFP', 'EMG', 'EOG', 'MEGsrc']

    if ks is None:
        ks = raws.keys()
        
    print('Starting spectrograms computation')
    specgrams = {}
    for k in ks:
        raw = raws[k]
        
        #if (not forceRecalc) and ('specgram' in f.keys()):
        #    continue
        #sampleFreq = int( raw.info['sfreq'] )  # use global instead now

        assert NFFT <= sampleFreq 

        chnames = raw.info['ch_names']
        orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)

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
            chdata, chtimes = getData(k,chns,ts,te)
            specgramsComputed = getAllSpecgrams(chdata,sampleFreq,NFFT,specgramoverlap)
        
            for i,chn in enumerate(chns):
                specdict[chn] = specgramsComputed[i]
        #raw['specgram'] = specdict
        specgrams[k] = specdict
        
    print('Spectrograms computation finished')
    return specgrams
        

def getData(rawname,chns,ts=None, te=None):
    '''
    ts,te -- timebins
    '''
    assert isinstance(chns,list)
    r = []
    curraw = raws[rawname]

    MEGsrc_inds = []
    nonMEGsrc_inds = []
    chdatas = []
    times = None
    for i,chn in enumerate(chns):
        if chn.find('MEGsrc') >= 0:
            MEGsrc_inds += [i]
            chdata, times = MEGsrcChname2data(rawname,chn,ts=ts,te=te,rettimes=True)  # we want to replace times, since we assume equal sampling for all channels and all sources
        else:
            nonMEGsrc_inds += [i]
            chdata, chtimes = curraw [chn,ts:te]
            times = chtimes

        chdatas += [chdata]

    return np.vstack(chdatas), times

########  Gen stats across conditions
def getStatPerChan(time_start,time_end,freq_min,freq_max,modalities=None):
    if modalities is None:
        modalities = ['LFP', 'EMG', 'EOG', 'MEGsrc']

    glob_stats = {}
    for subj in subjs_analyzed:   # over subjects 
        stat_persubj = {}
        #raws_from_subj = subjs_analyzed[subj]['rawnames']
        for medcond in subjs_analyzed[subj]['medconds'] :
            stat_permed_perchan = {}
            raws_from_subj = subjs_analyzed[subj][medcond]
            for k in raws_from_subj.values():   # over raws
                sp = specgrams[k]  # spectrogram from a given raw file

                raw = raws[k]
                #chnames = list( sp.keys() )
                chnames = raw.info['ch_names'] 
                orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)
                #chnames2 = []
                for side_ind in range(len(orderEMG) ): 
                    #chnames2 = chnames_tuples[side_ind]['LFP'] + chnames_tuples[side_ind]['EMG']
                    chnames2 = []
                    for modality in modalities:
                        chnames2 += chnames_tuples[side_ind][modality]
                    #chnamesEMG = chnames_tuples[side_ind]['EMG']

                    #chis = mne.pick_channels(chnames,include=chnames2, ordered=True )
                    #ts,te = raw.time_as_index([time_start, time_end])
                    #chdata, chtimes = raw[chis,ts:te]
                    chdata, chtimes = getData(k, chnames2 )

                    specsTremorEMGcurSide = []
                    for chii,chn in enumerate(chnames2):
                        f,b,spdata = sp[chn]

                        tremDetTimeEnd = nonTaskTimeEnd
                        #if k.find('rest'):
                        ft,bt,spdatat = getSubspec(f,b,spdata,tremorBandStart, tremorBandEnd, 
                                0, tremDetTimeEnd)
                        assert spdatat.shape[0] > 0

                        st = {}

                        if chn.find('EMG') >= 0: 
                            specsTremorEMGcurSide += [ (chn, ft,bt, spdatat)  ]
                         
                            if tremrDet_clusterMultiFreq:
                                cnt = getDataClusters( spdatat ) 
                                st['tremorfreq_clusters'] = cnt
                            else:
                                thrs = [0] * len(ft)
                                for freqi in range( len(ft) ):  # separately per frequency subband
                                    #thr = calcSpecThr( spdatat[freqi,:]  )
                                    cnt = getDataClusters( spdatat[freqi,:] ) 
                                    thr = np.mean(cnt)

                                    m = tremorThrMult.get(subj,1.)  # maybe we have customized it a bit
                                    thr *= m
                                    thrs[freqi] = thr

                                st['thrPerFreq_trem'] = thrs

                        # get subspdata
                        st['max_spec'] = 0
                        st['min_spec'] = 1e8
                        st['mean_spec'] = 0

                        st['max_spec_trem'] = 0
                        st['min_spec_trem'] = 1e8
                        st['mean_spec_trem'] = 0

                        chdat = chdata[chii,:]
                        st['max']  = -1e8
                        st['min']  = 1e8
                        st['mean'] = 0
                        if chn in stat_permed_perchan:
                            st['max_spec'] = max( st['max_spec'], np.max(spdata))
                            st['min_spec'] = min( st['min_spec'], np.min(spdata))
                            st['mean_spec'] += np.mean(spdata) / len(raws_from_subj)

                            st['max_spec_trem'] = max( st['max_spec_trem'], np.max(spdatat))
                            st['min_spec_trem'] = min( st['min_spec_trem'], np.min(spdatat))
                            st['mean_spec_trem'] += np.mean(spdatat) / len(raws_from_subj)

                            st['max'] = max( st['max'], np.max(chdat))
                            st['min'] = min( st['min'], np.min(chdat))
                            st['mean'] += np.mean(chdat) / len(raws_from_subj)
                        else:
                            st['max_spec'] = np.max( spdata)
                            st['min_spec'] = np.min( spdata)
                            st['mean_spec'] = np.mean(spdata) / len(raws_from_subj)

                            st['max_spec_trem'] = np.max(spdatat)
                            st['min_spec_trem'] = np.min(spdatat)
                            st['mean_spec_trem'] = np.mean(spdatat) / len(raws_from_subj)

                            st['max']  = np.max( chdat)
                            st['min']  = np.min( chdat)
                            st['mean'] = np.mean(chdat) / len(raws_from_subj)

                            #if chn == 'EMG063_old':
                            #    import pdb; pdb.set_trace()

                        stat_permed_perchan[chn] = st

                
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
                        cnt = getDataClusters( rr )   #cnt is numcluster x dim
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
                            cntcur = getDataClusters( rr[i,:] )  
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

                            plt.savefig('Scatters_{}_{}.pdf'.format( k,orderEMG[side_ind]))
                            plt.close()

                        ythr = ythr_sep
                        yclust = yclust_sep
                        if 'thrPerCh_trem_allEMG' in stat_permed_perchan:
                            stat_permed_perchan['thrPerCh_trem_allEMG'].update(  ythr  )
                        else:
                            stat_permed_perchan['thrPerCh_trem_allEMG'] =   ythr   

                        if 'tremorfreq_clusters_allEMG' in stat_permed_perchan:
                            stat_permed_perchan['tremorfreq_clusters_allEMG'].update(  yclust )
                        else:
                            stat_permed_perchan['tremorfreq_clusters_allEMG'] = yclust 
                # end of cycle over channels
            # end of cycle over rawnames with the same medication state
            stat_persubj[medcond] = stat_permed_perchan
                    
        glob_stats[subj] = stat_persubj
        #glob_stats[subj] = stat_perchan
    return glob_stats
        
def getSubspec(freqs,bins,Sxx,freq_min,freq_max,time_start=None,time_end=None):
    #bininds = np.where( np.logical_and( bins >= time_start , bins < tetmp/sampleFreq) )[0]
    if time_start is not None or time_end is not None:
        if time_start is None:
            time_start = np.min(bins)
        if time_end is None:
            time_end = np.max(bins)
        bininds = np.where( np.logical_and( bins >= time_start , bins <= time_end) )[0]
        bins = bins[bininds]
        Sxx = Sxx[:,bininds]

    freqinds = np.where( np.logical_and(freqs >= freq_min,freqs <= freq_max) )[0]
    freqmini = freqinds[0]
    freqmaxi = freqinds[-1]
        
#     norm = None
#     if chname.find('LFP') >= 0:
#         freqinds = np.where( np.logical_and(freqs >= plot_minFreqInSpec,freqs < plot_maxFreqInSpec) )[0]
#         freqmini = freqinds[0]
#         freqmaxi = freqinds[-1]
#     else:
#         freqmini = 0
#         freqmaxi = len(freqs)

    #if len(freqinds) == 1:
    #else:
    #    freqs = freqs[freqmini:freqmaxi]
    #    Sxx = Sxx[freqmini:freqmaxi,:]
    freqs = freqs[freqinds]
    Sxx = Sxx[freqinds,:]
    
    return freqs,bins,Sxx

def getIntervals(bininds,width=100,thr=0.1, percentthr=0.8,inc=5, minlen=50, 
        extFactor = 0.25, endbin = None):
    '''
    tremini -- indices of timebins, where tremor was detected
    thr -- thershold for convolution to be larger then, for L\infty-normalized data
    output -- convolution, intervals (pairs of timebin indices)
    inc -- how much we extend the interval each time (larger means larger intervals, but worse boundaries)
    minlen -- minimum number of bins required to make the interval be included 
    percentthr -- min ratio of thr crossings within the window to continue extending the interval
    extFactor -- we'll extend found intervals by width * extFactor
    endbin -- max timebin
    '''
    if endbin is None:
        mt = np.max (bininds ) + 1
    else:
        mt = endbin
    raster = np.zeros( mt, dtype=np.int )
    raster[bininds] = 1
    #width = 100
    avflt = np.ones(width) #box filter
    #avflt = sig.gaussian(width, width/4)
    avflt /= np.sum(avflt) 
    cvl = np.convolve(raster,avflt,mode='same')
    
    #cvlskip = cvl[::skip]
    cvlskip = cvl
    thrcross = np.where( cvlskip > thr )[0]
    belowthr = np.where( cvlskip <= thr )[0]
    shift = int(width * extFactor )
    
    pairs = []
    gi = 0
    rightEnd = 0  #right end
    leftEnd = 0  #left end
    while gi < len(thrcross):
        #leftEnd = thrcross[gi]   # index in cvlskip arary
        searchres = np.where(np.logical_and(belowthr > leftEnd, belowthr > rightEnd))[0]  # find first failing
        if len(searchres) == 0:    # if we didn't find furhter negatives, perhaps we are at the end already and all the end is positive
            rightEnd = thrcross[-1]
        else:
            di = searchres[0]
            rightEnd = belowthr[di]
        subcvl = cvlskip[leftEnd:rightEnd]
        while rightEnd < len(cvlskip) + inc + 1: # try to enlarge until possible
            val = np.sum(subcvl > thr) / len(subcvl)
            if (rightEnd + inc) >= len(cvlskip) and (val > percentthr):
                rightEnd = len(cvlskip)
                subcvl = cvlskip[leftEnd:rightEnd]
                break

            if (rightEnd + inc) < len(cvlskip) and (val > percentthr):
                rightEnd += inc
                subcvl = cvlskip[leftEnd:rightEnd]
            else:
                break
            
        if rightEnd-leftEnd >= minlen:  
            newp = (  max(0, leftEnd-shift), max(0, rightEnd-shift) ) # extend the interval on both sides 
            pairs += [newp ]

        assert leftEnd < rightEnd
                
        searchres = np.where(thrcross > rightEnd)[0]  # _ind_ of next ind of thr crossing after end of current interval
        if len(searchres) == 0:
            break
        else:
            gi = searchres[0]
        leftEnd = thrcross[gi]  # ind of thr crossing after end of current interval
    
    return cvlskip,pairs

def getDataClusters(chdata, n_clusters=2,n_jobs=-1):
    '''
    arguments  dim x time   array
    returns    dim x cluster number
    '''
    #assert (chdata.ndim == 1) or (chdata.ndim == 2 and chdata.shape[0] == 1), 'too many dims, want just one'
    import sklearn.cluster as skc

    augment = 0
    if (chdata.ndim == 1) or (chdata.ndim == 2 and chdata.shape[0] == 1):
        x = chdata.flatten()
        X = np.array( list(zip(x,np.zeros(len(x))) ) )   # cannot cluster 1D
        augment = 1
    else:
        assert chdata.ndim == 2
        X = chdata.T
    kmeans = skc.KMeans(n_clusters=n_clusters, n_jobs=n_jobs).fit( X )

    if augment:
        cnt = kmeans.cluster_centers_[:,0]
    else:
        cnt = kmeans.cluster_centers_
        assert cnt.shape[0] == 2 and cnt.shape[1] == chdata.shape[0]
    return cnt

def calcSpecThr(chdata):
    cnt = getDataClusters(chdata)
    thr = np.mean(cnt)
    return thr

def getParamsFromRawname(rawname):
    r = re.match( "(S\d{2})_(.+)_(.+)*", rawname ).groups()
    subjstr = r[0]
    medcond = r[1]
    task = r[2]

    return subjstr,medcond,task

def getParamsFromSrcname(srcname):
    r = re.match( "srcd_(S\d{2})_(.+)_(.+)_(.+)", rawname ).groups()
    subjstr = r[0]
    medcond = r[1]
    task = r[2]
    roi = r[3]

    return subjstr,medcond,task,roi

def getRawname(subj,medcond,task):
    #fn = 'S{:02d}_{}_{}'.format(subjstr,medcond,task)
    if isinstance(subj,str):
        fn = '{}_{}_{}'.format(subj,medcond,task)
    elif isinstance(subj,int): 
        fn = 'S{:02d}_{}_{}'.format(subj,medcond,task)
    else:
        raise ValueError('Wrong subj type')
    return fn

def getSrcname(subj,medcond,task,roi):
    #fn = 'S{:02d}_{}_{}'.format(subjstr,medcond,task)
    if isinstance(subj,str):
        fn = 'srcd_{}_{}_{}_{}'.format(subj,medcond,task,roi)
    elif isinstance(subj,int): 
        fn = 'srcd_S{:02d}_{}_{}_{}'.format(subj,medcond,task,roi)
    else:
        raise ValueError('Wrong subj type')
    return fn



def findTremor(k,thrSpec = None, thrInt=0.13, width=40, percentthr=0.8, inc=1, minlen=50, extFactor=0.25):
    '''
    k is raw name
    output -- per raw, per side,  couples of start, end
    percentthr -- minimum percantage of time the events should happen within the interval for it to be considered continous
    '''
    sind_str,medcond,task = getParamsFromRawname(k)

    #gen_subj_info[sind_str]['tremor_side']
    #gen_subj_info[sind_str]['tremfreq']
    
    chnames = raws[k].info['ch_names'] 
    orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)
    
    tremorIntervals = {}
    for pair_ind in range(len(orderEMG)):
        side = orderEMG[pair_ind]
        chns = chnames_tuples[pair_ind]['EMG']
        sideinfo = {}
        for chn in chns:
            freq, bins, Sxx = specgrams[k][chn]
            freq,bins,bandspec = getSubspec(freq,bins,Sxx, tremorBandStart, tremorBandEnd)
            #freqinds = np.where( 
            #    np.logical_and(freq >= tremorBandStart,freq <= tremorBandEnd))
            #bandspec = Sxx[freqinds,:]
            if thrSpec is None:
                if tremorDetectUseCustomThr and 'tremorDetect_customThr' in gen_subj_info[sind_str]:
                    thr = gen_subj_info[sind_str]['tremorDetect_customThr'][medcond][chn]
                    logcond = np.sum( bandspec, axis=0 ) > thr # sum over freq
                    pp = np.where( logcond )
                    trembini = pp[0]
                else:
                    if tremrDet_clusterMultiMEG:
                        # thresholds per channel
                        thr = glob_stats[sind_str][medcond]['thrPerCh_trem_allEMG'][chn]  # computed using k-means
                        logcond = np.sum( bandspec, axis=0 ) > thr # sum over freq
                        pp = np.where( logcond )
                        #import pdb; pdb.set_trace()
                        trembini = pp[0]
                    else:
                        thrPerFreq = glob_stats[sind_str][medcond][chn]['thrPerFreq_trem']  # computed using k-means
                        assert len(thrPerFreq) == len(freq), '{}, {}'.format(thrPerFreq, freq)
                        logcond = bandspec[0,:] > thrPerFreq[0] 
                        for freqi in range(len(freq)): # indices of indices
                            logcond = np.logical_or(logcond, bandspec[freqi,:] > thrPerFreq[freqi]  )
                        pp = np.where( logcond )
                        trembini = pp[0]
            else:
                logcond = bandspec > thrSpec
                #me = np.mean(bandspec)
                #thrSpec = me/2
                pp = np.where( logcond )
            
                trembini = pp[2]  # indices of bins, regardless of power of which freq subband became higher
            pairs = []
            if len(trembini) > 0:
                cvlskip,pairs = getIntervals(trembini, width, thrInt,
                        percentthr, inc,  minlen, extFactor, endbin=len(bins) )
                
                pairs2 = [ (bins[a],bins[b]) for a,b in pairs    ] 
#                 for pa in pairs:
#                     a,b = pa
#                     a = bins[a]
#                     b = bins[b]
            else:
                pairs2 = []

            #else:
            #    print pp
            
            sideinfo[chn] = pairs2
            
        tremorIntervals[orderEMG[pair_ind]] = sideinfo
            
    return tremorIntervals
            #print(Sxx.shape, freqinds)
            #return yy
    #assert chdata.ndim == 1
    

def findAllTremor(thrSpec = None, thr=0.13, width=40, percentthr=0.8, inc=1, minlen=50, extFactor=0.25):
    tremPerRaw = {}
    for k in specgrams:
        tremIntervals = findTremor(k, thrSpec, thr, width, percentthr, inc)
        tremPerRaw[k] = tremIntervals
        
    return tremPerRaw

def mergeTremorIntervals(intervals, mode='intersect'):
    '''
    mode -- 'intersect' or 'join' -- how to deal with intervals coming from differnt muscles 
    '''
    newintervals = {}
    for k in intervals:
        ips = intervals[k]
        newips = {}
        for side in ips:

            intcur = ips[side]
            assert len(intcur) == 2
            chns = list( intcur.keys() )
            pairs1 = intcur[chns[0] ]
            pairs2 = intcur[chns[1] ]

            mask2 = np.ones(len(pairs2) )
            
            # currently tremor that appear only in one MEG, gets ignored (otherwise I'd have to add non-merged to the list as well)
            resp = []
            #for i1, (a1,b1) in enumerate(pairs):
            #    for i2,(a2,b2) in enumerate(pairs):
            for i1, (a1,b1) in enumerate(pairs1):
                mergewas = 0
                for i2,(a2,b2) in enumerate(pairs2):
                    #if i1 == i2 or mask[i2] == 0 or mask[i1] == 0:
                    #    continue
                    if mask2[i2] == 0:
                        continue

                    # if one of the end is inside the other interval
                    if (b1 <=  b2 and b1 >= a2) or ( b2 <= b1 and b2 >= a1 ) :
                        if mode == 'intersect':
                            newp = (max(a1,a2), min(b1,b2) )
                        else:
                            newp = (min(a1,a2), max(b1,b2) )
                        resp += [ newp ]
                        #mask[i1] = 0
                        #mask[i2] = 0  # we mark those who participated in merging
                        mask2[i2] = 0
                        mergewas = 1
                        break

                #resp += [ p for i,p in enumerate(pairs) if mask[i] == 1 ]  # add non-merged


            if mode == 'join':
                # now merging intersecting things that could have arised from joining
                pairs = resp 
                mask = np.ones(len(pairs) )
                resp2 = []
                for i1, (a1,b1) in enumerate(pairs):
                    for i2,(a2,b2) in enumerate(pairs):
                        if i1 == i2 or mask[i2] == 0 or mask[i1] == 0:
                            continue
                        if (b1 <=  b2 and b1 >= a2) or ( b2 <= b1 and b2 >= a1 ) :
                            resp2 += [ (min(a1,a2), max(b1,b2) ) ]
                            mask[i1] = 0
                            mask[i2] = 0  # we mark those who participated in merging
                            break
                resp2 += [ p for i,p in enumerate(pairs) if mask[i] == 1 ]  # add non-merged

                newips[side] = resp2
            else:
                newips[side] = resp
        newintervals[k] = newips
    return newintervals


            
def filterFavChnames(chnames,subj):
    '''
    input list of strings
    subj 'S??'
    '''
    if plot_LFP_onlyFavoriteChannels:
        # get intersection
        favchn = gen_subj_info[subj]['favoriteLFPch']
        chnames_new = []
        for chname in chnames:
            if chname.find('LFP') >= 0:
                if chname in favchn:
                    chnames_new += [chname]
            else:
                chnames_new += [chname]
        #lfpnames = list( set(lfpnames) & set( gen_subj_info[subj]['favoriteLFPch'] ) )
        #ch_toplot_timecourse += lfpnames
    else:
        chnames_new = chnames

    return chnames_new
                    
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
                if MEGsrc_inds_toshow is None or ind in MEGsrc_inds_toshow:
                    chn = 'MEGsrc_{}_{}'.format(roistr,ind)
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


    chnames_tuples = []
    chinds_tuples = []
    for sidei in range(len(order_by_EMGside)):
        side = order_by_EMGside[sidei]
        revside = order_by_EMGside[1-sidei]
        chnames_tuples += [{'EMG':EMGnames[side], 
            'LFP':LFPnames[revside], 'MEGsrc':MEGsrc_names[revside], 'EOG':eog_names }]
        chinds_tuples += [{'EMG':EMGinds[side],  
            'LFP':LFPinds[revside], 'MEGsrc':MEGsrc_inds[revside], 'EOG':eog_inds }]
        # both EOGs goe to both sides
    
#     chnames_tuples += [{'EMG':EMGnames['right'], 'LFP':LFPnames['left'] }]
#     chinds_tuples += [{'EMG':EMGinds['right'], 'LFP':LFPinds['left'] }]
    
    return order_by_EMGside, chinds_tuples, chnames_tuples, LFPnames, EMGnames, MEGsrc_names

def getTuples(sind_str):
    cht = gen_subj_info[sind_str]['chantuples']
    sides = []
    indtuples = []
    nametuples = []
    for side in cht:
        sides += [side]
        indtuples +=  cht[side]['indtuples']
        nametuples +=  cht[side]['nametuples']

    return sides, indtuples, nametuples

def getChannelSide(chname):
    # MEG meaning of sides
    sides = subjinfo['LFPnames_perSide'].keys()
    for side in sides:
        if chname in subjinfo['LFPnames_perSide'][side]:
            return side
        if chname in subjinfo['EMGnames_perSide'][side]:
            return side
        if chname in subjinfo['MEGsrcnames_perSide'][side]:
            return side
    return -1

def parseMEGsrcChname(chn):
    m = re.match('MEGsrc_(.+)_(.+)', chn)
    roistr = m.groups()[0]
    srci = int(m.groups()[1])
    return roistr,srci

def MEGsrcChname2data(rawname,chns,ts=None,te=None,rettimes=False):
    '''
    ts,te -- bindins
    '''
    if isinstance(chns,str):
        chns = [chns]

    f = None
    srcvals = []
    for i,chn in enumerate(chns):  
        #m = re.match('MEGsrc_(.+)_(.+)', chn)
        #roistr = m.groups()[0]
        #srci = int(m.groups()[1])
        roistr,srci = parseMEGsrcChname(chn)

        sind_str,medcond,task = getParamsFromRawname(rawname)
        srcname = 'srcd_{}_{}_{}_{}'.format(sind_str,medcond,task,roistr)
        f = srcs[srcname]
        src = f['source_data'] 
        ref = src['avg']['mom']     
        # maybe I can make more efficient data extraction of multiple channels at once 
        # later if needed, if I bundle all channels from the same source file 
        srcval = f[ref[0,srci] ][ts:te,0]   # 1D array with reconstructed source activity
        srcvals += [srcval]
    r = np.vstack(srcvals)

    if rettimes:
        times = f['source_data']['time'][ts:te,0]
        return r, times
    else:
        return r

def MEGsrcChind2data(rawname,chi):
    indshift = 0
    for roistr in sorted(MEGsrc_roi):  # very important to use sorted!
        sind_str,medcond,task = getParamsFromRawname(rawname)
        srcname = 'srcd_{}_{}_{}_{}'.format(sind_str,medcond,task,roistr)
        f = srcs[srcname]
        src = f['source_data'] 
        nsrc = src['avg']['mom'].shape[1]
        if chi < indshift + nsrc:
            srci = chi - indshift
            src = f['source_data'] 
            ref = src['avg']['mom']
            srcval = f[ref[0,srci] ][:,0]   # 1D array with reconstructed source activity
            return srcval
        else:
            indshift += nsrc

    return None

def plotSpectralData(plt,time_start = 0,time_end = 400, chanTypes_toshow = None, onlyTremor = False ):

    if chanTypes_toshow is None:
        #chanTypes_toshow = {'timecourse': ['EMG', 'LFP'], 'spectrogram': ['LFP'], 'bandpow': ['EMG', 'LFP'] }
        chanTypes_toshow = {} 
        if show_timecourse:
            chanTypes_toshow[ 'timecourse'] = {'chantypes':['EMG', 'LFP', 'MEGsrc', 'EOG'] }
        if show_bandpow:
            chanTypes_toshow[ 'bandpow']    = {'chantypes': ['EMG', 'LFP', 'MEGsrc' ] }
        if show_spec:
            chanTypes_toshow[ 'spectrogram'] = {'chantypes': ['LFP']    }
            if plot_EMG_spectrogram:
                chanTypes_toshow[ 'spectrogram']['chantypes'] += ['EMG']   
            if plot_MEGsrc_spectrogram:
                chanTypes_toshow[ 'spectrogram']['chantypes'] += ['MEGsrc']   


    
    
    #ks = sorted( ks[0:12] )

    #time_start,time_end = 0,150
    #time_start,time_end = 0,250

    mainSideColor = 'w'; otherSideColor = 'lightgrey'
    #normType='unifnorm' # or 'log'
    normType='log' # or 'log'

    ndatasets = len(raws)
    #nffts = [512,1024]
    tremor_intervals_use_merged = 1
    tremor_intervals_use_merged = 0

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

    nchans_perModality = {}
    nspec_EMGchans = 2 # per side
    #nLFPchans = 7 # max, per side
    if plot_LFP_onlyFavoriteChannels:
        nchans_perModality['LFP'] = 1
        nspec_LFPchans = 1
    else:
        nchans_perModality['LFP'] = 3
        nspec_LFPchans = 3 # max, per side

    nchans_perModality['EMG'] = 2
    nchans_perModality['EOG'] = 2

    # MEGsrcnames = list( gen_subj_info.values() )[0 ] ['MEGsrcnames_perSide']
    # assume that already taken into account MEGsrc_inds_toshow
    MEGsrcnames_subj0 = list( gen_subj_info.values() )[0] ['MEGsrcnames_perSide']
    n = max( len( MEGsrcnames_subj0['left'] ), len( MEGsrcnames_subj0['right'] )  )  # since we can have different sides for different subjects
    nchans_perModality['MEGsrc'] = n
    
    #nr_bandpow = 2

    nspect_per_pair = 0
    pt = chanTypes_toshow['spectrogram']
    for modality in pt['chantypes']:   
        nspect_per_pair += nchans_perModality[modality]  # 1 row per spectrogram

    chanTypes_toshow['spectrogram']['nplots'] = nspect_per_pair
    chanTypes_toshow['bandpow']['nplots'] = len(chanTypes_toshow['bandpow']['chantypes'] )
    if 'MEGsrc' in chanTypes_toshow['bandpow']['chantypes']:
        if plot_MEGsrc_rowPerBand:
            chanTypes_toshow['bandpow']['nplots'] += (-1) + len( plot_freqBandNames_perModality['MEGsrc'] )
        if plot_MEGsrc_separateRoi:
            chanTypes_toshow['bandpow']['nplots'] *= len(MEGsrc_roi)


    chanTypes_toshow['timecourse']['nplots'] = len(chanTypes_toshow['timecourse']['chantypes'] )

    print(  chanTypes_toshow )
    #if 'EMG' in chanTypes_toshow['spectrogram']:
    #    nspect_per_pair +=  nspec_EMGchans
    #if 'LFP' in chanTypes_toshow['spectrogram']   
    #    nspect_per_pair += nspec_LFPchans
    #if 'EMG' in chanTypes_toshow['bandpow']:
    #    nr_bandpow +=  1
    #if 'LFP' in chanTypes_toshow['bandpow']   
    #    nr_bandpow += 1
    nplots_per_side = sum( chanTypes_toshow[pt]['nplots'] for pt in chanTypes_toshow )

    nc = len(ks)
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
        

    fig, axs = plt.subplots(ncols = nc, nrows=nr, figsize= (ww*nc,hh*nr), sharey='none')
    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
    colind = 0
    for colind in range(nc):
        k = ks[colind]

        
        sind_str,medcond,task = getParamsFromRawname(k)
        deftimeint =  [(time_start,time_end) ]
        time_start, time_end =  plot_timeIntervalsPerSubj.get( sind_str, deftimeint  )   [0]

        #chnames = raws[k].info['ch_names']
        orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)
        
        if tremor_intervals_use_merged:
            tremorIntervals = tremIntervalMergedPerRaw[k]
        else:
            tremorIntervals = tremIntervalPerRaw[k]


        if plot_onlyMainSide:
            pair_inds = [orderEMG.index( gen_subj_info[sind_str]['tremor_side'] ) ]
        for channel_pair_ind in pair_inds:
            if plot_onlyMainSide:
                channel_pair_ind_forAx = 0
            else:
                channel_pair_ind_forAx = channel_pair_ind
            side_body = orderEMG[channel_pair_ind]

            #inds = chinds_tuples[channel_pair_ind]['LFP'] + chinds_tuples[channel_pair_ind]['EMG']
            #ch_toplot = [chnames[i] for i in inds]
            #lfpnames = chnames_tuples[channel_pair_ind]['LFP']
            #ch_toplot_timecourse = []
            #if 'LFP' in chanTypes_toshow['timecourse']['chantypes']:
            #    if plot_LFP_onlyFavoriteChannels:
            #       # get intersection
            #       lfpnames = list( set(lfpnames) & set( gen_subj_info[subj]['favoriteLFPch'] ) )
            #       ch_toplot_timecourse += lfpnames
            #if 'EMG' in chanTypes_toshow['timecourse']['chantypes']:
            #    ch_toplot_timecourse += chnames_tuples[channel_pair_ind]['EMG']
            spcht = chanTypes_toshow['timecourse']['chantypes']

            ch_toplot_timecourse = []
            for modality in spcht:
                ch_toplot_timecourse += chnames_tuples[channel_pair_ind][modality]
            ch_toplot_timecourse = filterFavChnames( ch_toplot_timecourse, sind_str )
            

            ts,te = raws[k].time_as_index([time_start, time_end])

            #chs = mne.pick_channels(chnames,include=ch_toplot_timecourse, ordered=True )
            #chdata, chtimes = raws[k][chs,ts:te]
            
            chdata, chtimes = getData(k, ch_toplot_timecourse, ts,te )

            mintime = min(chtimes)
            maxtime = max(chtimes)
            
            tremorIntervalsCurSide = tremorIntervals[orderEMG[channel_pair_ind]]

            ################## plot timecourse
            for modalityi,modality in enumerate(chanTypes_toshow['timecourse']['chantypes'] ):
                ax = axs[nplots_per_side*channel_pair_ind_forAx + modalityi,colind]   # one row per modality

                addstr = ''
                for i in range(chdata.shape[0] ):
                    addstr = ''
                    #chn = chnames[chs[i]]
                    chn = ch_toplot_timecourse[i]
                    if chn.find(modality) < 0:
                        continue
                    mn = np.mean( chdata[i,:] )

                    st = glob_stats[sind_str][medcond][chn]
                    pars = plot_paramsPerModality[modality]

                    ys = chdata[i,:]
                    if pars['shiftMean']:
                        ys -= mn
                        addstr += ', meanshited'
                    if 'bandpass_freq' in pars:
                        bandpassFltorder =  pars['bandpass_order'] 
                        bandpassFreq = pars['bandpass_freq']
                        sos = sig.butter(bandpassFltorder, bandpassFreq, 
                                'bandpass', fs=sampleFreq, output='sos')
                        ys = sig.sosfilt(sos, ys)

                        addstr += ', bandpass {}Hz'.format(bandpassFreq)
                    else:
                        if 'highpass_freq' in pars:
                            highpassFltorder =  pars['highpass_order'] 
                            highpassFreq = pars['highpass_freq']
                            sos = sig.butter(highpassFltorder, highpassFreq, 
                                    'highpass', fs=sampleFreq, output='sos')
                            ys = sig.sosfilt(sos, ys)

                            addstr += ', highpass {}Hz'.format(highpassFreq)
                        if 'lowpass_freq' in pars:
                            lowpassFltorder =  pars['lowpass_order'] 
                            lowpassFreq = pars['lowpass_freq']
                            sos = sig.butter(lowpassFltorder, lowpassFreq, 
                                    'lowpass', fs=sampleFreq, output='sos')
                            ys = sig.sosfilt(sos, ys)
                            addstr += ', lowpass {}Hz'.format(lowpassFreq)

                    axLims = pars['axLims']
                    ymin, ymax = axLims.get(sind_str, (st['min'],st['max'] ) )
                    if pars['shiftMean']:
                        axLims_meanshifted = pars['axLims_meanshifted']
                        if sind_str in axLims_meanshifted: 
                            ymin, ymax = axLims_meanshifted.get(sind_str )
                        else:
                            ymin -= mn
                            ymax -= mn
                    #else: # because if we filter, we remove highest components
                    #    ymin, ymax = -ymaxs[modality],ymaxs[modality]

                    ax.plot(chtimes, ys, label = '{}'.format(chn), alpha=0.7 )
                    ax.set_ylim(ymin,ymax)
                    ax.set_xlim(mintime,maxtime)
                    
                    if modality == 'EMG' and isinstance(tremorIntervalsCurSide, dict):
                        pairs = tremorIntervalsCurSide[chn]
                        for pa in pairs:
                            ax.fill_between( list(pa) , ymin, ymax, facecolor='red', alpha=0.2)

                if modality == 'EMG'  and isinstance(tremorIntervalsCurSide, list):
                    pairs = tremorIntervalsCurSide
                    for pa in pairs:
                        ax.fill_between( list(pa) , ymin, ymax, facecolor='red', alpha=0.2)
                    

                ax.legend(loc=legendloc)
                ax.set_title('{}, {} hand: {}{}'.format(k,side_body,modality,addstr) )
                ax.set_xlabel('Time, [s]')
                #ax.set_ylabel('{} Voltage, mean subtracted{}'.format(modality,addstr))
                ax.set_ylabel('{} Voltage'.format(modality))
                
                if orderEMG[channel_pair_ind] == gen_subj_info[sind_str]['tremor_side']:
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
            if orderEMG[channel_pair_ind] == gen_subj_info[sind_str]['tremor_side']:
                ax.patch.set_facecolor(mainSideColor)
            else:
                ax.patch.set_facecolor(otherSideColor)

            tetmp = min(te+NFFT,int(maxtime*sampleFreq) )

            ################  plot freq bands ################################

            if show_bandpow and ftc_instead_psd:
                print('Starting plotting frequency bands')
                specgramsComputed = specgrams[k]
                spcht = chanTypes_toshow['bandpow']['chantypes']
                for modalityi,modality in enumerate(spcht):
                    #ch_toplot_bandpow = []
                    #for modality in spcht:
                    #    ch_toplot_bandpow += chnames_tuples[channel_pair_ind][modality]
                    ch_toplot_bandpow = chnames_tuples[channel_pair_ind][modality] 

                    colorEMGind = 0
                    maxpow = 0
                    for chn in ch_toplot_bandpow:
                        freqs, bins, Sxx = specgramsComputed[chn]
                        if specgram_scaling == 'psd':
                            freqres = freqs[1]-freqs[0]
                        else:
                            freqres = 1.
                        assert chn.find(modality) >= 0 

                        #if chn.find(modality) < 0:
                        #    continue
                        # by default we show all interesting frequency bands
                        freqBands_names = plot_freqBandNames_perModality[modality]
                        # although for EMG only tremor band
                        for fbi,fbname in enumerate(freqBands_names):
                            axcoordy = modalityi
                            title = '{}, Freq bands powers'.format(modality)
                            if modality.find( 'MEGsrc') >= 0 :
                                if plot_MEGsrc_rowPerBand:   # assume MEGsrc are always the last to plot 
                                    axcoordy += fbi
                                    title = '{}, {} band power'.format(modality,fbname)
                                #getSrcname
                                if plot_MEGsrc_separateRoi:
                                    curroi,srci = parseMEGsrcChname(chn)
                                    axcoordy += MEGsrc_roi.index(curroi)     # assume MEGsrc are always the last to plot

                            #print('{} ,  {}'.format(chn, axcoordy ) )
                            ax = axs[rowind_shift + axcoordy,colind]

                            fbs,fbe = freqBands[fbname]
                            freqs_b, bins_b, Sxx_b = getSubspec(freqs,bins,Sxx, fbs,fbe, 
                                    time_start,tetmp/sampleFreq)
                            bandpower = np.sum(Sxx_b,axis=0) * freqres

                            color = None
                            if modality.find('EMG') >= 0:      # use fixed colors for EMG
                                color = plot_colorsEMG[colorEMGind]
                                colorEMGind += 1

                            #print('--- plotting {} {} max {}'.format(chn, fbname, np.max(bandpower) ) )
                            ax.plot(bins_b,bandpower,
                                    label='{}, {}'.format(chn,fbname), 
                                    ls = plot_freqBandsLineStyle[fbname], alpha=0.5, c=color )
                            logscale = 0
                            maxpow = max(maxpow, np.max(bandpower) )
                            if logscale:
                                ax.set_yscale('log')
                                ax.set_ylabel('logscale')
                            ax.legend(loc=legendloc)

                            pars = plot_paramsPerModality[modality]
                            deflims = (0, maxpow)
                            ymin,ymax = pars.get('axLims_bandPow',{}).get(sind_str, deflims )
                            ax.set_ylim(ymin,ymax)
                            ax.set_title(title )
                            ax.set_xlim(mintime,maxtime)

                        if modality.find('EMG') >= 0:
                            if tremrDet_clusterMultiMEG:
                                clust = glob_stats[sind_str][medcond]['tremorfreq_clusters_allEMG'][chn]  # computed using k-means
                                for clusti in range(len(clust) ):
                                    ax.axhline(y = clust[clusti], 
                                            label = '{} clust{}'.format(chn,clusti) ,
                                            #ls = plot_freqBandsLineStyle['tremor' ], 
                                            ls='--', 
                                            c=plot_colorsEMG[colorEMGind-1] )
                            else:
                                thrs = glob_stats[sind_str][medcond][chn]['thrPerFreq_trem'] 
                                ax.axhline(y = freqres * np.sum(thrs), 
                                        label = '{} tremor thr'.format(chn) ,ls = ltype_tremorThr)

                            if tremorDetectUseCustomThr:
                                thr = gen_subj_info[sind_str]['tremorDetect_customThr'][medcond][chn] 
                                ax.axhline(y=thr, ls = ltype_tremorThr,lw=2, c= plot_colorsEMG[colorEMGind-1], 
                                        label = '{} tremor thr'.format(chn) )
                                                                 

            rowind_shift += chanTypes_toshow['bandpow']['nplots']

            if show_spec:
                print('Starting plotting spectrum')
                spcht = chanTypes_toshow['spectrogram']['chantypes']
                ch_toplot_spec = []
                for modality in spcht:
                    ch_toplot_spec += chnames_tuples[channel_pair_ind][modality] 
                ch_toplot_spec = filterFavChnames( ch_toplot_spec, sind_str )
                #chs = mne.pick_channels(chnames,include=ch_toplot_spec, ordered=True )
                
                specgramsComputed = specgrams[k]

                for rowind,chname in enumerate(ch_toplot_spec):
                    freqs, bins, Sxx = specgrams[k][chname]
                
                    minfreq = 0
                    if chname.find('LFP') >= 0:
                        minfreq = plot_minFreqInSpec
                    freqs, bins, Sxx = getSubspec(freqs,bins,Sxx,
                                                  minfreq,plot_maxFreqInSpec,
                                                  time_start,tetmp/sampleFreq)
                    
                    ax = axs[rowind + rowind_shift,colind]
                    stats = glob_stats[sind_str][medcond][chname]
                    mx = stats['max_spec']; mn = stats['min_spec']; me = stats['mean_spec']                
                    #while (mx - me) > (mx - mn)*0.9:  # catch short highrise
                    #    mx *= 0.9
                    
                    if normType == 'uniform':
                        norm = mpl.colors.Normalize(vmin=0.,vmax=mx);
                    elif normType == 'log':
                        norm = mpl.colors.LogNorm(vmin=mn,vmax=mx);
                    #print(chname,Sxx.shape,len(freqs),mx, mn)
                    im = ax.pcolormesh(bins, freqs, Sxx, 
                            cmap=plot_specgramCmapPerModality[modality], norm=norm)

                    #fig.colorbar(im, ax=ax) #for debugging
                    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.specgram.html
                    ax.set_title('Spec {}: {}, min={:.6E}, max={:.6E}_NFFT={} overlap={:d} c={}'.
                            format(k, chname, mn,mx, NFFT, int(NFFT*specgramoverlap), c_parameter) )
                    ax.set_xlabel('Time, [s]')
                    ax.set_ylabel('Freq, [Hz] '+normType)
                    ax.set_xlim(mintime,maxtime)
#                     if chname.find('LFP') >= 0:
#                         ax.set_ylim(plot_minFreqInSpec,plot_maxFreqInSpec)
#                     else:
#                         ax.set_ylim(0,plot_maxFreqInSpec)
                        
                    if orderEMG[channel_pair_ind] == gen_subj_info[sind_str]['tremor_side']:
                        ax.patch.set_facecolor(mainSideColor)
                    else:
                        ax.patch.set_facecolor(otherSideColor)

        print('Plotting {} finished'.format(k))

    ss = []
    subjstr = ''
    for k in ks:
        sind_str,medcond,task = getParamsFromRawname(k)
        if sind_str in ss:
            continue
        ss += [sind_str]
        subjstr += '{},'.format( sind_str )

    plt.tight_layout()
    if savefig:
        #figname = 'Spec_{}_pairno{}_{} nr{}, {:.2f},{:.2f}, spec{} timecourse{} c{}.{}'. \
        #            format(subjstr, channel_pair_ind, data_type, nr, float(time_start), \
        #                   float(time_end),show_spec,show_timecourse,c_parameter,ext)
        figname = 'Spec_{}_pairno{}_{} nr{}, {:.2f}, spec{} timecourse{} c{}.{}'. \
                    format(subjstr, channel_pair_ind, data_type, nr , \
                           float(time_end),show_spec,show_timecourse,c_parameter,ext)
        plt.savefig( os.path.join(plot_output_dir, figname ) )
        print('Figure saved to {}'.format(figname) )
    else:
        print('Skipping saving fig')

    if not showfig:
        plt.close()
    else:
        plt.show()
        
    print('Plotting all finished')

def plotTimecourse(plt):
    ndatasets = len(raws)
    #nffts = [512,1024]


    ymaxEMG = 0.00035
    ymaxLFP = 0.00012

    chanTypes_toshow = ['EMG','LFP']
    ymaxs = {'EMG':ymaxEMG, 'LFP':ymaxLFP}
    #chanTypes_toshow = ['EMG']

    nc = 4
    nr = len(ks)        

    time_start,time_end = 0,1000
    #time_start,time_end = 0,300  # to get only rest part

    fig, axs = plt.subplots(ncols = nc, nrows=nr, figsize= (ww*nc,hh*nr), sharey='none')
    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
    axind = 0

    colind = 0

    for axind in range(nr):
        k = ks[axind]

        chnames = raws[k].info['ch_names']
        orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)
        
        for channel_pair_ind in [0,1]:
            inds = []
            for cti in range(len(chanTypes_toshow_timecourse) ):
                ct = chanTypes_toshow[cti]
                inds = chinds_tuples[channel_pair_ind][ct]
                #inds += lfp_inds[3:] + emgkil_inds[-2:] #+ eog_inds
                # convert indices to channel names
                ch_toplot = [chnames[i] for i in inds]
                chs = mne.pick_channels(chnames,include=ch_toplot, ordered=True )

                ts,te = f.time_as_index([time_start, time_end])

                chdata, chtimes = raws[k][chs,ts:te]
                mintime = min(chtimes)
                maxtime = max(chtimes)

                
                if nc == 1:
                    ax = axs[axind]
                else:
                    colind = channel_pair_ind * 2 + cti
                    ax = axs[axind,colind]
                
                for i in range(chdata.shape[0] ):
                    ax.plot(chtimes,chdata[i,:], label = '{}'.format(chnames[chs[i] ] ), alpha=0.7 )
                    ax.set_ylim(-ymaxEMG/2,ymaxs[ct])
                    ax.set_xlim(mintime,maxtime)

                ax.legend(loc=legendloc)
                side = orderEMG[channel_pair_ind]
                ax.set_title('{:10} {} {}'.format(k,ct,side)  )
                ax.set_xlabel('Time, [s]')
                ax.set_ylabel('Voltage')

    ss = []
    subjstr = ''
    for k in ks:
        sind_str,medcond,task = getParamsFromRawname(k)
        if sind_str in ss:
            continue
        ss += [sind_str]
        subjstr += '{},'.format( sind_str )

    plt.tight_layout()
    if savefig:
        figname = '_{}_{} nr{}, {:.2f},{:.2f}.{}'. \
                    format(subjstr, data_type, nr, float(time_start), \
                           float(time_end),ext)
        plt.savefig( os.path.join(plot_output_dir, figname ) )
    if not showfig:
        plt.close()
    else:
        plt.show()
        
    print('Plotting all finished')


####################################
####################################
####################################
####################################

if __name__ == '__main__':

    if os.environ.get('DATA_DUSS') is not None:
        data_dir = os.path.expandvars('$DATA_DUSS')
    else:
        data_dir = '/home/demitau/data'
                
    raws = {}
    srcs = {}

    plot_output_dir = 'output'

    specgramoverlap = 0.75
    c_parameter = 20.0  # for the spectrogram calc
    c_parameter = 3
    #NFFT = 256  # for spectrograms
    NFFT = 256  # for spectrograms
    nonTaskTimeEnd = 300
    specgram_scaling = 'spectrum'  # power, not normalized by freq band (i.e. not denisty)
 
    # which time range use to compute spectrograms
    spec_time_start=0 
    spec_time_end=nonTaskTimeEnd

    #tremorThrMult = {'S05': 20. }
    tremorThrMult = { }
    updateJSON = False
    useKilEMG = False
    plot_highPassLFP = True

    tremrDet_clusterMultiFreq = True
    tremrDet_clusterMultiMEG = True

    plot_LFP_onlyFavoriteChannels = 0
    plot_EMG_spectrogram          = False
    plot_MEGsrc_spectrogram       = True
    #plot_onlyMainSide             = True
    plot_onlyMainSide             = False
    # I have selected them myself, not the ones chosen by Jan
    favoriteLFPch_perSubj = {'S01': ['LFPR23', 'LFPL12' ], 'S02': ['LFPR12', 'LFPL12'], 'S03': ['LFPR12'] } 
    plot_time_start = 0
    plot_time_end = 100

    plot_minFreqInSpec = 6
    plot_minFreqInSpec = 0
    #plot_maxFreqInSpec = 50
    #plot_maxFreqInSpec = 80
    plot_maxFreqInSpec = 35


    EMGlimsBySubj =  { 'S01':(0,0.001) }  
    #EMGlimsBySubj_meanshifted =  { 'S01':(-0.0001,0.0001),  'S02':(-0.0002,0.0002)}   # without highpassing
    EMGlimsBySubj_meanshifted =  { 'S01':(-0.00005,0.00005),  'S02':(-0.0002,0.0002)}  
    EMGlimsBandPowBySubj = {}; # { 'S01':(0,1e-10), 'S02':(0,1e-10) }  
      
    LFPlimsBySubj = {}
    #LFPlimsBySubj =  { 'S01':(0,0.001) }  
    LFPlimsBySubj_meanshifted =  { 'S01':(-0.000015,0.000015), 'S02':(-0.000015,0.000015) }  
    LFPlimsBandPowBySubj =  {}; #{ 'S01':(0,1e-11), 'S02':(0,1e-11) }  

    MEGsrclimsBandPowBySubj =  { 'S01':(0,400),  'S02':(0,600), 'S03':(0,700)  }

    EMGplotPar = {'shiftMean':True, 
            'axLims':EMGlimsBySubj, 'axLims_meanshifted':EMGlimsBySubj_meanshifted,
            'axLims_bandPow':EMGlimsBandPowBySubj  }
    #EMGplotPar.update( {'lowpass_freq':15, 'lowpass_order':10} )
    EMGplotPar.update( {'bandpass_freq':(0.5,15), 'bandpass_order':10} )

    LFPplotPar = {'shiftMean':True, 'highpass_freq': 1.5, 'highpass_order':10, 
            'axLims':LFPlimsBySubj, 'axLims_meanshifted':LFPlimsBySubj_meanshifted,
            'axLims_bandPow':LFPlimsBandPowBySubj  }

    EOGplotPar = {'shiftMean':True, 
            'axLims': {}, 'axLims_meanshifted':{} } #'axLims_meanshifted':LFPlimsBySubj_meanshifted,
            #'axLims_bandPow':LFPlimsBandPowBySubj,
            # 'highpass_freq': 1.5, 'highpass_order':10 , }

    MEGsrcplotPar = {'shiftMean':True, 'highpass_freq': 1.5, 'highpass_order':10, 'axLims': {},
            'axLims_meanshifted':{}, 'axLims_bandPow': MEGsrclimsBandPowBySubj }
            #'axLims':LFPlimsBySubj, 'axLims_meanshifted':LFPlimsBySubj_meanshifted,
            #'axLims_bandPow':LFPlimsBandPowBySubj  }

    # those with more apparent contrast between background and beta and tremor
    plot_timeIntervalsPerSubj = { 'S01':[ (0,30) ], 'S02':[ (30,150) ], 'S03':[ (0,300) ]   }
    #plot_timeIntervalsPerSubj = {}

    plot_paramsPerModality = {}
    plot_paramsPerModality = {'EMG':EMGplotPar, 'LFP':LFPplotPar, 
            'EOG':EOGplotPar, 'MEGsrc':MEGsrcplotPar }
    plot_MEGsrc_rowPerBand = True


    tremorDetectUseCustomThr = 0
    tremorDetect_customThr = {}
    # left
    _off = { 'EMG063_old': 0.65e-10  , 'EMG064_old': 0.4e-10   }
    _on  = { 'EMG063_old': 0.65e-10  , 'EMG064_old': 0.4e-10   }
    # right
    _off.update( { 'EMG061_old': 5e-11  , 'EMG062_old': 4e-11   } )
    _on.update(  { 'EMG061_old': 5e-11  , 'EMG062_old': 4e-11   } )
    tremorDetect_customThr['S01'] = {'off':_off, 'on':_on }

    tremIntDef_convWidth = 10
    tremIntDef_convThr   = 0.13
    tremIntDef_incNbins   = 5
    tremIntDef_percentthr=0.8
    tremIntDef_minlen=50
    tremIntDef_extFactor=0.25

    ########################################## Data selection
    # used on the level of chanTuples generation first, so for unselected nothing will be computed 
    MEGsrc_inds_toshow = None
    MEGsrc_inds_toshow = np.arange(10)   
    MEGsrc_inds_toshow = np.arange(20)   
    #MEGsrc_inds_toshow = [74,75,76]   # used on the level of chanTuples generation first, so for unselected nothing will be computed

    # Set plot params
    show_timecourse = 1
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
    hh = 3
    ww = 20
    legendloc = 'lower right'

    MEGsrc_roi = ['Brodmann area 4']
    #MEGsrc_roi = ['Brodmann area 6']

    subjinds = [1,2,3,4,5,6,7,8,9,10]
    tasks = ['hold', 'move', 'rest']
    medconds = ['off', 'on']

    subjinds = [4,5,6]
    #tasks = ['hold']
    tasks = ['rest', 'move', 'hold']
    medconds = ['off', 'on']

    subjinds = [1,2,3]
    #tasks = ['hold']
    tasks = ['rest', 'move', 'hold']
    medconds = ['off', 'on']

    #tasks = ['hold']
    #subjinds = [1]

    #subjinds = [4]
    ###tasks = ['hold']
    #tasks = ['rest', 'move']
    #medconds = ['off', 'on']

    #subjinds = [3,4,5,6]
    ##tasks = ['hold']
    #tasks = ['rest', 'move', 'hold']
    #medconds = ['off', 'on']

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
    print('Filenames to be read ',fnames_noext)

    srcPerRawname = {}
    fnames_src_noext = []
    for subjind in subjinds:
        for medcond in medconds:
            for task in tasks:
                rawname = getRawname(subjind,medcond,task)
                srcPerRawname[rawname] = []
                for curroi in MEGsrc_roi:
                    if curroi.find('_') >= 0:
                        raise ValueError("Src roi contains underscore, we'll have poblems with parsing")

                    #fn = 'srcd_S{:02d}_{}_{}_{}'.format(subjind,medcond,task,curroi)
                    fn = getSrcname(subjind,medcond,task,curroi)
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
    subjs_analyzed = {}  # keys -- subjs,  vals -- arrays of keys in raws
    for k in raws:
        f = raws[k]
        sind_str,medcond,task = getParamsFromRawname(k)

        cursubj = {}
        if sind_str in subjs_analyzed:
            cursubj = subjs_analyzed[sind_str]

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

        subjs_analyzed[sind_str] =  cursubj
    print(subjs_analyzed)


    #####################   Read subject information, provided by Jan
    with open(os.path.join(data_dir,'Info.json') ) as info_json:
        gen_subj_info = json.load(info_json)
            
    ####################    generate some supplementary info, save to a multi-level dictionary 
    for k in raws:
        sind_str,medcond,task = getParamsFromRawname(k)
        subjinfo = gen_subj_info[sind_str]
        if 'chantuples' not in subjinfo: # assume human has fixed set of channels 
            chnames = raws[k].info['ch_names'] 
            rr = genChanTuples(k, chnames) 
            orderEMG, ind_tuples, name_tuples, LFPnames_perSide, EMGnames_perSide, MEGsrcnames_perSide = rr 
            chantuples = {}
            
            subjinfo['LFPnames_perSide'] = LFPnames_perSide 
            subjinfo['EMGnames_perSide'] = EMGnames_perSide 
            if len(srcs) > 0:
                subjinfo['MEGsrcnames_perSide'] = MEGsrcnames_perSide 
            for i in range(len(orderEMG) ):
                yy = {}
                yy['indtuples'] = ind_tuples 
                yy['nametuples'] = name_tuples 
                chantuples[ orderEMG[i] ] = yy

            subjinfo['chantuples' ] = chantuples

    if MEGsrc_inds_toshow is None:
        n = 0
        MEGsrcnames = list( gen_subj_info.values() )[0 ] ['MEGsrcnames_perSide']
        for side in MEGsrcnames:
            n+= len( MEGsrcnames[side] )
        MEGsrc_inds_toshow = np.arange(n)  # differnt roi's will have repeating indices? No because the way I handle it in getChantuples!
    print( 'MEGsrc_inds_toshow = ', MEGsrc_inds_toshow )


    for subj in gen_subj_info:
        if subj in tremorDetect_customThr:
            gen_subj_info[subj]['tremorDetect_customThr'] = tremorDetect_customThr[subj]
        if subj in favoriteLFPch_perSubj:
            gen_subj_info[subj]['favoriteLFPch'] = favoriteLFPch_perSubj[subj]

    # if we want to save the updated information back to the file
    if updateJSON:
        with open(os.path.join(data_dir,'Info.json'), 'w' ) as info_json:
            json.dump(gen_subj_info, info_json)

    # compute spectrograms for all channels (can be time consuming, so do it only if we don't find them in the memory)
    try: 
        specgrams
    except NameError:
        specgrams = precomputeSpecgrams(raws,NFFT=NFFT)
    else:
        print('----- Using previously precomuted spectrograms')

    #time_start, time_end = 0,1000
    time_start_forstats, time_end_forstats = 0,300
    freq_min_forstats, freq_max_forstats = 0, NFFT//2   #NFFT//2

    # check what Jan considered as tremor frequency 
    tfreqs = []
    for subj in subjs_analyzed:
        tfreq = gen_subj_info[subj]['tremfreq']
        tside = gen_subj_info[subj]['tremor_side']
        print('{} has tremor at {} side with freq {}'.format(subj,tside,tfreq) )
        tfreqs += [ tfreq]

        if subj not in favoriteLFPch_perSubj and plot_LFP_onlyFavoriteChannels:
            raise ValueError('Want to plot only favorite LFP they are not set for all subjects!' )

    # define tremor band with some additional margin
    safety_freq_shift = 0.5
    tremorBandStart = min(tfreqs) - safety_freq_shift #3.8 
    tremorBandEnd   = max(tfreqs) + safety_freq_shift #6.8
    if  tremorBandEnd - tremorBandStart < 1.5:
        tremorBandStart -= 0.8
        tremorBandEnd += 0.8
    print('Tremor freq band to be used: from {} to {}'.format(tremorBandStart,tremorBandEnd) )

    # define other bands that will be used
    betaBand = 15,30
    lowGammaBand = 31,45
    motorGammaBand = 30,100

    freqBands = {'tremor':(tremorBandStart,tremorBandEnd), 'beta':betaBand,
            'lowgamma':lowGammaBand, 'gamma_motor':motorGammaBand }
    plot_freqBandNames_perModality = {'EMG': ['tremor' ], 'LFP': ['beta','lowgamma'], 
            'MEGsrc':['tremor','beta'], 'EOG':['lowgamma'] }
    plot_freqBandsLineStyle = {'tremor':'-', 'beta':'--', 'lowgamma':':', 'gamma_motor':':'  }

    ############## plotting params 
    plot_colorsEMG = [ 'black', 'green' ] 
    ltype_Clust = '--'
    ltype_tremorThr = '-.'
    #plot_colorsEMGClust = [ 'blue', 'red' ] 

    plot_specgramCmapPerModality = {'LFP': 'hot', 'MEGsrc':'viridis' }
    plot_MEGsrc_separateRoi = False

    #cmap = 'hot'
    #cmap = 'gist_rainbow'
    #cmap = 'viridis'


    ##############   compute some statisitics, like max, min, mean, cluster in subbands, etc -- can be time consuming
    try:
        glob_stats
    except NameError:
        glob_stats = getStatPerChan(time_start_forstats,time_end_forstats,
                freq_min_forstats,freq_max_forstats)  # subj,channame, keys [! NO rawname!]
    else:
        print('----- Using previously computed stats!')

    ##############   find tremor intercals in all subjects and raws 
    try: 
        tremIntervalPerRaw
    except NameError:
        tremIntervalPerRaw = findAllTremor(width=tremIntDef_convWidth, inc=tremIntDef_incNbins, 
                thr=tremIntDef_convThr, minlen=tremIntDef_minlen, extFactor=tremIntDef_extFactor,
                percentthr=tremIntDef_percentthr)
    else:
        print('----- Using previously tremor intervals!')


    # merge intervals within side
    mergeMode = 'union' 
    #mergeMode = 'intersect' 
    tremIntervalMergedPerRaw = mergeTremorIntervals(tremIntervalPerRaw, mode=mergeMode)


    # Set matplotlib params
    #inferno1 = mpl.cm.get_cmap('viridis', 256)
    #inferno2 = mpl.colors.ListedColormap(inferno1(np.linspace(0, 1, 128)))

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
        plotSpectralData(plt,time_start=plot_time_start,time_end=plot_time_end ) 
