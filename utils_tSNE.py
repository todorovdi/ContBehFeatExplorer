import numpy as np
import udus_dataproc as mdp # main data proc
import re

import globvars as gv
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

import multiprocessing as mpr
import mne


def plotEvolutionMultiCh(dat, times, chan_names=None, interval = None, extend=5,
                         yshift_stdmult = 4, bnd_toshow = 'start', rawname='',
                         ax = None, save=1, prefix='', ww=None, dat_ext=None,
                         chan_names_ext=None, yshifts=None, interval_for_stats=None):
    '''
    plots on a single axis
    dat = n_chans x time
    interval  =  (start,end,int_type)
    ext chans are assumed to be scaled
    '''
    feat_types = ['con']
    bands = ['beta']

    if ww is None:
        ww = extend * 6

    n_channels = dat.shape[0]
    assert len(chan_names) == n_channels, '{} {}'.format(len(chan_names) , n_channels )
    if chan_names is None:
        chan_names = map(str, range(n_channels ) )
    if not isinstance(chan_names,list):
        chan_names = list(chan_names)

    if dat_ext is not None:
        assert dat_ext.shape[1] == dat.shape[1]
        n_channels_ext = dat_ext.shape[0]
        if chan_names_ext is not None:
            assert len(chan_names_ext) == dat_ext.shape[0]
        else:
            chan_names_ext = map(lambda x: 'ext_' + str(x), range(n_channels_ext))
            chan_names_ext = list(chan_names_ext)
    else:
        n_channels_ext = 0
        chan_names_ext = []

    if yshifts is not None:
        assert len(yshifts) == n_channels + n_channels_ext
        use_given_yshifts = 1
    else:
        use_given_yshifts = 0

    #feats_toshow = feature_names_all[:20] + feature_names_all[-20:]
    #feats_toshow = feature_names_all


    if ax is None:
        plt.figure(figsize=(ww,len(chan_names)))
        ax = plt.gca()

    if interval is None:
        interval = 0, times-1, 'entire'
    start,end,int_type = interval

    assert start <= end
    if bnd_toshow == 'start':
        tv = start
    elif bnd_toshow == 'end':
        tv = end
    else:
        tv = None

    if bnd_toshow == '*':
        time_wnd = start, end
    else:
        time_wnd = max(0,tv-extend), min(times[-1], tv+extend)


    timinds = np.where( (times >= time_wnd[0])  * (times <= time_wnd[1])  )[0]
    times_inwnd = times[timinds]

    if interval_for_stats is not None:
        t0s,t1s = interval_for_stats
        assert t1s >= t0s

        timinds_stats = np.where( (times >= t0s)  * (times <= t1s)  )[0]
        #times_inwnd = times[timinds]
    else:
        timinds_stats = slice(None)

    yshift = 0
    yshifti = 0
    if not use_given_yshifts:
        yshifts = []

    mn = np.inf
    mx = -np.inf
    for chi in range(n_channels):
        if use_given_yshifts:
            yshift = yshifts[yshifti]
            yshifti += 1
        else:
            yshifts += [yshift]
        curdat = dat[chi,timinds]
        cd  = curdat + yshift
        p = ax.plot( times_inwnd, cd, alpha=0.8)
        mn = min(np.min(cd), mn)
        mx = max(np.max(cd), mx)
        curdat_forstats = dat[chi,timinds_stats]
        ax.axhline(y= yshift + np.mean(curdat_forstats), c = p[0].get_color(), ls=':')
        yshift += np.std(curdat_forstats) * yshift_stdmult

    for chi in range(n_channels_ext):
        if use_given_yshifts:
            yshift = yshifts[yshifti]
            yshifti += 1
        else:
            yshifts += [yshift]
        curdat = dat_ext[chi,timinds]
        cd  = curdat + yshift
        p = ax.plot( times_inwnd,  cd,  alpha=0.8)
        mn = min(np.min(cd), mn)
        mx = max(np.max(cd), mx)
        ax.axhline(y= yshift , c = p[0].get_color(), ls=':') # here we want to plot 0, not mean
        curdat_forstats = dat_ext[chi,timinds_stats]
        yshift += np.quantile(curdat_forstats,0.97) * yshift_stdmult

    #ax.axvline(x = start, c='r', ls=':')
    if tv is not None:
        ax.axvline(x = tv, c='r', ls=':', label='{} {}'.format(int_type,bnd_toshow))
    ax.set_xlim(time_wnd)

    ax.set_yticks(yshifts)
    ax.set_yticklabels(chan_names + chan_names_ext)

    #ax.set_xlabel('time in [s], shifted by {} edgebins'.format(nedgeBins))

    figname = '{}_{}_{}ch_evol {} {}_({:.1f},{:.1f}).png'.format(rawname,prefix,
                                                    n_channels,int_type,bnd_toshow,
                                                    time_wnd[0], time_wnd[1])
    if save:
        plt.savefig(figname)
        print('Plot saved to ',figname)

    return yshifts, mn, mx
    #plt.close()


def plotBasicStatsMultiCh(dats, chan_names=None, printMeans = True, singleAx = 1,
                          shift_std_mult = 6):
    ''' dats is a list of ndarrays n_chans x ntimes'''
    if isinstance(dats,np.ndarray):
        dats = [dats]
    assert isinstance(dats,list), 'We need list as input!'
    n_channels = dats[0].shape[0]
    assert len(chan_names) == n_channels
    if chan_names is None:
        chan_names = map(str, range(n_channels ) )

    nr = dats[0].shape[0]; nc=len(dats);  ww = 5; hh = 2
    if singleAx :
        ww = min( len(chan_names)*2, 60)
        hh = 5
        fig,axs = plt.subplots(nrows=nc, ncols=1, figsize = (ww, hh))
        if nc == 1:
            axs = [axs]
        xshift = 0
        xshifts = []
        for dati,dat in enumerate(dats):
            ax = axs[dati]
            for ind in range(nr ):
                xshifts += [xshift]
                hist, bin_edges = np.histogram(dat[ind], bins=50, density=False)
                m = np.mean(dat[ind]);
                s = np.std(dat[ind]);
                bin_edges += xshift

                ax.bar(bin_edges[0:-1],hist, alpha=0.7)

                #ax.hist(dat[ind], bins=100, alpha=0.7 )
                if printMeans:
                    print(chan_names[ind],m)
                ax.axvline(x=xshift + m,c='r',ls=':')
                ax.set_title(chan_names[ind])

                xshift    += s*shift_std_mult

            ax.set_xticks(xshifts)
            ax.set_xticklabels(chan_names,rotation=90)
    else:
        fig,axs = plt.subplots(nrows=nr, ncols=nc, sharex='col',  figsize = (ww*nc, hh*nr))
        axs = axs.reshape( (nr,nc ))
        for dati,dat in enumerate(dats):
            for ind in range(nr ):
                ax = axs[ind,dati]
                ax.hist(dat[ind], bins=100, alpha=0.7 )
                m = np.mean(dat[ind]);
                if printMeans:
                    print(chan_names[ind],m)
                ax.axvline(x=m,c='r',ls=':')
                ax.set_title(chan_names[ind])

def plotIntervalData(dat, chan_names, interval, times = None, raw=None, plot_types = ['psd'], extend = 3, sfreq=256,
                     dat_ext = None, chan_names_ext=None, ww=8, hh=3, fmax=None):
    start,end,int_type  = interval

    assert raw is not None or (times is not None)

    ts, inds, sliceNames = getIntervalSurround( start,end, extend, raw=raw, times=times)
    #inds = raw.time_as_index(
    #    [start - extend, start, start+extend, end-extend, end, end+extend])
    prestarti, starti, poststarti, preendi, endi, postendi = inds
    print('Interval {} duration is {}'.format(int_type, end-start) )

    if times is None:
        times = np.arange( dat.shape[1] )  / sfreq

    n_channels = dat.shape[0]
    assert len(chan_names) == n_channels

    #sliceNames = ['prestart', 'poststart',  int_type, 'preend', 'postend']
    sliceNames[2] = int_type
    tuples = [ (times[inds[i] ], times[inds[i+1] ],sliceNames[i] ) for i in range(len(inds) -1 ) ]
    slices = [ slice(inds[i], inds[i+1]) for i in range(len(inds) -1 ) ]

    nr = 1 ; #len(n_channels)
    nc = len(slices) * len(plot_types)

    pt = plot_types[0]

    if pt == 'psd':
        fig,axs = plt.subplots( nrows=nr, ncols=nc, figsize=(nc*ww, nr*hh), sharey='row' )
    elif pt =='timecourse':
        hh = len(chan_names)
        fig,axs = plt.subplots( nrows=nr, ncols=nc, figsize=(nc*ww, nr*hh), sharey='row' )

    left_space = 0.05
    bottom_space = None
    top_space = None
    if pt == 'timecourse':
        left_space = 0.11
        top_space = 1 - min(0.1, 1 / len(chan_names) )
        bottom_space = min(0.1, 1 / len(chan_names) )
    plt.subplots_adjust(left=left_space, bottom=bottom_space,
                        right=0.98, top=top_space, wspace=0.01, hspace=0.01)

    axs = axs.reshape( (nr,nc) )
    fig.suptitle('{} duration = {:.1f}s, extend = {}'.format(int_type, end-start, extend) )

    # mne.time_frequency.psd_array_multitaper(x, sfreq, fmin=0, fmax=inf, bandwidth=None, adaptive=False, low_bias=True, normalization='length', n_jobs=1, verbose=None)
    #import mne.time_frequency.psd_array_multitaper as mnepsd

    from mne.time_frequency import psd_array_multitaper as mnepsd

    if fmax is None:
        fmax = sfreq/2

    if pt == 'psd':
        for chi in range(n_channels):
            sfname = chan_names[chi]
            ls = '-'
            if sfname.find('msrc') >= 0:
                ls = '--'

            assert len(sliceNames) == len(slices)
            for i in range(nc):
                ax = axs[0,i ]
                slicei = i // len(plot_types)
                t0,t1,sln =  tuples[i]
                ax.set_title(sliceNames[slicei] + '_({:.2f},{:.2f})'.format(t0,t1)  )
                plot_typei = nc % len(plot_types)
                pt = plot_types[plot_typei]
                if t1 - t0 <= 0.1:
                    continue

                psdres,freq = mnepsd(dat[chi, slices[slicei] ],
                                     sfreq, fmin=1, fmax=fmax, verbose=False)
                ax.semilogy( freq,psdres, label=sfname, ls=ls)
                #ax.psd( dat[chi, slices[slicei] ], Fs=sfreq )
                ax.grid()

            axs[0,0].legend(loc='upper right')
    elif pt =='timecourse':
        interval_for_stats = times[inds[0] ], times[inds[-1] ]
        yshifts = None
        mn = np.inf
        mx = -np.inf
        for i in range(nc):
            ax = axs[0,i ]
            slicei = i // len(plot_types)
            ax.set_title(sliceNames[slicei]  )
            plot_typei = nc % len(plot_types)
            pt = plot_types[plot_typei]
            t0,t1,sln =  tuples[i]
            if t1 - t0 <= 0.1:
                continue
            yshifts, mn_,mx_ = plotEvolutionMultiCh( dat, times, chan_names, interval=tuples[i],
                                 extend=0, bnd_toshow='*', ax=ax, dat_ext = dat_ext,
                                 chan_names_ext = chan_names_ext, save=0, yshifts=yshifts,
                                           interval_for_stats = interval_for_stats)
            mn = min(mn,mn_)
            mx = max(mx,mx_)

        for i in range(nc):
            ax = axs[0,i ]
            ax.set_ylim(mn,mx)

        #ax = plt.subplot(1,5,1); ax.set_title('prestart')
        #ax.psd(dat[chi, prestarti:starti], Fs=256, ls=ls);
        #ax = plt.subplot(1,5,2); ax.set_title('poststart')
        #ax.psd(dat[chi, starti:poststarti], Fs=256, ls=ls);
        #ax = plt.subplot(1,5,3); ax.set_title(int_type)
        #ax.psd(dat[chi, poststarti:preendi], Fs=256, ls=ls);
        #ax= plt.subplot(1,5,4); ax.set_title('preend')
        #ax.psd(dat[chi, preendi:endi], Fs=256, ls=ls);
        #ax = plt.subplot(1,5,5); ax.set_title('postend')
        #ax.psd(dat[chi, endi:postendi], Fs=256, ls=ls);
    return times

def plotCSD(csd, fbs_list, channel_names, timebins, sfreq=256, intervalMode=1,
            int_names = None):
    '''
    timebins is list of time timbin indices or list of 2-tuples with time rnages
    '''
    assert isinstance(timebins,list), 'want list'
    N = len(fbs_list)
    nc = N; nr = len(timebins)


    indtuples = [ (timebins[i], timebins[i+1]) for i in range(len(timebins) -1 ) ]

    ww = 3; hh = 3
    if intervalMode:
        nr -= 1
    fig,axs = plt.subplots(ncols = nc, nrows =nr, figsize = (ww*N,hh*nc), sharey = 'row',
                           sharex='col')

    n_channels = len(channel_names)

    mn = np.inf
    mx = 0
    MM = []
    stack = []
    for tbi,tb in enumerate(timebins):
        for freqi in range(N):
            if intervalMode:
                if tbi == len(timebins) - 1:
                    break
                itpl = indtuples[tbi]
                tb = slice( itpl[0], itpl[1] )
            M = utils.getFullCSDMat(csd,freqi,tb, n_channels)
            M = np.abs(M)
            stack.append(M)
            MM += [M.flatten() ]
            #mn = min(np.min(M), mn)
            #mx = max(np.max(M), mx)
    MM = np.hstack(MM)
    mn = np.quantile(MM,0.05)
    mx = np.quantile(MM,0.95)

    from collections.abc import Iterable
    for tbi,tb in enumerate(timebins):
        for freqi in range(N):
            if intervalMode:
                if tbi == len(timebins) - 1:
                    break
                itpl = indtuples[tbi]
                tb = np.array(itpl )
            elif not isinstance(tb, Iterable):
                tb = np.array([tb])
            M = stack.pop()
            #M = utils.getFullCSDMat(csd,freqi,tb, n_channels)
            #M = np.abs(M)
            fbname = fbs_list[freqi]
            ax = axs[tbi,freqi]

            norm = mpl.colors.LogNorm(vmin=mn,vmax=mx)
            ax.pcolormesh( np.abs(M ) ,norm=norm);
            #ax.set_title( fbname  )
            ax.set_xticks(range(n_channels))
            ax.set_xticklabels(channel_names, rotation=90)

            ax.set_yticks(range(len(channel_names)))
            ax.set_yticklabels(channel_names)
            inn = ''
            if int_names is not None:
                inn = '\n{} '.format(int_names[tbi])
            ttl =  '{}, {} time={}s'.format(fbname, inn, tb // sfreq)
            ax.set_title( ttl )

    fig.suptitle( 'abs of CSD for timebins min,max=  {},{} s'.
                 format( timebins[0]// sfreq, timebins[-1]// sfreq ))


def plotMultiMarker(ax,dat1, dat2, c, m, alpha=None, s=None, picker=False):
    '''  returns tuple -- list , list of lists , list of artists '''
    resinds = []
    markerset = list( set( m ) )
    scs = []
    m = np.array(m)
    c = np.array(c)
    for curm in markerset:
        inds = np.where(m==curm)[0]

        #print(curm, len(inds))
        if len(inds):
            #print(inds)
            sc = ax.scatter(dat1[inds],dat2[inds],c=c[inds],marker=curm,alpha=alpha,
                       s=s,picker=picker)
            resinds += [ inds.tolist() ]  # yes, I want list of lists
            scs += [sc]
    return markerset,resinds,scs


def getIntervalSurround(start,end, extend, raw=None, times=None, verbose = False):
    assert raw is not None or (times is not None)
    '''
    times can start not from zero, but start and end assume we start from zero
    '''

    assert end >= start

    if times is None:
        times = raw.times
    ge = times[-1]
    gs = times[0]

    if end == start:
        ts = [end] * 6
    else:
        extendIn = extend
        extendOut = extend
        end = min(end,ge)
        start = max(gs,start)
        ts = [max(gs,start - extendOut), start, min(ge,start+extendIn),
            max(gs,end-extendIn), end, min(ge,end+extendOut) ]

        while (not np.all( np.diff(ts) >= 0) ):
            extendIn /= 1.5
            ts = [max(gs,start - extendOut), start, min(ge,start+extendIn),
                max(gs,end-extendIn), end, min(ge,end+extendOut) ]

    if verbose:
        print('extendIn used', extendIn)
    assert np.all( np.diff(ts) >= 0)
    if times is None:
        tsis = raw.time_as_index(ts)
    else:
        bins = ( times / (times[1] - times[0] ) ).astype(int)
        ts_ = ( ts / (times[1] - times[0] ) ).astype(int)
        tsis = np.searchsorted(bins, ts_)
        #tsis = np.where(   )

    subint_names = ['prestart','poststart','main','preend','postend']
    return ts, tsis, subint_names

def selFeatsRegexInds(names, regexs, unique=1):
    import re
    if isinstance(regexs,str):
        regexs = [regexs]

    inds = []
    for namei,name in enumerate(names):
        for pattern in regexs:
            r = re.match(pattern, name)
            if r is not None:
                inds += [namei]
                if unique:
                    break

    return inds


def selFeatsRegex(data, names, regexs, unique=1):
    import re
    if isinstance(regexs,str):
        regexs = [regexs]
    assert len(data) == len(names)

    inds = selFeatsRegexInds( names, regexs, unique)
    namesel = np.array(names) [inds]


    #namesel = []
    #inds = []
    #for namei,name in enumerate(names):
    #    for pattern in regexs:
    #        r = re.match(pattern, name)
    #        if r is not None:
    #            namesel += [name]
    #            inds += [namei]
    #            if unique:
    #                break


    if data is None:
        datsel = None
    else:
        datsel = data[inds]

    return datsel, namesel

def robustMean(dat,axis=None, q=0.05):
    if q < 1e-10:
        return np.mean(dat)
    qvmn = np.quantile(dat, q,   axis=axis)
    qvmx = np.quantile(dat, 1-q, axis=axis)
    return np.mean(  dat[ (dat<=qvmx) * (dat>=qvmn) ] )
    #np.mean(  dat[dat<q

def robustMeanPos(dat,axis=None, q=0.05):
    if q < 1e-10:
        return np.mean(dat)
    else:
        qvmx = np.quantile(dat, 1-q, axis=axis)
        # dat[dat<=qvmx] is a flattened array anyway
        return np.mean(  dat[dat<=qvmx] )

from scipy.stats import pearsonr
def _feat_correl(arg):
    bn_from,bn_to,fromi,toi,name,window_starts,windowsz,dfrom,dto,oper,pos = arg

    corr_window = []
    q = 0.05
    #q = 0
    if pos:
        mfrom = robustMeanPos(dfrom, q=q)    # global mean
        mto   = robustMeanPos(dto, q=q)
    else:
        mfrom = robustMean(dfrom, q=q)    # global mean
        mto   = robustMean(dto, q=q  )
    for wi in range(len(window_starts)):
        ws = window_starts[wi]
        sl = slice(ws,ws+windowsz)

        #r = np.correlate(datsel_from[fromi,sl], datsel_to[toi,sl] )
        if oper == 'corr':
            #r,pval = pearsonr(dfrom[sl], dto[sl])
            # want to measure deviations from global means
            r =  np.mean( (dfrom[sl] - mfrom) *(dto[sl] - mto) )  #mean in time
        elif oper == 'div':
            r = dfrom[sl] / dto[sl]
        corr_window += [r]
    rr = np.hstack(corr_window)

    return rr,bn_from,bn_to,fromi,toi,window_starts,name,oper,pos

def computeFeatOrd2(dat, names, skip, windowsz, band_pairs,
                      n_free_cores=2, positive=1, templ = None,
                    templ_exclude = None):
    '''
    Xfull is chans x timebins
    windowz is in bins of Xtimes_full
    '''
    #e.g  bandPairs = [('tremor','beta'), ('tremor','gamma'), ('beta','gamma') ]
    # compute Pearson corr coef between different band powers all chan-chan
    # parhaps I don't need corr between LFP chans
    #bandPairs = bandPairs[0:1]
    #bandPairs = bandPairs[1:2]
    #bandPairs = bandPairs[2:3]

    if templ_exclude is not None:
        assert len(templ_exclude) == 2


    #window_starts = np.arange(len(Xtimes_full) )[::skip]
    window_starts = np.arange( 0, dat.shape[-1],  skip, dtype=int)
    #window_starts_tb = raw_lfponly
    #assert len(window_starts) == len(Xtimes)


    if templ is None:
        templ = r'con_{}.*:\s(.*),\1'

    cors = []
    cor_names = []
    args = []
    for bn_from,bn_to,oper in band_pairs:
        templ_from = templ.format(bn_from)
        templ_to   = templ.format(bn_to)

        #re.match(templ_exclude[0], name )

        datsel_from,namesel_from = selFeatsRegex(dat, names, templ_from)
        datsel_to,namesel_to = selFeatsRegex(dat, names, templ_to)


        for fromi in range(len(namesel_from)):
            for toi in range(len(namesel_to)):
                # within same band we don't want to compute BOTH i->j and j->i
                if bn_from == bn_to and fromi < toi:
                    continue

                nfrom,nto = namesel_from[fromi], namesel_to[toi]
                if nfrom.find('LFP') >= 0 and nto.find('LFP') >= 0:
                    continue

                nfrom = nfrom.replace('con_',''); nfrom = nfrom.replace('allf_','');
                nto = nto.replace('con_',''); nto = nto.replace('allf_','')
                name = '{}_{},{}'.format(oper,nfrom,nto)

                dfrom = datsel_from[fromi]
                dto = datsel_to[toi]
                assert dfrom.size > 1
                assert dto.size > 1
                arg = bn_from,bn_to,fromi,toi,name,window_starts,windowsz,dfrom,dto,oper,positive
                args += [arg]
                #corr_window = []
                #for wi in range(len(window_starts)):
                #    ws = window_starts[wi]
                #    sl = slice(ws,ws+windowsz)

                #    #r = np.correlate(datsel_from[fromi,sl], datsel_to[toi,sl] )
                #    r,pval = pearsonr(datsel_from[fromi,sl], datsel_to[toi,sl])
                #    corr_window += [r]
                #cors += [np.hstack(corr_window) ]
                #cor_names += [(name)]

    ncores = max(1, min(len(args) , mpr.cpu_count()-n_free_cores) )
    if ncores > 1:
        #if ncores > 1:
        pool = mpr.Pool(ncores)
        print('high ord feats:  Starting {} workers on {} cores'.format(len(args), ncores))
        res = pool.map(_feat_correl, args)

        pool.close()
        pool.join()
    else:
        res = []
        for arg in args:
            res += [ _feat_correl(arg) ]

    cors = []
    cor_names = []
    for tpl in res:
        rr,bn_from,bn_to,fromi,toi,window_starts,name,oper,positive = tpl
        cors += [rr]
        cor_names += [name]

    return cors,cor_names

def prepareLegendElements(mrk,mrknames,colors,tasks, s=8, m_unlab='o', skipExt = False):
    legend_elements = []
    if tasks is not None:
        if isinstance(tasks,str):
            tasks = [tasks]
        else:
            assert isinstance(tasks,list)

    unset = []
    for clrtype in colors.keys():
        if clrtype == 'neut':
            continue
        if clrtype in ['move', 'hold'] and clrtype not in tasks:
            unset += [clrtype]
            continue

        for m,mn in zip(mrk,mrknames):
            if skipExt and len(mn) > 0:  #mn == '' for the "meat" part of the interval
                continue

            legel_ = mpl.lines.Line2D([0], [0], marker=m, color='w', label=clrtype+mn,
                                        markerfacecolor=colors[clrtype], markersize=s)
            #print(clrtype+mn)


            legend_elements += [legel_]

    #if len(unset)> 0:
    #    print('!! Warning: found {} but it was unselected (thus not put in legend)'.format(unset) )

    legel_unlab = mpl.lines.Line2D([0], [0], marker=m_unlab, color='w', label='unlab',
                                markerfacecolor=colors['neut'], markersize=s)
    legend_elements += [legel_unlab]
    return legend_elements

def prepColorsMarkers(side_letter, anns, Xtimes,
               nedgeBins, windowsz, sfreq, totskip, mrk,mrknames,
               color_per_int_type, extend = 3, defmarker='o', neutcolor='grey',
                     convert_to_rgb = False ):
    #windowsz is in 1/sfreq bins
    #Xtimes_almost is in bins whose size comes from gen_features
    #Xtimes is in possible smaller bins
    hsfc = side_letter

    if convert_to_rgb:
        import matplotlib.colors as mcolors
        for k in color_per_int_type:
            cname = color_per_int_type[k]
            if isinstance(cname,str):
                color_per_int_type[k] = mcolors.to_rgb(cname)


    #hsfc = 'L'; print('Using not hand side (perhabs) for coloring')
    annot_colors_cur = {}
    for k in color_per_int_type:
        annot_colors_cur[ '{}_{}'.format(k,hsfc)   ] = color_per_int_type[k]
    #for task in tasks:
    #    annot_colors_cur[ '{}_{}'.format(task, hsfc) ] = color_per_int_type[task]

    colors =  [neutcolor] * len(Xtimes)
    markers = [defmarker] * len(Xtimes)

    for an in anns:
        for descr in annot_colors_cur:
            if an['description'] == descr:
                col = annot_colors_cur[descr]

                start = an['onset']
                end = start + an['duration']

                timesBnds, indsBnd, sliceNames = getIntervalSurround( start,end, extend,
                                                                    times=Xtimes)
                #print('indBnds in color prep ',indsBnd)
                for ii in range(len(indsBnd)-1 ):
                    # do not set prestart, poststart for left recording edge
                    if start <= nedgeBins/sfreq and ii in [0,1]:
                        continue
                    # do not set preend, posted for right recording edge
                    #globend = Xtimes_almost[-1] + nedgeBins/sfreq
                    globend = Xtimes[-1] + nedgeBins/sfreq
                    if  globend - end <= nedgeBins/sfreq and ii in [3,4]:
                        continue
                    # window size correction because it goes _before_
                    bnd0 = min(len(Xtimes)-1, indsBnd[ii]   + windowsz // totskip -1   )
                    bnd1 = min(len(Xtimes)-1, indsBnd[ii+1] + windowsz // totskip -1   )
                    #inds2 = slice( bnd0, bnd1 )
                    inds2 = range( bnd0, bnd1 )

                    #inds2 = slice( indsBnd[ii], indsBnd[ii+1] )
                    #markers[inds2] = mrk[ii]

                    for jj in inds2:
                        colors [jj] = col
                        markers[jj] = mrk[ii]
                    #print(len(inds2))
    return colors,markers

def plotPCA(pcapts,pca, nPCAcomponents_to_plot,feature_names_all, colors, markers,
            mrk, mrknames, color_per_int_type, task,
            pdf=None,neutcolor='grey'):

    ##################  Plot PCA
    nc = min(nPCAcomponents_to_plot, pcapts.shape[1] );
    nr = 1; ww = 5; hh = 4
    fig,axs = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*ww,nr*hh))
    ii = 0
    while ii < nc:
        indx = 0
        indy = ii+1
        ax = axs[ii];  ii+=1
        plotMultiMarker(ax, pcapts[:,indx], pcapts[:,indy], c=colors, m = markers, alpha=0.5);
        ax.set_xlabel('PCA comp {}'.format(indx) )
        ax.set_ylabel('PCA comp {}'.format(indy) )

    legend_elements = prepareLegendElements(mrk,mrknames,color_per_int_type, task )

    plt.legend(handles=legend_elements)
    plt.suptitle('PCA')
    #plt.show()
    if pdf is not None:
        pdf.savefig()
    plt.close()

    ######################### Plot PCA components structure

    nr = min(nPCAcomponents_to_plot, pcapts.shape[1] )
    nc = 1
    hh=2
    ww = max(14 , min(40, len(feature_names_all)/3 ) )
    fig,axs = plt.subplots(nrows=nr, ncols=nc, figsize=(ww*nc, hh*nr), sharex='col')
    for i in range(nr):
        ax = axs[i]
        dd = np.abs(pca.components_[i] )
        ax.plot( dd )
        ax.set_title('(abs of) PCA component {}, expl {:.2f} of variance (ratio)'.format(i, pca.explained_variance_ratio_[i]))

        ax.grid()
        ax.set_xlim(0, len(dd) )

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
    if pdf is not None:
        pdf.savefig()
    plt.close()


def findOutlierInds(X, qvmult = 2, qshift = 1e-2, printLog = False):
    assert qshift <= 1
    assert qvmult >= 1e-2
    dim = X.shape[1]

    qs = 100*np.array( [qshift, 1-qshift] )
    #print(qs)
    bad_inds = []
    for i in range(dim):
        # percentile can take more than one Q
        curd = X[:,i]
        qmn,qmx = np.percentile(curd, qs)
        #print(qmn,qmx)
        #qvals = qvals_ + (qvmult-1) * qvals_ * np.sign(qvals_)
        qmn = qmn - (qvmult-1) * np.abs(qmn)
        qmx = qmx + (qvmult-1) * np.abs(qmx)
        #print(qmn,qmx)

        mask_less = curd < qmn
        mask_more = curd > qmx
        #inds = np.where(np.logical_or(curd < qvals[0], curd > qvals[1] ) )[0]
        #print(len(inds), inds)
        l,m = np.sum(mask_less), np.sum(mask_more)
        if l + m > 0:
            if printLog:
                print(feature_names_all[i], l,m)
            bad_inds_cur = np.where( np.logical_or(mask_less,mask_more) )[0]
            bad_inds += list(bad_inds_cur)


    bad_inds = np.sort( np.unique(bad_inds) )
    return bad_inds

def findOutlierLimitDiscard(X, discard=1e-2, qshift = 1e-2, qvmult_start=2, printLog =False):
    '''
    discard is the proportion of the data I am ready to mark as outliers
    '''
    assert discard <= 1
    assert qshift <= 1
    assert qvmult_start >= 1e-2
    qvmult = qvmult_start
    while True:
        bad_inds = findOutlierInds(X,qvmult)
        discard_ratio = len(bad_inds)/X.shape[0]
        if printLog:
            print('qvmult={:.3f}, len(bad_inds)={} of {}, discard_ratio(pct)={:.3f}'.
                format(qvmult, len(bad_inds), X.shape[0], 100 * discard_ratio ) )
        if discard_ratio < discard:
            break
        else:
            qvmult *= 1.1

    return bad_inds, qvmult, discard_ratio

def downsample(X, skip):
    outlen = (X.shape[0] // skip  ) * skip
    res = np.zeros((X.shape[0] // skip, X.shape[1]), dtype=X.dtype)
    for i in range(skip):
        r = X[i:i+outlen:skip]
        res += r
    res /= skip

    return X
