import numpy as np
import udus_dataproc as mdp # main data proc
import re

import globvars as gv
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

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

    for chi in range(n_channels):
        if use_given_yshifts:
            yshift = yshifts[yshifti]
            yshifti += 1
        else:
            yshifts += [yshift]
        curdat = dat[chi,timinds]
        p = ax.plot( times_inwnd, curdat + yshift, alpha=0.8)
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
        p = ax.plot( times_inwnd, curdat + yshift, alpha=0.8)
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

    return yshifts
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
        for i in range(nc):
            ax = axs[0,i ]
            slicei = i // len(plot_types)
            ax.set_title(sliceNames[slicei]  )
            plot_typei = nc % len(plot_types)
            pt = plot_types[plot_typei]
            t0,t1,sln =  tuples[i]
            if t1 - t0 <= 0.1:
                continue
            yshifts = plotEvolutionMultiCh( dat, times, chan_names, interval=tuples[i],
                                 extend=0, bnd_toshow='*', ax=ax, dat_ext = dat_ext,
                                 chan_names_ext = chan_names_ext, save=0, yshifts=yshifts,
                                           interval_for_stats = interval_for_stats)

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
    for curm in markerset:
        inds = np.where(m==curm)[0]
        if len(inds):
            #print(inds)
            sc = ax.scatter(dat1[inds],dat2[inds],c=c[inds],marker=curm,alpha=alpha,
                       s=s,picker=picker)
            resinds += [ inds.tolist() ]
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

    extendIn = extend
    extendOut = extend
    ts = [max(gs,start - extendOut), max(gs,start), min(ge,start+extendIn),
          max(gs,end-extendIn), min(end,ge), min(ge,end+extendOut) ]

    while (not np.all( np.diff(ts) >= 0) ):
        extendIn /= 1.5
        ts = [max(gs,start - extendOut), max(gs,start), min(ge,start+extendIn),
            max(gs,end-extendIn), min(end,ge), min(ge,end+extendOut) ]

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


def selFeatsRegex(data, names, regexs, unique=1):
    import re
    if isinstance(regexs,str):
        regexs = [regexs]
    assert len(data) == len(names)

    namesel = []
    inds = []
    for namei,name in enumerate(names):
        for pattern in regexs:
            r = re.match(pattern, name)
            if r is not None:
                namesel += [name]
                inds += [namei]
                if unique:
                    break


    if data is None:
        datsel = None
    else:
        datsel = data[inds]

    return datsel, namesel

def computeFeatCorrel(Xfull, Xtimes_full, feature_names_all, skip, windowsz, bandPairs):
    #e.g  bandPairs = [('tremor','beta'), ('tremor','gamma'), ('beta','gamma') ]
    # compute Pearson corr coef between different band powers all chan-chan
    # parhaps I don't need corr between LFP chans
    #bandPairs = bandPairs[0:1]
    #bandPairs = bandPairs[1:2]
    #bandPairs = bandPairs[2:3]


    window_starts = np.arange(len(Xtimes_full) )[::skip]
    #window_starts_tb = raw_lfponly
    #assert len(window_starts) == len(Xtimes)

    from scipy.stats import pearsonr

    cors = []
    cor_names = []
    for bn_from,bn_to in bandPairs:
        templ_from = r'con_{}.*:\s(.*),\1'.format(bn_from)
        templ_to = r'con_{}.*:\s(.*),\1'.format(bn_to)

        datsel_from,namesel_from = selFeatsRegex(Xfull.T, feature_names_all, templ_from)
        datsel_to,namesel_to = selFeatsRegex(Xfull.T, feature_names_all, templ_to)


        for fromi in range(len(datsel_from)):
            for toi in range(len(namesel_to)):
                nfrom,nto = namesel_from[fromi], namesel_to[toi]
                if nfrom.find('LFP') >= 0 and nto.find('LFP') >= 0:
                    continue

                nfrom = nfrom.replace('con_',''); nfrom = nfrom.replace('allf_','');
                nto = nto.replace('con_',''); nto = nto.replace('allf_','')
                name = 'corr_{},{}'.format(nfrom,nto)

                corr_window = []
                for wi in range(len(window_starts)):
                    ws = window_starts[wi]
                    sl = slice(ws,ws+windowsz)

                    #r = np.correlate(datsel_from[fromi,sl], datsel_to[toi,sl] )
                    r,pval = pearsonr(datsel_from[fromi,sl], datsel_to[toi,sl])
                    corr_window += [r]
                cors += [np.hstack(corr_window) ]
                cor_names += [(name)]

    return cors,cor_names

def prepareLegendElements(mrk,mrknames,colors,task, skipExt = False):
    legend_elements = []
    if task is not None:
        if isinstance(task,str):
            task = [task]
        else:
            assert isinstance(task,list)

    for clrtype in colors.keys():
        if clrtype == 'neut':
            continue
        if clrtype in ['move', 'hold'] and clrtype not in task:
            continue

        for m,mn in zip(mrk,mrknames):
            if skipExt and len(mn) > 0:  #mn == '' for the "meat" part of the interval
                continue

            legel_ = mpl.lines.Line2D([0], [0], marker=m, color='w', label=clrtype+mn,
                                        markerfacecolor=colors[clrtype], markersize=8)
            #print(clrtype+mn)


            legend_elements += [legel_]

    legel_unlab = mpl.lines.Line2D([0], [0], marker='o', color='w', label='unlab'+mn,
                                markerfacecolor=colors['neut'], markersize=8)
    legend_elements += [legel_unlab]
    return legend_elements
