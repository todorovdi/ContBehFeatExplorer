import numpy as np
import udus_dataproc as mdp # main data proc
import re

import globvars as gv
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils

import globvars as gv
import os

import multiprocessing as mpr
import mne

from scipy.stats import pearsonr

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
        plt.savefig(os.path.join(gv.dir_fig,figname))
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
        hh = 8
        fig,axs = plt.subplots(nrows=nc, ncols=1, figsize = (ww, hh))
        plt.subplots_adjust(left=0.01, right=0.99,top=0.99, bottom = 0.4)
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
                    print(chan_names[ind],m,s)
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
                s = np.std(dat[ind]);
                if printMeans:
                    print(chan_names[ind],m,s)
                ax.axvline(x=m,c='r',ls=':')
                ax.set_title(chan_names[ind])

def plotIntervalData(dat, chan_names, interval, times = None, raw=None, plot_types = ['psd'],
                     extend = [3,3,3,3], sfreq=256,
                     dat_ext = None, chan_names_ext=None, ww=8, hh=3, fmax=None):
    start,end,int_type  = interval

    assert raw is not None or (times is not None)

    ts, inds, sliceNames = getIntervalSurround( start,end, extend, raw=raw, times=times)
    #inds = raw.time_as_index(
    #    [start - extend, start, start+extend, end-extend, end, end+extend])
    prestarti, starti, poststarti, preendi, endi, postendi = inds
    print('Interval {} duration is {:.2f}, extend={}'.format(int_type, end-start,extend) )

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


def plotMultiMarker(ax,dat1, dat2, c, m, alpha=None, s=None, picker=False,
                    emph_inds=[], custom_marker_size={}):
    '''  returns tuple -- list , list of lists , list of artists '''
    assert len(dat1) == len(dat2)
    assert len(c) == len(m)
    assert len(c) == len(dat1)
    resinds = []
    markerset = list( set( m ) )  # remove duplicates
    scs = []
    m = np.array(m)
    c = np.array(c)
    if isinstance(s,list):
        s = np.array(s)
    #if len(emph_inds) > 0:

    for curm in markerset:
        inds = np.where(m==curm)[0]

        #print(curm, len(inds))
        if len(inds):
            #print(inds)
            if isinstance(s,int):
                #s_cur = s
                s_cur = custom_marker_size.get(curm, s)
                ss = [s_cur]*len(inds)
            elif s is None:
                ss = None
            else:
                ss = s[inds]

            sc = ax.scatter(dat1[inds],dat2[inds],c=c[inds],marker=curm,alpha=alpha,
                       s=ss,picker=picker)
            resinds += [ inds.tolist() ]  # yes, I want list of lists
            scs += [sc]
    return markerset,resinds,scs


def getIntervalSurround(start,end, extend, raw=None, times=None, verbose = False,
                        wbd_sec=None):
    assert raw is not None or (times is not None)
    '''
    start and end are in seconds
    times are in seconds, can have large (but constant) gaps
    times can start not from zero, but start and end assume we start from zero

    returns BOUNDARIES of surrdound intervals

    it uses dt computed from times (if we don't give raw). It's dangerous if there are gaps
    '''

    assert end >= start

    if not (isinstance(extend, list) ):
        extend = 4*[extend]
    else:
        assert len(extend) == 4

    if times is None:
        assert raw is not None
        times = raw.times

    if wbd_sec is None:
        dtimes = np.diff(times)
        mndt,mxdt = np.min( dtimes), np.max(dtimes)
        assert (mxdt-mndt) < 1e-10, (mxdt-mndt)
        dt = mxdt
        wbd_sec = np.vstack( [times, times] )
        wbd_sec[1] += dt

    assert wbd_sec.shape[1] == len(times)

    ge = times[-1]
    gs = times[0]

    if end == start:
        ts = [end] * 6
    else:
        #extendIn = extend
        #extendOut = extend
        extendOutL = extend[0]
        extendInL  = extend[1]
        extendInR  = extend[2]
        extendOutR = extend[3]
        end = min(end,ge)
        start = max(gs,start)
        ts = [max(gs,start - extendOutL), start, min(ge,start+extendInL),
            max(gs,end-extendInR), end, min(ge,end+extendOutR) ]

        while (not np.all( np.diff(ts) >= 0) ):
            extendInL /= 1.5
            extendInR /= 1.5
            ts = [max(gs,start - extendOutL), start, min(ge,start+extendInL),
                max(gs,end-extendInR), end, min(ge,end+extendOutR) ]

    if verbose:
        print('extendIn used', extendInL, extendInR)
    assert np.all( np.diff(ts) >= 0)
    if times is None:
        tsis = raw.time_as_index(ts)
    else:
        # return window indices where intervals start
        ts = np.array(ts)
        mask = np.logical_and(ts >= wbd_sec[0],ts < wbd_sec[1])
        tsis = np.where(mask)
        assert len(tsis) == len(ts)
        assert np.diff(tsis) >= 0

        #allbins = ( times / dt  ).astype(int)
        #ts_ =  ( ts /    dt  ).astype(int)
        #tsis = np.searchsorted(allbins, ts_)

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
    '''data can be None if I only want to select feat names'''
    import re
    if isinstance(regexs,str):
        regexs = [regexs]
    if data is not None:
        if isinstance( data, np.ndarray):
            assert len(data) == len(names)
        elif isinstance( data, list):
            assert len(data[0]) == len(names)
        else:
            raise ValueError('Wrong data type {}'.format(type(data) ) )

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
        if isinstance( data, np.ndarray):
            datsel = data[inds]
        elif isinstance( data, list):
            datsel = []
            for subdat in data:
                datsel += [subdat[inds] ]

    return datsel, namesel

def robustMean(dat,axis=None, q=0.05, q_compl= None,
               ret_aux = False, ret_std=False, per_dim = False, pos = False):
    if q_compl is None:
        q_compl = q
    if max(q,q_compl) < 1e-10:
        return np.mean(dat)

    pcts = np.array([q, 1-q_compl])  * 100
    if pos:
        pcts[0 ] = 0
    assert pcts[1] <= 100
    r = np.percentile(dat, pcts,   axis=axis)
    if r.ndim > 1:
        qvmn = r[0,:]
        qvmx = r[1,:]
    else:
        qvmn,qvmx = r

    res = None
    res_std = None
    if per_dim and dat.ndim > 1:
        res = []
        res_std = []
        for d in range(dat.shape[0]):
            dcur = dat[d]
            seldat =dcur[ (dcur<=qvmx[d]) * (dcur>=qvmn[d]) ]
            rr = np.mean( seldat  )
            res.append(rr)
            if ret_std:
                rr_std = np.std(  seldat )
                res_std.append(rr_std)
        res = np.array(res)
        if ret_std:
            res_std = np.array(res_std)
    else:
        seldat = dat[ (dat<=qvmx) * (dat>=qvmn) ]
        res = np.mean(  seldat )
        if ret_std:
            res_std = np.std(  seldat )

    if ret_aux:
        if ret_std:
            return res, res_std, qvmn, qvmx
        else:
            return res, qvmn, qvmx
    else:
        if ret_std:
            return res, res_std
        else:
            return res
    #np.mean(  dat[dat<q

def robustMeanPos(dat,axis=None, q=0.05, per_dim=False):
    return robustMean(dat,axis=axis, per_dim=per_dim, q=0., q_compl=q)
    #if q < 1e-10:
    #    return np.mean(dat)
    #else:
    #    qvmx = np.quantile(dat, 1-q, axis=axis)
    #    # dat[dat<=qvmx] is a flattened array anyway
    #    return np.mean(  dat[dat<=qvmx] )

def _feat_correl(arg):
    bn_from,bn_to,fromi,toi,name,window_starts,windowsz,skip,dfrom,dto,oper,pos = arg

    corr_window = []
    q = 0.05
    #q = 0
    if pos:
        mfrom = robustMeanPos(dfrom, q=q)    # global mean
        mto   = robustMeanPos(dto, q=q)
    else:
        mfrom = robustMean(dfrom, q=q)    # global mean
        mto   = robustMean(dto, q=q  )

    import utils
    stride_ver = True
    if stride_ver:
        win = (windowsz,)
        step = (skip,)
        stride_view_dfrom = utils.stride(dfrom, win=win, stepby=step )
        stride_view_dto   = utils.stride(dto, win=win, stepby=step )
        rr = np.mean( (dfrom - mfrom[:,None] ) * (dto - mto[:,None] ) , axis=-1)
    else:
        for wi in range(len(window_starts)):
            ws = window_starts[wi]
            sl = slice(ws,ws+windowsz)

            #r = np.correlate(datsel_from[fromi,sl], datsel_to[toi,sl] )
            if oper == 'corr':
                #r,pval = pearsonr(dfrom[sl], dto[sl])
                # want to measure deviations from global means
                r =  np.mean( (dfrom[sl] - mfrom) *(dto[sl] - mto) )  #mean in time
            elif oper == 'div':
                r = np.mean( dfrom[sl] / dto[sl] )
            corr_window += [r]
        rr = np.hstack(corr_window)

    return rr,bn_from,bn_to,fromi,toi,window_starts,name,oper,pos

def _feat_correl3(arg):
    resname,bn_from,bn_to,pc,fromi,toi,windowsz,skip,dfrom,dto,mfrom,mto,oper,pos = arg

    q = 0.05
    if mfrom is None:
        if pos:
            mfrom = robustMeanPos(dfrom, q=q)    # global mean
        else:
            mfrom = robustMean(dfrom, q=q)    # global mean
    if mto is None:
        if pos:
            mto   = robustMeanPos(dto, q=q)
        else:
            mto   = robustMean(dto, q=q  )

    import utils

    assert dfrom.size == dto.size, (dfrom.shape, dto.shape, bn_from,bn_to,fromi,toi,resname)
    assert dfrom.ndim == 1

    # to agree with Hjorth
    ndb = len(dfrom)
    padlen = windowsz-skip
    #if  int(ndb / windowsz) * windowsz - ndb == 0:  # dirty hack to agree with pandas version
    #    padlen += 1
    dfrom = np.pad(dfrom, [(0), (padlen) ], mode='edge' )
    dto = np.pad(dto, [(0), (padlen) ], mode='edge' )

    win = (windowsz,)
    step = (skip,)
    stride_view_dfrom = utils.stride(dfrom, win=win, stepby=step )
    stride_view_dto   = utils.stride(dto, win=win, stepby=step )
    if oper == 'corr':
        rr = np.mean( (stride_view_dfrom - mfrom ) * (stride_view_dto - mto ) , axis=-1)
    elif oper == 'div':
        rr = stride_view_dfrom / stride_view_dto
    else:
        raise ValueError('wrong oper {}'.format(oper) )
    #import ipdb; ipdb.set_trace()

    # removing invalid due to padding
    #rr = rr[0:-windowsz//skip]
    sl = slice(windowsz//skip - 1,None,None)
    rr = rr[sl]

    #ndb = len(dfrom)
    #padlen = windowsz-skip
    #if  int(ndb / windowsz) * windowsz - ndb == 0:  # dirty hack to agree with pandas version
    #    padlen += 1

    del stride_view_dfrom
    del stride_view_dto
    del dfrom
    del dto

    wbd = None  # to save memory
    #pred = np.arange(ndb + padlen)
    #wnds = utils.stride(pred, (windowsz,), (skip,) )
    #wbd = np.vstack( [ wnds[:,0], wnds[:,-1] + 1 ] )
    #wbd = wbd[:,sl] - padlen

    #window_boundaries_st =  np.arange(0,ndb - windowsz, skip ) # use before padding
    #window_boundaries_end = window_boundaries_st + windowsz
    #wbd = np.vstack( [ window_boundaries_st, window_boundaries_end] )
    #print(dfrom.shape, stride_view_dto.shape)

    return rr,resname,bn_from,bn_to,pc,fromi,toi,wbd,oper,pos

def computeFeatOrd3(raws, names, defnames, skip, windowsz, band_pairs,
                    parcel_couplings, LFP2parcel_couplings, LFP2LFP_couplings,
                    res_group_id = 9, n_jobs=None, positive=1, templ = None,
                    roi_labels = None, sort_keys=None, printLog=False, means=None):
    '''
    dat is chans x timebins
    windowz is in bins of Xtimes_full
    returns lists
    '''
    #e.g  bandPairs = [('tremor','beta'), ('tremor','gamma'), ('beta','gamma') ]
    # compute Pearson corr coef between different band powers all chan-chan

    #TODO: the names I receive here are not pure channel names so it won't work. Or maybe I don't need it
    #sides_,groupis_,parcelis_,compis_ = utils.parseMEGsrcChnamesShortList(names)

    # this is if we want to really compute second order features. I don't use
    # it this way
    if templ is None:
        templ = r'con_{}.*:\s(.*),\1'

    cors = []
    cor_names = []
    args = []
    ctr = 0
    for bn_from,bn_to,oper in band_pairs:
        templ_from = templ.format(bn_from)
        templ_to   = templ.format(bn_to)

        #datsel_from,namesel_from = selFeatsRegex(dat, names, templ_from)
        #datsel_to,namesel_to = selFeatsRegex(dat, names, templ_to)

        # note that these names CAN be different if we couple LFP HFO to src
        # normal freqs
        names_from = names[bn_from]
        names_to   = names[bn_to]

        if bn_from.find('HFO') < 0 and bn_to.find('HFO') < 0:
            for pc in parcel_couplings:
                (pi1,pi2) = pc
                ind_pairs = parcel_couplings[pc]  # ind_pairs won't work because it is indices not in band array
                for (i,j) in ind_pairs:
                    rev = (pi1 != pi2) and (bn_from != bn_to)

                    dfrom = raws[bn_from][i][0][0]
                    dto   = raws[bn_to][j][0][0]
                    assert dfrom.size > 1
                    assert dto.size > 1

                    chn1 = names_from[i]
                    chn2 = names_to[j]
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn1)
                    side2, gi2, parcel_ind2, si2  = utils.parseMEGsrcChnameShort(chn2)

                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_from,side1,res_group_id,pi1,0)
                    newchn2 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,  side2,res_group_id,pi2,0)

                    resname = '{}_{},{}'.format(oper,newchn1,newchn2)
                    # we won't have final name before averaging
                    if means is not None:
                        #m1 = means[bn_from][chn1]
                        #m2 = means[bn_to][chn2]
                        m1 = means[bn_from][i]
                        m2 = means[bn_to][j]
                    else:
                        m1,m2=None,None
                    arg = resname,bn_from,bn_to,pc,i,j,windowsz,skip,dfrom,dto,m1,m2,oper,positive
                    args += [arg]

                    if rev: # reversal of bands, not indices (they are symmetric except when division)
                        # change bands. So I need same channels (names and
                        # data) but with exchanged bands
                        dfrom = raws[bn_to][i][0][0]
                        dto   = raws[bn_from][j][0][0]

                        # change bands
                        chn1 = names_to[i]
                        chn2 = names_from[j]
                        side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn1)
                        side2, gi2, parcel_ind2, si2  = utils.parseMEGsrcChnameShort(chn2)

                        # change bands
                        newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,side1,res_group_id,pi1,0)
                        newchn2 = '{}_msrc{}_{}_{}_c{}'.format(bn_from,  side2,res_group_id,pi2,0)

                        resname = '{}_{},{}'.format(oper,newchn1,newchn2)
                        # we won't have final name before averaging
                        if means is not None:
                            #m1 = means[bn_to][chn1]
                            #m2 = means[bn_from][chn2]
                            m1 = means[bn_to][i]
                            m2 = means[bn_from][j]
                        else:
                            m1,m2=None,None
                        arg = resname,bn_to,bn_from,pc,i,j,windowsz,skip,dfrom,dto,m1,m2,oper,positive
                        args += [arg]

                #newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_from,side1,res_group_id,pi1,0)
                #newchn2 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,  side2,res_group_id,pi2,0)

        #XOR instead of AND, allow one to has HFO
        #if bool(bn_from.find('HFO') < 0) ^ bool(bn_to.find('HFO') <= 0):
        for pc in LFP2parcel_couplings:
            (chn1,pi2) = pc
            ind_pairs = LFP2parcel_couplings[pc]

            # do I need beta LFP vs tremor src AND  beta src vs tremor LFP ?
            # maybe yes because if I duplicate paris in the input I'd have to
            # filter out it in LFP2LFP
            for (i,j) in ind_pairs:
                rev = False
                ind_lfp = i
                ind_src = j
                # I cannot reverse then
                # i -- always LFP, j -- alwyas src
                if bn_from.find('HFO') >= 0:
                    # effind_from index we'll use to access raws[bn_from]
                    effind_from = names_from.index( defnames[ind_lfp] )
                    effind_to   = ind_src

                    chn_src = names_to[ind_src]
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn_src)
                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,side1,res_group_id,parcel_ind1,0)

                    chn = names_from[effind_from]
                    newchn2 = '{}_{}'.format(bn_from, chn )
                # I cannot reverse then
                elif bn_to.find('HFO') >= 0:
                    effind_from = ind_src
                    effind_to   = names_to.index( defnames[ind_lfp] )

                    chn_src = names_from[ind_src]
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn_src)
                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_from,side1,res_group_id,parcel_ind1,0)

                    chn = names_to[effind_to]
                    newchn2 = '{}_{}'.format(bn_to, chn )
                elif bn_to.find('HFO') < 0 and  bn_to.find('HFO') < 0 :
                    effind_from = ind_src
                    effind_to   = ind_lfp

                    chn_src = names_from[ind_src]
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn_src)
                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_from,side1,res_group_id,parcel_ind1,0)

                    chn = names_to[effind_to]
                    newchn2 = '{}_{}'.format(bn_to, chn )

                    rev = bn_from != bn_to  # if bands are the same, no need to reverse
                else:
                    raise ValueError('smoething is wrong {},{}'.format(bn_from,bn_to) )

                resname = '{}_{},{}'.format(oper,newchn1,newchn2)
                # we won't have final name before averaging

                dfrom = raws[bn_from][effind_from][0][0]
                dto   = raws[bn_to][effind_to][0][0]
                assert dfrom.size > 1
                assert dto.size > 1

                #name = '{}_{},{}'.format(oper,nfrom,nto)
                # we won't have final name before averaging
                if means is not None:
                    #m1 = means[bn_from][names_from[effind_from] ]
                    #m2 = means[bn_to][names_to[effind_to]]
                    m1 = means[bn_from][effind_from ]
                    m2 = means[bn_to][effind_to]
                else:
                    m1,m2=None,None
                arg = resname,bn_from,bn_to,pc,effind_from,effind_to,windowsz,skip,dfrom,dto,m1,m2,oper,positive
                args += [arg]

                ######################3
                if rev:
                    # change order
                    effind_from = ind_lfp
                    effind_to   = ind_src

                    chn_src = names_from[ind_src]  # we still need source index
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn_src)
                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,side1,res_group_id,pi1,0)

                    chn = names_to[effind_to]
                    newchn2 = '{}_{}'.format(bn_from, chn )

                    resname = '{}_{},{}'.format(oper,newchn1,newchn2)

                    # not that we exchanged effinds
                    d1   = raws[bn_to][effind_from][0][0]
                    d2   = raws[bn_from] [effind_to][0][0]
                    #d1 = raws[bn_to][effind_from][0][0]
                    #d2   = raws[bn_from][effind_to][0][0]
                    assert d1.size > 1
                    assert d2.size > 1

                    #name = '{}_{},{}'.format(oper,nfrom,nto)
                    # we won't have final name before averaging
                    if means is not None:
                        #m1 = means[bn_to][names_to[effind_from] ]
                        #m2 = means[bn_from][names_from[effind_to]]
                        m1 = means[bn_to][effind_from ]
                        m2 = means[bn_from][effind_to]
                    else:
                        m1,m2=None
                    arg = resname,bn_to,bn_from,pc,effind_from,effind_to,windowsz,skip,d1,d2,m1,m2,oper,positive
                    args += [arg]

        for pc in LFP2LFP_couplings:
            (chn1,chn2) = pc
            if oper == 'div' and chn1 == chn2 and bn_from == bn_to:
                continue
            ind_pairs = LFP2LFP_couplings[pc]
            for (i,j) in ind_pairs:
                rev = (i != j)  # here it makes sense only if I compute cross-LFP (which I usually don't)
                # I cannot reverse then
                if bn_from.find('HFO') >= 0:
                    effind_from = names_from.index( defnames[i] )
                    effind_to   = j
                # I cannot reverse then
                elif bn_to.find('HFO') >= 0:
                    effind_from = i
                    effind_to   = names_to.index( defnames[j] )
                elif bn_to.find('HFO') < 0 and  bn_to.find('HFO') < 0 :
                    effind_from = i
                    effind_to   = j
                elif bn_to.find('HFO') >= 0 and bn_to.find('HFO') >= 0:
                    effind_from = names_from.index( defnames[i] )
                    effind_to   = names_to.index( defnames[j] )
                else:
                    raise ValueError('smoething is wrong {},{}'.format(bn_from,bn_to) )

                newchn1 = '{}_{}'.format(bn_from, names_from[effind_from] )
                newchn2 = '{}_{}'.format(bn_to, names_to[effind_to] )

                resname = '{}_{},{}'.format(oper,newchn1,newchn2)

                dfrom = raws[bn_from][effind_from][0][0]
                dto   = raws[bn_to][effind_to][0][0]
                assert dfrom.size > 1
                assert dto.size > 1

                #name = '{}_{},{}'.format(oper,nfrom,nto)
                # we won't have final name before averaging
                if means is not None:
                    #m1 = means[names_from[effind_from] ]
                    #m2 = means[names_to[effind_to]]
                    m1 = means[bn_from][effind_from ]
                    m2 = means[bn_to][effind_to]
                else:
                    m1,m2=None,None
                arg = resname,bn_from,bn_to,pc,effind_from,effind_to,windowsz,skip,dfrom,dto,m1,m2,oper,positive
                args += [arg]

                if rev:
                    newchn1 = '{}_{}'.format(bn_to,   names_from[effind_from] )
                    newchn2 = '{}_{}'.format(bn_from, names_to[effind_to] )
                    resname = '{}_{},{}'.format(oper,newchn1,newchn2)

                    d1 = raws[bn_to][effind_from][0][0]
                    d2 = raws[bn_from][effind_to][0][0]
                    assert d1.size > 1
                    assert d2.size > 1

                    if means is not None:
                        #m1 = means[bn_to][names_to[effind_to] ]
                        #m2 = means[bn_from][names_from[effind_to]]
                        m1 = means[bn_to][effind_from ]
                        m2 = means[bn_from][effind_to]
                    else:
                        m1,m2=None
                    #name = '{}_{},{}'.format(oper,nfrom,nto)
                    # we won't have final name before averaging
                    arg = resname,bn_to,bn_from,pc,effind_from,effind_to,windowsz,skip,d1,d2,m1,m2,oper,positive
                    args += [arg]

        # then I need to do within parcel averaging

    # for debug which names get included only
    #return None, sorted(set([arg[0] for arg in args] ))

    if n_jobs is None:
        from globvars import gp
        ncores = max(1, min(len(args) , mpr.cpu_count()-gp.n_free_cores) )
    elif n_jobs == -1:
        ncores = mpr.cpu_count()
    else:
        ncores = n_jobs
    if ncores > 1:
        #if ncores > 1:
        pool = mpr.Pool(ncores)
        print('high ord feats:  Sending {} tasks to {} cores'.format(len(args), ncores))
        res = pool.map(_feat_correl3, args)

        pool.close()
        pool.join()
    else:
        res = []
        for arg in args:
            res += [ _feat_correl3(arg) ]

    dct = {}
    dct_nums = {}
    # wbd is same for all
    for r in res:
        rr,resname,bn_from,bn_to,pc,fromi,toi,wbd,oper,pos  = r
        # make averages

        #nfrom = pc[0]
        #nto = pc[1]
        #name = '{}_{},{}'.format(oper,nfrom,nto)
        dct_key = resname
        if dct_key not in  dct:
            dct[dct_key] = rr
            dct_nums[dct_key] = 1
        else:
            dct[dct_key] += rr
            dct_nums[dct_key] += 1

    min_num = np.inf
    resname_min_num = ''
    for dct_key in dct:
        rr = dct[dct_key]
        num =dct_nums[dct_key]
        if num < min_num:
            min_num = num
            resname_min_num = dct_key
        cors += [  rr / num  ]
        cor_names += [ dct_key ]

    ############
    ndb = len(dfrom)
    padlen = windowsz-skip
    #if  int(ndb / windowsz) * windowsz - ndb == 0:  # dirty hack to agree with pandas version
    #    padlen += 1

    sl = slice(windowsz//skip - 1,None,None)
    pred = np.arange(ndb + padlen)
    wnds = utils.stride(pred, (windowsz,), (skip,) )
    wbd = np.vstack( [ wnds[:,0], wnds[:,-1] + 1 ] )
    wbd = wbd[:,sl] - padlen

    #print( dct_nums )
    #print(resname_min_num, num)

    return cors,cor_names, dct_nums, wbd


def computeFeatOrd2(dat, names, skip, windowsz, band_pairs,
                      n_free_cores=2, positive=1, templ = None,
                     roi_pairs=None,
                    roi_labels = None, sort_keys=None, printLog=False):
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


    #window_starts = np.arange(len(Xtimes_full) )[::skip]
    window_starts = np.arange( 0, dat.shape[-1],  skip, dtype=int)
    #window_starts_tb = raw_lfponly
    #assert len(window_starts) == len(Xtimes)

    #TODO: the names I receive here are not pure channel names so it won't work. Or maybe I don't need it
    #sides_,groupis_,parcelis_,compis_ = utils.parseMEGsrcChnamesShortList(names)

    # this is if we want to really compute second order features. I don't use
    # it this way
    if templ is None:
        templ = r'con_{}.*:\s(.*),\1'

    cors = []
    cor_names = []
    args = []
    ctr = 0
    for bn_from,bn_to,oper in band_pairs:
        templ_from = templ.format(bn_from)
        templ_to   = templ.format(bn_to)

        datsel_from,namesel_from = selFeatsRegex(dat, names, templ_from)
        datsel_to,namesel_to = selFeatsRegex(dat, names, templ_to)

        for fromi in range(len(namesel_from)):
            for toi in range(len(namesel_to)):
                # within same band we don't want to compute BOTH i->j and j->i
                if bn_from == bn_to and fromi < toi:
                    continue

                nfrom,nto = namesel_from[fromi], namesel_to[toi]
                if nfrom.find('LFP') >= 0 and nto.find('LFP') >= 0:
                    regex = '.*(LFP.[0-9]+).*'
                    r = re.match(regex, nfrom)
                    assert len(r.groups() ) == 1
                    rfrom = r.groups()[0]

                    r = re.match(regex, nto)
                    assert len(r.groups() ) == 1
                    rto = r.groups()[0]

                    if not ( (oper == 'div') and rfrom == rto ):
                        continue

                #if nfrom.find('msrc') >= 0 and nto.find('msrc') >= 0:
                nfrom_nice = utils.getMEGsrc_chname_nice(nfrom,roi_labels, sort_keys, preserve_prefix=0)
                nto_nice   = utils.getMEGsrc_chname_nice(nto,  roi_labels, sort_keys, preserve_prefix=0)
                if nfrom.find('LFP') < 0:
                    nfrom_nice = 'msrc_' + nfrom_nice
                else:
                    nfrom_nice = nfrom[nfrom.find('LFP') : ]


                if nto.find('LFP') < 0:
                    nto_nice = 'msrc_' + nto_nice
                else:
                    nto_nice = nto[nto.find('LFP') : ]

                #if printLog:
                #    print('from {} = {}, to {} = {}'.format(nfrom, nfrom_nice, nto, nto_nice) )

                roi_pair_allowed = False
                for roi_from,roi_to in roi_pairs:
                    match_from = re.match(roi_from, nfrom_nice)
                    match_to = re.match(roi_to, nto_nice)
                    #if nfrom_nice.find(roi_from) >= 0 and nto_nice.find(roi_to) >= 0:
                    if match_from is not None and match_to is not None:
                        roi_pair_allowed  = True
                        break


                if not roi_pair_allowed:
                    break

                if printLog:
                    ctr += 1
                    print('{}: **from {} = {}, to {} = {}'.format(ctr, nfrom, nfrom_nice, nto, nto_nice) )

                nfrom = nfrom.replace('con_',''); nfrom = nfrom.replace('allf_','');
                nto = nto.replace('con_',''); nto = nto.replace('allf_','')
                name = '{}_{},{}'.format(oper,nfrom,nto)

                dfrom = datsel_from[fromi]
                dto = datsel_to[toi]
                assert dfrom.size > 1
                assert dto.size > 1
                arg = bn_from,bn_to,fromi,toi,name,window_starts,windowsz,skip,dfrom,dto,oper,positive
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

    print(len(args) )
    return 0,0

    ncores = max(1, min(len(args) , mpr.cpu_count()-n_free_cores) )
    if ncores > 1:
        #if ncores > 1:
        pool = mpr.Pool(ncores)
        print('high ord feats:  Sending {} tasks to {} cores'.format(len(args), ncores))
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

def colNames2Rgba(cols):
    assert isinstance(cols,list) or isinstance(cols,np.ndarray)
    import matplotlib.colors as mcolors
    res = [0]*len(cols)
    for k in range(len(cols)):
        cname = cols[k]
        if isinstance(cname,str):
            res[k] = mcolors.to_rgba(cname)
    return res

def getImporantCoordInds(components, nfeats_show = 140, q=0.8, printLog = 1):
    # components is N x nfeats
    assert components.shape[0] <= components.shape[1]
    strong_inds_pc = []
    strongest_inds_pc = []
    nfeats_show_pc = min(nfeats_show, components.shape[1] // len(components) )
    if printLog:
        print('Per component we use {} feats'.format(nfeats_show_pc) )
    inds_toshow = []
    for i in range(len(components) ):
        dd = np.abs(components[i  ] )

        inds_sort = np.argsort(dd)  # smallest go first
        inds_toshow_cur = inds_sort[-nfeats_show_pc:]
        inds_toshow += [inds_toshow_cur]

        #dd_toshow = dd[inds_toshow_cur]
        strong_inds = np.where(dd   > np.quantile(dd,q) ) [0]
        #print(i, strong_inds )
        strongest_ind = np.argmax(dd)
        assert  strongest_ind == inds_toshow_cur[-1]
        strongest_inds_pc += [strongest_ind]

        #strong_inds_pc += [strong_inds.copy() ]
        strong_inds_pc += [inds_sort[-nfeats_show_pc//2:]  ]

    inds_toshow = np.sort( np.unique( inds_toshow) )

    return inds_toshow, strong_inds_pc, strongest_inds_pc

def prepColorsMarkers(side_letter, anns, Xtimes,
               nedgeBins, windowsz, sfreq, totskip, mrk,mrknames,
               color_per_int_type, extend = 3, defmarker='o', neutcolor='grey',
                     convert_to_rgb = False, dataset_bounds = None, wbd=None ):
    # Xtimes are in seconds, dataset_bounds too
    # Xtimes are window starts
    # wbd is 2 x len(Xtimes)
    #windowsz is in 1/sfreq bins
    #Xtimes_almost is in bins whose size comes from gen_features
    #Xtimes is in possible smaller bins
    '''
        mrknames not used
        output length = len(Xtimes)
    '''
    hsfc = side_letter

    if not (isinstance(extend, list) ):
        extend = 4*[extend]
    else:
        assert len(extend) == 4

    extendInL  = extend[0]
    extendOutL = extend[1]
    extendInR  = extend[2]
    extendOutR = extend[3]

    if wbd is None:
        wbd = np.vstack( [Xtimes, Xtimes] )
        wbd[1] += windowsz

    if convert_to_rgb:
        import matplotlib.colors as mcolors
        for k in color_per_int_type:
            cname = color_per_int_type[k]
            if isinstance(cname,str):
                color_per_int_type[k] = mcolors.to_rgb(cname)


    #hsfc = 'L'; print('Using not hand side (perhabs) for coloring')
    annot_color_perit = {}
    for k in color_per_int_type:
        annot_color_perit[ '{}_{}'.format(k,hsfc)   ] = color_per_int_type[k]
    #for task in tasks:
    #    annot_color_perit[ '{}_{}'.format(task, hsfc) ] = color_per_int_type[task]

    assert Xtimes[0] <= 1e-10

    colors =  [neutcolor] * len(Xtimes)
    markers = [defmarker] * len(Xtimes)

    globend = Xtimes[-1]
    globstart = 0

    ivalis = utils.ann2ivalDict(anns)
    #print( ivalis)
    # assume that intervals for each label are ordered and don't overlap

    #for it in ivalis:
    for it in annot_color_perit:
        if it not in ivalis:  #neut_{} will not be
            continue
        descr = it
        #prev_interval = None
        inds_set = []
        for ivli,interval in enumerate(ivalis[it] ):
            #for an in anns:
            #   for descr in annot_color_perit:
            #       if an['description'] != descr:
            #           continue
            col = annot_color_perit[descr]

            #start = an['onset']
            #end = start + an['duration']
            start,end, it_ = interval

            if dataset_bounds is not None:
                interval_dataset_ind = None
                for di in range(len(dataset_bounds)):
                    dlb,drb = dataset_bounds[di]
                    if start >= dlb and end <= drb:
                        interval_dataset_ind = di
                        break
                assert interval_dataset_ind  is not None
                globend = drb     # in seconds
                globstart = dlb

            # TODO: give original times here
            # not it based on window starts only
            timesBnds, indsBnd, sliceNames = getIntervalSurround( start,end, extend,
                                                                times=Xtimes)

            #cycle over surround intervals
            for ii in range(len(indsBnd)-1 ):
                # do not set prestart, poststart for left recording edge
                if start <= globstart + nedgeBins/sfreq and ii in [0,1]:
                    continue
                # do not set preend, posted for right recording edge
                #globend = Xtimes_almost[-1] + nedgeBins/sfreq
                if  globend - end <= nedgeBins/sfreq and ii in [3,4]:
                    continue

                # for notrem we won't make outer surrounds
                if  it.startswith('notrem') and \
                        sliceNames[ii] not in ['poststart', 'main', 'preend']:  #but not 2!
                    continue

                # if we prestart and prev one start is close
                # I could also add windowsz_sec - windowsz_sec // totskip
                if ivli > 0: # assume that intervals are consequtive
                    prev_interval = ivalis[it][ivli-1]
                    start_prev, end_prev, it_prev = prev_interval
                    assert start - end_prev >= 0
                    if it.startswith('trem') and sliceNames[ii] == 'prestart' and \
                            (start - end_prev) < (extendOutL + extendOutR):
                        print('Skipping', interval)
                        continue

                # if we postend and next one start is close
                if ivli < len(ivalis[it]) - 1:
                    next_interval = ivalis[it][ivli+1]
                    start_next, end_next, it_next = next_interval
                    assert start_next - end >= 0
                    if it.startswith('trem') and sliceNames[ii] == 'postend' and \
                            (start_next - end) < (extendOutL + extendOutR):
                        print('Skipping2', interval)
                        continue

                # window size correction because it goes _before_
                #bnd0 = min(len(Xtimes)-1, indsBnd[ii]   + windowsz // totskip -1   )
                #bnd1 = min(len(Xtimes)-1, indsBnd[ii+1] + windowsz // totskip -1   )

                # window size correction because it goes _after_
                bnd0 = min(len(Xtimes)-1, indsBnd[ii]   )
                bnd1 = min(len(Xtimes)-1, indsBnd[ii+1] )
                #
                window_inds = np.arange( bnd0, bnd1 )

                #inds2 = slice( indsBnd[ii], indsBnd[ii+1] )
                #markers[inds2] = mrk[ii]

                #if sliceNames[ii] == 'main':
                #    print(ivli,interval, timesBnds, indsBnd,inds2)
                for jj in window_inds:
                    colors [jj] = col
                    markers[jj] = mrk[ii]

                inds_set += list(window_inds)
        #print('!!!!! ',it,len(inds_set), inds_set)
    return colors,markers

def plotPCA(pcapts,pca, nPCAcomponents_to_plot,feature_names_all, colors, markers,
            mrk, mrknames, color_per_int_type, task,
            pdf=None,neutcolor='grey', nfeats_show = 50, q = 0.8, title_suffix = ''):


    if hasattr(pca, 'components_'):
        pt_type = 'PCA'
        components = pca.components_
    elif hasattr(pca, 'coefs_'):
        pt_type = 'LDA'
        components = pca.coefs_
    else:
        pt_type = 'unk'
        components = None

    toshow_decide_0th_component = 0

    ##################  Plot PCA
    nc = min(nPCAcomponents_to_plot, pcapts.shape[1] );
    #if nc == 1 and pcapts.shape[1] == 2:
    #    nc == 2
    nr = 1; ww = 5; hh = 4
    if nc == 1:
        ww = 12
    fig,axs = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*ww,nr*hh))
    if not isinstance(axs,np.ndarray):
        axs = np.array([axs])
    ii = 0
    while ii < nc:
        indx = ii
        indy = ii+1
        if indy >= pcapts.shape[1]:
            indy = 0
        ax = axs[ii];  ii+=1
        plotMultiMarker(ax, pcapts[:,indx], pcapts[:,indy], c=colors, m = markers, alpha=0.5);
        ax.set_xlabel('{} comp {}'.format(pt_type,indx) )
        ax.set_ylabel('{} comp {}'.format(pt_type,indy) )

    legend_elements = prepareLegendElements(mrk,mrknames,color_per_int_type, task )

    plt.legend(handles=legend_elements)
    plt.suptitle(pt_type + title_suffix)
    #plt.show()
    if pdf is not None:
        pdf.savefig()
        plt.close()

    ######################### Plot PCA components structure
    if components is None:
        return None
    dd = np.abs(components[0] )

    nr = min(nPCAcomponents_to_plot, pcapts.shape[1] )
    if toshow_decide_0th_component:
        print('0th component')
        inds_sort = np.argsort(dd)  # smallest go first
        inds_toshow = inds_sort[-nfeats_show:]

        dd_toshow = dd[inds_toshow]
        #strong_inds = np.where(dd_toshow   > np.quantile(dd_toshow,q) ) [0]
        strong_inds = inds_toshow
        strongest_ind = np.argmax(dd_toshow)
        strong_inds_pc = [strong_inds]
        strongest_inds_pc = [strongest_ind]
    else:
        strong_inds_pc = []
        strongest_inds_pc = []
        nfeats_show_pc = nfeats_show // nPCAcomponents_to_plot
        print('Per component we will plot {} feats'.format(nfeats_show_pc) )
        inds_toshow = []
        for i in range(nr):
            dd = np.abs(components[i  ] )

            inds_sort = np.argsort(dd)  # smallest go first
            inds_toshow_cur = inds_sort[-nfeats_show_pc:]
            inds_toshow += [inds_toshow_cur]

            #dd_toshow = dd[inds_toshow_cur]
            strong_inds = np.where(dd   > np.quantile(dd,q) ) [0]
            #print(i, strong_inds )
            strongest_ind = np.argmax(dd)
            assert  strongest_ind == inds_toshow_cur[-1]
            strongest_inds_pc += [strongest_ind]

            #strong_inds_pc += [strong_inds.copy() ]
            strong_inds_pc += [inds_sort[-nfeats_show_pc//2:]  ]

        inds_toshow = np.sort( np.unique( inds_toshow) )

    #print(inds_toshow, strong_inds_pc, strongest_inds_pc)


    nc = 1
    hh=4
    ww = max(14 , min(40, components.shape[1]/3 ) )
    fig,axs = plt.subplots(nrows=nr, ncols=nc, figsize=(ww*nc, hh*nr), sharex='col')
    if nr == 1:
        axs = [axs]
    for i in range(nr):
        ax = axs[i]
        #dd = np.abs(pca.components_[i] )
        dd = np.abs(components[i,inds_toshow  ] )
        ax.plot( dd )
        ax.axhline( np.quantile(dd, q), ls=':', c='r' )
        ttl = '(abs of) component {}' .format(i)
        if hasattr(pca, 'explained_variance_ratio_'):
            ttl += ', expl {:.2f} of variance (ratio)'.format(pca.explained_variance_ratio_[i])
        ax.set_title(ttl)

        ax.grid()
        ax.set_xlim(0, len(inds_toshow) )


    ax.set_xticks(np.arange(len(inds_toshow) ))
    if feature_names_all is not None:
        ax.set_xticklabels(feature_names_all[inds_toshow], rotation=90)

    tls = ax.get_xticklabels()
    ratio = 0.5 * len(inds_toshow) / len(strong_inds_pc )
    for compi in range(len(strong_inds_pc ) ):
        sipc = 0
        #print(compi, strong_inds_pc[compi] )
        si_cur = strong_inds_pc[compi]
        for i in si_cur[::-1]:
            ii = np.where(inds_toshow == i)[0]
            #print(ratio, sipc  )
            if len(ii) > 0 and (sipc < ratio ):
                sipc += 1
                ii = ii[0]
                tls[ii].set_color("purple")
    for compi in range(len(strong_inds_pc ) ):
        ii = np.where(inds_toshow == strongest_inds_pc[compi])[0][0]
        tls[ii ].set_color("red")

    plt.tight_layout()
    #plt.suptitle('PCA first components info')
    #plt.savefig('PCA_info.png')
    if pdf is not None:
        pdf.savefig()
        plt.close()

    return strong_inds_pc


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
            #if printLog:
            #    print(feature_names_all[i], l,m)
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

def downsample(X, skip, axis=0, mean=True, printLog = 1):
    if skip == 1:
        return X
    if X.ndim == 1:
        X = X[:,None]
    if axis != 0:
        X = np.swapaxes(X, 0,axis)
    outlen = (X.shape[0] // skip  ) * skip
    res = np.zeros((X.shape[0] // skip, X.size // X.shape[0]), dtype=X.dtype)
    sh = list(X.shape)
    if len(sh) > 0:
        sh[0] //= skip
        res = res.reshape( sh  )
    for i in range(skip):
        r = X[i:i+outlen:skip]  # using just X[i::skip] would give different lengths for differnt i-s
        #print(i, len(r),  X[i::skip].shape )
        #r = X.take(indices =slice(i:i+outlen:skip), aixs=axis)
        res += r


    #raise ValueError('ff')

    if mean:  #sometimes we might not want mean to save perfrmance -- we'll scale later anyway
        res /= skip

    if printLog:
        print('_X and res shapes ',X.shape,res.shape)

    dif = len(X) - len(res) * skip
    if dif >0 :
        res = np.vstack( [ res, np.mean(X[-dif:] , axis=0)[None,:] ] )
        dif2 = len(X) - len(res) * skip
        if printLog:
            print('Warning downsample killed {} samples, but we put them at then end and get {}'.format(dif,dif2) )



    if printLog:
        print('X and res shapes ',X.shape,res.shape)

    if axis != 0:
        res = np.swapaxes(res, axis,0)

    return res

def findByPrefix(data_dir, rawname, prefix, ftype='PCA',regex=None):
    #returns relative path
    import os, re
    if regex is None:
        regex = '{}_{}_{}_[0-9]+chs_nfeats([0-9]+)_pcadim([0-9]+).*'.format(rawname, prefix,ftype)
    fnfound = []
    for fn in os.listdir(data_dir):
        r = re.match(regex,fn)
        if r is not None:
            #n_feats,PCA_dim = r.groups()
            if prefix in ['move', 'hold', 'rest']:
                continue
            #print(fn,r.groups())
            fnfound += [fn]
    return fnfound


def concatArtif(rawnames,Xtimes_pri, crop=(None,None), artif_mod = None,
                allow_missing=False, side_rev_pri=None, sfreq=None, wbd_pri=None ):
    if artif_mod is None:
        artif_mod_str = [ '_ann_LFPartif', '_ann_MEGartif' ]
    else:
        artif_mod_str = [ '_ann_{}artif'.format(mod) for mod in artif_mod ]

    return concatAnns(rawnames,Xtimes_pri, artif_mod_str, crop=crop,
                     allow_missing= allow_missing,
                      side_rev_pri=side_rev_pri, sfreq=sfreq, wbd_pri=wbd_pri )

def concatAnnsNaive(rawnames,true_times_pri, suffixes=['_anns']):
    '''
    true_times_pri shoud not have gaps! (no edge bins removal was applied)
    '''
    import globvars as gv
    data_dir = gv.data_dir

    assert len(rawnames) == len(true_times_pri)
    anns_pri = []
    for rawni,rawname_ in enumerate(rawnames):
        subj,medcond,task  = utils.getParamsFromRawname(rawname_)
        #tasks += [task]

        anns = mne.Annotations([],[],[])
        for suffix in suffixes:
            anns_fn = rawname_ + suffix + '.txt'
            anns_fn_full = os.path.join(data_dir, anns_fn)
            anns += mne.read_annotations(anns_fn_full)
        #raw.set_annotations(anns)
        anns_pri += [anns]

        assert true_times_pri[rawni] [0] < 1e-10

    #dt = np.min( np.diff(true_times_pri[0] ) )
    #assert abs( (true_times_pri[1][0] - true_times_pri[0][-1]) - dt) <= 1e-10

    anns = mne.Annotations([],[],[])
    tshift = 0
    for anni,ann in enumerate(anns_pri):
        anns.append(ann.onset + tshift, ann.duration, ann.description)
        tshift += true_times_pri[anni][-1]

    return anns

def _rev_descr_side(descr):
    assert isinstance(descr,str)
    if descr.startswith('BAD_LFPR'):
        newdescr = 'BAD_LFPL'
    elif descr.startswith('BAD_LFPL'):
        newdescr = 'BAD_LFPR'
    if descr.startswith('BAD_MEGR'):
        newdescr = 'BAD_MEGL'
    elif descr.startswith('BAD_MEGL'):
        newdescr = 'BAD_MEGR'
    elif descr.endswith('_L'):
        newdescr = descr[-1] + 'R'
    elif descr.endswith('_R'):
        newdescr = descr[-1] + 'L'
    else:
        raise ValueError('wrong descr {} !'.format(descr) )

    return newdescr

def revAnnSides(anns):
    descr_old = anns.description
    descr_new = []
    for de in descr_old:
        revde = _rev_descr_side(de)
        descr_new += [revde]
    newann = mne.annotations(anns.onset,anns.duration,descr_new)
    return newann

def concatAnns(rawnames, Xtimes_pri, suffixes=['_anns'], crop=(None,None),
               allow_short_intervals = False, allow_missing=False, dt_sec=None,
               side_rev_pri=None, sfreq=None, wbd_pri=None):
    '''
    Xtimes_pri can have gaps (usually by endge bins removal) and start not from zero
     althogh much better use them without gaps and then use smarter window bounds
    output Xtimes will not have gaps (thus there are will be some shifts in ann times as well)
        and start from zero
    dataset_bounds -- times (not bins)
    allow_short_intervals -- what to do if interval is
    shorter than dt (can happen if we have short artifacts and spaced windows)
    sfreq is sampling freq of what is in wbd_pri

    returns
       merged anns (shfited by Xtimes_cur[0] ) ,
       anns_cur (unshifted),
       Xtimes_almost ( concatenated shifted Xtimes ),
       dataset_bounds ( dataset bounds in sec if we were to join dataset naively )
    '''
    import globvars as gv
    import os
    data_dir = gv.data_dir


    if isinstance(rawnames,str):
        rawnames = [rawnames]
    if isinstance(Xtimes_pri,np.ndarray):
        Xtimes_pri = [Xtimes_pri]
    if wbd_pri is not None:
        if isinstance(wbd_pri,np.ndarray):
            wbd_pri = [wbd_pri]
        assert len(wbd_pri) == len(Xtimes_pri)

    if side_rev_pri is None:
        side_rev_pri = [0] * len(rawnames)
    elif isinstance(side_rev_pri,int) or isinstance(side_rev_pri,bool):
        side_rev_pri = [side_rev_pri]

    if dt_sec is None and sfreq is not None:
        dt_sec = 1/sfreq
    elif sfreq is None and dt_sec is not None:
        sfreq = int(1/dt_sec)

    assert len(rawnames) == len(Xtimes_pri)

    # first we just read annotations and do the reversal if needed
    anns_pri = []
    dt_pri = []
    for rawi,rawname_ in enumerate(rawnames):
        subj,medcond,task  = utils.getParamsFromRawname(rawname_)
        #tasks += [task]

        anns = mne.Annotations([],[],[])
        for suffix in suffixes:
            anns_fn = rawname_ + suffix + '.txt'
            anns_fn_full = os.path.join(data_dir, anns_fn)
            if os.path.exists(anns_fn_full):
                #print('concatAnns: reading {}'.format(anns_fn) )
                anns_cur = mne.read_annotations(anns_fn_full)
                if side_rev_pri[rawi]:
                    anns_cur = revAnnSides(anns_cur)
                anns += anns_cur
                if crop[0] is not None or crop[1] is not None:
                    anns.crop(crop[0],crop[1] )
            else:
                warnstr = 'ConcatAnns: Missing file '+anns_fn_full
                print(warnstr)
                if not allow_missing:
                    raise ValueError(warnstr)
        #raw.set_annotations(anns)
        anns_pri += [anns]

        dt = Xtimes_pri[0][1] - Xtimes_pri[0][0]  #note that this is not 1/sfreq (since we skipped)
        dt_pri += [dt]

    assert np.max(dt_pri) - np.min(dt_pri) < 1e-10

    anns = mne.Annotations([],[],[])

    dataset_bounds = []
    Xtimes_almost = []
    timeshift = 0    # in seconds
    for xti in range(len(Xtimes_pri)):

        Xtimes_cur = Xtimes_pri[xti]
        # since we had shifted by nedgeBins, we have to correct for it
        timeshift += -Xtimes_cur[0]  # in case if we start not from zero

        #print(timeshift)
        if wbd_pri is not None:
            wbdcur = wbd_pri[xti]
            lastwnd_end = wbdcur[1,-1] / sfreq

            inds = np.where(Xtimes_cur < lastwnd_end)[0] # strong ineq is important here
            assert np.max(np.diff(inds) ) == 1, np.min(np.diff(inds) ) == 1
            Xtimes_cur = Xtimes_cur[inds]   # crop to the last window border
        #else:
        #    lastwnd_end = np.max(Xtimes_cur + dt)

        Xtimes_shifted = Xtimes_cur + timeshift
        Xtimes_almost += [Xtimes_shifted]
        ann_cur = anns_pri[xti]
        #print(ann_cur.onset, ann_cur.duration )
        if len(ann_cur):
            #print(ann_cur.onset, ann_cur.description)
            onset_ = np.maximum(Xtimes_shifted[0],ann_cur.onset + timeshift)   # kill negatives
            onset_ = np.minimum(Xtimes_shifted[-1], onset_ )
            end_ = np.minimum(onset_ + ann_cur.duration, Xtimes_shifted[-1] )  # kill over-end

            duration_ = np.maximum(0, end_ - onset_)
            if (not allow_short_intervals) and (dt_sec is not None):
                assert np.all(duration_ > dt_sec )
            anns.append(onset_ , duration_,ann_cur.description)

        #print(Xtimes_shifted[0],timeshift,Xtimes_cur[-1], Xtimes_cur[-1] -Xtimes_cur[0])
        dataset_bounds += [ (Xtimes_shifted[0], Xtimes_shifted[-1] ) ]

        timeshift += Xtimes_cur[-1] + dt

    #print('fd',Xtimes_almost[0][0],Xtimes_almost[0][-1],Xtimes_almost[1][0],Xtimes_almost[1][-1] )
    Xtimes_almost = np.hstack(Xtimes_almost)
    df = np.diff(Xtimes_almost)
    assert abs( np.max(df) - np.min(df) ) <= 1e-10,  (np.max(df), np.min(df))
    assert np.all( anns.onset >= 0 ), anns.onset

    return anns, anns_pri, Xtimes_almost, dataset_bounds

def getAnnBins(ivalis,Xtimes_almost,nedgeBins, sfreq,totskip, windowsz, dataset_bounds,
               set_empty_arrays = 0, force_all_arrays_nonzero=1):
    '''
    nedgeBins -- currently unused
    windowsz -- in bins, we'll shift this forward
    totskip -- in bins
    ivalis -- dict interval types -> list of 3-tuples
    '''
    #ivalis is dict of lists of tuples (beg,start, itype)
    ivalis_tb = {}  # only bounds
    ivalis_tb_indarrays = {}  # entire arrays
    #globend = Xtimes[-1] + nedgeBins/sfreq
    sfreq = int(sfreq)

    #assert( Xtimes_almost[0] < 1e-10 )

    maxind = len(Xtimes_almost)-1


    edge_window_nbins = windowsz
    #
    for itype in ivalis:
        intervals = ivalis[itype]
        intervals_bins = []
        cur_indarr = []
        #print(itype)
        for interval in intervals:
            st,end,it_ = interval

            if dataset_bounds is not None:
                interval_dataset_ind = None
                # find out whic dataset our interval belongs to
                for di in range(len(dataset_bounds)):
                    dlb,drb = dataset_bounds[di]
                    if st >= dlb and end <= drb:
                        interval_dataset_ind = di
                        break
                assert interval_dataset_ind  is not None, (interval, dataset_bounds)
                globend = drb
                globstart = dlb
            else:
                dlb = Xtimes_almost[0]
                drb = Xtimes_almost[-1]

            if st <= dlb + edge_window_nbins / sfreq:  # if interval left bound larger than dataset bound, just use left bound
                bnd0 = int(dlb * sfreq) // totskip
                #print('lbam ',bnd0, dlb, interval)
                #print('1')
            else:  # if interval left bound larger than dataset bound (easy)
                # just make sure it is not too far right
                bnd0 = int( min(maxind, (st * sfreq   + edge_window_nbins) // totskip -1   ) )
                #print('2')

            if drb - end <= edge_window_nbins / sfreq:
                bnd1 = int(drb * sfreq) // totskip
                bnd1 = min(bnd1, maxind)     # this is important to have otherwise we have err for S04
                #print('3')
            else:
                bnd1 = int( min(maxind, (end *sfreq   + edge_window_nbins) // totskip -1   ))
                #print('4')

            assert bnd1 <= maxind
            if force_all_arrays_nonzero and bnd1 == bnd0 and bnd0 + 1 <= maxind:
                bnd1 = bnd0+1

            #print('fdfsdlbam ',bnd0, bnd1, interval)
            if (bnd1 - bnd0 > 0) or (bnd1 == bnd0 and set_empty_arrays):
                intervals_bins += [(bnd0,bnd1,it_)]
                ar = np.arange(bnd0,bnd1, dtype=int)
                if len(ar) or set_empty_arrays:
                    cur_indarr += [ar]
                else:
                    print('skip', it_)
            else:
                print('  {}: bnd1 ({}) <= bnd0 ({})  :( '.format(interval, bnd1, bnd0) )

            if len(cur_indarr) or set_empty_arrays:
                ivalis_tb_indarrays[itype] = cur_indarr
        if len(intervals_bins) >= 0 or set_empty_arrays:
            #print('__', itype, len(ivalis_tb_indarrays), intervals_bins )
            ivalis_tb[itype] =  intervals_bins
        else:
            print('getAnnBins: zero interval bins, skip', itype)

    return ivalis_tb, ivalis_tb_indarrays

def mergeAnnBinArrays(ivalis_tb_indarrays):
    '''
    takes dict of lists of binind arrays
    returns dict of bininds arrays
    '''
    ivalis_tb_indarrays_merged = {}
    for k in ivalis_tb_indarrays:
        ivalis_tb_indarrays_merged[k] = np.hstack(ivalis_tb_indarrays[k])
    return ivalis_tb_indarrays_merged

#def specificity_score(y_true,y_pred):
#    ''' arra-ylike inputs '''
#    total = len(y_true)
#    TN = sum(Neg != true_ind)
#    spec = TN / len(Neg)
#    return spec

def sprintfPerfs(perfs):
    p = np.array(perfs) * 100
    perfs_str = '{:.2f}%,{:.2f}%,{:.2f}%'.format(p[0], p[1], p[2])
    return perfs_str

from sklearn.feature_selection import mutual_info_classif

def _MI(arg):
    inds,X,y = arg
    r = mutual_info_classif(X,y, discrete_features = False)
    return inds,r

def getMIs(X,class_labels,class_ind,n_jobs = None):
    mask     = class_labels == class_ind
    mask_inv = class_labels != class_ind
    tmp = np.zeros( len(class_labels), dtype=bool)
    #tmp = class_labels[:]
    tmp[mask] = 1
    tmp[mask_inv] = 0
    args = []
    nfeats = X.shape[1]

    from globvars import gp

    if n_jobs is not None:
        max_cores = n_jobs
    else:
        max_cores = mpr.cpu_count()-gp.n_free_cores

    if max_cores == 1:
        arg = np.arange(nfeats),X,class_labels
        args += [arg]
    else:
        bnds = np.arange(0, nfeats, nfeats//max_cores )
        for bndi,bnd in enumerate(bnds):
            if bndi < len(bnds) - 1:
                right = bnds[bndi+1]
            else:
                right = nfeats
            inds = np.arange(bnd,right )
            arg = inds,X[:,inds],class_labels
            args += [arg]

    n_jobs = max_cores

    pool = mpr.Pool(n_jobs)
    res = pool.map(_MI, args)
    #if printLog:
    #    print('getPredPowersCV:  Sending {} tasks to {} cores'.format(len(args), n_jobs))
    pool.close()
    pool.join()

    mic = np.zeros(nfeats)
    for r in res:
        inds,mic_cur = r
        mic[inds] = mic_cur
    return mic

def getLDApredPower(clf,X,class_labels,class_ind, printLog = False):
    '''
    LDA perf in detecting class_ind
    - class_ind  is an interger class id
    '''
    true_ind = class_ind
    mask = (class_labels == true_ind)
    mask_inv = np.logical_not(mask)

    if np.sum(mask) == 0 or np.sum(mask_inv) == 0:
        s = 'one of masks is bad '.format(np.sum(mask), np.sum(mask_inv) )
        print('getLDApredPower: WARNING {}'.format(s) )
        #raise ValueError(s)
        return np.nan, np.nan, np.nan

    ntot = len(class_labels)

    X_P = X[mask]
    Pos = clf.predict(X_P)
    TP = sum(Pos == true_ind)
    sens = TP / len(Pos)

    X_N = X[mask_inv]
    Neg = clf.predict(X_N)
    TN = sum(Neg != true_ind)
    spec = TN / len(Neg)
    #spec = specificity_score(y_true,y_pred)

    FP = len(Pos) - TP
    FN = len(Neg) - TN
    F1 =  TP / (TP + 0.5 * ( FP + FN ) )

    if printLog:
        print('getLDApredPower: True pos {} ({:.3f}), all pos {} ({:.3f})'.format(TP, TP/ntot, len(Pos), len(Pos)/ntot ) )
        print('getLDApredPower: True neg {} ({:.3f}), all neg {} ({:.3f})'.format(TN, TN/ntot, len(Neg), len(Neg)/ntot ) )

    #if n_KFold_splits is not None:
    #    from sklearn.model_selection import KFold
    #    kf = KFold(n_splits=n_KFold_splits)
    #    res = kf.split(X)

    #    #KFold(n_splits=2, random_state=None, shuffle=False)
    #    for train_index, test_index in res:

    return sens,spec, F1

def _getPredPower_singleFold(arg):
    from numpy.linalg import LinAlgError
    (clf,add_clf_creopts,add_fitopts,X_train,X_test,y_train,y_test,class_ind,printLog)  = arg
    model_cur = type(clf)(**add_clf_creopts)  # I need num LDA compnents I guess
    try:
        model_cur.fit(X_train, y_train, **add_fitopts)
        perf_cur = getLDApredPower(model_cur,X_test,y_test,
                                    class_ind, printLog=printLog)
        if printLog:
            #print('getPredPowersCV: CV {}/{} pred powers {}'.format(-1,n_splits,cur) )
            print('getPredPowersCV: current fold pred powers {}'.format(perf_cur) )
    except LinAlgError as e:
        print( str(e) )
        model_cur, perf_cur = None, None

    return model_cur,perf_cur


def getPredPowersCV(clf,X,class_labels,class_ind, printLog = False, n_splits=None,
                    return_clf_obj=False, skip_noCV =False, add_fitopts={},
                   add_clf_creopts ={} ):
    # clf is assumed to be already fitted here
    # TODO: maybe I need to adapt for other classifiers
    # ret = [perf_nocv, perfs_CV, perf_aver ] and maybe list of classif objects
    # obtained during CV
    from globvars import gp
    if skip_noCV:
        assert n_splits is not None
        perf_nocv = None
    else:
        perf_nocv = getLDApredPower(clf,X,class_labels,class_ind, printLog=printLog)
        if printLog:
            print('getPredPowersCV: perf_nocv ',perf_nocv, X.shape)

    #for model_cur in cv_results['estimator']
    if n_splits is not None:
        #if n_KFold_splits is not None:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True)
        split_res = kf.split(X)

        n_jobs_perrun = add_clf_creopts.get('n_jobs', 1)

        #KFold(n_splits=2, random_state=None, shuffle=False)
        perfs_CV = []
        Xarr = np.array(X)
        models = []
        #indcv_indset = 0
        args = []

        for train_index, test_index in split_res:
            #print(train_index )
            X_train, X_test = Xarr[train_index], Xarr[test_index]
            y_train, y_test = class_labels[train_index], class_labels[test_index]
            if len(set(y_train)) <= 1 or len(set(y_test)) <= 1:
                continue

            arg = (clf,add_clf_creopts,add_fitopts,X_train,X_test,y_train,y_test,class_ind, printLog)
            args += [arg]

        if n_jobs_perrun > 1:
            n_jobs = 1
            for arg in args:
                r = _getPredPower_singleFold(arg)
                model_cur,perfs_cur = r
                models += [model_cur]
                perfs_CV += [perfs_cur]
        else:
            n_jobs = max(1, min(len(args) , mpr.cpu_count()-gp.n_free_cores) )

            pool = mpr.Pool(n_jobs)
            if printLog:
                print('getPredPowersCV:  Sending {} tasks to {} cores'.format(len(args), n_jobs))
            res = pool.map(_getPredPower_singleFold, args)
            pool.close()
            pool.join()
            for r in res:
                model_cur,perfs_cur = r
                if (model_cur is not None) and (perfs_cur is not None):
                    models += [model_cur]
                    perfs_CV += [perfs_cur]
                    #indcv_indset += 1

        perfarr = np.vstack(perfs_CV)
        not_nan_fold_inds = np.where(  np.max( np.isnan(perfarr).astype(int) , axis= 1) == 0 )[0]
        assert len(not_nan_fold_inds) > 0
        perf_aver = np.mean(perfarr[not_nan_fold_inds] , axis = 0)
        ret = [perf_nocv, perfs_CV, perf_aver ]
        if return_clf_obj:
            ret += [models]
    else:
        ret = perf_nocv, [perf_nocv] , perf_nocv
        if return_clf_obj:
            ret += [ [clf] ]

    return tuple(ret)

def calcLDAVersions(X_to_fit, X_to_transform, class_labels,n_components_LDA,
                    class_ind_to_check, revdict, n_splits=4, calcName = ''):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=min(X_to_fit.shape[1], n_components_LDA ) )
    # main LDA with all features
    lda.fit(X_to_fit, class_labels)

    res = {}

    print('---------------- Start LDA examination {},  X_to_fit.shape = {}'.format(calcName, X_to_fit.shape ) )

    print('LDA var explained = ', lda.explained_variance_ratio_)
    print('LDA priors ', list(zip( [revdict[cid] for cid in lda.classes_],lda.priors_) ) )

    X_LDA = lda.transform(X_to_transform)  # we transform all points, even bad and ulabeled ones. Transform is done using scalings


    # Compute training on training (separability)
    sens,spec,F1 = getLDApredPower(lda,X_to_fit, class_labels, class_ind_to_check)
    print('-- LDA on train sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens,spec,F1) )

    subres = {}
    subres['ldaobj'] = lda
    subres['X_transformed'] = X_LDA
    subres['perfs'] = sens,spec,F1
    res['fit_to_all_data'] = subres

    ########## Compute with CV
    perf_noCV, perfs_CV, res_aver_LDA, ldas_CV = \
        getPredPowersCV(lda, X_to_fit,  class_labels, class_ind_to_check,
                                printLog=False, n_splits=n_splits, return_clf_obj=True,
                        skip_noCV=1)
    sens_cv,spec_cv,F1_cv = res_aver_LDA

    subres = {}
    subres['ldaobjs']     = ldas_CV
    subres['CV_perfs']     = perfs_CV
    subres['CV_perf_aver'] = sens_cv,spec_cv,F1_cv
    subres['n_splits'] = n_splits
    res['CV'] = subres

    # Now compute average coefficients over folds
    print('-- LDA CV       sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens_cv,spec_cv,F1_cv) )
    scalings_list      = [lda_cur.scalings_ for lda_cur in ldas_CV   ]
    xbars_list         = [lda_cur.xbar_     for lda_cur in ldas_CV   ]
    intercept_list     = [lda_cur.intercept_     for lda_cur in ldas_CV   ]
    coef_list          = [lda_cur.coef_     for lda_cur in ldas_CV   ]
    #rotations_list = [lda_cur.rotations_  for lda_cur in ldas_CV   ]
    #means_list     = [lda_cur.means_  for lda_cur in ldas_CV   ]
    scalings_aver = sum(scalings_list) / len(scalings_list)
    intercept_aver = sum(intercept_list) / len(intercept_list)
    coef_aver = sum(coef_list) / len(coef_list)
    #rotations_aver = sum(rotations_list) / len(rotations_list)
    #means_aver = sum(means_list) / len(means_list)

    from copy import deepcopy
    lda_aver = deepcopy(lda)
    lda_aver.coef_ = coef_aver
    lda_aver.intercept_ = intercept_aver

    sens_avCV,spec_avCV,F1_avCV = getLDApredPower(lda_aver,X_to_fit, class_labels, class_ind_to_check)
    print('-- LDA avCV on train sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens,spec,F1) )

    perf_nocv_LDA_avCV, results_LDA_avCV, res_aver_LDA_avCV, ldas_CV_avCV = \
        getPredPowersCV(lda_aver, X_to_fit,class_labels, class_ind_to_check,
                                printLog=False, n_splits=n_splits, return_clf_obj=True, skip_noCV=1)
    sens_cv_avCV,spec_cv_avCV,F1_cv_avCV = res_aver_LDA_avCV

    print('-- LDA CV _avCV sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens_cv_avCV,spec_cv_avCV,F1_cv_avCV) )
    X_LDA_CV = lda_aver.transform(X_to_transform)

    subres = {}
    subres['ldaobj'] = lda_aver
    subres['X_transformed'] = X_LDA_CV
    subres['perfs'] = sens_avCV,spec_avCV,F1_avCV
    res['CV_aver'] = subres

    return res

def selMinFeatSet(clf, X, class_labels, class_ind, sortinds, drop_perf_pct = 5, n_splits=4,
                  verbose=1, add_fitopts={}, add_clf_creopts={}, check_CV_perf = False,
                  nfeats_step = 3, nsteps_report=1, max_nfeats=100):
    '''
    sortind -- sorted increasing importance (i.e. the most imporant is the last one)
    it is assumed that clf.fit has already been made
    last feature is the most significant
    returns list of tuples, the first one is for all features,
        the last one is for the best found set of features

        the last one is alwasy cross-validated
    '''
    from globvars import gp

    if check_CV_perf:
        s = 'CV'
    else:
        s = 'noCV'
    print('selMinFeatSet: --- starting {} comp pred powers for X.shape={}, step={}, max_nfeats={}'.
          format(s,X.shape,nfeats_step,max_nfeats) )

    perf_nocv_, results_, res_aver_ = getPredPowersCV(clf,X,class_labels,class_ind, verbose >=3,
                           n_splits=n_splits, add_fitopts=add_fitopts, add_clf_creopts=add_clf_creopts)
    if check_CV_perf:
        sens_full,spec_full,F1_full = res_aver_
    else:
        sens_full,spec_full,F1_full = perf_nocv_


    Xarr = np.array(X)
    #X_red = Xarr[:,sortinds[-2:].tolist()]
    model_red = type(clf)(**add_clf_creopts)

    # red = utsne.getLDApredPower(model_red,X_red,y,
    #                             gp.class_ids_def['trem_' + mts_letter], printLog=1)
    # print(red)

    perfs = []
    nfeats = X.shape[1]

    if verbose >= 1:
        print('selMinFeatSet: --- all feats give perf={}, check_CV_perf = {}'.
            format( sprintfPerfs([sens_full,spec_full,F1_full] ),check_CV_perf) )

    perfs += [ (-1, sortinds.tolist(), perf_nocv_, res_aver_)   ]
    sens_prev,spec_prev = 0,0
    if check_CV_perf:
        n_splits_cycle =  n_splits
    else:
        n_splits_cycle =  None

    converge_thr = drop_perf_pct / 100
    close_to_full_thr = drop_perf_pct / 100
    for i in range(1,max_nfeats+1,nfeats_step):
        # counting backwards
        inds = sortinds[-i:].tolist()
        X_red = Xarr[:,inds]
        model_red.fit(X_red, class_labels, **add_fitopts)
        perf_nocv, results, res_aver = \
            getPredPowersCV(model_red,X_red,class_labels, class_ind,
                        printLog=(verbose >= 3), n_splits=n_splits_cycle,
                        add_fitopts=add_fitopts,
                        add_clf_creopts=add_clf_creopts)
        sens,spec,F1 = perf_nocv

        perfs += [ (i,inds, perf_nocv,res_aver)   ]
        if verbose >= 2 and ( int(i-1 / nfeats_step) % nsteps_report == (nsteps_report-1) ):
            print('selMinFeatSet: --- search of best feat set, len(inds)={}, perf={}'.
                  format(len(inds), sprintfPerfs(res_aver) ) )

        # the last one if always CV even though if we checking only training
        # data perf when selecting feats
        cond_conv = ( (sens - sens_prev) <  converge_thr ) and ( (spec - spec_prev) <  converge_thr )
        cond_close = (sens_full - sens  < close_to_full_thr) and (spec_full- spec  < close_to_full_thr)
        if cond_close and cond_conv:
            perf_nocv, results, res_aver = getPredPowersCV(model_red,X_red,class_labels,
                                class_ind, printLog=verbose >= 3, n_splits=n_splits, add_fitopts=add_fitopts,
                                                            add_clf_creopts=add_clf_creopts)
            sens,spec,F1 = res_aver
            perfs[-1] =  (i,inds, perf_nocv,res_aver)

            if verbose >= 1:
                print('selMinFeatSet: --- ENDED search of best feat set, len(inds)={}, perf={}'.
                      format(len(inds), sprintfPerfs(res_aver) ) )
            break

        sens_prev,spec_prev = sens,spec


    return perfs



def makeClassLabels(sides_hand, grouping, int_types_to_distinguish, ivalis_tb_indarrays,
                    good_inds, num_labels_tot, rem_neut=1):
    '''
        sides_hand = list of one-char strings

        returns
    '''
    from globvars import gp
    import copy
            # prepare class_ids (merge)
    class_ids_grouped = copy.deepcopy(gp.class_ids_def)
    if len(grouping) > 1:
        for side_letter in ['L', 'R']:
            # the class label I'll assign to every class to be merged
            main = '{}_{}'.format(grouping[0],side_letter)
            for int_type_cur in grouping:
                cur = '{}_{}'.format(int_type_cur,side_letter)
                class_ids_grouped[cur] = gp.class_ids_def[main]
    print(class_ids_grouped)

    #TODO: make possible non-main side


    #class_labels = np.repeat(gp.class_id_neut,len(Xconcat_imputed))
    class_labels = np.repeat(gp.class_id_neut,num_labels_tot)
    assert gp.class_id_neut == 0

    #old_ver = 0
    #if old_ver:
    #    int_types = set()
    #    for itb in int_types_to_distinguish:
    #        for side in sides_hand:
    #            assert len(side) == 1
    #            int_types.update(['{}_{}'.format(itb,side)])
    #    #int_types = ['trem_L', 'notrem_L', 'hold_L', 'move_L']
    #    int_types = list(int_types)
    #    #print(int_types)

    #    classes = [k for k in ivalis_tb_indarrays.keys() if k in int_types]  #need to be ordered
    #    #classes

    #    for i,k in enumerate(classes):
    #        #print(i,k)
    #        for bininds in ivalis_tb_indarrays[k]:
    #            #print(i,len(bininds), bininds[0], bininds[-1])
    #            class_labels[ bininds ] = i + 1
    from collections.abc import Iterable

    revdict = {}
    # set class label for current interval types
    bincounts_per_class_name = {}
    for itb in int_types_to_distinguish:
        for side in sides_hand:
            class_name = '{}_{}'.format(itb,side)
            # look at bininds of this type that were found in the given dataset
            bininds_ = ivalis_tb_indarrays.get(class_name,None )
            if bininds_ is None:
                print(': WARNING 1, no bininds_ for {} found'.format(class_name)   )
                bincounts_per_class_name[class_name] = 0
                continue
            elif isinstance(bininds_,Iterable):
                if not isinstance(bininds_[0],Iterable ) :  # then we have merged array
                    bininds_ = [bininds_]

                totlen = 0
                for bininds in bininds_:
                    #print(i,len(bininds), bininds[0], bininds[-1])
                    cid = class_ids_grouped[class_name]
                    totlen += len(bininds)
                    class_labels[ bininds ] = cid
                    if cid not in revdict:
                        revdict[cid] = class_name
                    elif revdict[cid].find(class_name) < 0:
                        revdict[cid] += '&{}'.format(class_name)

                bincounts_per_class_name[class_name] = totlen
                if totlen == 0:
                    print(': WARNING 2, no bininds_ for {} found'.format(class_name)   )



    class_labels_good = class_labels[good_inds]

    # if I don't want to classify against points I am not sure about (that
    # don't have a definite label)
    if rem_neut:
        neq = class_labels_good != gp.class_id_neut
        inds = np.where( neq)[0]
        class_labels_good = class_labels_good[inds]
    #else:
    #    classes = ['neut'] + classes  # will fail if run more than once

    return class_labels, class_labels_good, revdict, class_ids_grouped

def countClassLabels(class_labels_good, class_ids_grouped):
    if isinstance (class_labels_good,np.ndarray):
        assert class_labels_good.ndim == 1
    elif not isinstance (class_labels_good,list):
        raise ValueError('Wrong type')
    counts = {}
    for class_name in class_ids_grouped:
        cid = class_ids_grouped[class_name]
        #print(cid)
        num_cur = np.sum(class_labels_good == cid)
        counts[class_name] = num_cur
    return counts

#def checkClassLabelsCompleteness(class_labels_good, class_ids_grouped, class_to_check,
#                                 min_num = 1):
#    countClassLabels(class_labels_good, class_ids_grouped)
#    for class_name in class_ids_grouped:
#        cid = class_ids_grouped[class_name]
#        num_cur = np.sum(class_labels_good == cid)
#        if num_cur < min_num:
#            s = 'Class {} (cid={}) is not present at all'.format(class_name,cid)
#            raise ValueError(s)
