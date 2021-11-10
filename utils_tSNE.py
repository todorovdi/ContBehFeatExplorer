import numpy as np
#import udus_dataproc as mdp # main data proc
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

from featlist import selFeatsRegexInds

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
    else:
        assert wbd_sec.dtype == np.float

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
        tsis = []
        for tcur in ts:
            #ts = np.array(ts)
            mask = np.logical_and( tcur >= wbd_sec[0], tcur < wbd_sec[1] )
            tsis_cur = np.where(mask)[0]
            assert len(tsis_cur) > 0
            tsis += [tsis_cur[0] ]
        assert len(tsis) == len(ts)
        assert np.all(np.diff(tsis) >= 0), np.diff(tsis)

        #allbins = ( times / dt  ).astype(int)
        #ts_ =  ( ts /    dt  ).astype(int)
        #tsis = np.searchsorted(allbins, ts_)

    subint_names = ['prestart','poststart','main','preend','postend']
    return ts, tsis, subint_names

#def selFeatsRegexInds(names, regexs, unique=1):
#    # return indices of names that match at least one of the regexes
#    import re
#    if isinstance(regexs,str):
#        regexs = [regexs]
#
#    inds = []
#    for namei,name in enumerate(names):
#        for pattern in regexs:
#            r = re.match(pattern, name)
#            if r is not None:
#                inds += [namei]
#                if unique:
#                    break
#
#    return inds


def selFeatsRegex(data, names, regexs, unique=1, copy=False):
    '''
    data can be None if I only want to select feat names
    data can be either numpy array or list of numpy arrays
    '''
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
            if copy:
                datsel = datsel.copy()
        elif isinstance( data, list):
            datsel = []
            for subdat in data:
                if copy:
                    datsel += [subdat[inds].copy() ]
                else:
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
                arg = bn_from,bn_to,fromi,toi,name,\
                    window_starts,windowsz,skip,dfrom,dto,oper,positive
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

def prepColorsMarkers(anns, Xtimes,
               nedgeBins, windowsz, sfreq, totskip, mrk,mrknames,
               color_per_int_type, extend = 3, defmarker='o', neutcolor='grey',
                     convert_to_rgb = False, dataset_bounds = None, wbd=None,
                     side_letter=None  ):
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
    if not (isinstance(extend, list) ):
        extend = 4*[extend]
    else:
        assert len(extend) == 4

    if side_letter is None:
        side_letter = ['R', 'L']

    extendInL  = extend[0]
    extendOutL = extend[1]
    extendInR  = extend[2]
    extendOutR = extend[3]

    if wbd is None:
        wbd = np.vstack( [Xtimes*sfreq, Xtimes*sfreq] ).astype(int)
        wbd[1] += windowsz

    assert wbd.dtype == np.int
    wbd_sec = (wbd / sfreq).astype(float)

    if convert_to_rgb:
        import matplotlib.colors as mcolors
        for k in color_per_int_type:
            cname = color_per_int_type[k]
            if isinstance(cname,str):
                color_per_int_type[k] = mcolors.to_rgb(cname)


    #side_letter = 'L'; print('Using not hand side (perhabs) for coloring')
    annot_color_perit = {}
    for k in color_per_int_type:
        if isinstance(side_letter,str):
            let = side_letter
            assert len(let) == 1
            it_lab = '{}_{}'.format(k,side_letter)
            annot_color_perit[ it_lab   ] = color_per_int_type[k]
        elif isinstance(side_letter,list) and isinstance(side_letter[0],str):
            for let in side_letter:
                assert len(let) == 1
                it_lab = '{}_{}'.format(k,side_letter)
                annot_color_perit[ it_lab   ] = color_per_int_type[k]
        else:
            raise ValueError('Wrong side_letter {}'.format(side_letter) )
    #for task in tasks:
    #    annot_color_perit[ '{}_{}'.format(task, side_letter) ] = color_per_int_type[task]

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
                                                                times=Xtimes, wbd_sec=wbd_sec)

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
    from plots import plotComponents
    strong_inds_pc = plotComponents(pca.components_, feature_names_all,
                                    nPCAcomponents_to_plot, nfeats_show, q,
                  toshow_decide_0th_component, pca.explained_variance_ratio_)

    plt.tight_layout()
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
        print('downsample: _X and res shapes ',X.shape,res.shape)

    dif = len(X) - len(res) * skip
    if dif >0 :
        res = np.vstack( [ res, np.mean(X[-dif:] , axis=0)[None,:] ] )
        dif2 = len(X) - len(res) * skip
        if printLog:
            print('Warning downsample killed {} samples, but we put them at then end and get {}'.format(dif,dif2) )



    if printLog:
        print('downsample:X and res shapes ',X.shape,res.shape)

    if axis != 0:
        res = np.swapaxes(res, axis,0)

    return res

def findByPrefix(data_dir, rawname, prefix, ftype='PCA',regex=None, ret_aux=0):
    #returns relative path
    import os, re
    if regex is None:
        regex = '{}_{}_{}_[0-9]+chs_nfeats([0-9]+)_pcadim([0-9]+).*'.format(rawname, prefix,ftype)
    fnfound = []
    match_infos = []
    ntot = 0
    for fn in os.listdir(data_dir):
        ntot += 1
        r = re.match(regex,fn)
        if r is not None:
            #n_feats,PCA_dim = r.groups()
            if prefix in ['move', 'hold', 'rest']:
                continue
            #print(fn,r.groups())
            if ret_aux:
                match_infos += [r]
            fnfound += [fn]
    print(f'findByPrefix: selected {len(fnfound)} files among {ntot}')
    if ret_aux:
        return fnfound, match_infos
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
    tpls0 = [ ('BAD_LFPR','BAD_LFPL'), ('BAD_MEGR', 'BAD_MEGL')    ]
    tpls1 = []
    for tpl in tpls0:
        tpls1 += [ tpl,  (tpl[1],tpl[0]) ]
    #d = dict(tpls1)

    was_some = False
    for d1,d2 in tpls1:
        if descr.startswith(d1):
            add_str = ''
            # if the artifact specification is more specific
            if len(descr) > len(d1):
                add_str = descr[ len(d1): ]
            newdescr = d2 + add_str
            was_some = True

    #if descr.startswith('BAD_LFPR'):
    #    newdescr = 'BAD_LFPL'
    #elif descr.startswith('BAD_LFPL'):
    #    newdescr = 'BAD_LFPR'
    #elif descr.startswith('BAD_MEGR'):
    #    newdescr = 'BAD_MEGL'
    #elif descr.startswith('BAD_MEGL'):
    #    newdescr = 'BAD_MEGR'
    if not was_some:
        if descr.endswith('_L'):
            tmp = list(descr)
            tmp[-1] = 'R'
            newdescr = ''.join(tmp)
        elif descr.endswith('_R'):
            tmp = list(descr)
            tmp[-1] = 'L'
            newdescr = ''.join(tmp)
        else:
            raise ValueError('wrong descr {} !'.format(descr) )

    return newdescr

def revAnnSides(anns):
    descr_old = anns.description
    descr_new = []
    for de in descr_old:
        revde = _rev_descr_side(de)
        descr_new += [revde]
    newann = mne.Annotations(anns.onset,anns.duration,descr_new)
    return newann

def concatAnns(rawnames, Xtimes_pri, suffixes=['_anns'], crop=(None,None),
               allow_short_intervals = False, allow_missing=False, dt_sec=None,
               side_rev_pri=None, sfreq=None, wbd_pri=None,
               remove_gaps_between_datasets = False, ret_wbd_merged=False):
    '''
    Xtimes_pri can have gaps (usually by endge bins removal) and start not from zero
     althogh much better use them without gaps and then use smarter window bounds
    output Xtimes will not have gaps (thus there are will be some shifts in ann times as well)
        and start from zero IF remove_gaps_between_datasets
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

    assert Xtimes_pri[0].dtype == float
    if wbd_pri is not None:
        if isinstance(wbd_pri,np.ndarray):
            wbd_pri = [wbd_pri]
        assert len(wbd_pri) == len(Xtimes_pri)
        assert wbd_pri[0].dtype == int

    if side_rev_pri is None:
        side_rev_pri = [0] * len(rawnames)
    elif isinstance(side_rev_pri,int) or isinstance(side_rev_pri,bool):
        side_rev_pri = [side_rev_pri] * len(rawnames)

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

        #print(anns.onset,anns.duration,anns)

        dt = Xtimes_pri[0][1] - Xtimes_pri[0][0]  #note that this is not 1/sfreq (since we skipped)
        dt_pri += [dt]

    assert np.max(dt_pri) - np.min(dt_pri) < 1e-10


    cur_zeroth_bin = 0
    wbds = []

    anns = mne.Annotations([],[],[])

    dataset_bounds = []
    Xtimes_almost = []
    timeshift = 0    # in seconds
    for xti in range(len(Xtimes_pri)):
        Xtimes_cur = Xtimes_pri[xti]
        #print(timeshift)
        # only keep Xtimes that are inside wbd (i.e. remove edge bins outside
        # windows)
        if wbd_pri is not None:
            wbdcur = wbd_pri[xti].copy()

            if remove_gaps_between_datasets:
                sh = wbdcur[0,0]
                wbdcur -= sh
                timeshift -= sh / sfreq

            firstwind_start = wbdcur[0,0] / sfreq  # convert bins to times
            lastwnd_end     = wbdcur[1,-1] / sfreq # convert bins to times

            cnd0 = Xtimes_cur >= firstwind_start
            cnd1 = Xtimes_cur < lastwnd_end
            cnd = np.logical_and(cnd0, cnd1)
            inds = np.where(cnd)[0] # strong ineq is important here
            assert len(inds) > 0
            assert np.max(np.diff(inds) ) == 1, np.min(np.diff(inds) ) == 1
            Xtimes_cur = Xtimes_cur[inds]   # crop to the last window border

            wbd = wbd_pri [xti]
            wbds += [wbd + cur_zeroth_bin]
            skip = wbd[0,1] - wbd[0,0]
        elif remove_gaps_between_datasets:
            timeshift += -Xtimes_cur[0]  # in case if we start not from zero



        Xtimes_shifted = Xtimes_cur + timeshift
        Xtimes_almost += [Xtimes_shifted]
        ann_cur = anns_pri[xti]
        #print(ann_cur.onset, ann_cur.duration )
        if len(ann_cur):
            #print(ann_cur.onset, ann_cur.description)
            go = []
            for oi,onset_cur in enumerate(ann_cur.onset):
                if onset_cur + timeshift  >  Xtimes_shifted[-1]:
                    continue
                elif onset_cur + ann_cur.duration[oi] + timeshift  <  Xtimes_shifted[0]:
                    continue
                else:
                    go += [oi]

            #print(go,anns)

            # max between start of the current dataset and onset
            onset_ = np.maximum(Xtimes_shifted[0],ann_cur.onset[go] + timeshift)   # kill negatives
            # min between end of the current dataset and onset
            onset_ = np.minimum(Xtimes_shifted[-1], onset_ )
            end_ = np.minimum(onset_ + ann_cur.duration[go], Xtimes_shifted[-1] )  # kill over-end

            duration_ = np.maximum(0, end_ - onset_)
            if (not allow_short_intervals) and (dt_sec is not None):
                assert np.all(duration_ > dt_sec )
            anns.append(onset_ , duration_,ann_cur.description[go] )

        if wbd_pri is not None:
            cur_zeroth_bin += wbd[1,-1] + skip
            timeshift = cur_zeroth_bin/sfreq
        else:
            timeshift += Xtimes_cur[-1] + dt   #Xtimes mean actually start of the time bin

        #print(Xtimes_shifted[0],timeshift,Xtimes_cur[-1], Xtimes_cur[-1] -Xtimes_cur[0])
        dataset_bounds += [ (Xtimes_shifted[0], Xtimes_shifted[-1] ) ]


    #print('fd',Xtimes_almost[0][0],Xtimes_almost[0][-1],Xtimes_almost[1][0],Xtimes_almost[1][-1] )
    Xtimes_almost = np.hstack(Xtimes_almost)
    df = np.diff(Xtimes_almost)
    if remove_gaps_between_datasets:
        assert abs( np.max(df) - np.min(df) ) <= 1e-10,  (np.max(df), np.min(df))
    assert np.all( anns.onset >= 0 ), anns.onset

    if wbd_pri is not None:
        wbd_merged = np.hstack(wbds)
        d = np.diff(wbd_merged, axis=1)
        assert np.min(d)  > 0, d   # make sure there are no zero-sized gaps

    if ret_wbd_merged and wbd_pri is not None:
        return anns, anns_pri, Xtimes_almost, dataset_bounds, wbd_merged
    else:
        return anns, anns_pri, Xtimes_almost, dataset_bounds

def getAnnBins(ivalis,Xtimes_almost,nedgeBins, sfreq,totskip, windowsz, dataset_bounds,
               set_empty_arrays = 0, force_all_arrays_nonzero=1):
    '''
      returns dict of bin indices (both boundaries and filled) for each interval in ivalis
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
    for itype, intervals in ivalis.items():
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
    p = np.array(list(perfs) ) * 100
    perfs_str = '{:.2f}%'.format(p[0])
    if len(perfs) > 1:
        perfs_str += ',{:.2f}%'.format(p[1])
    if len(perfs) > 2:
        perfs_str += ',{:.2f}%'.format(p[2])
    return perfs_str


def _MI(arg):
    from sklearn.feature_selection import mutual_info_classif
    inds,X,y = arg
    try:
        r = mutual_info_classif(X,y, discrete_features = False)
    except ZeroDivisionError as e:
        print(e)
        r = -1
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
        max_cores = max(1, mpr.cpu_count()-gp.n_free_cores )

    assert max_cores > 0
    if max_cores == 1:
        arg = np.arange(nfeats),X,class_labels
        args += [arg]
    else:
        bnds = np.arange(0, nfeats, max(1,nfeats//max_cores) )
        for bndi,bnd in enumerate(bnds):
            if bndi < len(bnds) - 1:
                right = bnds[bndi+1]
            else:
                right = nfeats
            inds = np.arange(bnd,right )
            arg = inds,X[:,inds],class_labels
            args += [arg]

    n_jobs = max_cores
    print('getMI:  Sending {} tasks to {} cores'.format(len(args), n_jobs))
    #pool = mpr.Pool(n_jobs)
    #res = pool.map(_MI, args)
    ##if printLog:
    #pool.close()
    #pool.join()

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs)(delayed(_MI)(arg) for arg in args)

    mic = np.zeros(nfeats)
    for r in res:
        inds,mic_cur = r
        mic[inds] = mic_cur
    return mic

def confmatNormalize(confmat, norm_type='true'):
    res = None
    if norm_type == 'true':
        ### shoud be equiv to normalize = 'true' in sklearn.confusion_matrix
        ### sum across columns to get total number of true i-th entries
        #totnums = np.sum(confmat, axis = 1)
        ### confmat_ratio[i,j] = ratio of true i-th predicted as j-th among total
        ### true i-th. I want diag elements to be close to 1 and off-diag to be close
        ### to 0
        #confmat_ratio = confmat / totnums[:,None]
        #res = confmat_ratio

        # divide by nums of true (for this we need to sum over all possible
        # predictions which is axis 1)
        res = confmat / confmat.sum(axis=1, keepdims=True)

    elif norm_type == 'all':
        totnums = np.sum(confmat)
        ## confmat_ratio[i,j] = ratio of true i-th predicted as j-th among total
        ## true i-th. I want diag elements to be close to 1 and off-diag to be close
        ## to 0
        confmat_ratio = confmat / totnums
        res = confmat_ratio
    else:
        raise ValueError('wrong norm_type')
    return res

def getClfPredPower(clf,X,class_labels,class_ind, label_ids_order = None,
                    printLog = False):
    '''
    LDA perf in detecting class_ind
    - class_ind  is an interger class id
    '''
    if label_ids_order is None:
        label_ids_order = list(sorted(set(class_labels) ) )

    from sklearn.metrics import confusion_matrix
    preds = clf.predict(X)
    # Confusion matrix whose i-th row and j-th column entry indicates the number
    # of samples with true label being i-th class and predicted label being j-th
    # class.   confmat[i,j] -- true i'th predicted being j'th
    # ordering: sorted(set()+set())
    confmat = confusion_matrix(class_labels, preds, labels=label_ids_order)

    ##tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    from sklearn.metrics import recall_score
    #sens_sk = recall_score(class_labels, preds)

    #recall_per_class = recall_score(class_labels, preds,
    #                                average = None, labels=label_ids_order)

    ind_to_check = class_ind
    ## pos_actual of really positive
    pos_actual = (np.array(class_labels,dtype=int) == \
                  int(ind_to_check) )
    ## mask of really negative
    neg_actual = np.logical_not(pos_actual)

    bin_ver = np.zeros( len(class_labels), dtype=int )
    bin_ver[pos_actual] = 1
    bin_ver[neg_actual] = 0
    bin_ver_pred = np.zeros( len(class_labels), dtype=int )
    bin_ver_pred[preds == ind_to_check] = 1
    bin_ver_pred[preds != ind_to_check] = 0
    ## yes, I want 1,0
    recall_per_class_bin = recall_score(bin_ver, bin_ver_pred,
                                    average = None, labels=[1,0])

    if np.sum(pos_actual) == 0 or np.sum(neg_actual) == 0:
        s = f'one of masks is bad {np.sum(pos_actual)}, {np.sum(neg_actual)}'
        print('getClfPredPower: WARNING {}'.format(s) )
        #raise ValueError(s)
        sens = np.nan
        spec = np.nan
        F1 = np.nan
    else:
        ntot = len(class_labels)

        #X_P = X[pos_actual]
        #Pos = clf.predict(X_P)
        #Pos = preds[pos_actual]
        #TP = sum(Pos == ind_to_check)
        #sens = TP / len(Pos)

        predicted_pos = (preds == ind_to_check)
        TP = sum(predicted_pos & pos_actual)
        sens = TP / sum(pos_actual)

        #X_N = X[neg_actual]
        #Neg = clf.predict(X_N)
        #Neg = preds[neg_actual]
        #TN = sum(Neg != ind_to_check)
        #spec = TN / len(Neg)
        #spec = specificity_score(y_true,y_pred)

        predicted_neg = (preds != ind_to_check)
        TN = sum(predicted_neg & neg_actual)
        spec = TN / sum(neg_actual)

        #FP = len(Pos) - TP
        #FN = len(Neg) - TN
        #F1 =  TP / (TP + 0.5 * ( FP + FN ) )

        FP = sum(pos_actual) - TP
        FN = sum(neg_actual) - TN
        F1 =  TP / (TP + 0.5 * ( FP + FN ) )

        #if printLog:
        #    print('getClfPredPower: True pos {} ({:.3f}), all pos {} ({:.3f})'.format(TP, TP/ntot, len(Pos), len(Pos)/ntot ) )
        #    print('getClfPredPower: True neg {} ({:.3f}), all neg {} ({:.3f})'.format(TN, TN/ntot, len(Neg), len(Neg)/ntot ) )

    #if n_KFold_splits is not None:
    #    from sklearn.model_selection import KFold
    #    kf = KFold(n_splits=n_KFold_splits)
    #    res = kf.split(X)

    #    #KFold(n_splits=2, random_state=None, shuffle=False)
    #    for train_index, test_index in res:

    return sens,spec, F1, confmat

def _getPredPower_singleFold(arg):
    from xgboost import XGBClassifier
    from interpret.glassbox import ExplainableBoostingClassifier
    from numpy.linalg import LinAlgError
    (fold_type,clf,add_clf_creopts,add_fitopts,X_train,X_test,y_train,y_test,class_ind,n_classes, split_ind, printLog)  = arg
    model_cur = type(clf)(**add_clf_creopts)  # I need num LDA compnents I guess
    try:
        if isinstance(clf, XGBClassifier) or isinstance(clf,ExplainableBoostingClassifier):
            #print('WEIGHTS COMPUTED')
            from sklearn.utils.class_weight import compute_sample_weight
            class_weights = compute_sample_weight('balanced', y_train)
            model_cur.fit(X_train, y_train, **add_fitopts, sample_weight=class_weights)
        else:
            model_cur.fit(X_train, y_train, **add_fitopts)
        perf_cur = getClfPredPower(model_cur,X_test,y_test,
                                    class_ind, printLog=printLog)
        if printLog:
            #print('getPredPowersCV: CV {}/{} pred powers {}'.format(-1,n_splits,cur) )
            print('getPredPowersCV: current fold pred powers {}'.format(perf_cur) )
    except LinAlgError as e:
        print( str(e) )
        model_cur, perf_cur = None, None

    return split_ind, fold_type, model_cur,perf_cur


def getPredPowersCV(clf,X,class_labels,class_ind, printLog = False, n_splits=None,
                    ret_clf_obj=False, skip_noCV =False, add_fitopts={},
                   add_clf_creopts ={}, train_on_shuffled =True, seed=0,
                    group_labels=None, stratified = True ):
    # clf is assumed to be already fitted on entire training data here
    # TODO: maybe I need to adapt for other classifiers
    # ret = [perf_nocv, perfs_CV, perf_aver, confmat_avGroupKFolder ] and maybe list of classif objects
    # obtained during CV
    ret = []
    from globvars import gp
    if skip_noCV:
        assert n_splits is not None
        perf_nocv = None
    else:
        perf_nocv = getClfPredPower(clf,X,class_labels,class_ind, printLog=printLog)
        if printLog:
            print('getPredPowersCV: perf_nocv ',perf_nocv, X.shape)


    retcur = {}
    retcur['perf_nocv'] = perf_nocv

    y = class_labels
    n_classes = len(set(y))

    #for model_cur in cv_results['estimator']
    if n_splits is not None:
        #if n_KFold_splits is not None:
        from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
        from sklearn.model_selection import train_test_split
        if group_labels is None:
            if stratified:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            else:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_res = kf.split(X,y)
        else:
            assert len(group_labels) == len(class_labels), (len(group_labels), len(class_labels) )
            #ngl = len(set(group_labels) )
            kf = GroupKFold(n_splits=n_splits  )  # no shuffling is possible here
            split_res = kf.split(X,y,groups=group_labels)  # trains on some groups then tests on other

        n_jobs_perrun = add_clf_creopts.get('n_jobs', 1)

        #KFold(n_splits=2, random_state=None, shuffle=False)
        perfs_CV = []
        Xarr = np.array(X)
        models = []
        #indcv_indset = 0
        confmats = []

        args = []

        class_labels_u = np.unique(class_labels)

        test_indices = []
        split_ind = 0
        for train_index, test_index in split_res:
            #print(train_index )
            X_train, X_test = Xarr[train_index], Xarr[test_index]
            y_train, y_test = class_labels[train_index], class_labels[test_index]

            assert n_classes == len(set(y_train) )
            assert n_classes == len(set(y_test) )

            class_labels_test_u = np.unique(y_test)
            assert len(class_labels_test_u) == len(class_labels_u)
            if len(set(y_train)) <= 1 or len(set(y_test)) <= 1:
                continue

            fold_type = 'regular'
            arg = (fold_type,clf,add_clf_creopts,add_fitopts,\
                   X_train,X_test,y_train,y_test,class_ind, n_classes, split_ind, printLog)
            args += [arg]

            split_ind += 1

            test_indices += [test_index]


        if train_on_shuffled:
            # train on shuffled labels to check overfitting
            class_labels_shuffled = class_labels.copy()
            np.random.shuffle(class_labels_shuffled)
            X_train, X_test, y_train, y_test = \
                train_test_split(X, class_labels_shuffled, test_size=0.25,
                                 random_state=0)

            fold_type_shuffled = 'train_on_shuffled_labels'
            fold_type = fold_type_shuffled
            arg = (fold_type, clf,add_clf_creopts,add_fitopts,X_train,X_test,y_train,y_test,class_ind, n_classes, -1, printLog)
            args += [arg]

        retcur['test_indices_list'] =  [0]*len(args)

        res_fold_type_spec = None
        if n_jobs_perrun > 1:
            n_jobs = 1
            for arg in args:
                r = _getPredPower_singleFold(arg)
                split_ind, fold_type, model_cur,perfs_cur = r
                if fold_type != 'regular':
                    res_fold_type_spec = r
                else:
                    models += [model_cur]
                    perfs_CV += [perfs_cur]
                retcur['test_indices_list'][split_ind] = test_indices[split_ind]
        else:
            n_jobs = max(1, min(len(args) , mpr.cpu_count()-gp.n_free_cores) )

            if printLog:
                print('getPredPowersCV:  Sending {} tasks to {} cores'.format(len(args), n_jobs))
            #pool = mpr.Pool(n_jobs)
            #res = pool.map(_getPredPower_singleFold, args)
            #pool.close()
            #pool.join()

            from joblib import Parallel, delayed
            res = Parallel(n_jobs=n_jobs)(delayed(_getPredPower_singleFold)(arg) for arg in args)

            for r in res:
                # perfs_cur - 4-tuple
                split_ind, fold_type, model_cur,perfs_cur = r
                if fold_type != 'regular':
                    res_fold_type_spec = r
                else:
                    if (model_cur is not None) and (perfs_cur is not None):
                        models += [model_cur]
                        perfs_CV += [perfs_cur]
                        #indcv_indset += 1

                retcur['test_indices_list'][split_ind] = test_indices[split_ind]


        # convert three performance measures to a single matrix for further
        # averaging
        perfarr = np.vstack( [ (p[0],p[1],p[2])  for p in perfs_CV]  )
        confmats = [ p[-1]  for p in perfs_CV]
        not_nan_fold_inds = np.where(  np.max( np.isnan(perfarr).astype(int) , axis= 1) == 0 )[0]
        assert len(not_nan_fold_inds) > 0
        perf_aver = np.mean(perfarr[not_nan_fold_inds] , axis = 0)
        # it is bad to averge non-normalized confmat. But I still keep full
        # ones as well
        confmats = [ confmatNormalize(cm, 'true') for cm in np.array(confmats)[not_nan_fold_inds] ]
        confmat_aver =  np.mean( np.array(confmats), axis=0 )
        #ret = [perf_nocv, perfs_CV, perf_aver, confmat_aver ]
        retcur['good_fold_inds'] = not_nan_fold_inds
        retcur['bad_fold_inds'] = np.setdiff1d(np.arange(len(perfarr)) , not_nan_fold_inds )
        retcur['perfs_CV'] = perfs_CV
        retcur['perf_aver'] = perf_aver
        retcur['confmat_aver'] = confmat_aver
        if ret_clf_obj:
            #ret += [models]
            retcur['clf_objs'] = models
        #ret += [retcur]
    else:
        #ret = perf_nocv, [perf_nocv] , perf_nocv
        retcur['perfs_CV'] = [perf_nocv]
        retcur['perf_aver'] = perf_nocv
        retcur['confmat_aver'] = None
        if ret_clf_obj:
            retcur['clf_objs'] = [clf]
        #    ret += [ [clf] ]
    if res_fold_type_spec is not None:
        res_fold_type_spec_no_clfobj =  res_fold_type_spec[0], *res_fold_type_spec[2:]
    else:
        res_fold_type_spec_no_clfobj = None
    retcur['fold_type_shuffled' ] = res_fold_type_spec_no_clfobj

    #return tuple(ret)
    return retcur

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

    if X_to_transform is not None:
        X_LDA = lda.transform(X_to_transform)  # we transform all points, even bad and ulabeled ones. Transform is done using scalings
    else:
        X_LDA = None

    #classification_report(y_true, y_pred, target_names=target_names)

    # Compute prediction on training (separability)
    sens_train,spec_train,F1_train, confmat_train = getClfPredPower(lda,X_to_fit, class_labels, class_ind_to_check)
    print('-- LDA on train sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens_train,spec_train,F1_train) )

    subres = {}
    subres['ldaobj'] = lda
    subres['X_transformed'] = X_LDA
    subres['perfs'] = sens_train,spec_train,F1_train,confmat_train
    res['fit_to_all_data'] = subres

    # Compute prediction on training, shuffled labels
    class_labels_shuffled = class_labels.copy()
    np.random.shuffle(class_labels_shuffled)
    sens,spec,F1,confmat = getClfPredPower(lda,X_to_fit, class_labels_shuffled,
                                           class_ind_to_check)
    print('-- LDA check_on_shuffle sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens,spec,F1) )

    subres = {}
    subres['perfs'] = sens,spec,F1,confmat
    res['fit_to_all_data_check_on_shuffle'] = subres

    ##################
    lda_shuffled = type(lda)()
    lda_shuffled.fit(X_to_fit, class_labels_shuffled)
    sens,spec,F1,confmat = getClfPredPower(lda_shuffled,X_to_fit,
                                           class_labels_shuffled,
                                           class_ind_to_check)
    print('-- LDA train_on_shuffle labels sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens,spec,F1) )

    subres = {}
    subres['perfs'] = sens,spec,F1,confmat
    res['fit_to_all_data_train_on_shuffle'] = subres

    ########## Compute with CV
    r = getPredPowersCV(lda, X_to_fit,  class_labels, class_ind_to_check,
                                printLog=False, n_splits=n_splits, ret_clf_obj=True,
                        skip_noCV=1, train_on_shuffled=False)

    #perf_noCV, perfs_CV, res_aver_LDA, confmat_aver_LDA, ldas_CV = \

    sens_cv,spec_cv,F1_cv = r['perf_aver'] #res_aver_LDA
    ldas_CV = r['clf_objs']

    subres = {}
    #subres['ldaobjs']     = ldas_CV
    #subres['CV_perfs']     = perfs_CV
    #subres['CV_perf_aver'] = sens_cv,spec_cv,F1_cv, confmat_aver_LDA
    subres['ldaobjs']     = ldas_CV
    subres['CV_perfs']     = r['perfs_CV']
    subres['CV_perf_aver'] = sens_cv,spec_cv,F1_cv, r['confmat_aver']

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

    sens_avCV,spec_avCV,F1_avCV, confmat_avCV = \
        getClfPredPower(lda_aver,X_to_fit, class_labels, class_ind_to_check)
    print('-- LDA avCV on train sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens_avCV,spec_avCV,F1_avCV) )

    #perf_nocv_LDA_avCV, results_LDA_avCV, res_aver_LDA_avCV, confmat_av_avCV, ldas_CV_avCV = \
    r2 = getPredPowersCV(lda_aver, X_to_fit,class_labels, class_ind_to_check,
                         printLog=False, n_splits=n_splits, ret_clf_obj=True,
                         skip_noCV=1, train_on_shuffled = False)
    sens_cv_avCV,spec_cv_avCV,F1_cv_avCV = r2['perf_aver'] #res_aver_LDA_avCV

    print('-- LDA CV _avCV sens {:.2f} spec {:.2f} F1 {:.2f}'.format(sens_cv_avCV,spec_cv_avCV,F1_cv_avCV) )
    if X_to_transform is not None:
        X_LDA_CV = lda_aver.transform(X_to_transform)
    else:
        X_LDA_CV = None

    subres = {}
    subres['ldaobj'] = lda_aver
    subres['X_transformed'] = X_LDA_CV
    #subres['perfs'] = sens_avCV,spec_avCV,F1_avCV, confmat_avCV
    # here we save CV-performance of averaged LDA
    subres['perfs'] = sens_cv_avCV,spec_cv_avCV,F1_cv_avCV, r2['confmat_aver'] #_av_avCV
    res['CV_aver'] = subres

    return res


import copy
def selMinFeatSet(clf, X, class_labels, class_ind, sortinds, drop_perf_pct = 4,
                  conv_perf_pct = 2, n_splits=4, verbose=1,
                  add_fitopts={},
                  add_clf_creopts={}, check_CV_perf = False, nfeats_step = 3,
                  nsteps_report=1, max_nfeats=100, stop_if_boring = True,
                  ret_clf_obj=False, stop_cond = ['sens', 'spec', 'F1'],
                  seed=0, featnames=None):
    '''
    sortinds -- sorted increasing importance (i.e. the most imporant is the last one, as given by argsort)
    it is assumed that clf.fit has already been made
    last feature is the most significant
    returns list of tuples, the first one is for all features,
        the last one is for the best found set of features

        the last one is alwasy cross-validated
    '''
    args = copy.deepcopy(locals() )
    del args['X']
    del args['class_labels']
    from globvars import gp

    if check_CV_perf:
        s = 'CV'
    else:
        s = 'noCV'

    max_nfeats = min(max_nfeats,X.shape[1] )

    print('selMinFeatSet: --- starting {} for X.shape={}, step={}, max_nfeats={}, drop_perf_thr={}, conv_thr={}'.
          format(s,X.shape,nfeats_step,max_nfeats,drop_perf_pct,conv_perf_pct) )

    r0 = getPredPowersCV(clf,X,class_labels,class_ind, verbose >=3,
                         n_splits=n_splits, add_fitopts=add_fitopts,
                         add_clf_creopts=add_clf_creopts,
                         ret_clf_obj=ret_clf_obj,seed=seed)
    #perf_nocv_, results_, res_aver_, confmat_ = r0[:4]
    # used for stopping later
    # performance using all features
    if check_CV_perf:
        sens_full,spec_full,F1_full = r0['perf_aver']  # res_aver_
    else:
        sens_full,spec_full,F1_full = r0['perf_nocv']

    perfvec_full = []
    if 'sens' in stop_cond:
        perfvec_full += [sens_full]
    if 'spec' in stop_cond:
        perfvec_full += [spec_full]
    if 'F1' in stop_cond:
        perfvec_full += [F1_full]
    perfvec_full = np.array(perfvec_full)

    Xarr = np.array(X)
    #X_red = Xarr[:,sortinds[-2:].tolist()]
    model_red = type(clf)(**add_clf_creopts)

    # red = utsne.getClfPredPower(model_red,X_red,y,
    #                             gv.class_ids_def['trem_' + mts_letter], printLog=1)
    # print(red)

    perfs = []
    nfeats = X.shape[1]

    if featnames is not None:
        featnames = np.array(featnames)
        assert len(featnames) == nfeats

    if verbose >= 1:
        print('selMinFeatSet: --- all {} feats give perf={}, check_CV_perf = {}'.
            format( len(sortinds), sprintfPerfs([sens_full,spec_full,F1_full] ),check_CV_perf) )

    rrcur = {}
    rrcur['fold_type'] = 'all_features_present'
    rrcur['fold_ind'] = -1
    rrcur['sortinds'] = sortinds.tolist()
    rrcur['perf_nocv'] = r0['perf_nocv']
    rrcur['perf_aver'] = r0['perf_aver']
    rrcur['confmat'] = r0['confmat_aver']
    rrcur['featinds_present'] = np.arange(len(sortinds) )
    rrcur['fold_type_shuffled' ] = r0['fold_type_shuffled']
    rrcur['args']  = args

    #rrcur['max_nfeats'] = max_nfeats
    #rrcur['n_splits'] = n_splits
    #rrcur['check_CV_perf'] = check_CV_perf
    #rrcur['stop_cond']
    #rrcur['seed']
    #rrcur['drop_perf_pct']
    #rrcur['conv_perf_pct'] =

    #rr = [-1, sortinds.tolist(), perf_nocv_, res_aver_]
    if ret_clf_obj:
        rrcur['clf_objs'] = r0['clf_objs']
    #    models_cur = r0[-1]
    #    rr += [models_cur]
    perfs += [ rrcur    ]
    #perfs += [ (-1, sortinds.tolist(), perf_nocv_, res_aver_)   ]

    sens_prev,spec_prev = 0,0
    perfvec_prev = np.zeros(len(stop_cond) )
    if check_CV_perf:
        n_splits_cycle =  n_splits
    else:
        n_splits_cycle =  None

    stop_now = False
    converge_thr = conv_perf_pct / 100
    close_to_full_thr = drop_perf_pct / 100
    nsteps = 0
    inds_printed = set([])
    for i in range(1,max_nfeats+1,nfeats_step):
        # counting backwards
        inds = sortinds[-i:].tolist()[::-1] # reversal or order here is only since Jan 21. It is cosmetic
        X_red = Xarr[:,inds]
        model_red.fit(X_red, class_labels, **add_fitopts)
        # don't train on shuffled in intermediate steps
        r= getPredPowersCV(model_red,X_red,class_labels, class_ind,
                    printLog=(verbose >= 3), n_splits=n_splits_cycle,
                    add_fitopts=add_fitopts,
                    add_clf_creopts=add_clf_creopts,ret_clf_obj=ret_clf_obj,
                           train_on_shuffled = False, seed=seed)
        #perf_nocv, results, res_aver,confmat = r[:4]

        if check_CV_perf:
            sens,spec,F1 = r['perf_aver']
        else:
            sens,spec,F1,confmat_nocv = r['perf_nocv']
        perfvec_cur = []
        if 'sens' in stop_cond:
            perfvec_cur += [sens]
        if 'spec' in stop_cond:
            perfvec_cur += [spec]
        if 'F1' in stop_cond:
            perfvec_cur += [F1]
        perfvec_cur = np.array(perfvec_cur)

        rrcur = {}
        rrcur['fold_type'] = 'some_features_present'
        rrcur['fold_ind'] = i
        rrcur['perf_nocv'] = r['perf_nocv']
        rrcur['perf_aver'] = r['perf_aver']
        rrcur['confmat'] = r['confmat_aver']
        rrcur['featinds_present'] = inds

        #rr = [i,inds, perf_nocv,res_aver]
        if ret_clf_obj:
            rrcur['clf_objs'] = r['clf_objs']
            #rr += [models_cur]
        perfs += [ rrcur  ]
        #perfs += [ tuple(rr)    ]

        # the last one if always CV even though if we checking only training
        # data perf when selecting feats
        # I do not want to use abs() here
        #cond_conv = ( (sens - sens_prev) <  converge_thr ) and ( (spec - spec_prev) <  converge_thr )
        conv_dist_Linf = np.max( np.abs(perfvec_cur - perfvec_prev) )
        cond_conv = conv_dist_Linf  <  converge_thr
        #cond_close = (sens_full - sens  < close_to_full_thr) and (spec_full- spec  < close_to_full_thr)
        close_dist_Linf = np.max(perfvec_full - perfvec_cur ) # allow signed. Negative means improvement, I allow it
        cond_close = close_dist_Linf  < close_to_full_thr
        # if even the full sunsitivity was low, don't bother
        #cond_boring = ( sens_full < 0.4) and nsteps > 1 and (not stop_if_boring)
        cond_boring = ( np.max(perfvec_cur) < 0.4) and nsteps > 1 and (not stop_if_boring)
        stop_now = (cond_close and cond_conv) or cond_boring


        if verbose >= 2 and ( int(i-1 / nfeats_step) % nsteps_report == (nsteps_report-1) ):
            print('selMinFeatSet: --- search of best feat set, len(inds)={}, perf={}'.
                  format(len(inds), sprintfPerfs(r['perf_aver'],  ) ) )
            print(f'____::{len(inds)}: dist_close={100*close_dist_Linf:.2f}%,  dist_conv={100*conv_dist_Linf:.2f}%')
            inds_not_printed = list( set(inds) - inds_printed)
            if featnames is not None:
                print(featnames[inds_not_printed] )
                inds_printed = inds_printed | set(inds_not_printed)


        #print( (f'___::{len(inds)}: sens_full = {sens_full*100:.2f} %,  '
        #        f'sens_nocv = {sens*100:.2f} %, '
        #        f'sens_nocv_prev = {sens_prev:.2f} % stop_now = {stop_now}') )
        if stop_now:
            if n_splits_cycle is None:
                rstop = getPredPowersCV(model_red,X_red,class_labels, class_ind,
                                    printLog=verbose >= 3, n_splits=n_splits,
                                    add_fitopts=add_fitopts,
                                    add_clf_creopts=add_clf_creopts,
                                    ret_clf_obj=ret_clf_obj, seed=seed)
                #perf_nocv, results, res_aver, confmat = r[:4]
                sens,spec,F1 = rstop['perf_aver']

                #rr = [i,inds, perf_nocv,res_aver]
                rrcur['perf_nocv'] = rstop['perf_nocv']
                rrcur['perf_aver'] = rstop['perf_aver']
                rrcur['confmat'] = rstop['confmat_aver']
                rrcur['fold_type_shuffled' ] = rstop['fold_type_shuffled']
                if ret_clf_obj:
                    #models_cur = r[-1]
                    #rr += [models_cur]
                    rrcur['clf_objs'] = rstop['clf_objs']
                #perfs[-1] = tuple(rr)

                rrcur['stopped_natually'] = stop_now
                rrcur['cond_close'] = cond_close
                rrcur['cond_conv'] =  cond_conv
                rrcur['cond_boring'] = cond_boring

                perfs[-1] = rrcur

            if verbose >= 1:
                print('selMinFeatSet: --- ENDED search of best feat set, len(inds)={}, perf={}, {}'.\
                      format(len(inds), sprintfPerfs(rrcur['perf_aver'] ),
                      f' boring={cond_boring}, close={cond_close}, conv={cond_conv}'  ) )

            break

        sens_prev,spec_prev,F1_prev = sens,spec,F1
        perfvec_prev = perfvec_cur.copy()
        nsteps += 1

    if not stop_now:  # if we stopped because cycle has ended
        if verbose >= 1:
            print('selMinFeatSet: max number of features reached, adding full to the end')


        rrcur = {}
        rrcur['fold_type'] = 'all_features_present'
        rrcur['fold_ind'] = i+1
        rrcur['sortinds'] = sortinds.tolist()
        rrcur['perf_nocv'] = r0['perf_nocv']
        rrcur['perf_aver'] = r0['perf_aver']
        rrcur['confmat'] = r0['confmat_aver']
        rrcur['featinds_present'] = np.arange(max_nfeats)
        rrcur['fold_type_shuffled' ] = r0['fold_type_shuffled']
        rrcur['stopped_natually'] = stop_now
        rrcur['cond_close'] = cond_close
        rrcur['cond_conv'] =  cond_conv
        rrcur['cond_boring'] = cond_boring


        #rr = [i+1, sortinds.tolist(), r0['perf_nocv'], r0['perf_aver'] ]
        if ret_clf_obj:
            #models_cur = r0[-1]
            #rr += [models_cur]
            rrcur['clf_objs'] = r0['clf_objs']
        #perfs += [ tuple(rr)  ]
        perfs += [ rrcur ]

    return perfs



def makeClassLabels(sides_hand, grouping, int_types_to_distinguish,
                    ivalis_tb_indarrays, good_inds, num_labels_tot,
                    rem_neut=1):
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
    print(f'class_ids_grouped = {class_ids_grouped}')

    #TODO: make possible non-main side


    # first fill everthing with neutral label
    #class_labels = np.repeat(gv.class_id_neut,len(Xconcat_imputed))
    class_labels = np.repeat(gp.class_id_neut,num_labels_tot)
    assert gp.class_id_neut == 0

    from collections.abc import Iterable

    revdict = {}
    # set class label for current interval types
    bincounts_per_class_name = {}
    # over all interval_types
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
    inds_not_neut = None
    if rem_neut:
        neq = class_labels_good != gp.class_id_neut
        inds_not_neut = np.where( neq)[0]
        class_labels_good = class_labels_good[inds_not_neut]
    #else:
    #    classes = ['neut'] + classes  # will fail if run more than once

    return class_labels, class_labels_good, revdict, class_ids_grouped, inds_not_neut

def countClassLabels(class_labels_good, class_ids_grouped=None,revdict=None):
    #assert revdict is not None or (class_ids_grouped is not None)
    if isinstance (class_labels_good,np.ndarray):
        assert class_labels_good.ndim == 1
    elif not isinstance (class_labels_good,list):
        raise ValueError('Wrong type')
    counts = {}
    if revdict is not None:
        for cid in set(class_labels_good):
            num_cur = np.sum(class_labels_good == cid)
            lbl = revdict.get(cid, 'cid={}'.format(cid) )
            counts[lbl ] = num_cur
    else:
        if class_ids_grouped is None:
            for cid in set(class_labels_good):
                num_cur = np.sum(class_labels_good == cid)
                counts[cid ] = num_cur
        else:
            raise ValueError('to be debugged')
            cids_used = []
            class_names_used = []
            for class_name in class_ids_grouped:
                cid = class_ids_grouped[class_name]
                if cid in cids_used:
                    class_name
                else:
                    lbl = class_name
                #print(cid)
                num_cur = np.sum(class_labels_good == cid)
                counts[lbl] = num_cur
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


#trem_class_label = 'trem_L'
#global main_class_id = redict[trem_class_label]
def balanced_accuracy_pseudo_binary(y_true, y_pred, *, sample_weight=None,
                        adjusted=False, main_class_id=None):
    #global main_class_id
    assert main_class_id is not None
    y_true2 = np.ones(len(y_true))
    y_true2[y_true != main_class_id] = 0

    y_pred2 = np.ones(len(y_pred))
    y_pred2[y_pred != main_class_id] = 0

    y_true = y_true2
    y_pred = y_pred2

    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        import warnings
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

def getScoresPerClass(class_ids,scores, ret_bias=False):
    # assumes shape  numpoints x numclasses x (numfeats  + 1)
    if scores.ndim == 2:
        lblinds = np.sort( np.unique(class_ids) )
    elif scores.ndim == 3:
        lblinds = range(scores.shape[1] )
    else:
        raise ValueError(f'Wrong ndim, {scores.shape}')
    assert scores.shape[0] == len(class_ids), (scores.shape[0], len(class_ids) )
    r = np.zeros( (len(lblinds) ,scores.shape[-1] - 1 ) )
    biases = np.zeros( len(lblinds) )
    for lblind in lblinds:
        # select points where true class is like the current one
        ptinds = np.where(class_ids == lblind)[0]
        classid_enconded = lblind
        if scores.ndim == 3:
            # XGB doc: Note the final column is the bias term
            sc = scores[ptinds,lblind,0:-1]
            bias_cur = np.mean(scores[ptinds,lblind,-1] )
        elif scores.ndim == 2:
            # XGB doc: Note the final column is the bias term
            sc = scores[ptinds,0:-1]
            bias_cur = np.mean(scores[ptinds,-1] )
        scores_cur = np.mean(sc, axis=0)
        r[lblind,:]  = scores_cur

        biases[lblind] = bias_cur
    rr = r
    if ret_bias:
        rr = r,biases
    return rr

def sklearn_VIF(X, exogs,n_jobs, VIF_thr=10, search_worst=True,
                return_obj = 'no', printLog= 0):
    from sklearn.linear_model import LinearRegression
    import gc
    # data is nbins, nchans x
    # exogs -- subset of indices from range(X.shape[1])
    # if search_worse is False, returns first of exogs that is bad

    if isinstance(exogs , int):
        exogs = [exogs]

    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}
    if return_obj == 'no':
        linreg_dict = None
    else:
        linreg_dict = {}

    worst_VIF = -1
    worst_VIF_exog = -1

    colinds_all = np.arange(X.shape[1])  # it can be longer than len(exogs)!
    inds_regress_from_dict = {}

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in colinds_all if i != exog]
        assert len(not_exog) > 0
        #print(not_exog,exog)
        Xcur, ycur = X[:,not_exog], X[:,exog]
        #print(Xcur.shape,ycur.shape)

        # extract r-squared from the fit
        linreg = LinearRegression(n_jobs=n_jobs)
        r_squared = linreg.fit(Xcur, ycur).score(Xcur, ycur)

        if printLog:
            print(f'sklearn_VIF: exog={exog} not_exog = {not_exog} coef.shape={linreg.coef_.shape}')
        # can happen on test data
        if 1 - r_squared < 1e-10:
            vif = 1e10
        else:
            # calculate VIF
            vif = 1/(1 - r_squared)

        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

        inds_regress_from_dict[exog] = not_exog

        if return_obj == 'all':
            linreg_dict[exog] = linreg

        if vif > worst_VIF :
            worst_VIF = vif
            worst_VIF_exog = exog
            if return_obj == 'worst':
                # we want to forget the others (to save mem)
                linreg_dict = {exog:linreg}

        if not search_worst and vif > VIF_thr:
            break

        gc.collect()
    #    The cutoff to detect multicollinearity: VIF > 10 or Tolerance < 0.1
    return vif_dict,tolerance_dict,linreg_dict

def findBadColumnsVIF(X,VIF_thr=10,n_jobs=-1, search_worst=False, featnames=None,
                      printLog=1, rev=False):
    # finds iteratively what to get rid of
    #for coi,col_ind in enumerate(col_ordering):
    # search_worst -- whether we throw away worst or just first bad
    # order in colinds_bad, vfs_list and lingreg_obj is the same
    # colind_bad is the index in global numbering


    if featnames is not None:
        assert len(featnames) == X.shape[1]
    import gc;
    colinds_all = np.arange(X.shape[1])

    if len(colinds_all) < 2:
        return [],colinds_all,[],[],[],[]

    if rev:
        revinds = np.arange(X.shape[1])[::-1]
        r = findBadColumnsVIF(X[:,revinds],VIF_thr,n_jobs,search_worst,
                              np.array(featnames)[revinds],printLog,rev=0)
        colinds_bad,cols_good, vfs_list, featsets_list, linreg_objs, exogs_list = r

        colinds_bad = revinds[colinds_bad]
        if len(colinds_bad) == 0:
            cols_good = colinds_all
        else:
            cols_good = np.setdiff1d(colinds_all,colinds_bad)

        for i in range(len(featsets_list)) :
            fs = featsets_list[i]
            if i < len(exogs_list):
                exogs = exogs_list[i]
                fs_rev = revinds[ fs[exogs] ] # elements of colinds_good
            featsets_list[i] = revinds[fs][::-1]
            if i < len(exogs_list):
                exogs_list[i] = [list(featsets_list[i]).index(e) for e in fs_rev][::-1]
            linreg_objs[i].coef_ = linreg_objs[i].coef_[::-1]

        return colinds_bad,cols_good, vfs_list, featsets_list, linreg_objs, exogs_list


    colinds_bad = []
    vfs_list = []
    featsets_list = []
    linreg_objs = []
    exogs_list = []
    for iter_num in colinds_all:
        colinds_good = np.setdiff1d(colinds_all,colinds_bad)  # inds in orig
        if len(colinds_good) <= 1:  # then we could not select not_exog
            break
        Xcur = X[:,colinds_good]

        if len(colinds_bad) >= 2:  # assert we have increasing order
            bads_increasing_order = np.min( np.diff(colinds_bad) ) > 0
        else:
            bads_increasing_order = True

        # in general exogs are indices in colinds_good, NOT in original array
        if len(colinds_bad) == 0:
            exogs = np.arange( len(colinds_good) )
        else:
            # bad inds were added in increasing order
            # so before the last bad ind all other inds were not dependent of
            # following columns, so we should start checking next after last
            # bad since by removing we cannot increase number of regressable
            # so take only columns of (X[:,colinds_good]) with lower

            if not search_worst:
                assert bads_increasing_order

            # if we lucky for search_worst we can also use this trick
            if bads_increasing_order:
                exogs = np.where(colinds_good > np.max(colinds_bad) )[0]
            else:
                exogs = np.arange( len(colinds_good) )

        if len(exogs) == 0:
            print('len(exogs) == 0, exiting')
            break
        #exogs = np.arange(Xcur.shape[1])
        ret_obj = 'worst'
        vfs,tol_dict,linreg_dict = sklearn_VIF(Xcur, exogs,
                            n_jobs=n_jobs, VIF_thr=VIF_thr,
                            search_worst=search_worst, return_obj=ret_obj)
        gc.collect()
        vfs_list += [vfs]
        exog_outs,values = zip(*vfs.items())
        # mi is an index of element in exog_outs, NOT in orig array, NOT in colinds_good
        mi = np.argmax(values)
        vf_worst = values[mi]
        if ret_obj == 'all':
            raise ValueError('not implemented')
        linreg_obj = linreg_dict[exog_outs[mi] ]
        linreg_objs += [linreg_obj]
        exogs_list += [exogs]  # indices of elements of colinds_good

        colind_bad = colinds_good [ exog_outs[mi] ]
        featsets_list += [colinds_good]

        featname_info = ''
        if featnames is not None:
            featname_info = featnames[colind_bad]
        if printLog:
            print(f'findBadColumnsVF: iter_num={iter_num} '
                f'len(colinds_bad)={len(colinds_bad)}/{len(colinds_all)} '
                  f'worst(-ish) VIF is {vf_worst:.2f}, badind={colind_bad}{featname_info} (search_worst={int(search_worst)})', flush=True)
            #print(f'linreg shape {linreg_obj.coef_.shape}, '
            #      f'colinds_good[exogs={exogs}]={colinds_good[exogs]}, '
            #      f'colinds_good={colinds_good}, exog_outs={exog_outs}, {values}' )
        if vf_worst >= VIF_thr:
            colinds_bad += [colind_bad]
        else:
            break

        gc.collect()

    if len(colinds_bad) == 0:
        cols_good = colinds_all
    else:
        cols_good = np.setdiff1d(colinds_all,colinds_bad)
    return colinds_bad,cols_good, vfs_list, featsets_list, linreg_objs, exogs_list


def reconstructFullScoresFromVIFScores(scores_per_class_VIF, nfeats_total,
                                      colinds_bad,cols_good, featsets_list, linreg_objs, exogs_list,
                                       printLog=0, ):

    if len(colinds_bad) == 0:
        return scores_per_class_VIF
    #scores_reconstructed = scores_per_class.copy() * np.nan # I only need shape
    #exogs_glob_inds = featsets_list[-1][exogs_list[-1]]
    #actual_exogs = np.setdiff1d(exogs_glob_inds, [colinds_bad[-1]])
    if scores_per_class_VIF.ndim == 2:
        scores_reconstructed = np.ones( (scores_per_class_VIF.shape[0] , nfeats_total)  ) *np.nan
        scores_reconstructed[:,cols_good] = scores_per_class_VIF
    elif scores_per_class_VIF.ndim == 3:
        scores_reconstructed = np.ones( (scores_per_class_VIF.shape[0],
                                         scores_per_class_VIF.shape[1] , nfeats_total + 1)  ) *np.nan
        assert np.max(cols_good) < nfeats_total
        # bias should go to the end separately
        scores_reconstructed[:,:,cols_good] = scores_per_class_VIF[:,:,:-1]
        scores_reconstructed[:,:,-1] = scores_per_class_VIF[:,:,-1]

    for i,badind in list(enumerate(colinds_bad) )[::-1]:
        linreg_coef = linreg_objs[i].coef_
        intercept = linreg_objs[i].intercept_
        # these are indices (global) which we tried to regress FROM featsets_list[i]
        exogs_glob_inds = featsets_list[i][exogs_list[i]]  # colinds_good
        inds_regressed_from = np.setdiff1d( featsets_list[i] , [badind] )
        #actual_exogs = np.setdiff1d(exogs_glob_inds, [badind])
        if printLog:
            print(f'ordinal={i},badind={badind}, exogs_glob_inds={exogs_glob_inds},'
                  f'regressed from { inds_regressed_from }')
            print(linreg_coef,intercept)

        assert len(inds_regressed_from) == len(linreg_coef)
        #reconstracted_perf = sc[]
        if scores_per_class_VIF.ndim == 2:
            val = np.dot( scores_reconstructed[:,inds_regressed_from], linreg_coef)
            scores_reconstructed[:,badind] = val
        elif scores_per_class_VIF.ndim == 3:
            val = np.dot( scores_reconstructed[:,:,inds_regressed_from], linreg_coef)
            scores_reconstructed[:,:,badind] = val
        if printLog:
            print(val)


    #if scores_per_class_VIF.ndim == 3:
    #    scores_reconstructed

    return scores_reconstructed

def selFeatsBoruta(X,y,verbose = 2,add_clf_creopts=None, n_jobs = -1, random_state=0):
    from boruta import BorutaPy
    from xgboost import XGBClassifier
    #from sklearn.utils.class_weight import compute_sample_weight
    #class_weights = compute_sample_weight('balanced', y)

    ######

    if add_clf_creopts is None:
        add_clf_creopts={ 'n_jobs':n_jobs, 'use_label_encoder':False,
                        'importance_type': 'total_gain' }
        tree_method = 'exact'
        method_params = {'tree_method': tree_method}

        add_clf_creopts.update(method_params)
    clf_XGB = XGBClassifier(**add_clf_creopts)

    ##########

    # define Boruta feature selection method
    feat_selector = BorutaPy(clf_XGB, n_estimators='auto',
                             verbose=verbose, random_state=random_state)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)

    # check selected features - first 5 features are selected
    #feat_selector.support_

    # call transform() on X to filter it down to selected features
    #X_filtered = feat_selector.transform(X)

    # check ranking of features
    return np.where(feat_selector.support_)[0], feat_selector.ranking_

#def getFeatIndsRelToChn(featnames, chnpart='LFP'):
#    from featlist import  parseFeatNames
#    r = parseFeatNames(featnames)
#    chnames = [chn for chn in r['ch1'] if chn.find(chnpart) >= 0] + [chn for chn in r['ch2'] if (chn is not None and chn.find(chnpart) >= 0) ]
#    chnames = list(sorted(set(chnames)))
#    return chnames


def selBestLFP(output_cur, clf_type = 'XGB', chnames_LFP = None, s= '',
               featnames=None, nperfs = 2):
    #output_cur = output_per_int_types[int_type]
    #s = '{}:{}:{}:{}'.format(k,prefix,grouping,int_type)
    #s = '{}:{}:{}:{}'.format(ki,prefix,grouping,int_type)

    featnames = output_cur['feature_names_filtered']
    if chnames_LFP is None:
        from featlist import getChnamesFromFeatlist
        chnames_LFP = getChnamesFromFeatlist(featnames, mod='LFP')

    anvers = output_cur['{}_analysis_versions'.format(clf_type)]
    anver_full = anvers['all_present_features']
    if 'CV_aver' in anver_full:
        perfs_full = anver_full['CV_aver']['perfs']   # exclude conf matrix
    else:
        perfs_full = anver_full['perf_dict']['perf_aver']   # exclude conf matrix

    perfs_str_full = sprintfPerfs(perfs_full[:3])
    perfs_full = np.array(perfs_full[:nperfs])
    print('selBestLFP {}:: Full avCV perfs {}'.format(s,perfs_str_full))
    pdrop = {}
    for chn in chnames_LFP:
        key = 'all_present_features_but_{}'.format(chn)
        anver = anvers.get(key,None)
        if anver is None:
            print(f'selBestLFP: {chn} anver is None')
            break
        #perfs = [p[:nperfs] for p in anver['CV']['CV_perfs'] ]
        #perfs = [p[:nperfs] for p in anver['CV']['CV_perfs'] ]
        if 'CV_aver' in anver:
            perfs = anver['CV_aver']['perfs']
        else:
            perfs = anver['perf_dict']['perf_aver']

        #print(perfs_full, perfs)
        perfs_str = sprintfPerfs(perfs[:3] )
        perfs = np.array(perfs[:nperfs] )

        print('selBestLFP {}:: No {} avCV perfs {}'.format(s,chn,perfs_str))
        perf_drop = perfs_full[:nperfs] - perfs

        pdrop[chn] = perf_drop
        print(f'selBestLFP: no {chn} perf drop: {sprintfPerfs(perf_drop)}' )
    if len(pdrop) == 0:
        pdrop = None
        winning_chan = None
    else:
        winning_chan = chooseBestLFPchan(pdrop, chnames_LFP)

    return pdrop, winning_chan

# not used

def chooseBestLFPchan(pdrop, chnames_LFP):
    #winnder_chans = {}
    pds = []
    # I want ordered access across
    for chn in chnames_LFP:
        pds += [pdrop[chn]]
    pds = np.vstack(pds)
    # axis 0 -- index of channel
    maxdrop_perchan = np.max(pds,axis=1)
    maxdrop_perchan = np.maximum(maxdrop_perchan,0)

    mindrop_perchan = np.min(pds,axis=1)
    mindrop_perchan = np.minimum(mindrop_perchan,0)
    # dropping channel should maximum worsen and minimum improve (mindrop is neg)
    inds = np.argsort(maxdrop_perchan + mindrop_perchan)
    win_ind = inds[-1]
    return chnames_LFP[win_ind]
    #winnder_chans[s] = chnames_LFP[win_ind]
    #print(pds*100)
    #print('{:50} {} {}'.format( s, chnames_LFP[ win_ind ], sprintfPerfs( pds[win_ind] ) ) )

#def chooseBestLFPchan(perf_drops, chnames_LFP):
#    winnder_chans = {}
#    for s,pdcurd in perf_drops.items():
#        pds = []
#        # I want ordered access across
#        for chn in chnames_LFP:
#            pds += [pdcurd[chn]]
#        pds = np.vstack(pds)
#        # axis 0 -- index of channel
#        maxdrop_perchan = np.max(pds,axis=1)
#        maxdrop_perchan = np.maximum(maxdrop_perchan,0)
#
#        mindrop_perchan = np.min(pds,axis=1)
#        mindrop_perchan = np.minimum(mindrop_perchan,0)
#        # dropping channel should maximum worsen and minimum improve (mindrop is neg)
#        inds = np.argsort(maxdrop_perchan + mindrop_perchan)
#        win_ind = inds[-1]
#        winnder_chans[s] = chnames_LFP[win_ind]
#        #print(pds*100)
#        print('{:50} {} {}'.format( s, chnames_LFP[ win_ind ], sprintfPerfs( pds[win_ind] ) ) )


def selLFP_calcPerfDrops_multi(output_per_raw, rawnames_to_use = None, groupings_to_use = None,
                         prefixes_to_use = None, clf_type = 'XGB' ):
    perf_drops = {}
    from featlist import getChnamesFromFeatlist
    # if all raws were processed together, they'll have same performances saved, no need to repeat
    if rawnames_to_use is None:
        rawnames_to_use = output_per_raw.keys()

    perf_drops_res = {}
    best_chans = {}

    #for k in output_per_raw:
    for ki,k in enumerate(rawnames_to_use):
        output_per_prefix = output_per_raw[k]
        #for prefix in output_per_prefix:
        prefixes_to_use_cur = prefixes_to_use
        if prefixes_to_use_cur is None:
            prefixes_to_use_cur = output_per_prefix.keys()
        for prefix in prefixes_to_use_cur:
            output_per_grouping = output_per_prefix.get(prefix,None)
            if output_per_grouping is None:
                continue
            #for grouping in output_per_grouping:
            groupings_to_use_cur = groupings_to_use
            if groupings_to_use_cur is None:
                groupings_to_use_cur = output_per_grouping.keys()
            for grouping in groupings_to_use_cur:
                output_per_int_types = output_per_grouping[grouping]
                for int_type in output_per_int_types:
                    output_cur = output_per_int_types[int_type]
                    #s = '{}:{}:{}:{}'.format(k,prefix,grouping,int_type)
                    s = '{}:{}:{}:{}'.format(ki,prefix,grouping,int_type)
                    if output_cur is None:
                        #print(k,prefix,grouping,int_type,'=None')
                        continue

                    featnames = output_cur['feature_names_filtered']
                    chnames_LFP = getChnamesFromFeatlist(featnames, mod='LFP')

                    lda_anver = output_cur['{}_analysis_versions'.format(clf_type)]
                    anver_full = lda_anver['all_present_features']
                    perfs_full = anver_full['CV_aver']['perfs']
                    perfs_full = np.array(perfs_full)
                    perfs_str_full = sprintfPerfs(perfs_full)
                    print('{}:: Full avCV perfs {}'.format(s,perfs_str_full))
                    perf_drops[s] = {}
                    for chn in chnames_LFP:
                        key = 'all_present_features_but_{}'.format(chn)
                        anver = lda_anver.get(key,None)
                        if anver is None:
                            #print(s,' fail')
                            break
                        perfs = np.mean(anver['CV']['CV_perfs'], axis=0)
                        perfs = np.array(perfs)

                        #print(perfs_full, perfs)
                        perfs_str = sprintfPerfs(perfs)

                        print('{}:: No {} avCV perfs {}'.format(s,chn,perfs_str))
                        perf_drop = perfs_full - perfs

                        perf_drops[s][chn] = perf_drop
                        print('  Perf drop: ', sprintfPerfs(perf_drop) )
                    if len(perf_drops[s]) == 0:
                        del perf_drops[s]
                        winning_chan = None
                    else:
                        winning_chan = chooseBestLFPchan(perf_drops[s], chnames_LFP)

                best_chans[k][prefix][grouping][int_type] = winning_chan


                    #print(lda_anver.keys())

    return perf_drops, best_chans

def genParList(param_grids, keys=None):
    # taken from sklearn CV
    #print('fds')
    if not isinstance(param_grids,list):
        param_grids = [param_grids]
    from itertools import product
    for p in param_grids:

        # Always sort the keys of a dictionary, for reproducibility
        if keys is None:
            keys = p.keys()

        items = [(k,v) for k,v in p.items() if k in keys]
        items = sorted(items)
        if items is None:
            yield {}
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params

def gridSearch(dtrain, params, param_grids, keys, num_boost_round=100,
              early_stopping_rounds=10, nfold=5, seed=0, shuffle=True,
               printLog = False, main_metric = 'mae'):
    search_grid_cur = list ( genParList(param_grids, keys) )
    # Define initial best params and MAE
    import xgboost
    from time import time


    min_mae = float("Inf")
    best_params = None
    cv_results_best = None
    for pd in search_grid_cur:
        params_cur = dict(params.items()) #copy

        for k in ['use_label_encoder', 'importance_type', 'n_estimators']:
            if k in params_cur:
                del params_cur[k]

        for parname,parval in pd.items():
            params_cur[parname] = parval

        #print(params_cur)

        time_start = time()
        #metrics = {'mae','rmse','logloss'}
        if main_metric == 'mlogloss':
            second_metric = 'merror'
        else:
            second_metric = 'mae'
        metrics = {main_metric, second_metric}
        cv_results = xgboost.cv(params_cur,
            dtrain,
            num_boost_round=num_boost_round,
            seed=seed,
            nfold=nfold, shuffle=shuffle,
            metrics=metrics,
            early_stopping_rounds=early_stopping_rounds )


        time_end = time()
        time_passed = time_end - time_start
#         display(cv_results)
#         return

        # Update best MAE
        mean_mm = cv_results[f'test-{main_metric}-mean'].min()
        boost_rounds = cv_results[f'test-{main_metric}-mean'].argmin()
        mean_secmet = cv_results[f'test-{second_metric}-mean'].min()
        if printLog:
            print(f"{pd} \t{main_metric}: mae={mean_secmet:.4f} for {boost_rounds} rounds, {time_passed:.3f}s", flush=True)
        if mean_mm < min_mae:
            min_mae = mean_mm
            best_params = pd
            cv_results_best = cv_results
    return best_params, cv_results_best

def gridSearchSeq(X,y,params,search_grid,param_list_search_seq,
                  num_boost_round=100,
              early_stopping_rounds=10, nfold=5, seed=0, shuffle=True,
                  printLog=False, sel_num_boost_round = False,
                  main_metric='mae', test_dataset_prop = 0.2):

    assert X.shape[0] == len(y), (X.shape[0], len(y)  )

    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    X_train, X_test, y_train, y_test =\
        train_test_split(X,y,test_size=test_dataset_prop, random_state=seed,
                         shuffle=True)
    #assert len(set(y_train) ) == len(set(

    dtrain = xgb.DMatrix(X, y)

    best_params_list = []
    cv_resutls_best_list = []
    params_mod = dict( params.items() )
    for parlist in param_list_search_seq:
        best_params,cv_results_best= gridSearch(dtrain, params_mod,
                search_grid, parlist, num_boost_round = num_boost_round,
                early_stopping_rounds = early_stopping_rounds,
                nfold=nfold, seed=seed, printLog= printLog, main_metric=main_metric )
        best_params_list += [best_params]
        cv_resutls_best_list += [cv_results_best]
        params_mod.update(best_params)
    params_final = params_mod


    num_boost_round_best = None
    if sel_num_boost_round:
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest  = xgb.DMatrix(X_test,  y_test)
        #evals=[(dtest, "Test")],

        params_cur = dict(params_mod.items()) #copy
        for k in ['use_label_encoder', 'importance_type', 'n_estimators']:
            if k in params_cur:
                del params_cur[k]


        model = xgb.train( params_cur,
                dtrain, num_boost_round=num_boost_round,
                evals=[(dtest, "Test")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval = printLog)
        num_boost_round_best = model.best_iteration + 1
        params_final['n_estimators'] = num_boost_round_best
        if printLog:
            print("boost rounds sel: best score: {:.2f} in {} rounds".format(model.best_score,
                                                                             num_boost_round_best))

    return params_final, best_params_list, cv_resutls_best_list, num_boost_round_best

def shapr_proxy(X_train, y_train, colnames=None, groups = None,
                n_samples=200, n_batches=1, class_weights=None,
                add_clf_creopts={}, n_combinations=None):
    #colnames = None
    ##groups = [["0"],["1","2"]]
    #d = { 'A':["0"], 'B':["1","2"] }  # does not work
    #n_samples = 200
    #n_batches = 1
    #add_clf_creopts

    '''
    n_batches: Positive integer. Specifies how many batches the total
            number of feature combinations should be split into when
            calculating the contribution function for each test
            observation. The default value is 1. Increasing the number of
            batches may significantly reduce the RAM allocation for
            models with many features. This typically comes with a small
            increase in computation time.

    n_samples: Positive integer. Indicating the maximum number of samples
            to use in the Monte Carlo integration for every conditional
            expectation. See also details.

    '''

    if groups is not None:
        assert isinstance(groups,dict)
        import rpy2.rlike.container as rlc
        tags,list_items = list( zip(*groups.items()) )
        groups = rlc.TaggedList(list_items, tags=tuple(tags) )

    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri

    numpy2ri.activate()
    #import feather as ft

    assert X_train.ndim == 2
    assert X_train.shape[0] == len(y_train)
    if  class_weights is not None:
        assert  len( class_weights) == len(y_train)

    if colnames is None:
        colnames = map(str,range(X_train.shape[1]))
    assert len(colnames) == X_train.shape[1]
    df = pd.DataFrame(X_train, columns = colnames  )

    xgboost = importr('xgboost')
    shapr = importr('shapr')
    base= importr('base')

    # arrow_feather = importr('arrow')
    # arrow_feather.read_feather(fn)


    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(df)

    X_train_r = robjects.r['matrix'](X_train, ncol=X_train.shape[1])
    dtrain = xgboost.xgb_DMatrix(X_train, label = y_train)
    #robjects.r('colnames(X_train_r) <- range(10)' )


    num_class = len(set(y_train) )
    if num_class > 2:
        objective = "multi:softprob"

        raise ValueError('shapr will fail')

        model = xgboost.xgboost(dtrain, nround = 20, verbose = False,
                weight=class_weights, objective = objective , num_class=num_class,
                                **add_clf_creopts)#, **add_fitopts)
    else:
        objective = "binary:logistic"

        model = xgboost.xgboost(dtrain, nround = 20,
        verbose = False, weight=class_weights,
                            objective = objective ,
                                **add_clf_creopts)#, **add_fitopts)

    p = base.mean(y_train)   # bias
    if groups is None:
        print('Running shapr.shapr withOUT groups')

        # Prepare the data for explanation
        if n_combinations is not None:
            explainer = shapr.shapr(r_from_pd_df, model, n_combinations = n_combinations)
        else:
            explainer = shapr.shapr(r_from_pd_df, model)
        #> The specified model provides feature classes that are NA. The classes of data are taken as the truth.

        # Specifying the phi_0, i.e. the expected prediction without any features

        print('Starting corrected kernelShap (no groups)')

        # Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
        # the empirical (conditional) distribution approach with bandwidth parameter sigma = 0.1 (default)
        explanation = shapr.explain(
        r_from_pd_df,
        approach = "empirical",
        explainer = explainer,
        prediction_zero = p,
        n_samples = n_samples,
        n_batches = n_batches
        )

    else:
        #c("gaussian", rep("empirical", 4), #      rep("copula", 5))
        #group <- list(A = c("lstat", "rm"), B = c("dis", "indus"))
        #explainer_group <- shapr(x_train, model, group = group)
        print('Running shapr.shapr with groups')

        if n_combinations is not None:
            explainer_group = shapr.shapr(r_from_pd_df, model, group=groups, n_combinations = n_combinations)
        else:
            explainer_group = shapr.shapr(r_from_pd_df, model, group=groups)

        print(f'Starting corrected kernelShap (on {len(groups)} groups)')

        explanation = shapr.explain(
            r_from_pd_df,
            explainer_group,
            approach = "empirical",
            prediction_zero = p,
            n_samples = n_samples,
            n_batches = n_batches
        )
    #  print(explain_groups$dt)



    numpy2ri.deactivate()

    return explanation


def classSubsetInds(y,cid):
    if isinstance(y,list):
        y = np.array(y)
    inds0 = np.where(y == cid) [0]
    #print(inds0)
    uy = np.unique(y)
    inds_per_class = {}
    for class_id in (set(uy) - set([cid])):
        inds_cur = np.where(y == class_id)[0]
        inds_cur_full = np.append(inds0,inds_cur)
        #print(inds_cur.shape,inds_cur_full.shape,inds0.shape)
        inds_per_class[(cid,class_id)] = inds_cur_full
    return inds_per_class


def _computeEBM(X,y,EBM,ebm_creopts,revdict, class_ind_to_check_lenc, n_splits=5,
               EBM_CV=0, featnames_ebm=None, tune_params = False,
                params_space = None, max_evals = 20):
    import itertools
    from interpret.glassbox.ebm.utils import EBMUtils
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.model_selection import train_test_split
    from utils_postprocess_HPC import EBMlocExpl2scores

    from interpret.glassbox import ExplainableBoostingClassifier as _EBM
    from interpret.privacy import DPExplainableBoostingClassifier as _DPEBM

    do_oversample = 1
    if do_oversample:
        from imblearn.over_sampling import RandomOverSampler
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_orig,y_orig = X,y
        X,y = oversample.fit_resample(X,y)

    if tune_params:
        ebm_creopts_loc = dict(ebm_creopts.items())  # make a copy
        from hyperopt import fmin,tpe, Trials, STATUS_OK
        if EBM == _DPEBM:
            q = 1e-1
            qX = np.quantile(X,[q,1-q], axis=0) # 2 x X.shape[0]
            privacy_schema = dict( zip( range(X.shape[0]), list( zip(*qX) )  ) )
            privacy_schema['target'] = (0,0)  # it will not be used
            ebm_creopts_loc['composition'] = 'gdp'
            ebm_creopts_loc['privacy_schema'] = privacy_schema

        from sklearn.model_selection import cross_val_score

        def objective(space_loc):
            import time
            st_time = time.time()
            ebm_creopts_locloc = dict(ebm_creopts_loc.items())
            ebm_creopts_locloc.update(space_loc)

            #model = EBM(**space_loc)
            model = EBM(**ebm_creopts_locloc)
            accuracy = cross_val_score(model,X,y,cv=4, scoring='balanced_accuracy').mean()

            return {'loss':-accuracy, 'status':STATUS_OK,
                    'creopts':ebm_creopts_loc, 'effparams':space_loc,
                    'time':time.time() - st_time}

        trials = Trials()
        best = fmin(fn=objective, space=params_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        trial_ind = np.argmin( trials.losses() )
        best_detailed = trials.results[ trial_ind ]['creopts']
        ebm_creopts_loc.update(best_detailed )
        ebm_creopts = ebm_creopts_loc
    else:
        trials = None

    r0_ebm = None
    perf_per_cp = None
    ebm_merged = None
    if EBM_CV:
        ebm = EBM(**ebm_creopts)
        if do_oversample:
            class_weights_cur = None
        else:
            class_weights_cur = compute_sample_weight('balanced',y)

        ebm.fit(X, y, sample_weight=class_weights_cur)

        r0_ebm = getPredPowersCV(ebm,X,y,
                class_ind_to_check_lenc, printLog = True, n_splits=n_splits,
                ret_clf_obj=True, skip_noCV =False, add_fitopts={},
                add_clf_creopts =ebm_creopts, train_on_shuffled =False, seed=0,
                group_labels=None )
        sens,spec, F1 = r0_ebm['perf_aver']
        confmat  = r0_ebm['confmat_aver']

        perf_per_cp = {}
        uls = list(set(y))
        if len( uls  ) == 2:
            for icv,ebm_cur in enumerate(r0_ebm['clf_objs']):
                inds_cur = r0_ebm['test_indices_list' ][icv]
                inds_per_cp = classSubsetInds(y[inds_cur], class_ind_to_check_lenc)
                for cp,inds_cur_cp in inds_per_cp.items():
                    if cp not in perf_per_cp:
                        perf_per_cp[cp] = []

                    XX,yy = X[inds_cur_cp],y[inds_cur_cp]
                    #print('fdsfs')
                    perf_cur = getClfPredPower(ebm_cur,XX,yy,
                        class_ind_to_check_lenc, printLog=False)
                    #import pdb; pdb.set_trace()
                    perf_per_cp[cp] += [perf_cur]

            for cp,inds_cur_cp in inds_per_cp.items():
                confmats = [p[-1] for p in perf_per_cp[cp] ]
                confmats = [ confmatNormalize(cm, 'true') for cm in np.array(confmats) ]
                confmat_aver =  np.mean( np.array(confmats), axis=0 )

                perfarr = np.vstack( [ (p[0],p[1],p[2])  for p in perf_per_cp[cp]]  )
                perf_aver = np.mean(perfarr , axis = 0)

                d = {'confmat_aver':confmat_aver, 'perf_aver':perf_aver }
                perf_per_cp[cp] += [  d  ]
        else:
            from utils_postprocess_HPC import perfFromConfmat
            class_pairs = list(itertools.combinations(uls, 2))
            perf_per_cp = {}
            for cp in class_pairs:
                c1,c2 = cp
                confmat_aver = confmat[[c1,c2],:][:,[c1,c2] ]
                ind = c1
                if c2 == class_ind_to_check_lenc:
                    ind = c2
                perf_aver = perfFromConfmat(confmat,ind)
                d = {'confmat_aver':confmat_aver, 'perf_aver':perf_aver }
                perf_per_cp[cp] = [ d  ]


        merge_EBMs = 0
        if merge_EBMs:
            ebm_merged = EBMUtils.merge_models(models=r0_ebm['clf_objs'])
            local_exp = ebm_merged.explain_local(X,y)
            scores_cur,true_labels,predicted_labels, featnames_out = \
                EBMlocExpl2scores(local_exp.data(), inc_interactions=False)
            scores = np.mean(scores_cur)
        else:
            scores_list = []
            for ebm_cur in r0_ebm['clf_objs']:
                # don't forget that we explain model, not data
                local_exp = ebm_cur.explain_local(X,y)
                #local_exp = ebm_cur.explain_local(X,y)
                expl_data = local_exp.data()
                if expl_data is None:
                    expl_data = ebm_cur.explain_local(X,y)._internal_obj['specific']
                #return ebm_cur,expl_data
                scores_cur,true_labels,predicted_labels, featnames_out = \
                    EBMlocExpl2scores(expl_data, inc_interactions=False)
                assert tuple(featnames_out) == tuple(featnames_ebm)
                scores_list += [scores_cur]
                assert scores_cur.shape[0] == X.shape[0]
            scores = np.mean(np.array(scores_list), axis=0)

        confmat_normalized = r0_ebm['confmat_aver']
        #print('scores cur shape ', scores_cur.shape)
        #print('scores ', scores_cur.shape)
        #import pdb; pdb.set_trace()
    else:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.20, random_state=0, shuffle=True)
        ebm = EBM(**ebm_creopts)
        if do_oversample:
            class_weights_cur = None
        else:
            class_weights_cur = compute_sample_weight('balanced',y_train)
        ebm.fit(X_train, y_train, sample_weight=class_weights_cur)
        local_exp = ebm.explain_local(X,y)
        if expl_data is None:
            expl_data = ebm_cur.explain_local(X,y)._internal_obj['specific']

        sens,spec, F1, confmat  = \
            getClfPredPower(ebm,X_test,y_test,class_ind_to_check_lenc, printLog=False)
        confmat_normalized = confmatNormalize(confmat) * 100
        print(f'confmat_normalized_true (pct) = {confmat_normalized}')

        inds_per_cp = classSubsetInds(y_test, class_ind_to_check_lenc)
        perf_per_cp = {}
        for cp,inds_cur_cp in inds_per_cp.items():
            perfs = []
            XX,yy = X_test[inds_cur_cp],y_test[inds_cur_cp]
            perf_cur = getClfPredPower(ebm,XX,yy,
                class_ind_to_check_lenc, printLog=False)
            perfs += [perf_cur]
            perf_per_cp[cp] = perfs

        for cp,inds_cur_cp in inds_per_cp.items():
            confmats = [p[-1] for p in perf_per_cp[cp] ]
            confmats = [ confmatNormalize(cm, 'true') for cm in np.array(confmats) ]
            confmat_aver =  np.mean( np.array(confmats), axis=0 )

            perfarr = np.vstack( [ (p[0],p[1],p[2])  for p in perf_per_cp[cp]]  )
            perf_aver = np.mean(perfarr , axis = 0)

            d = {'confmat_aver':confmat_aver, 'perf_aver':perf_aver }
            perf_per_cp[cp] += [  d  ]

    global_exp = ebm.explain_global()
    # extracting data from explainer
    scores = global_exp.data()['scores']
    names  = global_exp.data()['names']
    sis = np.argsort(scores)[::-1]
    featnames_srt = np.array(names)[sis]
    nfs = np.nan
    if len(sis) > 1:
        nfs = scores[ sis[1] ]
    print(f'EBM: Strongest feat is {featnames_srt[0]}'
            f'with score {scores[sis[0] ]}'
            f' ,next feat score is {nfs}')


    info_cur = {}
    info_cur['scores'] = scores
    info_cur['ebmobj'] = ebm
    info_cur['ebm_mergedobj'] = ebm_merged
    info_cur['explainer'] = global_exp
    info_cur['explainer_loc'] = local_exp
    info_cur['perf'] = sens,spec, F1, confmat
    info_cur['perf_dict'] = r0_ebm
    info_cur['confmat_normalized'] = confmat_normalized
    info_cur['feature_names']= names
    info_cur['perf_per_cp' ] = perf_per_cp
    info_cur['ebm_creopts'] = ebm_creopts
    info_cur['hyperopt_trials'] = trials
    res = info_cur

        #featsel_info.update(info_cur)
    return res


def computeEBM(X,y,EBM,ebm_creopts,revdict, class_ind_to_check_lenc, n_splits=5,
               EBM_compute_pairwise=0,EBM_CV=0, featnames_ebm=None,
               tune_params = False, params_space = None, max_evals = 20 ):
    import itertools
    from interpret.glassbox.ebm.utils import EBMUtils
    from sklearn.utils.class_weight import compute_sample_weight

    if EBM_compute_pairwise:
        uls = list(set(y))
        class_pairs = list(itertools.combinations(uls, 2))
        print(class_pairs)

        info_per_cp = {}
        #cpi = 0
        for cpi,(c1,c2) in enumerate(class_pairs):
            inds1 = np.where(y == c1)[0]
            inds2 = np.where(y == c2)[0]

            inds = np.append(inds1,inds2)
            indpair_names = revdict[c1],revdict[c2]

            print(f'Starting computing EBM for class pair {indpair_names}, in total {len(inds)}'+
                f'=({len(inds1)}+{len(inds2)}) data points')
            # filter classes
            X_cur_cp = X[inds]
            y_cur_cp = y[inds]

            info_cur_cp = _computeEBM(X,y,EBM,ebm_creopts,revdict, class_ind_to_check_lenc,
                n_splits=n_splits, EBM_CV=EBM_CV, featnames_ebm=featnames_ebm,
                                      tune_params = tune_params,
                                      params_space = params_space, max_evals = max_evals)

            #global_exp = info_cur_cp['explainer']
            ## extracting data from explainer
            #scores = global_exp.data()['scores']
            #names  = global_exp.data()['names']
            #sis = np.argsort(scores)[::-1]
            #featnames_srt = np.array(names)[sis]
            #print(f'EBM: Strongest feat is {featnames_srt[0]}')

            info_per_cp['data_point_inds'] = inds
            info_per_cp[indpair_names ] = info_cur_cp

        res = info_per_cp
    else:
        print('Starting computing EBM for all classes')

        # filter classes

        info_cur = _computeEBM(X,y,EBM,ebm_creopts,revdict, class_ind_to_check_lenc,
            n_splits=n_splits, EBM_CV=EBM_CV, featnames_ebm=featnames_ebm,
                        tune_params = tune_params, params_space = params_space, max_evals = max_evals)
        res = info_cur

        #featsel_info.update(info_cur)
    return res
