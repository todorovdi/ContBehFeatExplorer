import numpy as np
import udus_dataproc as mdp # main data proc
import re

import globvars as gv
import scipy.signal as sig
import matplotlib.pyplot as plt
import globvars as gv
import os

def getIntervalIntersection(a1,b1, a2, b2):
    assert b1 >= a1
    assert b2 >= a2
    if b1 < a2:
        return []
    if a1 > b2:
        return []
    r =  [ max(a1,a2), min(b1,b2) ]
    if r[1] - r[0] > 1e-2:
        return r
    else:
        return []

def findIntersectingAnns(ts,te,anns, ret_type='types'):
    # by default returns only types
    assert te >= ts
    if te-ts <= 1e-10:
        print('Warning, using empty interval {},{}'.format(ts,te) )
    d = []
    inds = []
    import mne
    newanns = mne.Annotations([],[],[])
    for ani,an in enumerate(anns):
        isect = getIntervalIntersection(ts,te,an['onset'],an['onset']+an['duration'])
        #print(isect)
        if len(isect) > 0:
            d.append( an['description'] )
            inds.append(ani)
            newanns.append( isect[0], isect[1] - isect[0], an['description'] )

    if ret_type == 'indices':
        return ani
    elif ret_type == 'anns':
        return newanns
    elif ret_type == 'types':
        return d

def unpackTimeIntervals(trem_times_byhand, mainSide = True, gen_subj_info=None,
                        skipNotLoadedRaws=True, printArtifacts=False):
    # unpack tremor intervals sent by Jan
    artif = {}
    tremIntervalJan = {}
    if gen_subj_info is None:
        gen_subj_info = gv.gen_subj_info
    for subjstr in trem_times_byhand:
        maintremside = gen_subj_info[subjstr]['tremor_side']
        if mainSide:
            side = maintremside
        else:
            side = getOppositeSideStr(maintremside)

        s = trem_times_byhand[subjstr]
        for medcond in s:
            ss = s[medcond]
            for task in ss:
                sss = ss[task]
                rawname = getRawname(subjstr,medcond,task )
                if skipNotLoadedRaws and rawname not in gv.raws:
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
                        if printArtifacts:
                            print(rawname,intType, s4p)
                        r = re.match( "artifact_(.+)", intType ).groups()
                        chn = r[0]
                        if chn == 'MEG':
                            chn += side
                        if rawname in artif:
                            gai = artif[rawname]
                            if chn not in gai:
                                artif[rawname][chn] = s4p
                            else:
                                invalids = []
                                for a,b in gai[chn]:
                                    for ii,ival in enumerate(s4p):
                                        aa,bb = ival
                                        if abs(a-aa) < 1e-6 and abs(b-bb) < 1e-6:
                                            invalids += [ii]
                                validinds = list( set(range(len(s4p) ) ) - set(invalids) )
                                if len(validinds) > 0:
                                    artif[rawname][chn] += [s4p[ii] for ii in validinds]
                        else:
                            artif[ rawname] ={ chn: s4p }

                    tremdat[intType] = s4p  # array of 2el lists

                tremIntervalJan[rawname] = { 'left':[], 'right':[] }
                tremIntervalJan[rawname][side] =  tremdat
                #print(subjstr,medcond,task,kt,len(s4), s4)
                #print(subjstr,medcond,task,kt,len(s4p), s4p)
    return tremIntervalJan, artif

def removeBadIntervals(intervals ):
    '''
    remove pre and post that intersect tremor
    '''
    #if len(timeIntervalPerRaw_processed) == 0:
    #    return []
    #intervals = timeIntervalPerRaw_processed[rawname]
    if len(intervals) == 0:
        return []

    ivalis = {}  # dict of indices of interval
    for itype in gv.gparams['intTypes']:
        ivit = []
        for i,interval in enumerate(intervals):
            t1,t2,it = interval
            if it == itype:
                ivit += [i]
        if len(ivit) > 0:
            ivalis[itype] = ivit

    tremIntInds = ivalis.get( 'middle_full', [] )
    if len(tremIntInds) > 0:
        for tii in tremIntInds:
            t1,t2,it = intervals[tii]
            for itype in ['pre','post']:
                tiis = ivalis.get(itype, None)
                if tiis is None:
                    continue

                indsToRemove = []
                for tii2 in tiis:
                    tt1,tt2,tit = intervals[tii2]
                    isect = getIntervalIntersection(t1,t2,tt1,tt2)
                    if len(isect) > 0:
                        indsToRemove += [tii2]
                        print(indsToRemove, isect, t1,t2,tt1,tt2)
                        intervals[ tii2] = (tt1,tt2, tit+'_isecTrem' )
                remainInds = list( set( tiis ) - set(indsToRemove) )
                ivalis[itype] = remainInds

    return intervals

def processJanIntervals( intervalData, intvlen, intvlenStats, loffset, roffset, time_end, mvtTypes = ['tremor'], leftoffset = 0.2  ):
    '''
     intervalData -- dict of dicts, each subdict has 'left' and 'right keys,
      each containing lists of triples (a,b,intervalType)
    '''

    # in sec. Don't want the interval to start with recording (there are artifacts)
    def intervalParse(a,b, itype = 'incPre' ):
        #if a <= loffset / 2:   # if a is close to the recoding beginning
        #    intType = 'incPost'
        #else:
        #    if itype = 'incPre':
        #        intType = 'incPre'
        #    else:
        #        intType = 'incPost'
        a = max(leftoffset, a)  # we don't want a to be precisely zero to avoid artifacts
        assert b > a

        if itype == 'middle':
            b = min(b, a + intvlen)
            return [( a,b, 'middle' ) ]
        if itype == 'middle_full':
            return [( a,b, 'middle_full' ) ]
        #if itype == 'unk_activity_full':
        #    return [( a,b, 'unk_activity_full' ) ]

        if a < loffset:  # if tremor starts at the recording beginning
            if itype == 'incPre':
                return []
        elif time_end - b < roffset:  #if tremor ends at the recording end
            if itype == 'incPost':
                return []

        # now if the tremor is really long we can extract 3 plots from the same tremor interval

        if itype == 'incPre':
            a1 = a - loffset
            b1 = min( a1 + intvlen , time_end )
            intType = 'incPre'
        elif itype == 'incPost':
            b1 = b + roffset
            a1 = max(leftoffset, b1 - intvlen )
            intType = 'incPost'
        elif itype == 'pre':
            a1 = max(leftoffset, a - intvlenStats)
            b1 = a
            intType = 'pre'
            #print('preee')
        elif itype == 'post':
            a1 = b
            b1 = min(b + intvlenStats, time_end)
            intType = 'post'
            #print('postt')
        elif itype == 'initseg':
            a1 = a
            b1 = min( a1 + intvlenStats , time_end )
            intType = 'initseg'
        elif itype == 'endseg':
            b1 = b
            a1 = max(leftoffset, b1 - intvlenStats )
            intType = 'endseg'
        else:
            raise ValueError('bad itype!')

        if a1 <= a and b1 >= b: #if old interval is inside the new one
            intType = 'incBoth'

        if b1 - a1 < 1.:  # if less than 1 second, discard
            return []
        return [( a1,b1, intType ) ]

    tipr = {}  # to plot all intervals, tipr[rawname] is a list of 2-el list with interval ends
    for k in intervalData:
        ti = intervalData[k]
        for side in ['left','right']:
            if side not in ti:
                tipr[k][side] = []
                continue

            tis = ti[side]
            if not isinstance(tis,dict):
                continue
            if k not in tipr:
                tipr[k] = {}
            tipr[k][side] = {}

            #import pdb; pdb.set_trace()

            tipr[k][side] =  []
            for mvtType in mvtTypes:
                if mvtType not in tis:
                    continue

                ti2 = tis[mvtType]
                if len(ti2) == 0:
                    continue

                #i = 0
                for p in ti2:  # list of interval ends
                    intType = ''
                    if len(p) != 2:
                        raise ValueError('{}, {}, {}: {} error in the interval specification'.
                                format(k,side,mvtType,p) )
                    a,b = p
                    if a >= time_end:
                        continue

                    #if k == 'S05_off_move':
                    #    print(a,b)

                    if mvtType == 'tremor':
                        intsToAdd = []
                        #if b - a >= intvlen * 3:
                        # these are for plotting
                        intsToAdd += intervalParse(a,b, 'incPre')
                        intsToAdd += intervalParse(a,b, 'middle')
                        intsToAdd += intervalParse(a,b, 'incPost')
                        # these are of stats
                        intsToAdd += intervalParse(a,b, 'middle_full')
                        intsToAdd += intervalParse(a,b, 'pre')
                        intsToAdd += intervalParse(a,b, 'post')
                        #
                        intsToAdd += intervalParse(a,b, 'endseg')
                        intsToAdd += intervalParse(a,b, 'initseg')


                    elif mvtType == 'unk_activity':
                        intsToAdd = [ (a,b, 'unk_activity_full') ]

                    elif mvtType == 'no_tremor':
                        a = max(leftoffset, a)
                        assert b > a
                        #b = min(b, a+ intvlen)
                        intType = 'no_tremor'
                        intsToAdd = [ (a,b,intType) ]

                    #if k == 'S05_off_move':
                    #    import pdb; pdb.set_trace()

                    tipr[k][side]  += intsToAdd
            if len(tipr[k][side] ) == 0:
                #tipr[k]  = [ (0,time_end,'unk') ]
                tipr[k][side]  = [ (0,time_end,'entire') ]

            #if k == 'S05_off_move':
            #    print('fdsfsd ',tipr[k])

    return tipr

def getMEGsrc_chname_nice(chn):
    name = chn
    if chn.find('HirschPt2011') >= 0:
        r = re.match( ".*_([0-9]+)", chn ).groups()
        num = int( r[0] )
        nlabel = num // 2
        nside =  num % 2
        # even indices -- left [no! even correspond to right]
        side = ['right', 'left'] [nside]
        label = gv.gparams['coord_labels'][nlabel]

        name = '{}_{}'.format( label,side)
    return name

def getMEGsrc_contralatSide(chn):
    # return no the brain side
    nicen = getMEGsrc_chname_nice(chn)
    for side in ['right', 'left']:
        if nicen.find(side) >= 0:
            return getOppositeSideStr(side)

def chname2modality(chn):
    modalities = ['LFP', 'EMG', 'EOG', 'MEGsrc']
    for modality in modalities:
        if chn.find(modality) >= 0:
            return modality
    raise ValueError('Bad chname, no modality understood')

def getChannelSide(rawname,chname):
    # MEG meaning of sides
    sind_str,medcond,task = getParamsFromRawname(rawname)
    subjinfo = gv.gen_subj_info[sind_str]
    sides = subjinfo['LFPnames_perSide'].keys()
    for side in sides:
        if chname in subjinfo['LFPnames_perSide'][side]:
            return side
        if chname in subjinfo['EMGnames_perSide'][side]:
            return side
        if chname in subjinfo['MEGsrcnames_perSide_all'][side]:
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
    times = None
    for i,chn in enumerate(chns):
        #m = re.match('MEGsrc_(.+)_(.+)', chn)
        #roistr = m.groups()[0]
        #srci = int(m.groups()[1])
        roistr,srci = parseMEGsrcChname(chn)

        sind_str,medcond,task = getParamsFromRawname(rawname)
        srcname = 'srcd_{}_{}_{}_{}'.format(sind_str,medcond,task,roistr)
        f = gv.srcs[srcname]
        times = f['source_data']['time'][ts:te,0]

        src = f['source_data']
        ref = src['avg']['mom'] [0,srci]
        # maybe I can make more efficient data extraction of multiple channels at once
        # later if needed, if I bundle all channels from the same source file
        if f[ref].size > 10:
            srcval = f[ref ][ts:te,0]   # 1D array with reconstructed source activity
        else:
            srcval = np.ones( len(times) ) * -1e-20
            print('{}  {} does not contain stuff'.format(srcname, srci) )
        srcvals += [srcval]
    r = np.vstack(srcvals)

    if rettimes:
        return r, times
    else:
        return r

def isMEGsrcchanBad(rawname, chn):
    roistr,srci = parseMEGsrcChname(chn)
    sind_str,medcond,task = getParamsFromRawname(rawname)

    srcname = 'srcd_{}_{}_{}_{}'.format(sind_str,medcond,task,roistr)
    if srcname not in gv.srcs:
        return -1
    f = gv.srcs[srcname]
    src = f['source_data']
    ref = src['avg']['mom'] [0,srci]
    if  (f[ref].size < 10):
        return 1
    else:
        return 0

def MEGsrcChind2data(rawname,chi, MEGsrc_roi):
    indshift = 0
    for roistr in sorted(MEGsrc_roi):  # very important to use sorted!
        sind_str,medcond,task = getParamsFromRawname(rawname)
        srcname = 'srcd_{}_{}_{}_{}'.format(sind_str,medcond,task,roistr)
        f = gv.srcs[srcname]
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

def filterFavChnames(chnames,subj):
    '''
    ! so far works only for LFP chnames
    input list of strings
    subj 'S??'
    '''
    if gv.gparams['plot_LFP_onlyFavoriteChannels']:
        # get intersection
        favchn = gv.gen_subj_info[subj]['favoriteLFPch']
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

def getParamsFromRawname(rawname):
    r = re.match( "(S\d{2})_(.+)_(.+)*", rawname ).groups()
    subjstr = r[0]
    medcond = r[1]
    task = r[2]

    return subjstr,medcond,task

def getParamsFromSrcname(srcname):
    r = re.match( "srcd_(S\d{2})_(.+)_(.+)_(.+)", srcname ).groups()
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


def getData(rawname,chns,ts=None, te=None, raws_=None):
    '''
    ts,te -- timebins
    '''
    assert isinstance(chns,list)
    if raws_ is None:
        raws = gv.raws
    else:
        raws = raws_
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

def sortChans(subj, modality, fbname, replace=False, numKeep = 3):
    cht = gv.gen_subj_info[ subj ]['chantuples']
    fbs,fbe = gv.freqBands[fbname]

    subj_info = gv.subjs_analyzed[subj]
    medconds = subj_info['medconds']
    tasks = subj_info['tasks']

    for side in cht:
        chtcur = cht[side]['nametuples']
        if modality not in chtcur :
            raise ValueError('No such modality!')
        chnames = chtcur[modality]
        bpt = np.zeros( len(chnames) )
        for chi,chn in enumerate(chnames):
            bandpow_total = 0  # across conditions
            wasBadChan = 0
            for medcond in medconds:
                bandpow_cur = 0
                for task in tasks:
                    rawname = '{}_{}_{}'.format(subj,medcond,task)
                    if rawname not in gv.raws:
                        continue

                    if modality.find('MEGsrc') >= 0:
                        bad = isMEGsrcchanBad(rawname,chn)
                    else:
                        bad = 0

                    if bad == 1:
                        bandpow_cur = 0
                        wasBadChan = 1
                    elif bad == -1:
                        bandpow_cur = 0
                        #print('Found bad channel ',chn)
                    else:
                        freq, bins, Sxx = gv.specgrams[rawname][chn]
                        sspec = getSubspec(freq,bins,Sxx, fbs,fbe)
                        if sspec is None:
                            continue
                        else:
                            f,b,bandspec = sspec
                            #bandpow = np.sum( bandspec, axis=0 )
                            bandpow_cur = np.sum(bandspec) * ( b[1] - b[0] )  * ( f[1] - f[0] )
                            #print('{} chn {} totalpow {}'.format(rawname,chn,bandpow_cur) )
                bandpow_total += bandpow_cur
            if wasBadChan:
                bandpow_total = 0
            bpt[chi] = bandpow_total
        #sortinds = np.argsort(bpt)
        #srt = chtcur[modality][sortinds][::-1]   # first the largest one
        ss = '{}_{}_sorted'.format(modality,fbname )

        from operator import itemgetter
        srt_ = sorted( zip(bpt,chnames), key = itemgetter(0) )[::-1]
        srt = [ chn for bp,chn in srt_ ]

        chtcur[ ss  ] = srt
        if replace:
            chtcur[modality] = srt[:numKeep]
            print(subj,  srt_[:numKeep] )

def getTuples(sind_str):
    cht = gv.gen_subj_info[sind_str]['chantuples']
    sides = []
    indtuples = []
    nametuples = []
    for sidei,side in enumerate(cht.keys() ):
        sides += [side]
        indtuples +=  [ cht[side]['indtuples'] ]
        nametuples += [ cht[side]['nametuples'] ]

    #chns = chnames_tuples[pair_ind]['EMG']
    return sides, indtuples, nametuples

def getOppositeSideStr(side ):
    sides_full = ['left', 'right' ]
    sides_onelet = ['L', 'R' ]
    sides_ind = [0, 1 ]

    sds = [sides_full, sides_onelet, sides_ind ]

    if side in sides_full:
        arr = sides_full
    elif side in sides_onelet:
        arr = sides_onelet
    elif side in sides_ind:
        arr = sides_ind
    else:
        raise ValueError('wrong side',side)

    return arr[ 1 - arr.index(side)]

def getBinsInside(bins, b1, b2, retBool = True, rawname = None):
    '''
    rawname not None should be used only when time_start == 0 and there is no 'holes' in bins array
    b1,b2 are times, not bin indices
    '''
    if rawname is None or bins[0] >= 1/gv.gparams['sampleFreq']:
        binsbool  = np.logical_and(bins >= b1 , bins <= b2)
    else:
        ts,te = gv.raws[rawname].time_as_index([b1,b2])
        if retBool:
            binsbool = np.zeros( len(bins) , dtype=bool)
            binsbool[ ts: te] = 1
            return binsbool
        else:
            return np.arange(ts,te,dtype=int)

    if retBool:
        return binsbool
    else:
        return np.where(binsbool)

def filterArtifacts(k, chn, bins, retBool = True, rawname = None):
    validbins_bool = np.ones( len(bins) , dtype=bool)
    if gv.artifact_intervals is not None and (k in gv.artifact_intervals):
        if chn.find('MEGsrc') >= 0:
            side = getMEGsrc_contralatSide(chn)
            chneff = 'MEG' + side
        else:
            chneff = chn
        cond = ( chneff in gv.artifact_intervals[k])

        if cond:
            artifact_intervals = gv.artifact_intervals[k][chneff]
            if rawname is None:
                for a,b in artifact_intervals:
                    #if chneff.find('MEG') >= 0:
                    #    print(' chn {} artifact {} : {},{}'.format(chn, chneff, a,b ) )
                    validbins_bool = np.logical_and( validbins_bool , np.logical_or(bins < a, bins > b)  )
            else:
                indsBad = []
                for a,b in artifact_intervals:
                    ts,te = gv.raws[rawname].time_as_index([a,b])
                    #indsBad += np.arange(ts,te,dtype=int).tolist()
                    validbins_bool[ts:te] = 0


        #if chneff.find('MEG') >= 0:
        #    print('filterArtifacts {} ({}), got {} of total {} bins'.format( chn, chneff, np.sum(validbins_bool) , len(bins) ) )

    if retBool:
        return validbins_bool
    else:
        return np.where( validbins_bool )[0]

def filterRareOutliers(dat, nbins=200, thr=0.01, takebin = 20, retBool = False):
    '''
    for non-negative data, 1D only [so flatten explicitly before]
    thr is amount of probability [=proportion of time] we throw away
    returns indices of good bins
    '''
    assert dat.ndim == 1
    mn,mx = getSpecEffMax(dat, nbins, thr, takebin)
    binBool = dat <= mx
    if retBool:
        return binBool
    else:
        bininds = np.where( binBool )[0]
        return bininds

#minNbins = 3
#minNbins * sampleFreq

def filterRareOutliers_specgram(Sxx, nbins=200, thr=0.01, takebin = 20, retBool = False):
    '''
    for non-negative data
    removesOutliers per frequency
    first dim is freqs
    '''
    numfreqs = Sxx.shape[0]
    binBool = [0] * numfreqs
    ntimebins =  Sxx.shape[1]
    for freqi in range(numfreqs):
        r = filterRareOutliers( Sxx[freqi], nbins, thr, takebin, retBool = True )
        binBool[freqi] = r

    binBoolRes = np.ones(ntimebins )
    for freqi in range(numfreqs):
        r = binBool[freqi]
        binBoolRes = np.logical_and(r  , binBoolRes )
        #ratio = np.sum(r) / ntimebins
        #ratio2 = np.sum(binBoolRes) / ntimebins
        #print(ratio, ratio2)

    indsRes = np.where(binBoolRes)[0]
    ratio = np.sum(binBoolRes) / ntimebins
    assert ratio  > 0.95, str(ratio)  # check that after intersecting we still have a lot of data

    if retBool:
        return binBoolRes
    else:
        return indsRes


def calcNoutMMM_specgram(Sxx, nbins=200, thr=0.01, takebin = 20, retBool = False):
    '''
    for non-negative data
    computes effective max and min per freq
    first dim is freqs
    '''
    numfreqs = Sxx.shape[0]
    binBool = [0] * numfreqs
    ntimebins =  Sxx.shape[1]
    mn = np.zeros(numfreqs)
    mx = np.zeros(numfreqs)
    me = np.zeros(numfreqs)
    for freqi in range(numfreqs):
        dat = Sxx[freqi,:]
        ret = getSpecEffMax(dat, nbins, thr, takebin)
        if ret is not None:
            mn_,mx_ = ret
            #r = filterRareOutliers( Sxx[freqi, :], nbins, thr, takebin, retBool = True )
            r =  dat <= mx_
            mn[freqi] = mn_
            mx[freqi] = mx_
            me[freqi] = np.mean( dat[r] )

    return mn,mx,me

#10-20, 20-30
def prepFreqs(min_freq = 3, max_freq = 400, min_freqres=2, gamma_bound = 30, HFO_bound = 90, frmult_scale=1 ):
    # gamma_bound is just bound when we change freq res to a coarser one

    pi5 = 2*np.pi / 5    # 5 is used in MNE as multiplier of std of gaussian used for Wavelet

    assert max_freq > gamma_bound, 'we want {} > {} '.format(max_freq, gamma_bound)
    assert min_freq < gamma_bound, 'we want {} < {} '.format(min_freq, gamma_bound)
    fbands = [ [min_freq,gamma_bound], [gamma_bound,HFO_bound], [HFO_bound,max_freq]  ]
    freqres = np.array( [ 1, 2, 4  ] ) * min_freqres
    frmults = frmult_scale * np.array( [pi5, pi5/2, pi5/4] )
    if max_freq <= HFO_bound:
        fbands = fbands[:2]
        freqres = freqres[:2]
        frmults = frmults[:2]

    freqs = []
    n_cycles  = []
    prev_fe = -1
    for fb,freq_step,fm in zip(fbands,freqres,frmults):
        if prev_fe < 0:
            fbstart = fb[0]
        else:
            fbstart = prev_fe + freq_step/2
        freqs_cur = np.arange(fbstart, fb[1], freq_step)
        freqs += freqs_cur.tolist();
        n_cycles += (freqs_cur * fm).tolist()
        prev_fe = fb[1]


    freqs = np.array(freqs)
    n_cycles = np.array(n_cycles)

    return freqs, n_cycles

def tfr(dat, sfreq, freqs, n_cycles, decim=1, n_jobs = None):
    import multiprocessing as mpr
    import mne
    if n_jobs is None:
        n_jobs = max(1, mpr.cpu_count() - 2 )
    elif n_jobs == -1:
        n_jobs = mpr.cpu_count()

    #dat has shape n_chans x ntimes // decim
    #returns n_chans x freqs x dat.shape[-1] // decim
    assert dat.ndim == 2
    assert len(freqs) == len(n_cycles)
    if abs(sfreq - int(sfreq) ) > 1e-5:
        raise ValueError('Integer sfreq is required')
    sfreq = int(sfreq)

    dat_ = dat[None,:]
    tfrres = mne.time_frequency.tfr_array_morlet(dat_, sfreq, freqs, n_cycles,
                                                 n_jobs=n_jobs, decim =decim)
    tfrres = tfrres[0]
    return tfrres




############################# Tremor-related

def getIntervals(bininds,width=100,thr=0.1, percentthr=0.8,inc=5, minlen=50,
        extFactorL = 0.25, extFactorR  = 0.25, endbin = None, cvl=None,
                 percent_check_window_width=256, min_dist_between = 100,
                 include_short_spikes=0, printLog = 0, minlen_ext=None):
    '''
    bininds either indices of bins where interesting happens OR the mask
    width -- number of bins for the averaging filter
    tremini -- indices of timebins, where tremor was detected
    thr -- thershold for convolution to be larger then, for L\infty-normalized data
        (works together with width arg). Max sensitive thr is 1/width
    output -- convolution, intervals (pairs of timebin indices)
    inc -- how much we extend the interval each time (larger means larger intervals, but worse boundaries)
    minlen -- minimum number of bins required to make the interval be included
    minlen_ext -- same as minlin, but after extFactors were applied
    percentthr -- min ratio of thr crossings within the window to continue extending the interval
    extFactor[R,L] -- we'll extend found intervals by width * extFactor (negative allowed)
    endbin -- max timebin
    '''

    if cvl is None:
        if np.max(bininds) == 1:
            if endbin is not None:
                assert len(bininds) == endbin
            raster = bininds
        else:
            raster = None

        if endbin is None:
            mt = np.max (bininds ) + 1
        else:
            mt = endbin

        if raster is None:
            raster = np.zeros( mt, dtype=np.int )
            raster[bininds] = 1
        #width = 100
        avflt = np.ones(width) #box filter
        #avflt = sig.gaussian(width, width/4)
        avflt /= np.sum(avflt)   # normalize
        cvl = np.convolve(raster,avflt,mode='same')
    else:
        assert cvl.ndim == 1

    if minlen_ext is None:
        minlen_ext = minlen
    #cvlskip = cvl[::skip]
    cvlskip = cvl
    thrcross = np.where( cvlskip >= thr )[0]
    belowthr = np.where( cvlskip < thr )[0]
    shiftL = int(width * extFactorL )
    shiftR = int(width * extFactorR )

    assert thr > 0
    #print('cross len ',len(thrcross), thrcross )
    #print('below len ',len(belowthr), belowthr )

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

        if printLog:
            print('searchres={}, rightEnd={}'.format(searchres, rightEnd) )
        subcvl = cvlskip[leftEnd:rightEnd]
        rightEnd_hist = []
        success = True
        while rightEnd < len(cvlskip) + inc + 1: # try to enlarge until possible
            rightEnd_hist += [rightEnd]

            if len(rightEnd_hist) > 10 and np.std(rightEnd_hist[-5:] ) < 1e-10:
                assert False
            if printLog:
                print('Extending: rightEnd={}, {}'.format(rightEnd, rightEnd_hist ) )
            if len(subcvl) <= percent_check_window_width:
                tmp = subcvl
            else:
                tmp = subcvl[-percent_check_window_width:]
            val = np.sum(tmp > thr) / len(tmp)
            if (rightEnd + inc) >= len(cvlskip) and (val > percentthr):
                rightEnd = len(cvlskip)
                subcvl = cvlskip[leftEnd:rightEnd]
                break

            if (rightEnd + inc) < len(cvlskip) and (val > percentthr):
                rightEnd += inc
                subcvl = cvlskip[leftEnd:rightEnd]
            else:
                if not (include_short_spikes and cvlskip[rightEnd] > thr ):
                    break
                if (len(rightEnd_hist) > 4) and (np.std(rightEnd_hist[-2:] ) < 1e-10):
                    success = False # the found segment has too low percentage
                    #rightEnd += 1
                    break
                #, (rightEnd_hist, subcvl)
                if printLog:
                    print('Ext: val={}'.format(val), (len(rightEnd_hist) , np.std(rightEnd_hist[-5:] )   ) )

        newp = (  max(0, leftEnd-shiftL), max(0, rightEnd+shiftR) )
        if rightEnd-leftEnd >= minlen and (newp[1]-newp[0]) >= minlen_ext and success:
            newp = (  max(0, leftEnd-shiftL), max(0, rightEnd+shiftR) ) # extend the interval on both sides
            if len(pairs):  #maybe we'd prefer to glue together two pairs
                prev_st,prev_end = pairs[-1]
                if newp[0] - prev_end > min_dist_between:
                    pairs += [newp ]
                else:
                    pairs[-1] = ( prev_st, newp[1] )
            else:
                pairs += [newp ]

        assert leftEnd <= rightEnd,  (leftEnd,rightEnd)

        searchres = np.where(thrcross > rightEnd)[0]  # _ind_ of next ind of thr crossing after end of current interval
        if len(searchres) == 0:
            break
        else:
            gi = searchres[0]
        leftEnd = thrcross[gi]  # ind of thr crossing after end of current interval

    return cvlskip,pairs

def findTremor(k,thrSpec = None, thrInt=0.13, width=40, percentthr=0.8, inc=1, minlen=50,
        extFactorL=0.25, extFactorR=0.25, tremor_band=[3,10], tremorDetectUseCustomThr=1,
              tremrDet_clusterMultiEMG = 1 ):
    '''
    k is raw name
    output -- per raw, per side,  couples of start, end
    percentthr -- minimum percantage of time the events should happen within the interval for it to be considered continous
    '''
    sind_str,medcond,task = getParamsFromRawname(k)

    #gen_subj_info[sind_str]['tremor_side']
    #gen_subj_info[sind_str]['tremfreq']

    chnames = gv.raws[k].info['ch_names']
    orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)

    tremorBandStart, tremorBandEnd = tremor_band
    specgramoverlap = 0.5

    tremorIntervals = {}
    cvlPerChan = {}
    for pair_ind in range(len(orderEMG)):
        side = orderEMG[pair_ind]
        chns = chnames_tuples[pair_ind]['EMG']
        sideinfo = {}
        for chn in chns:
            if gv.gparams['tremDet_useTremorBand']:
                freq, bins, Sxx = gv.specgrams[k][chn]
                freq,bins,bandspec = getSubspec(freq,bins,Sxx, tremorBandStart, tremorBandEnd,
                        rawname=k)
                #freqinds = np.where(
                #    np.logical_and(freq >= tremorBandStart,freq <= tremorBandEnd))
                #bandspec = Sxx[freqinds,:]
                if thrSpec is None:
                    if tremorDetectUseCustomThr and 'tremorDetect_customThr' in gv.gen_subj_info[sind_str]:
                        thr = gv.gen_subj_info[sind_str]['tremorDetect_customThr'][medcond][side][chn]
                        #import pdb; pdb.set_trace()
                        logcond = np.sum( bandspec, axis=0 ) > thr # sum over freq
                        pp = np.where( logcond )
                        trembini = pp[0]
                    else:
                        if tremrDet_clusterMultiEMG:
                            # thresholds per channel
                            thr = gv.glob_stats[sind_str][medcond]['thrPerCh_trem_allEMG'][chn]  # computed using k-means
                            logcond = np.sum( bandspec, axis=0 ) > thr # sum over freq
                            pp = np.where( logcond )
                            #import pdb; pdb.set_trace()
                            trembini = pp[0]
                        else:
                            thrPerFreq = gv.glob_stats[sind_str][medcond][chn]['thrPerFreq_trem']  # computed using k-means
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
                    widthBins = int( width * (1-specgramoverlap) / ( bins[1]-bins[0] ) )
                    cvlskip,pairs = getIntervals(trembini, widthBins, thrInt,
                            percentthr, inc,  minlen, extFactorL, extFactorR, endbin=len(bins) )

                    pairs2 = [ (bins[a],bins[b]) for a,b in pairs    ]
    #                 for pa in pairs:
    #                     a,b = pa
    #                     a = bins[a]
    #                     b = bins[b]
                    cvlPerChan[chn] = cvlskip
                else:
                    pairs2 = []
            else:
                highpassFltorder = 10
                highpassFreq =  20.
                sos = sig.butter(highpassFltorder, highpassFreq,
                        'highpass', fs=gv.gparams['sampleFreq'], output='sos')
                ts,te = gv.raws[k].time_as_index([gv.gparams['tremDet_timeStart'],
                    gv.gparams['tremDet_timeEnd']])
                chdata,chtimes = getData(k,[chn],ts,te)
                chdata = chdata[0]
                filtered = sig.sosfilt(sos,chdata )
                filtered /= max(np.max(filtered), np.min(filtered), key = abs ) # normalize
                rect = np.maximum(0, filtered)

                widthBins = int( width / ( chtimes[1]-chtimes[0] ) )
                avflt = np.ones(widthBins) #box filter
                #avflt = sig.gaussian(width, width/4)
                avflt /= np.sum(avflt)   # normalize
                cvl = np.convolve(rect,avflt,mode='same')

                #print('{} cvl max {}, widthBins {}'.format( chn,  np.max(cvl), widthBins) )

                cvlskip,pairs = getIntervals(None, widthBins, thrInt,
                        percentthr, inc,  minlen, extFactorL, extFactorR, endbin=None, cvl=cvl )
                #from IPython.core.debugger import Tracer; debug_here = Tracer(); debug_here()

                pairs2 = [ (chtimes[a],chtimes[b]) for a,b in pairs    ]
                cvlPerChan[chn] = cvlskip

            #else:
            #    print pp

            sideinfo[chn] = pairs2

        tremorIntervals[orderEMG[pair_ind]] = sideinfo

    return tremorIntervals, cvlPerChan
    #print(Sxx.shape, freqinds)
    #return yy
    #assert chdata.ndim == 1


def findAllTremor(thrSpec = None, thr=0.13, width=40, percentthr=0.8, inc=1, minlen=50,
        extFactorL=0.25, extFactorR=0.25):
    tremPerRaw = {}
    cvls = {}
    for k in gv.specgrams:
        tremIntervals,cvlPerChan = findTremor(k, thrSpec, thr, width, percentthr, inc)
        tremPerRaw[k] = tremIntervals

        cvls[k] = cvlPerChan

    return tremPerRaw, cvls

def anns2intervals(anns, tuple_len=2):
    import mne
    assert isinstance(anns,mne.Annotations)
    ivals = []
    for an in anns:
        tpl_ = [an['onset'], an['onset']+an['duration']]
        if tuple_len == 3:
            tpl_.append(an['description'] )
        tpl = tuple(tpl_)
        ivals += [tpl  ]
    return ivals

def mergeTremorIntervalsRawSide(intervals, mode='intersect'):
    '''
    mode -- 'intersect' or 'union' -- how to deal with intervals coming from differnt muscles
    intervals -- dict (rawname) of dicts (side) of dicts (chns) of lists of tuples
    '''
    import mne
    ann_mode = False
    if isinstance(intervals, mne.Annotations):
        intervals = anns2intervals(intervals)
        ann_mode = True
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


            if mode == 'union':
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

def mergeTremorIntervals(pairs1, pairs2, mode='intersect'):
    '''
    Mergint intervals _of the same type_
    mode -- 'intersect' or 'union' -- how to deal with intervals coming from differnt muscles
    '''
    import mne
    ann_mode = False
    if isinstance(pairs1, mne.Annotations) or isinstance(pairs2, mne.Annotations):
        assert  isinstance(pairs1, mne.Annotations) and isinstance(pairs2, mne.Annotations)
        pairs1 = anns2intervals(pairs1, tuple_len=3)
        pairs2 = anns2intervals(pairs2, tuple_len=3)
        ann_mode = True

    #intcur = intervals
    #assert len(intcur) == 2
    #chns = list( intcur.keys() )
    #pairs1 = intcur[chns[0] ]
    #pairs2 = intcur[chns[1] ]

    mask2 = np.ones(len(pairs2) )

    # currently tremor that appear only in one MEG, gets ignored (otherwise I'd have to add non-merged to the list as well)
    resp = []
    #for i1, (a1,b1) in enumerate(pairs):
    #    for i2,(a2,b2) in enumerate(pairs):
    for i1, p1 in enumerate(pairs1):
        (a1,b1) = p1[0],p1[1]
        mergewas = 0
        for i2,p2 in enumerate(pairs2):
            (a2,b2) = p2[0],p2[1]
            #if i1 == i2 or mask[i2] == 0 or mask[i1] == 0:
            #    continue
            if mask2[i2] == 0:
                continue

            if len(p1) == 3 or len(p2) == 3:
                #if p1[2] != p2[2]:
                #    continue
                assert p1[2] == p2[2]

            # if one of the end is inside the other interval
            if (b1 <=  b2 and b1 >= a2) or ( b2 <= b1 and b2 >= a1 ) :
                if mode == 'intersect':
                    newp = (max(a1,a2), min(b1,b2) )
                elif mode == 'union':
                    newp = (min(a1,a2), max(b1,b2) )
                resp += [ newp ]
                #mask[i1] = 0
                #mask[i2] = 0  # we mark those who participated in merging
                mask2[i2] = 0
                mergewas = 1
                break

        #resp += [ p for i,p in enumerate(pairs) if mask[i] == 1 ]  # add non-merged

    result = []

    if mode == 'union':
        result = mergeIntervalsWithinList(resp,pairs1,pairs2)
    elif mode == 'intersect':
        result = resp
    else:
        raise ValueError('wrong mode')

    return result

def mergeIntervalsWithinList(pairs,pairs1=None,pairs2=None, printLog=False):
    # now merging intersecting things that could have arised from joining
    if pairs1 is None:
        pairs1 = pairs
    if pairs2 is None:
        pairs2 = pairs

    import mne
    ann_mode = False
    if isinstance(pairs1, mne.Annotations) or isinstance(pairs2, mne.Annotations):
        assert  isinstance(pairs1, mne.Annotations) and isinstance(pairs2, mne.Annotations)
        pairs1 = anns2intervals(pairs1, tuple_len=3)
        pairs2 = anns2intervals(pairs2, tuple_len=3)
        ann_mode = True
    if isinstance(pairs, mne.Annotations):
        pairs = anns2intervals(pairs, tuple_len=3)

    n_merges = 0

    mask = np.ones(len(pairs) )
    resp2 = []
    for i1, p1 in enumerate(pairs1):
        (a1,b1) = p1[0],p1[1]
        for i2,p2 in enumerate(pairs2):
            (a2,b2) = p2[0],p2[1]
            if i1 == i2 or mask[i2] == 0 or mask[i1] == 0:
                continue
            if len(p1) == 3 and p1[2] != p2[2]:      # if descriptions are different
                continue
            if (b1 <=  b2 and b1 >= a2) or ( b2 <= b1 and b2 >= a1 ) :
                tpl_ = [min(a1,a2), max(b1,b2) ]
                if ann_mode:
                    tpl_.append(p1[2] )
                resp2 +=  [ tuple(tpl_)  ]
                mask[i1] = 0
                mask[i2] = 0  # we mark those who participated in merging
                if printLog:
                    print('merged {} and {}'.format(p1,p2) )
                n_merges += 1
                break
    resp2 += [ p for i,p in enumerate(pairs) if mask[i] == 1 ]  # add non-merged

    res  = resp2
    if ann_mode:
        res = intervals2anns(resp2)
    return res

############################# Spec helper funcions
def getBandpow(k,chn,fbname,time_start,time_end, mean_corr = False, spdat=None):
    '''
    can return not all bins, because filters artifacts!
    '''
    if spdat is None:
        specgramsComputed  = gv.specgrams[k]
        freqs, bins, Sxx = specgramsComputed[chn]
    else:
        freqs, bins, Sxx = spdat
    fbs,fbe = gv.freqBands[fbname]
    r = getSubspec(freqs,bins,Sxx, fbs,fbe,
            time_start,time_end,rawname=k)


    if r is not None:
        freqs_b, bins_b, Sxx_b = r
        if Sxx_b.size == 0:
            return None

        if isinstance(Sxx_b[0,0], complex):
            Sxx_b = np.abs(Sxx_b)

        #gv.gparams['artifacts'][k][chn]
        #validbins_bool = np.ones( len(bins_b) )
        #if gv.artifact_intervals is not None and k in gv.artifact_intervals and chn in gv.artifact_intervals[k]:
        #    artifact_intervals = gv.artifact_intervals[k][chn]
        #    validbins_bool_ = [True] * len(bins_b)
        #    for a,b in artifact_intervals:
        #        validbins_bool_ = np.logical_and( validbins , np.logical_or(bins_b < a, bins_b > b)  )

        #    validbins_bool = validbins_bool_
        #    validbininds = np.where( validbins_bool )[0]
        #    bins_b = bins_b[validbins_bool]
        #    Sxx_b = Sxx_b[validbininds]

        spec_thrBadScaleo = gv.gparams['spec_thrBadScaleo']
        validbins_bool0 = getBinsInside(bins_b,
                max(time_start, gv.gparams['spec_time_start'] + spec_thrBadScaleo),
                min(gv.gparams['spec_time_end']- spec_thrBadScaleo, time_end),
                retBool = True)

        validbins_bool1 = filterArtifacts(k,chn,bins_b)
        validbins_bool = np.logical_and( validbins_bool0, validbins_bool1)
        #if chn.find('LFP') >= 0:
        #    print('getBandpow: {}_ invalid bins {} of {}'.format( chn, len(bins_b) - np.sum(validbins_bool), len(bins_b) ) )
        bins_b = bins_b[validbins_bool]
        Sxx_b = Sxx_b[:,validbins_bool]

        if mean_corr:
            #time_start_mecomp = max(time_start, b[0] + thrBadScaleo)
            #time_end_mecomp = min(time_end, b[-1] - thrBadScaleo)

            #goodinds = filterRareOutliers_specgram(Sxx_b)
            #goodinds = np.where( goodinds )[0]
            #me = np.mean(Sxx_b[:,goodinds] , axis=1)
            sind_str, medcond, task = getParamsFromRawname(k)
            st = gv.glob_stats[sind_str][medcond][task][chn]
            me = st.get( 'mean_spec_nout_full' , None )
            if me is None:
                return None
            me = me[ np.logical_and( freqs >= fbs, freqs <= fbe ) ]
            Sxx_b =  ( Sxx_b -  me[:,None] ) / me[:,None]    # divide by mean


        if gv.gparams['specgram_scaling'] == 'psd':
            freqres = freqs[1]-freqs[0]
        else:
            freqres = 1.

        # divide by number of freqs to later compare between different bands, which have dif width
        bandpower = np.sum(Sxx_b,axis=0) * freqres / Sxx_b.shape[0]

        return bins_b, bandpower
    else:
        return None


def getSubspec(freqs,bins,Sxx,freq_min,freq_max,time_start=None,time_end=None,rawname=None):
    #bininds = np.where( np.logical_and( bins >= time_start , bins < tetmp/gv.gparams[sampleFreq]) )[0]
    '''
    rawname not None should be used only when time_start == 0 and there is no 'holes' in bins array
    '''
    if time_start is not None or time_end is not None:
        if time_start is None:
            time_start = np.min(bins)
        if time_end is None:
            time_end = np.max(bins)

        if rawname is None or bins[0] >= 1/gv.gparams['sampleFreq']:
            bininds = np.where( np.logical_and( bins >= time_start , bins <= time_end) )[0]
        else:
            ts,te = gv.raws[rawname].time_as_index( [time_start,time_end] )
            bininds = np.arange(ts,te,dtype=int)
        bins = bins[bininds]
        Sxx = Sxx[:,bininds]

    freqinds = np.where( np.logical_and(freqs >= freq_min,freqs <= freq_max) )[0]
    if len(freqinds) == 0 and gv.spec_specgramtype in ['scaleogram', 'mne.wavelet']:
        print('WARNING:  getSubspec did not find any spegram info for freq band {}-{}'.format( freq_min, freq_max) )
        #import pdb; pdb.set_trace()
        return None

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

##############################  general data sci

def getSpecEffMax(dat, nbins=200, thr=0.01, takebin = 20):
    '''for non-negative data'''

    short = False
    dat = dat.flatten()
    if len(dat) / 10 < nbins:
        short = True
        nbins = int(len(dat) / 10)
    maxnbins = 1e4
    checkval = 1
    bins = None
    cs = None

    if nbins == 0:
        return None

    while checkval > 1 - thr and (short or nbins <= len(dat) / 10):
        #print('nbins = ',nbins)
        if bins is not None:
            inds = np.where( dat < bins[takebin] )[0]
            dat = dat[inds]

        r, bins = np.histogram(dat, bins = nbins, density=False)
        cs = np.cumsum(r)
        cs = cs / cs[-1]
        checkval = cs[0] # prob accumulated in 1st bin
        nbins *= 2

    first = np.where(cs > 1 - thr)[0][0]
    bincoord = bins[first]

    return np.min(bins), bincoord

def getDataClusters(chdata, n_clusters=2,n_jobs=-1):
    '''
    arguments  dim x time   array
    returns    dim x cluster number
    '''
    #assert (chdata.ndim == 1) or (chdata.ndim == 2 and chdata.shape[0] == 1), 'too many dims, want just one'
    import sklearn.cluster as skc

    x = chdata
    if isinstance( x.flatten()[0] , complex):
        x = np.abs(x)

    augment = 0
    if (chdata.ndim == 1) or (chdata.ndim == 2 and chdata.shape[0] == 1):
        x = x.flatten()
        X = np.array( list(zip(x,np.zeros(len(x))) ) )   # cannot cluster 1D
        augment = 1
    else:
        assert x.ndim == 2
        X = x.T
    kmeans = skc.KMeans(n_clusters=n_clusters, n_jobs=n_jobs).fit( X )

    if augment:
        cnt = kmeans.cluster_centers_[:,0]
    else:
        cnt = kmeans.cluster_centers_
        assert cnt.shape[0] == 2 and cnt.shape[1] == chdata.shape[0]
    return cnt

def getDataClusters_fromDistr(data, n_clusters=2, n_splits = 25, percentOffset = 30,
                              clusterType = 'outmostPeaks',
                              debugPlot = False, limOffestPercent = 10, lbl = None):
    '''
    percentOffset -- how much percent of the span (distance between peaks) we offset to right
    '''
    # 1D clusterisation from distribution
    df = data.flatten()
    mn = np.min(data)
    mx = np.max(data)
    #bins = np.linspace(mn,mx,nsplits)
    #r = np.digitize(df, bins)   # histogram -- numbers of values falling within each bin
    r, bins = np.histogram(df,bins = n_splits,density=False)
    bins = np.array(bins)

    # find loc max
    locMaxInds = []
    locMaxVals = []
    for bini,n in enumerate(r):
        if bini > 0 and bini < len(r) - 1:
            n_next = r[bini+1]
            n_prev = r[bini-1]
            #print(n_prev,n,n_next)
            if n > n_next and n > n_prev:
                locMaxInds += [bini]
                locMaxVals += [n]
                #print('huu!')
        elif bini > 0 and bini == len(r) - 1:
            n_prev = r[bini-1]
            if n > n_prev:
                locMaxInds += [bini]
                locMaxVals += [n]
        elif bini == 0 and bini < len(r) - 1:
            n_next = r[bini+1]
            if n > n_next:
                locMaxInds += [bini]
                locMaxVals += [n]


    if clusterType == 'highestPeaks':
        sortinds = np.argsort( locMaxVals )  # zero el or sortinds -- index of lowest el in locMaxVals (same as ind in locMaxInds)
        m = min(len(sortinds), n_clusters)
        res = bins[  np.array(locMaxInds) [  sortinds[-m:]  ] ]

        if len(res) == 2 and percentOffset > 0:
            #res = res[::-1]
            span = res[1] - res[0]
            assert span > 0
            shift = span * percentOffset / 100
            res[0] = max( mn,  res[0] - shift  )
            res[1] = min( mx,  res[1] + shift  )
    elif clusterType == 'outmostPeaks':
        if len(locMaxInds) >= 2:
            i1 = min(locMaxInds)
            i2 = max(locMaxInds)
            ncl_left = int( n_clusters / 2 )
            ncl_right = int( n_clusters / 2 )
            if ncl_left + ncl_right < n_clusters:
                ncl_left += 1
                assert ncl_left + ncl_right == n_clusters
            res = np.append( bins[  locMaxInds[:ncl_left] ], bins[  locMaxInds[-ncl_right:] ] )
            assert len(res) == n_clusters
            span = res[-1] - res[0]
            assert span > 0
            shift = span * percentOffset / 100
            shiftLim = span * limOffestPercent / 100

            res[0] = max( mn + shiftLim,  res[0] - shift  )
            res[1] = min( mx - shiftLim,  res[1] + shift  )
        else:
            if lbl is not None:
                print('{} has only {} loc maximums'.format(lbl,len(locMaxInds) ) )
            res = np.ones(n_clusters) * ( (mx-mn) / 2 )
            span = mx - mn
            shift = span * limOffestPercent / 100
            res[0] = mn + shift
            res[1] = mx - shift
    else:
        raise ValueError('clusterType not implemented')




    if debugPlot:
        print('res is ',res)
        ax = plt.gca()
        ax.plot(bins, np.concatenate( (r, [0] )  ) )
        if len(res) > 0:
            ax.axvline(x = res[0])
        if len(res) > 1:
            ax.axvline(x = res[1])

    if len(res) > 0:
        assert res[-1] > res[0]
    return res

def calcSpecThr(chdata):
    '''
    calc spectral threshold
    '''
    cnt = getDataClusters(chdata)
    thr = np.mean(cnt)
    return thr

##############################################

def _flt(data,sfreq,lowcut,highcut,bandpass_order = 5):
    assert data.ndim == 1
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(bandpass_order, [low, high], btype='band')
    y = sig.lfilter(b, a, data)
    return y

def _hlbt(data,sfreq,l_freq,h_freq, ret_flt = 0):
    low,high = l_freq,h_freq #fbans[bandname]
    fltdata = _flt(data,sfreq,low,high)
    datcur = sig.hilbert(fltdata)
#     instphase = np.unwrap(np.angle(datcur))
#     instfreq = (np.diff(instphase) / (2.0*np.pi) * sfreq)
#     instampl = np.abs(datcur)
    if ret_flt:
        return datcur, fltdata
    else:
        return datcur


def getBandHilbDat(data, sfreq, l_freq,h_freq, ret_flt=0):
    assert data.ndim == 1
    hdat, fltdata = _hlbt(data,sfreq, l_freq,h_freq , ret_flt = 1)
    ang = np.angle(hdat)
    instphase = np.unwrap(ang)
    instfreq = (np.diff(instphase) / (2.0*np.pi) * sfreq)
    instampl = np.abs(hdat)

    if ret_flt:
        return ang, instfreq, instampl, fltdata
    else:
        return ang, instfreq, instampl


def H_difactmob(dat,dt, windowsz = None):
    import pandas as pd
    # last axis is time axis
    dif = np.diff(dat,axis=-1, prepend=dat[:,0][:,None] ) / dt
    if windowsz is None:
        activity = np.var(dat, axis=-1)
        vardif = np.var(dif, axis=-1)
    else:
        #raise ValueError('not implemented yet')
        if dat.ndim > 1:
            activity = []
            vardif = []
            for dim in range(dat.shape[0]):
                act = pd.Series(dat[dim]).rolling(windowsz).var()
                var   = pd.Series(dif[dim]).rolling(windowsz).var()  # there is one-bin shift here, better to remove..
                activity += [act]
                vardif += [var]
            activity = np.vstack(activity)
            vardif = np.vstack(vardif)
        else:
            raise ValueError('wrong!')

    mobility = np.sqrt( vardif / activity )

    return dif,activity, mobility

def Hjorth(dat, dt, windowsz = None):
    # if windowsz is not None:
    #     raise ValueError('not implemented yet')
    # activity = np.var(dat, axis=-1)
    # dif = np.diff(dat,axis=-1) / dt
    # vardif = np.var(dif)
    # mobility = np.sqrt( vardif / activity )
    dif, activity, mobility = H_difactmob(dat,dt, windowsz=windowsz)
    #dif2 = np.diff(dif) / dt


    dif2, act2, mob2 = H_difactmob(dif,dt, windowsz=windowsz)
    complexity = mob2 / mobility

    return activity, mobility, complexity


def tfr2csd(dat, sfreq, returnOrder = False, skip_same = []):
    ''' csd has dimensions Nchan x nfreax x nbins
    returns n x (n+1) / 2  x nbins array
    skip same = indices of channels for which we won't compute i->j correl (usually LFP->LFP)
      note that it ruins getCsdVals
    '''
    assert dat.ndim == 3
    n_channels = dat.shape[0]
    csds = []

    order  = []
    for chi in range(n_channels):
        if len(skip_same) > 0:
            good_sec_inds = [chi]
            for chi2 in range(chi+1,n_channels):
                if not ( (chi in skip_same) and (chi2 in skip_same) ):
                    good_sec_inds += [chi2]
            r = np.conj ( dat[[chi]] ) *  ( dat[good_sec_inds] )    # upper diagonal elements only, same freq cross-channels
            secarg = np.array( good_sec_inds, dtype=int )
        else:
            r = np.conj ( dat[[chi]] ) *  ( dat[chi:] )    # upper diagonal elements only, same freq cross-channels
            secarg   = np.arange(chi,n_channels)
        firstarg = np.ones(len(secarg), dtype=int ) * chi
        curindPairs = np.vstack( [firstarg,secarg] )
        order += [curindPairs]

        #print(r.shape)
        csds += [r  ]

    order = np.hstack(order)

    csd = np.vstack( csds )
    csd /= sfreq

    ret = csd
    if returnOrder:
        ret = csd, order
    return ret

def getCsdVals(csd,i,j, n_channels, updiag = 0):
    #swap
    assert i < n_channels
    assert j < n_channels

    if j < i:
        tmp = i
        i = j
        j = tmp

    idx = 0
    for ii in range(i):
        idx += (n_channels - ii - updiag)

    dist_to_diag = j - i - updiag
    idx += dist_to_diag
    #print(idx)
    return  csd[idx]
#i,j ->

def getFullCSDMat(csd, freq_index, time_index, n_channels, updiag=0):
    from collections.abc import Iterable
    # TODO: if time_index is iterable
    M = np.zeros( (n_channels, n_channels), dtype = csd.dtype)
    for i in range(n_channels):
        for j in range(n_channels):
            r = getCsdVals(csd,i,j, n_channels, updiag =updiag)[freq_index,time_index]
            if isinstance(r, Iterable):
                r = np.mean(r)
            M[i,j] = r

    return M

#########################

def removeAnnsByDescr(anns, anns_descr_to_remove):
    ''' decide by using find (so can be both sides of just one)'''
    #anns_descr_to_remove = ['endseg', 'initseg', 'middle', 'incPost' ]
    assert isinstance(anns_descr_to_remove,list)
    anns_upd = anns.copy()

    #anns_descr_fine = []
    #anns_onset_fine = []
    #anns_dur_fine = []
    Nbads = 0
    for ind in range(len(anns.description))[::-1]:
        wasbad = 0
        cd = anns.description[ind]
        for ann_descr_bad in anns_descr_to_remove:
            if cd.find(ann_descr_bad) >= 0:
                wasbad = 1
        if wasbad:
            print('Removing ',cd)
            anns_upd.delete(ind)
            Nbads += 1
#         if not wasbad:
#             anns_descr_fine += [cd]
#             anns_onset_fine += [anns.onset[ind]]
#             anns_dur_fine += [anns.duration[ind]]

#     onsets_upd = anns_onset_fine
#     duration_upd = anns_dur_fine
#     descrs_upd = anns_descr_fine

#     anns_upd = mne.Annotations( onset=onsets_aupd,
#                            duration = duration_upd,
#                            description=descrs_upd,
#                            orig_time=None)
    print('Overall removed {} annotations'.format(Nbads) )

    return anns_upd

def renameAnnDescr(anns,n2n):
    import mne
    assert isinstance(n2n,dict)

    anns_upd = mne.Annotations([],[],[])

    Nbads = 0
    for ind in range(len(anns.description))[::-1]:
        wasbad = 0
        cd = anns.description[ind]
        newcd = cd
        was = 0
        for str0 in n2n:
            if cd.find(str0) >= 0:
                was += 1
            newcd = newcd.replace(str0, n2n[str0])
            #if cd.find(ann_descr_bad) >= 0:
            #    wasbad = 1
        if was:
            print('{} --> {}'.format(cd,newcd) )
        dur = anns.duration[ind]
        onset = anns.onset[ind]

        anns_upd.append(onset,dur,newcd)
    return anns_upd

def ann2ivalDict(anns):
    ivalis = {}
    for i,an in enumerate(anns ):
        descr = an['description']
        if descr not in ivalis:
            ivalis[descr] = []
        tpl = an['onset'], an['onset']+ an['duration'], descr
        ivalis[descr] += [ tpl  ]#bandnames = ['tremor', 'beta', 'gamma', 'allf']

    return ivalis

#############################

def plotTopomapTau(ax,tfr,timeint,fb,vmin,vmax,contours = 8, logscale=False, colorbar=False):
    # tfr.data >= 0
    import mne
    from mne.viz.topomap import _prepare_topomap_plot,_get_ch_type,_make_head_outlines,_hide_frame
    from mne.viz.topomap import _set_contour_locator, partial, plot_topomap,_onselect
    from mne.viz.topomap import _setup_cmap, _merge_ch_data, _handle_default, _add_colorbar
    from mne.viz.topomap import _setup_vmin_vmax
    timin,timax,tiname = timeint
    fbname, fbmin, fbmax = fb

    ch_type=None
    ch_type = _get_ch_type(tfr, ch_type)
    layout = None
    sphere = np.array([0,0,0,0.9])
    picks, pos, merge_channels, names, _, sphere, clip_origin = \
        _prepare_topomap_plot(tfr, ch_type, layout, sphere=sphere)

    outlines = 'head'
    head_pos = None
    show_names = False
    outlines = _make_head_outlines(sphere, pos, outlines, clip_origin,
                                   head_pos)

    if not show_names:
        names = None

    freqi = np.where( (tfr.freqs <= fbmax) * (tfr.freqs >=fbmin) )[0]
    timei = np.where( (tfr.times <= timax) * (tfr.times >=timin) )[0]
    data = tfr.data[picks]
    if logscale:
        data = np.log(np.abs(data) )
        data -= np.min(data,axis=2)[:,:,None]

    norm = np.min(data) >= 0  # assumes data is non-neg
    cmap = _setup_cmap(None, norm=norm)

    if merge_channels:
        data, names = _merge_ch_data(data, ch_type, names, method='mean')

    #data = mne.baseline.rescale(data, tfr.times, baseline=None, mode='mean', copy=True)

    data = data[:,freqi[0]:freqi[-1]+1,timei[0]:timei[-1]+1]
    data = np.mean(np.mean(data, axis=2), axis=1)[:, np.newaxis]

    _hide_frame(ax)


    #vmin = None; vmax = None
    #vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    #print(vmin,vmax, np.max(data) )

    locator = None
    if not isinstance(contours, (list, np.ndarray)):
        locator, contours = _set_contour_locator(vmin, vmax, contours)

    ax.set_title('{}_{}'.format(fbname,tiname) )
    fig_wrapper = list()
    #selection_callback = partial(_onselect, tfr=tfr, pos=pos, ch_type=ch_type,
    #                             itmin=timei[0], itmax=timei[-1]+1,
    #                             ifmin=freqi[0], ifmax=freqi[-1]+1,
    #                             cmap=cmap[0], fig=fig_wrapper,
    #                             layout=layout)
    selection_callback = None

    if not isinstance(contours, (list, np.ndarray)):
        _, contours = _set_contour_locator(vmin, vmax, contours)

    # data[:, 0]
    #print(vmin,vmax, data.shape, data)
    #print(np.min(data), np.max(data), vmin,vmax)
    im,_ = plot_topomap(data[:,0], pos, vmin=vmin, vmax=vmax,
                            axes=ax, cmap=cmap[0], image_interp='bilinear',
                            contours=contours, names=names, show_names=show_names,
                            show=False, onselect=selection_callback,
                            sensors=True, res=64, head_pos=head_pos,
                            outlines=outlines, sphere=sphere)
    cbar_fmt='%1.1e'

    if colorbar:
        from matplotlib import ticker
        unit = None
        unit = _handle_default('units', unit)['misc']
        cbar, cax = _add_colorbar(ax, im, cmap, title=unit, format=cbar_fmt)
        if locator is None:
            locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = locator
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=12)

def plotBandLocations(tfr,timeints,fbs,prefix='', logscale=False,
                      colorbar=False, anns=None, qoffset=5e-2):
    '''
    power output of mne.time_frequency applied to Epochs object
    '''
    import mne
    nc = len(fbs); nr = len(timeints)
    ww = 4; hh = 3
    headsph = np.array([0,0,0,0.9])
    fig,axs = plt.subplots( nrows = nr, ncols = nc, figsize= (nc*ww, nr*hh))
    plt.subplots_adjust(top=0.97, bottom=0.02, right=0.95, left=0.01)
    dat = np.abs(tfr.data)

    picks = mne.pick_types(tfr.info, meg='mag', ref_meg=False,exclude='bads')
    #timei = np.where( (tfr.times <= timax) * (tfr.times >=timin) )[0]
    data = tfr.data[picks]

    if logscale:
        data = np.log(data)

    for j,fb in enumerate(fbs):
        print('Plotting band {} out of {}'.format(j+1, len(fbs) ) )
        fbname, fbmin, fbmax = fb

        freqi = np.where( (tfr.freqs <= fbmax) * (tfr.freqs >=fbmin) )[0]
        data_band = data[:,freqi[0]:freqi[-1]+1,:]
        data_band_me = np.mean(data_band, axis=1)  #mean over freq
        #data_band_me = np.mean(np.mean(data, axis=2), axis=1)

        mn,mx = np.percentile(data_band_me, 100*np.array([qoffset, 1-qoffset]) )        # over time and sensors
        for i,ti in enumerate(timeints):
            timin,timax,tiname = ti
            ax = axs[i,j]
            ttl = '{}:{}\n{:.1f},  {:.1f}'.format(tiname,fbname,timin,timax,logscale=logscale)
            if anns is not None and tiname != 'entire':
                ann_descrs = findIntersectingAnns(timin,timax,anns)
                if len(ann_descrs) > 0 :
                    ttl += '\n' + ','.join(ann_descrs)

            if i == 0:  # first raw always gets a colorbar
                colorbar_cur = True
            else:
                colorbar_cur = colorbar
            plotTopomapTau(ax,tfr,ti,fb,mn,mx,contours = 8,colorbar=colorbar_cur)
            ax.set_title(ttl)

            #tfr.plot_topomap(sensors=True, contours=8, tmin=timin, tmax=timax,
            #                   fmin=fbmin, fmax=fbmax, axes=ax, colorbar=True,
            #                   size=40, res=100, show=0, sphere=headsph,vmin=mn,vmax=mx);
            #plt.gcf().suptitle('{} : {}'.format(tiname,fbname))
    #plt.tight_layout()


    plt.savefig(os.path.join(gv.dir_fig,\
        '{}_bandpow_concentr_nints{}_nbands{}.pdf'.
                             format(prefix,len(timeints), len(fbs) )))
    plt.close()

#def plotTimecourse(plt):
#    ndatasets = len(gv.raws)
#    #nffts = [512,1024]
#
#
#    ymaxEMG = 0.00035
#    ymaxLFP = 0.00012
#
#    chanTypes_toshow = ['EMG','LFP']
#    ymaxs = {'EMG':ymaxEMG, 'LFP':ymaxLFP}
#    #chanTypes_toshow = ['EMG']
#
#    nc = 4
#    nr = len(ks)
#
#    time_start,time_end = 0,1000
#    #time_start,time_end = 0,300  # to get only rest part
#
#    fig, axs = plt.subplots(ncols = nc, nrows=nr, figsize= (ww*nc,hh*nr), sharey='none')
#    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
#    axind = 0
#
#    colind = 0
#
#    for axind in range(nr):
#        k = ks[axind]
#
#        chnames = gv.raws[k].info['ch_names']
#        orderEMG, chinds_tuples, chnames_tuples = getTuples(sind_str)
#
#        for channel_pair_ind in [0,1]:
#            inds = []
#            for cti in range(len(chanTypes_toshow_timecourse) ):
#                ct = chanTypes_toshow[cti]
#                inds = chinds_tuples[channel_pair_ind][ct]
#                #inds += lfp_inds[3:] + emgkil_inds[-2:] #+ eog_inds
#                # convert indices to channel names
#                ch_toplot = [chnames[i] for i in inds]
#                chs = mne.pick_channels(chnames,include=ch_toplot, ordered=True )
#
#                ts,te = f.time_as_index([time_start, time_end])
#
#                chdata, chtimes = gv.raws[k][chs,ts:te]
#                mintime = min(chtimes)
#                maxtime = max(chtimes)
#
#
#                if nc == 1:
#                    ax = axs[axind]
#                else:
#                    colind = channel_pair_ind * 2 + cti
#                    ax = axs[axind,colind]
#
#                for i in range(chdata.shape[0] ):
#                    ax.plot(chtimes,chdata[i,:], label = '{}'.format(chnames[chs[i] ] ), alpha=0.7 )
#                    ax.set_ylim(-ymaxEMG/2,ymaxs[ct])
#                    ax.set_xlim(mintime,maxtime)
#
#                ax.legend(loc=legendloc,framealpha = legalpha)
#                side = orderEMG[channel_pair_ind]
#                ax.set_title('{:10} {} {}'.format(k,ct,side)  )
#                ax.set_xlabel('Time, [s]')
#                ax.set_ylabel('Voltage')
#
#    ss = []
#    subjstr = ''
#    for k in ks:
#        sind_str,medcond,task = getParamsFromRawname(k)
#        if sind_str in ss:
#            continue
#        ss += [sind_str]
#        subjstr += '{},'.format( sind_str )
#
#    plt.tight_layout()
#    if savefig:
#        figname = '_{}_{} nr{}, {:.2f},{:.2f}.{}'. \
#                    format(subjstr, data_type, nr, float(time_start), \
#                           float(time_end),ext)
#        plt.savefig(os.path.join(gv.dir_fig, os.path.join(plot_output_dir, figname ) ))
#    if not showfig:
#        plt.close()
#    else:
#        plt.show()
#
#    print('Plotting all finished')

def collectChnamesBySide(info):
    #Axis X: From the origin towards the RPA point (exactly through)  #ears
    #Axis Y: From the origin towards the nasion (exactly through)     #nose
    #Axiz Z: From the origin towards the top of the head
    #Origin: Intersection of the line L through LPA and RPA and a plane orthogonal
    #to L and passing through the nasion.
    # positive Y means more frontal

    res = {'left':[], 'right':[] }
    res_inds = {'left':[], 'right':[] }
    chs = info['chs']
    for chi,ch in enumerate(chs):
        loc = ch['loc'][:3]
        if loc[0] >= 0:
            side  = 'right'
        else:
            side = 'left'
        res[side] += [ch['ch_name']]
        res_inds[side] += [chi]

    return res, res_inds

def intervals2anns(intlist, int_name=None, times=None):
    '''
    int_name -- str or list of strs or None; if None, description is taken as third tuple element
    if times is None, intlist items interpreted as pairs of times, else as paris of timebins
    '''
    assert isinstance(intlist,list)
    import mne
    anns = mne.Annotations([],[],[])
    for ivli,ivl in enumerate(intlist):
        b0,b1 = ivl[0], ivl[1]
        if times is not None:
            b0t,b1t = times[b0], times[b1]
        else:
            b0t,b1t = b0,b1
        if isinstance(int_name,str):
            int_name_cur = int_name
        elif isinstance(int_name,list):
            int_name_cur = int_name[ivli]
        elif int_name is None:
            assert len(ivl) > 2
            int_name_cur = ivl[2]
        anns.append([b0t],[b1t-b0t], [int_name_cur ]  )
    return anns

def findMEGartifacts(filt_raw , thr_mult = 2.5, thr_use_mean=0):
    raw_only_meg = filt_raw.copy()
    raw_only_meg.pick_types(meg=True)

    assert len(raw_only_meg.info['bads']) == 0, 'There are bad channels!'

    chnames_perside_meg, chis_perside_meg = collectChnamesBySide(filt_raw.info)
    import utils_tSNE as utsne
    import mne

    fig,axs = plt.subplots(2,1, figsize=(14,7), sharex='col')

    sides = sorted(chnames_perside_meg.keys())
    anns = mne.Annotations([],[],[])
    cvl_per_side = {}
    for sidei,side in enumerate(sides ):
        chnames_curside = chnames_perside_meg[side]
        megdat, times = raw_only_meg[chnames_curside]
        #megdat = raw_only_meg.get_data()
        me, mn,mx = utsne.robustMean(megdat, axis=1, per_dim =1, ret_aux=1, q = .25)
        megdat_scaled = ( megdat - me[:,None] ) / (mx-mn)[:,None]
        megdat_sum = np.sum(np.abs(megdat_scaled),axis=0)
        me_s, mn_s,mx_s = utsne.robustMean(megdat_sum, axis=None, per_dim =1, ret_aux=1, pos = 1)

        if thr_use_mean:
            mask= megdat_sum > me_s * thr_mult
        else:
            mask= megdat_sum > mx_s * thr_mult
        cvl,ivals_meg_artif = getIntervals(np.where(mask)[0] ,\
            include_short_spikes=1, endbin=len(mask), minlen=2, thr=0.001, inc=1,\
            extFactorL=0.1, extFactorR=0.1 )
        cvl_per_side[side] = cvl

        print('MEG artifact intervals found ' ,ivals_meg_artif)
        #import ipdb; ipdb.set_trace()

        ax = axs[sidei]
        ax.plot(filt_raw.times,megdat_sum)
        ax.axhline( me_s , c='r', ls=':')
        ax.axhline( mx_s , c='purple', ls=':')
        ax.axhline( me_s * thr_mult , c='r', ls='--')
        ax.axhline( mx_s * thr_mult , c='purple', ls='--')
        ax.set_title('{} MEG artif'.format(side) )

        for ivl in ivals_meg_artif:
            b0,b1 = ivl
            #b0t,b1t = filt_raw.times[b0], filt_raw.times[b1]
            #anns.append([b0t],[b1t-b0t], ['BAD_MEG{}'.format( side[0].upper() ) ]  )
            ax.axvline( filt_raw.times[b0] , c='r', ls=':')
            ax.axvline( filt_raw.times[b1] , c='r', ls=':')

        anns = intervals2anns(ivals_meg_artif,  'BAD_MEG{}'.format( side[0].upper() ), filt_raw.times )

        ax.set_xlim(filt_raw.times[0], filt_raw.times[-1]  )

    return anns, cvl_per_side


def plotICAdamage(filt_raw, rawname_, ica, comp_inds_intr,xlim, nr=20,
                  do_plot=1, ecg_comp_ind=-1, mult=20/100, multECG = 35/100,
                  fbands=None, pctshift=20, onlyVarInc = 1):

    icacomp = ica.get_sources(filt_raw)

    pcts = [pctshift,100 - pctshift]
    ext = 'png'
    figname = '{}_ICA_damage_dur{:.1f}s.{}'.format(rawname_, xlim[1]-xlim[0],ext)
    skip = 5
    if xlim[1] - xlim[0] > 100:
        skip = 40

    if fbands is None:
        fbands = {'tremor': [3,10], 'beta':[15,30],   'gamma':[30,100] }

    if do_plot:
        plt.close()
        nc = len(fbands)
        ww = 14; hh = 2
        fig,axs = plt.subplots(nr,nc, figsize=(nc*ww, nr*hh), sharex='col')
        plt.subplots_adjust(top=0.97,bottom=0.02,right=0.98, left=0.03, wspace=0.03)
    use_MNE_infl_calc = 1

    rawdat = filt_raw.get_data()
    rawdat_meg,times = filt_raw[ica.ch_names]

    import utils_tSNE as utsne

    ii = 0
    printLog =1


    sfreq = filt_raw.info['sfreq']

    for fbi,fbname in enumerate(fbands.keys() ):
        b0,b1 = fbands[fbname]
        ii = 0
        for compi in comp_inds_intr:
            curcomp = icacomp[compi][0][0]
            #efflims = np.percentile(curcomp, [5,95])

            curcomp_flt = _flt(curcomp, sfreq, b0,b1)
            ax = axs[ii,fbi]; ii+= 1
            ax.plot(filt_raw.times,curcomp_flt, label='comp')
            ax.set_xlim(xlim)
            ax.set_title('{} band of component {}'.format(fbname, compi) )

            if use_MNE_infl_calc:
                raw_cur = filt_raw.copy()
                ica.apply(raw_cur,exclude=[compi])
            else:
                src_ica = None
                import utils_preproc as upre
                infl = upre.getCompInfl(ica, src_ica, comp_inds_intr)
                contrib = infl[compi] # nchans x ntimebins

            stop = False
            for chi in range(len(rawdat_meg)):
                if ii >= nr and do_plot:
                    stop = True
                    break
                large = 0

                curdat = rawdat[chi]
                if use_MNE_infl_calc:
                    corrected_dat = raw_cur[chi][0][0]
                else:
                    corrected_dat = curdat - contrib[chi]

                curdatl       = _flt(curdat, sfreq, b0,b1)
                corrected_dat = _flt(corrected_dat, sfreq, b0,b1)

                #print(curdat.shape, corrected_dat.shape)
                #efflims_cur = efflims[chi]
                efflims_cur = np.percentile(curdat, pcts)
                efflims_cur_corr = np.percentile(corrected_dat, pcts)
                d = (efflims_cur[1] - efflims_cur[0])
                #range of corrected - range of current
                reduction = (efflims_cur_corr[1] - efflims_cur_corr[0]) - d
                change= reduction

                mult_cur = mult
                if compi == ecg_comp_ind:
                    mult_cur = multECG

                #Axis X: From the origin towards the RPA point (exactly through)  #ears
                #Axis Y: From the origin towards the nasion (exactly through)     #nose
                #Axiz Z: From the origin towards the top of the head
                #Origin: Intersection of the line L through LPA and RPA and a plane orthogonal
                #to L and passing through the nasion.
                # positive Y means more frontal

                if onlyVarInc:
                    large = (-reduction)  > mult_cur*d
                else:
                    large = abs(change)  > mult_cur*d

                loc = filt_raw.info['chs'][chi]['loc'][:3]
                coord_str = ', '.join( map(lambda x: '{:.4f}'.format(x),  loc.tolist() ) )
                coord_str = '[{}]'.format(coord_str)

                if large or printLog:
                    print('comp {}_chi {} {} large={} change {:.3f}%'.format(compi,chi, coord_str,
                                                                               large, 100*change/d) )

                if do_plot and large:
                    ax = axs[ii,fbi]; ii+= 1
                    curdat_ds = utsne.downsample(curdat, skip ,printLog=0 )
                    corrected_dat_ds = utsne.downsample(corrected_dat, skip ,printLog=0 )
                    ax.plot(filt_raw.times[::skip],curdat_ds, label='orig')
                    ax.plot(filt_raw.times[::skip],corrected_dat_ds, label='corr',alpha=0.5)
                    #ax.set_xlim(raw.times[0],raw.times[-1])
                    #ica.plot_overlay(filt_raw,picks=[ica.ch_names[chi]], exclude=[compi])

                    ax.set_title('{}: comp {} chanind {} {}:  {:.3f}%'.format(fbname, compi,chi,
                                                                               coord_str, 100*change/d))
                    ax.legend(loc='upper right')
                    ax.set_xlim(xlim)

                #print('comp {}_chi {} change ratio {:.3f}%'.format(compi,chi, 100*change/d) )
            if stop:
                break
        # I want to check if variance overall reduces (bad) or only during transient events
    plt.savefig(os.path.join(gv.dir_fig_preproc,figname), dpi=200)
    plt.close()
    import gc; gc.collect()

def artif2ann(art_dict, art_dict_nms, maintremside, side='main', LFPonly=1):
    # art_dict -- key: ch name -> list of 2-el-lists,
    onset = []
    duration = []
    description = []
    assert len(maintremside) > 1 and len(side) > 1

    if side == 'main' or side == maintremside:
        artifacts = [art_dict]
    elif side == 'other' or side == getOppositeSideStr(side):
        artifacts = [art_dict_nms]
    elif side == 'both':
        artifacts = [art_dict, art_dict_nms]
    for artifacts_cur in artifacts:
        for chn in artifacts_cur:
            intervals_cur = artifacts_cur[chn]
            curdescr = 'BAD_{}'.format(chn)
            for ivl in intervals_cur:
                onset += [ ivl[0]  ]
                duration += [ ivl[1]-ivl[0] ]
                description += [ curdescr ]

    import mne
    anns = mne.Annotations(onset, duration, description)
    return anns
# note than maintremside should correspond to the rawname!


def getEMGperHand(rectconvraw):
    EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
    rectconvraw.load_data()

    rectconvraw_perside = {}
    for side in EMG_per_hand:
        chns = EMG_per_hand[side]
        tmp = rectconvraw.copy()
        tmp.pick_channels(chns)

        assert len(tmp.ch_names) == 2
        rectconvraw_perside[side] = tmp

    for side in EMG_per_hand:
        badstr = '_' + getOppositeSideStr(side[0].upper())
        #print(badstr)
        anns_upd = removeAnnsByDescr(rectconvraw_perside[side].annotations, [badstr])
        rectconvraw_perside[side].set_annotations(anns_upd)

    return  rectconvraw_perside

def getLFPperSide(raw_lfponly, key='letter'):
    '''
    Also filter annotations
    '''
    #EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
    import mne
    raw_lfponly.load_data()

    raws_lfp_perside = {}
    for side in ['left', 'right' ]:
        sidelet = side[0].upper()
        if key == 'str':
            sidekey = side
        elif key == 'letter':
            sidekey = sidelet
        raws_lfp_perside[sidekey] = raw_lfponly.copy()
        chis = mne.pick_channels_regexp(raw_lfponly.ch_names, 'LFP{}.*'.format(sidelet))
        chnames_lfp = [raw_lfponly.ch_names[chi] for chi in chis]
        raws_lfp_perside[sidekey].pick_channels(   chnames_lfp  )

    for sidekey in raws_lfp_perside:
        # first we remove tremor annotations from other side. Since it is
        # brain, other side means ipsilateral
        sidelet = sidekey[0].upper()
        badstr = '_' + sidelet
        anns_upd = removeAnnsByDescr(raws_lfp_perside[sidekey].annotations, [badstr])

        # now we remove artifcats for other side of the brain
        badstr = 'LFP' + getOppositeSideStr(sidelet)
        anns_upd = removeAnnsByDescr(anns_upd, [badstr])

        # set result
        raws_lfp_perside[sidekey].set_annotations(anns_upd)

    return  raws_lfp_perside

def extendAnns(anns,ext_left, ext_right):
    # in sec
    import copy
    newanns = copy.deepcopy(anns)
    newanns.onset -= ext_left
    newanns.duration += ext_right
    return newanns

def findTaskStart(anns):
    r = None
    tasks = ['hold', 'move']
    for ann in anns:
        for task in tasks:
            if ann['description'].find(task) >= 0:
                r = ann
                break
        if r is not None:
            break
    if r is not None:
        return r['onset']
    else:
        return None

def getMainEMGcompOneSide(emg, side, store_both=False):
    # emg is raw with data from one side
    from sklearn.decomposition import PCA
    emgdat = emg.get_data()
    assert emgdat.shape[0] == 2
    pca = PCA(2)
    emgdat_rot = pca.fit_transform(emgdat.T)
    emgdat_rot = emgdat_rot.T

    #print( pca.explained_variance_ratio_ )
    import mne

    info_newemg = mne.create_info(ch_names = ['EMGmain{}'.format(side[0].upper())],
                sfreq = emg.info['sfreq'])
    if store_both:
        retchis = [0,1]
    else:
        retchis = [0]
    r = mne.io.RawArray(emgdat_rot[retchis], info_newemg)
    r.set_annotations(emg.annotations)
    return r, pca

def getMainEMGcomp(emg):
    if isinstance(emg,dict):
        ps = emg
    else:
        ps = getEMGperHand(emg)

    res = {}
    for side in ps:
        #names = mne.pick_channels_regexp(emg.ch_names,'EMG.*_old')
        #emgdat, times = emg[names]
        #assert emgdat.shape[0] == 2
        curraw = ps[side]

        r, pca = getMainEMGcompOneSide(curraw, side)
        res[side] = r

        print('-- On {} side, explain {} ratio by EMG components '.format(side, pca.explained_variance_ratio_ ) )
    return res

def getIntervalMaxs(raw,intervals, q=0.995):
    #m = np.zeros(2)
    alldat = []
    qpi = []
    for ivi,iv in enumerate(intervals):
        if len(iv) == 3:
            st,end,tt = iv
        elif len(iv ) == 2:
            st,end = iv
        else:
            raise ValueError('wrong len', len(iv))
        sti,endi = raw.time_as_index([st,end])
        dat,times = raw[:,sti:endi]
        alldat += [dat]
        mcur = np.quantile(dat,q,axis=1)
        qpi += [mcur]
        #m = np.maximum(m,mcur)
        print('mcur for interval {} is {}'.format(ivi,mcur) )
    alldat = np.hstack(alldat)
    m0 = np.max(alldat, axis=1)
    m = np.quantile(alldat, q, axis=1)
    return m,m0


def intervalJSON2Anns(rawname_, use_new_intervals = True, maintremside=None, return_artifacts = False):
    import json
    from copy import deepcopy

    with open('subj_info.json') as info_json:
        gen_subj_info = json.load(info_json)

    subj,medcond,task  = getParamsFromRawname(rawname_)
    subj_num = int(subj[1:])

    if maintremside is None:
        maintremside = gen_subj_info[subj]['tremor_side']
    nonmaintremside = getOppositeSideStr(maintremside)

    rawname_impr = rawname_
    # in Jan's file 8,9,10 are called 'hold' instead of 'rest', but in my json
    # it was corrected
    if subj_num > 7: # and not use_new_intervals:
        rawname_impr = '{}_{}_hold'.format(subj,medcond)

    if use_new_intervals:
        trem_times_fn = 'trem_times_tau.json'
    else:
        trem_times_fn = 'trem_time_jan_unmod.json'

    with open(trem_times_fn ) as jf:
        trem_times_byhand = json.load(jf)

    tremIntervalJan, artif_main         = unpackTimeIntervals(trem_times_byhand, mainSide = True,
                                                            gen_subj_info=gen_subj_info, skipNotLoadedRaws=0)

    if use_new_intervals:
        trem_times_nms_fn = 'trem_times_tau_nms.json'
        with open(trem_times_nms_fn ) as jf:
            trem_times_nms_byhand = json.load(jf)
        tremIntervalJan_nms, artif_nms = unpackTimeIntervals(trem_times_nms_byhand, mainSide = False,
                                                                gen_subj_info=gen_subj_info, skipNotLoadedRaws=0)

    #%debug

    artif = deepcopy(artif_main)
    if use_new_intervals:
        #for rawn in [rawname_]:
        if rawname_ in artif_nms and rawname_ not in artif:
            artif[rawname_] = artif_nms[rawname_]
        else:
            if rawname_ in artif_nms:
                artif[rawname_].update(artif_nms[rawname_] )

    #for rawn in tremIntervalJan:
    #    sind_str,medcond_,task_ = getParamsFromRawname(rawn)
    #    maintremside_cur = gen_subj_info[sind_str]['tremor_side']
    #    opside= getOppositeSideStr(maintremside_cur)
    #    if use_new_intervals:
    #        if rawn in tremIntervalJan_nms:
    #            tremIntervalJan[rawn][opside] = tremIntervalJan_nms[rawn][opside]
    if use_new_intervals:
        # first try to get the 'converted rawname' then the original (because I
        # was lazy when dealing with nms)
        if rawname_impr in tremIntervalJan_nms:
            tremIntervalJan[rawname_impr][nonmaintremside] = tremIntervalJan_nms[rawname_impr][nonmaintremside]
        else:
            assert rawname_ in tremIntervalJan_nms
            tremIntervalJan[rawname_impr][nonmaintremside] = tremIntervalJan_nms[rawname_][nonmaintremside]


    mvtTypes = ['tremor', 'no_tremor', 'unk_activity']

    plotTremNegOffset = 2.
    plotTremPosOffset = 2.
    maxPlotLen = 6   # for those interval that are made for plotting, not touching intervals for stats
    addIntLenStat = 5
    plot_time_end = 150

    #import ipdb; ipdb.set_trace()

    timeIntervalPerRaw_processed = processJanIntervals(tremIntervalJan, maxPlotLen, addIntLenStat,
                            plotTremNegOffset, plotTremPosOffset, plot_time_end, mvtTypes=mvtTypes)

    #print(timeIntervalPerRaw_processed[rawname_].keys() )
    intervals = timeIntervalPerRaw_processed[rawname_impr][maintremside]   #[rawn][side] -- list of tuples (beg,end, type string)]   #[rawn][side] -- list of tuples (beg,end, type string)
    if use_new_intervals:
        intervals_nms = timeIntervalPerRaw_processed[rawname_impr][nonmaintremside]   #[rawn][side] -- list of tuples (beg,end, type string)]   #[rawn][side] -- list of tuples (beg,end, type string)

    # convert to intervalType -> intervalInds
    import globvars as gv
    ivalis = {}  # dict of indices of intervals per interval type
    ivalis_nms = {}
    for itype in gv.gparams['intTypes']:
        ivit = []
        for i,interval in enumerate(intervals):
            t1,t2,it = interval

            if it == itype:
                ivit += [i]
        if len(ivit) > 0:
            ivalis[itype] = ivit

        ivit = []
        if use_new_intervals:
            for i,interval in enumerate(intervals_nms):
                t1,t2,it = interval

                if it == itype:
                    ivit += [i]
            if len(ivit) > 0:
                ivalis_nms[itype] = ivit

    #print('Main tremor side here is ',maintremside)

    #display('all intervals:' ,intervals)
    #display('intervals by type:', ivalis )

    # convert intervals to MNE type
    #annotation_desc_2_event_id = {'middle_full':0, 'no_tremor':1, 'endseg':2}
    #annotation_desc_2_event_id = {'middle_full':0, 'no_tremor':1}

    side2ivls = {maintremside:intervals}
    if use_new_intervals:
        side2ivls[nonmaintremside] = intervals_nms

    oldIntName2annDescr = {'middle_full': 'trem_{}' ,
                        'no_tremor':'notrem_{}',
                        'unk_activity_full': 'undef_{}'}

    onset = []
    duration = []
    description = []
    for side_hand in side2ivls:
        intervals_cur = side2ivls[side_hand]
        for ivl in intervals_cur:
            curdescr = ivl[2]
            if curdescr not in oldIntName2annDescr:
                continue
            onset += [ ivl[0]  ]
            duration += [ ivl[1]-ivl[0] ]
            newDescrName = oldIntName2annDescr[curdescr].format(side_hand[0].upper())
            description += [ newDescrName ]

    import mne
    anns_cnv = mne.Annotations(onset, duration, description)

    if return_artifacts:
        r =  anns_cnv, side2ivls , artif
    else:
        r =  anns_cnv, side2ivls
    return r

def getIntervalsTotalLens(ann, include_unlabeled = False, times=None):
    if isinstance(ann,dict):
        ivalis = ann
    else:
        ivalis = ann2ivalDict(ann)

    lens = {}
    totlen_labeled = {'_L':0, '_R':0 }
    for it in ivalis:
        lens[it] = 0
        for ival in ivalis[it]:
            a,b,it_ = ival
            len_cur = b-a
            lens[it] += len_cur
            for sidestr in totlen_labeled:
                if it.find(sidestr) >= 0:
                    totlen_labeled[sidestr] += len_cur

    if include_unlabeled:
        assert times is not None
        for sidestr in totlen_labeled:
            lens['nolabel'+sidestr] = times[-1] - times[0] - totlen_labeled[sidestr]

    return lens

def setArtifNaN(X, ivalis_artif_tb_indarrays_merged, feat_names):
    assert isinstance(X,np.ndarray)
    assert isinstance(ivalis_artif_tb_indarrays_merged, dict)
    assert isinstance(feat_names[0], str)
    Xout = X.copy()
    for interval_name in ivalis_artif_tb_indarrays_merged:
        templ = 'BAD_(.+)'
        matchres = re.match(templ,interval_name).groups()
        assert len(matchres) > 0
        artif_type = matchres[0]
        mode_MEG_artif = False
        mode_LFP_artif = False
        if artif_type.find('MEG') >= 0:
            mode_MEG_artif = True
            #print('MEG',artif_type)
        elif artif_type.find('LFP') >= 0:
            mode_LFP_artif = True
            artif_chn = artif_type
            #print('LFP',artif_type)
        interval_bins = ivalis_artif_tb_indarrays_merged[interval_name]

        for feati,featn in enumerate(feat_names):
            #print(featn)
            if mode_LFP_artif and featn.find(artif_chn) >= 0:
                Xout[interval_bins,feati] = np.nan
                #print('fd')
            elif mode_MEG_artif and featn.find('msrc') >= 0:
                Xout[interval_bins,feati] = np.nan
                #print('fd')

    return Xout
