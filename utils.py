import numpy as np
import udus_dataproc as mdp # main data proc
import re

import globvars as gv
import scipy.signal as sig

def getIntervalIntersection(a1,b1, a2, b2):
    if b1 < a2:
        return []
    if a1 > b2:
        return []
    r =  [ max(a1,a2), min(b1,b2) ]
    if r[1] - r[0] > 1e-2:
        return r
    else:
        return []

def unpackTimeIntervals(trem_times_byhand, mainSide = True, gen_subj_info=None,
        skipNotLoadedRaws=True):
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

def MEGsrcChind2data(rawname,chi):
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

############################# Tremor-related

def getIntervals(bininds,width=100,thr=0.1, percentthr=0.8,inc=5, minlen=50,
        extFactorL = 0.25, extFactorR  = 0.25, endbin = None, cvl=None,
                 percent_check_window_width=256, min_dist_between = 100):
    '''
    width -- number of bins for the averaging filter
    tremini -- indices of timebins, where tremor was detected
    thr -- thershold for convolution to be larger then, for L\infty-normalized data
    output -- convolution, intervals (pairs of timebin indices)
    inc -- how much we extend the interval each time (larger means larger intervals, but worse boundaries)
    minlen -- minimum number of bins required to make the interval be included
    percentthr -- min ratio of thr crossings within the window to continue extending the interval
    extFactor -- we'll extend found intervals by width * extFactor
    endbin -- max timebin
    '''
    assert cvl.ndim == 1

    if cvl is None:
        if endbin is None:
            mt = np.max (bininds ) + 1
        else:
            mt = endbin

        raster = np.zeros( mt, dtype=np.int )
        raster[bininds] = 1
        #width = 100
        avflt = np.ones(width) #box filter
        #avflt = sig.gaussian(width, width/4)
        avflt /= np.sum(avflt)   # normalize
        cvl = np.convolve(raster,avflt,mode='same')

    #cvlskip = cvl[::skip]
    cvlskip = cvl
    thrcross = np.where( cvlskip > thr )[0]
    belowthr = np.where( cvlskip <= thr )[0]
    shiftL = int(width * extFactorL )
    shiftR = int(width * extFactorR )

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

        #print(searchres, rightEnd)
        subcvl = cvlskip[leftEnd:rightEnd]
        while rightEnd < len(cvlskip) + inc + 1: # try to enlarge until possible
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
                break

        if rightEnd-leftEnd >= minlen:
            newp = (  max(0, leftEnd-shiftL), max(0, rightEnd-shiftR) ) # extend the interval on both sides
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
        extFactorL=0.25, extFactorR=0.25):
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
                        if tremrDet_clusterMultiMEG:
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

def mergeTremorIntervals(intervals, mode='intersect'):
    '''
    mode -- 'intersect' or 'union' -- how to deal with intervals coming from differnt muscles
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
    if len(freqinds) == 0 and spec_specgramtype in ['scaleogram', 'mne.wavelet']:
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
#     if windowsz is not None:
#         raise ValueError('not implemented yet')
#     activity = np.var(dat, axis=-1)
#     dif = np.diff(dat,axis=-1) / dt
#     vardif = np.var(dif)
#     mobility = np.sqrt( vardif / activity )
    dif, activity, mobility = H_difactmob(dat,dt, windowsz=windowsz)
    #dif2 = np.diff(dif) / dt

    dif2, act2, mob2 = H_difactmob(dif,dt, windowsz=windowsz)
    complexity = mob2 / mobility

    return activity, mobility, complexity


def tfr2csd(dat, sfreq, returnOrder = False):
    ''' csd has dimensions Nchan x nfreax x nbins
    returns n x (n+1) / 2  x nbins array
    '''
    assert dat.ndim == 3
    n_channels = dat.shape[0]
    csds = []

    order  = []
    for chi in range(n_channels):
        #r = dat[[chi]] * np.conj ( dat[chi:] )    # upper diagonal elements only, same freq cross-channels
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



def plotTimecourse(plt):
    ndatasets = len(gv.raws)
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

        chnames = gv.raws[k].info['ch_names']
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

                chdata, chtimes = gv.raws[k][chs,ts:te]
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

                ax.legend(loc=legendloc,framealpha = legalpha)
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
