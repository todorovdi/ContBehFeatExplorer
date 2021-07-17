import numpy as np
import re

import globvars as gv
import scipy.signal as sig
import matplotlib.pyplot as plt
import globvars as gv
import os
import mne

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

def genMEGsrcChnameShort(side,srcgroup_ind, ind, subind ):
    return 'msrc{}_{}_{}_c{}'.format(side,srcgroup_ind, ind, subind)

def parseMEGsrcChnameShort(chn):
    ''' <something>srcR_1_34_c54'''
    # actually '_' is part of '\w' but apparenty it is not a problem
    r = re.match('[\w_,]*src([RL])_([0-9]+)_c([0-9]+)',chn)
    if r is not None:
        assert len(r.groups() ) == 3
        side = r.groups()[0]
        ind = int(r.groups()[1])
        subind = int(r.groups()[2])
        srcgroup_ind = -1
    else:
        r = re.match('[\w_,]*src([RL])_([0-9]+)_([0-9]+)_c([0-9]+)',chn)
        if r is None:
            raise ValueError('wrong chn format {}'.format(chn))
        assert len(r.groups() ) == 4
        side = r.groups()[0]
        srcgroup_ind = int(r.groups()[1])
        ind = int(r.groups()[2])
        subind = int(r.groups()[3])

    return side,srcgroup_ind, ind, subind


def getMEGsrc_chname_nice(chn, roi_labels=None, keyorder=None, preserve_prefix = False):
    name = chn
    if chn.find('HirschPt2011') >= 0:  # old vere
        r = re.match( ".*_([0-9]+)", chn ).groups()
        num = int( r[0] )
        nlabel = num // 2
        nside =  num % 2
        # even indices -- left [no! even correspond to right]
        side = ['right', 'left'] [nside]
        label = gv.gparams['coord_labels'][nlabel]

        name = '{}_{}'.format( label,side)
    else:
        src_start_ind = chn.find('src')
        if src_start_ind >= 0:
            side, srcgroup_ind, ind, subind = parseMEGsrcChnameShort(chn)

            if srcgroup_ind < 0:
                name = '{}_c{}'.format(roi_labels[ind], subind)
            else:
                assert isinstance(roi_labels, dict) and keyorder is not None
                key = keyorder[srcgroup_ind]
                name = '{}_c{}'.format(roi_labels[key ][ind], subind)

            if preserve_prefix:
                assert chn[:4] == 'msrc'
                #we assume that the start of the thing is 'msrc'
                name = chn[:src_start_ind+5] + name


        #if isinstance(coord_labels_corresp_coord ,list):
        #    r = re.match('.+src._([0-9]+)_c([0-9]+)',chn)
        #    assert len(r.groups() ) == 2
        #    ind = int(r.groups()[0])
        #    subind = int(r.groups()[1])
        #    name = '{}_c{}'.format(coord_labels_corresp_coord[ind], subind)
        #else:
        #    assert isinstance(coord_labels_corresp_coord, dict) and keyorder is not None
        #    r = re.match('.+src._([0-9]+)_([0-9]+)_c([0-9]+)',chn)
        #    assert len(r.groups() ) == 3
        #    srcgroup_ind = int(r.groups()[0])
        #    ind = int(r.groups()[1])
        #    subind = int(r.groups()[2])
        #    key = keyorder[srcgroup_ind]
        #    name = '{}_c{}'.format(coord_labels_corresp_coord[key ][ind], subind)


    return name

def nicenMEGsrc_chnames(chns, roi_labels=None, keyorder=None, prefix = '', allow_empty=False):
    assert isinstance(roi_labels,dict)
    # nor MEGsrc chnames left unchanged
    if len(chns):
        assert isinstance(chns[0],str)
    else:
        return []
    chns_nicened = [0] * len(chns)
    for i,chn in enumerate(chns):
        chn_nice = chn
        if chn is None:
            if not allow_empty:
                raise ValueError('Found None!')
        else:
            fi = chn.find('src')
            if fi in [0,1]:   # we only accept .?src   kind of chnames here
                chn_nice = getMEGsrc_chname_nice(chn, roi_labels, keyorder)
                chn_nice = prefix + chn_nice
        chns_nicened[i] = chn_nice

    return chns_nicened

def nicenFeatNames(feat_names, roi_labels, keyorder):
    single = False
    if isinstance(feat_names, str):
        feat_names = [feat_names]
    r = []
    for fi,feat_name in enumerate(feat_names):
        fr = nicenFeatName(feat_name,roi_labels,keyorder)
        r += [fr]

    assert len(r) == len(feat_names)

    if single:
        return r[0]
    else:
        return r

def nicenFeatName(feat_name, roi_labels, keyorder):
    assert isinstance(roi_labels,dict)
    import re
    p = 'msrc._[0-9]+_[0-9]+_c[0-9]+'
    #p2 = 'msrc._[0-9]\+_[0-9]\+_c[0-9]\+'
    #r = re.match(p, feat_name)
    #print(r)
    source_chns = re.findall(p, feat_name)
    #print(source_chns)
    tmp = list(keyorder) * 10  # because we have 9 there
    nice_chns = nicenMEGsrc_chnames(source_chns,roi_labels,tmp,
                            prefix='msrc_')
    for src_chni,src_chn in enumerate(source_chns):
        feat_name = feat_name.replace(src_chn,nice_chns[src_chni], 1)

    return feat_name

def getMEGsrc_contralatSide(chn):
    assert isinstance(chn,str)
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
def prepFreqs(min_freq = 3, max_freq = 400, min_freqres=2,
              gamma_bound = 30, HFO_bound = 90, frmult_scale=2*np.pi / 5,
              freqres = [1 , 2 ,4 ], frmults = [1, 1/2, 1/4] , sfreq=None, separate_beta=True ):
    # gamma_bound is just bound when we change freq res to a coarser one
    # by default we have differnt freq steps for different bands and also different
    # window sizes (rusulting from different ratios of n_cycles and freq)
    # if we set all frmults to be equal, we get constant window size
    # (dpss windos would be of sfreq size then)

    pi5 = 2*np.pi / 5    # 5 is used in MNE as multiplier of std of gaussian used for Wavelet

    assert max_freq > gamma_bound, 'we want {} > {} '.format(max_freq, gamma_bound)
    assert min_freq < gamma_bound, 'we want {} < {} '.format(min_freq, gamma_bound)
    if separate_beta:
        b0,b1 = gv.fbands['beta']
        assert gamma_bound >= b1
        assert min_freq <= b0
        # duplicate 1
        freqres = freqres[0], freqres[1], freqres[1], freqres[2]
        frmults = frmults[0], frmults[1], frmults[1], frmults[2]
        fbands = [ [min_freq, b0], [b0,gamma_bound], [gamma_bound,HFO_bound], [HFO_bound,max_freq]  ]
    else:
        fbands = [ [min_freq,gamma_bound], [gamma_bound,HFO_bound], [HFO_bound,max_freq]  ]

    #freqres = np.array( [ 1, 2, 4  ] ) * min_freqres
    #frmults = frmult_scale * np.array( [pi5, pi5/2, pi5/4] )
    if max_freq <= HFO_bound:
        cutoff = 2
        if separate_beta:
            cutoff = 3
        fbands = fbands  [:cutoff]
        freqres = freqres[:cutoff]
        frmults = frmults[:cutoff]

    print(frmults)

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

    if sfreq is not None:
        Ws = mne.time_frequency.tfr._make_dpss(sfreq, freqs, n_cycles=n_cycles,
                                            time_bandwidth=4, zero_mean=False)
        lens = []
        for ww in Ws:
            for w in ww:
                lens += [len(w)]
        ulens = np.unique(lens)
        ulensl = len( ulens )
        if ulensl > 1:
            print('prepFreqs warning: found >1 windows size ',ulens)
        wsz_max = np.max(ulens)
    else:
        Ws = None
        wsz = None

    return freqs, n_cycles, Ws, wsz_max

def tfr(dat, sfreq, freqs, n_cycles, wsz, decim=1, n_jobs = None, mode ='valid'):
    '''
    takes 2-dim array dat nchans x nbins
    wsz is determined inside mne but I don't want to rewrite too much of their code
    returns complex output
    '''
    import multiprocessing as mpr
    import mne
    from globvars import gp
    if n_jobs is None:
        n_jobs = max(1, mpr.cpu_count() - gp.n_free_cores )
    elif n_jobs == -1:
        n_jobs = mpr.cpu_count()

    assert mode in ['valid','same','untouched']

    if dat.ndim == 1:
        dat = dat[None,:]
    #dat has shape n_chans x ntimes // decim
    #returns n_chans x freqs x dat.shape[-1] // decim
    assert dat.ndim == 2
    assert len(freqs) == len(n_cycles)
    if abs(sfreq - int(sfreq) ) > 1e-5:
        raise ValueError('Integer sfreq is required')
    sfreq = int(sfreq)

    #if n_jobs is None:
    # this does not work with CUDA :(
    #    if mne.utils.get_config('MNE_USE_CUDA') and gv.CUDA_state == 'ok':
    #        n_jobs = 'cuda'

    dat_ = dat[None,:]
    #tfrres = mne.time_frequency.tfr_array_morlet(dat_, sfreq, freqs, n_cycles,
    #                                             n_jobs=n_jobs, decim =decim)
    tfrres = mne.time_frequency.tfr_array_multitaper(dat_, sfreq, freqs, n_cycles,
                                                 n_jobs=n_jobs, decim =decim)
    tfrres = tfrres[0]
    # dpss window len = len(np.arange(0, n_cycles_cur/freq_cur, 1/sfreq) )

    # when we call cwt_gen_, we use in _compute_tfr we use mode == 'same'
    # calls _compute_tfr in tfr.py
    # calls _make_dpss
    #  t_win =  (n_cycles / freqs ) [i]
    #   t = np.arange(0, t_win, 1/sfreq)
    # _time_frequency_loop on preapred windows
    # n_times_out = X[:, decim_slice].shape[1],  where decim_slice = (None,None,int(decim) )
    # n_times = X.shape[1],  where decim_slice = (None,None,int(decim) )
    #    tapers, conc = dpss_windows(t.shape[0], half_nbw=time_bandwidth / 2.,
    #                                n_taps)
    #            #  dpss = scipy.signal.windows.dpss(N, half_nbw, Kmax)
    #    nfft = n_times + max_size - 1  (max_size is the max length of the
    #       windows) and then nfft = next_fast_len(nfft)  # 2 ** int(np.ceil(np.log2(nfft)))
    # and  epoch_ind,tfr = _cwt_gen(dta, W, fize=nfft)
    #   fft_Ws[i] = fft(W, fsize=nfft)
    #   ret = ifft(fft_x * fft_Ws[ii])[:n_times + W.size - 1]
    #   ret = _centered(ret[:,:,:decim] , data.shape )  # before decimation
    # and _centered  does     startind = (currsize - newsize) // 2
    # endind = startind + newsize
    # myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    # return arr[tuple(myslice)]

    #startind = (n_times + W.size - 1 - len(data) ) // 2
    #endind = startind + len(data)
    #and then we decimate

    # !!!!!!!! not that we have different relations between n_cycles
    # and freqs for two sets of freqs. So windows sizes will be different.
    # for lower freqs we have larger windows. But I'm ok for them being not
    # exactly aligned with high freq windows -- anyway machine learning later
    # will smear everything
    nbins_orig = dat.shape[-1]
    if mode == 'same':
        # in 'full' the first entries would be (index in convolution array + 1) = length of
        # intersection of the signals (=num of elemens of main array used)
        #wnd0 = [0, (wsz-1)//2 + 1]   # right boundary is not included, like [a,b)
        #wnd1 = [0, (wsz-1)//2 + decim + 1]   # right boundary is not included, like [a,b)
        #wnd2 = [0, (wsz-1)//2 + decim * 2 + 1]   # right boundary is not included, like [a,b)
        ##... until  (wsz-1)//2 + decim * 2 + 1 becomes >= wsz

        #wnd_prelast = [nbins_orig-(wsz-1)//2-decim , nbins_orig]
        #wnd_last = [nbins_orig-(wsz-1)//2 , nbins_orig]
        #end = (wsz-1)//2 + nbins_orig

        strt = (wsz - 1) // 2
        window_boundaries_st =  np.arange(strt - wsz + 1,nbins_orig, decim ) # we start from zero if wsz divide 2 and decim well
        #window_boundaries_st = np.maximum( window_boundaries_st - wsz, 0)
        window_boundaries_end = window_boundaries_st + wsz
        window_boundaries_end = np.minimum( window_boundaries_st, nbins_orig)
        window_boundaries = np.vstack( [ window_boundaries_st, window_boundaries_end] )
    elif mode == 'valid':
        # first index of window that has all bins valid
        #first_ind = (wsz // 2) // decim
        #last_offset = (wsz // 2) // decim
        #tfrres = tfrres[:,:,first_ind:-last_offset]
        # window_boundaries_st =  np.arange(0,nbins_orig - wsz, decim ) # we start from zero if wsz divide 2 and decim well
        # July 5:   actully first_ind = 0 and last_offset should be untouched as well
        #first_ind = 0

        first_ind = (wsz // 2) // decim
        last_offset = (wsz // 2) // decim
        tfrres = tfrres[:,:,first_ind:-last_offset]

        window_boundaries_st =  np.arange(0,nbins_orig, decim  ) # we start from zero if wsz divide 2 and decim well
        window_boundaries_end = window_boundaries_st + wsz
        window_boundaries = np.vstack( [ window_boundaries_st, window_boundaries_end] )

        window_boundaries = window_boundaries[:, :tfrres.shape[-1]  ]

        #end_cutoff = tfrres.shape[-1] - (wsz * 2) // decim + 1
        #tfrres =
        #window_boundaries

        #print( last_offset, tfrres.shape )
    elif mode == 'untouched':
        # in fact it should be shited by one bin right, but perhaps I'll ignore it
        window_boundaries_st =  np.arange(0,nbins_orig, decim  ) # we start from zero if wsz divide 2 and decim well
        window_boundaries_end = window_boundaries_st + wsz
        window_boundaries = np.vstack( [ window_boundaries_st, window_boundaries_end] )
    else:
        raise ValueError("Not implemented")

    if mode != 'untouched':
        assert window_boundaries.shape[-1] == tfrres.shape[-1], (window_boundaries.shape[-1], tfrres.shape[-1] )

    return tfrres, window_boundaries

#def getTFRWindows(datlen, sfreq, decim, mode, wsz):


# stolen from MNE so that I can explicitly control window locations
def _compute_tfr(epoch_data, freqs, sfreq=1.0, method='morlet',
                 n_cycles=7.0, zero_mean=None, time_bandwidth=None,
                 use_fft=True, decim=1, output='complex', n_jobs=1,
                 verbose=None, convolve_mode='same'):
    """Compute time-frequency transforms.
    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    freqs : array-like of floats, shape (n_freqs)
        The frequencies.
    sfreq : float | int, default 1.0
        Sampling frequency of the data.
    method : 'multitaper' | 'morlet', default 'morlet'
        The time-frequency method. 'morlet' convolves a Morlet wavelet.
        'multitaper' uses complex exponentials windowed with multiple DPSS
        tapers.
    n_cycles : float | array of float, default 7.0
        Number of cycles in the wavelet. Fixed number
        or one per frequency.
    zero_mean : bool | None, default None
        None means True for method='multitaper' and False for method='morlet'.
        If True, make sure the wavelets have a mean of zero.
    time_bandwidth : float, default None
        If None and method=multitaper, will be set to 4.0 (3 tapers).
        Time x (Full) Bandwidth product. Only applies if
        method == 'multitaper'. The number of good tapers (low-bias) is
        chosen automatically based on this to equal floor(time_bandwidth - 1).
    use_fft : bool, default True
        Use the FFT for convolutions or not.
    decim : int | slice, default 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].
        .. note::
            Decimation may create aliasing artifacts, yet decimation
            is done after the convolutions.
    output : str, default 'complex'
        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.
    %(n_jobs)s
        The number of epochs to process at the same time. The parallelization
        is implemented across channels.
    %(verbose)s
    Returns
    -------
    out : array
        Time frequency transform of epoch_data. If output is in ['complex',
        'phase', 'power'], then shape of out is (n_epochs, n_chans, n_freqs,
        n_times), else it is (n_chans, n_freqs, n_times). If output is
        'avg_power_itc', the real values code for 'avg_power' and the
        imaginary values code for the 'itc': out = avg_power + i * itc
    """
    from mne.time_frequency.tfr import _check_tfr_param
    from mne.time_frequency.tfr import _check_decim
    from mne.time_frequency.tfr import _make_dpss
    from mne.time_frequency.tfr import morlet
    from mne.time_frequency.tfr import _get_nfft
    from mne.time_frequency.tfr import _time_frequency_loop
    from mne import parallel_func
    # Check data
    epoch_data = np.asarray(epoch_data)
    if epoch_data.ndim != 3:
        raise ValueError('epoch_data must be of shape (n_epochs, n_chans, '
                         'n_times), got %s' % (epoch_data.shape,))

    # Check params
    freqs, sfreq, zero_mean, n_cycles, time_bandwidth, decim = \
        _check_tfr_param(freqs, sfreq, method, zero_mean, n_cycles,
                         time_bandwidth, use_fft, decim, output)

    decim = _check_decim(decim)
    if (freqs > sfreq / 2.).any():
        raise ValueError('Cannot compute freq above Nyquist freq of the data '
                         '(%0.1f Hz), got %0.1f Hz'
                         % (sfreq / 2., freqs.max()))

    # We decimate *after* decomposition, so we need to create our kernels
    # for the original sfreq
    if method == 'morlet':
        W = morlet(sfreq, freqs, n_cycles=n_cycles, zero_mean=zero_mean)
        Ws = [W]  # to have same dimensionality as the 'multitaper' case

    elif method == 'multitaper':
        Ws = _make_dpss(sfreq, freqs, n_cycles=n_cycles,
                        time_bandwidth=time_bandwidth, zero_mean=zero_mean)

    # Check wavelets
    if len(Ws[0][0]) > epoch_data.shape[2]:
        raise ValueError('At least one of the wavelets is longer than the '
                         'signal. Use a longer signal or shorter wavelets.')

    # Initialize output
    n_freqs = len(freqs)
    n_epochs, n_chans, n_times = epoch_data[:, :, decim].shape
    if output in ('power', 'phase', 'avg_power', 'itc'):
        dtype = np.float64
    elif output in ('complex', 'avg_power_itc'):
        # avg_power_itc is stored as power + 1i * itc to keep a
        # simple dimensionality
        dtype = np.complex128

    if ('avg_' in output) or ('itc' in output):
        out = np.empty((n_chans, n_freqs, n_times), dtype)
    else:
        out = np.empty((n_chans, n_epochs, n_freqs, n_times), dtype)

    # Parallel computation
    all_Ws = sum([list(W) for W in Ws], list())
    _get_nfft(all_Ws, epoch_data, use_fft)
    parallel, my_cwt, _ = parallel_func(_time_frequency_loop, n_jobs)

    # Parallelization is applied across channels.
    tfrs = parallel(
        my_cwt(channel, Ws, output, use_fft, convolve_mode, decim)
        for channel in epoch_data.transpose(1, 0, 2))

    # FIXME: to avoid overheads we should use np.array_split()
    for channel_idx, tfr in enumerate(tfrs):
        out[channel_idx] = tfr

    if ('avg_' not in output) and ('itc' not in output):
        # This is to enforce that the first dimension is for epochs
        out = out.transpose(1, 0, 2, 3)
    return out



############################# Tremor-related

def getIntervals(bininds,width=100,thr=0.1, percentthr=0.8,inc=5, minlen=50,
        extFactorL = 0.25, extFactorR  = 0.25, endbin = None, cvl=None,
                 percent_check_window_width=256, min_dist_between = 100,
                 include_short_spikes=0, printLog = 0, minlen_ext=None):
    '''
    bininds either indices of bins where interesting happens OR the mask
    width -- number of bins for the averaging filter (only matters for convolution computation and shiftL,shiftR)
    tremini -- indices of timebins, where tremor was detected
    thr -- thershold for convolution to be larger then, for L\infty-normalized data
        (works together with width arg). Max sensitive thr is 1/width
    output -- convolution, intervals (pairs of timebin indices)
    inc -- how much we extend the interval each time (larger means larger intervals, but worse boundaries)
    minlen -- minimum number of bins required to make the interval be included
    minlen_ext -- same as minlen, but after extFactors were applied
    percentthr -- min ratio of thr crossings within the window to continue extending the interval
    extFactor[R,L] -- we'll extend found intervals by width * extFactor (negative allowed)
    endbin -- max timebin

    returns cvlskip and pairs of bins

    [untested] to extend intervals I could raise inc, lower percentthr, change extFactor
    '''

    if cvl is None and (bininds is None or len(bininds) == 0 ):
        return np.array([]),[]

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
        mt = len(cvl)

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

        newp = (  max(0, leftEnd-shiftL), min(mt-1, rightEnd+shiftR) )
        if rightEnd-leftEnd >= minlen and (newp[1]-newp[0]) >= minlen_ext and success:
            newp = (  max(0, leftEnd-shiftL), min(mt-1, rightEnd+shiftR) ) # extend the interval on both sides
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

def fillBinsFromAnns(anns,  duration, sfreq=256,
                     descr_to_skip=['notrem_L', 'notrem_R'], printLog=0):
    # returns bin mask, where everything that belongs to at least one of the
    # annotations from the list (except descr_to_skip), is marked with 1
    if isinstance(anns,mne.Annotations):
        anns = [anns]

    nbins = int(duration * sfreq)
    binmask = np.zeros(nbins)

    for anns_cur in anns:
        ivalis = ann2ivalDict(anns_cur)
        for ann_type in ivalis:
            if ann_type in descr_to_skip:
                continue
            for start,end,it in ivalis[ann_type]:
                if end >= duration:
                    print('fillBinsFromAnns: Warning: one of anns ({},{},{}) is beyond duration boundary {}'.\
                            format(start,end,it,duration) )
                if printLog:
                    print('fillBinsFromAnns :',start,end)
                sl = slice(int(start*sfreq),int(end*sfreq),None )
                binmask[sl] = 1
    return binmask

def mergeAnns(anns, duration, sfreq=256, descr_to_skip=['notrem_L', 'notrem_R'],
             out_descr = 'merged'):
    '''
    merges all annotaions into one, skipping descr_to_skip
    '''
    if isinstance(anns,mne.Annotations):
        anns = [anns]
    if len(anns) == 0:
        return mne.Annotations([],[],[])

    bins = fillBinsFromAnns(anns, duration, sfreq, descr_to_skip)
    cvlskip,pairs = getIntervals(bininds=None,width=0,
                                       percentthr=1,inc=1,minlen=2,cvl=bins,
                   min_dist_between=1,include_short_spikes=1)

    ivals = []
    for s,e in pairs:
        ss,ee = s/sfreq,e/sfreq
        ivals += [(ss,ee,out_descr)]

    return intervals2anns(ivals)



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

def mergeTremorIntervals(pairs1, pairs2, mode='intersect'): #, only_merge_same_type = True):
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
    res_pairs = []
    ind_couplings = []
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
                if a1<=a2 and b1 <= b2:
                    intersect_type = '1_left_from_2'
                elif a2<=a1 and b2<=b1:
                    intersect_type = '2_left_from_1'
                elif a2<=a1 and b2>=b1:
                    intersect_type = '1_inside_2'
                elif a1<=a2 and b1>=b2:
                    intersect_type = '2_inside_1'
                else:
                    raise ValueError('uncounted :( {} {}'.format(p1,p2) )

                ind_couplings += [ (i1,i2,intersect_type) ]

    #for i1,i2,intersect_type in ind_couplings:

    for i1, p1 in enumerate(pairs1):
        (a1,b1) = p1[0],p1[1]
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
                ind_couplings += [ (i1,i2) ]


                if mode == 'intersect':
                    newp = (max(a1,a2), min(b1,b2) )
                elif mode == 'union':
                    newp = (min(a1,a2), max(b1,b2) )
                res_pairs += [ newp ]

                #mask[i1] = 0
                #mask[i2] = 0  # we mark those who participated in merging
                mask2[i2] = 0
                break

        #res_pairs += [ p for i,p in enumerate(pairs) if mask[i] == 1 ]  # add non-merged

    result = []

    if mode == 'union':
        result = mergeIntervalsWithinList(res_pairs,pairs1,pairs2)
    elif mode == 'intersect':
        result = res_pairs
    else:
        raise ValueError('wrong mode')

    return result

def mergeIntervalsWithinList(pairs,pairs1=None,pairs2=None, printLog=False):
    '''
    pairs --
    '''
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

def removeAnnsByDescr(anns, anns_descr_to_remove, printLog=True):
    '''
    anns_descr_to_remove -- list of SUBstrings annotation names (NOT regexs)
    decide by using find (so can be both sides of just one)
    full equality is searched for
    '''
    #anns_descr_to_remove = ['endseg', 'initseg', 'middle', 'incPost' ]
    if isinstance(anns_descr_to_remove,str):
        anns_descr_to_remove = [anns_descr_to_remove]
    assert isinstance(anns_descr_to_remove,list)
    anns_upd = anns.copy()

    #anns_descr_fine = []
    #anns_onset_fine = []
    #anns_dur_fine = []
    Nbads = 0
    remtype = []
    for ind in range(len(anns.description))[::-1]:
        wasbad = 0
        # current descr
        cd = anns.description[ind]
        for ann_descr_bad in anns_descr_to_remove:
            # if description contains as a subtring our string from the list
            if cd.find(ann_descr_bad) >= 0:
                wasbad = 1
        if wasbad:
            if printLog:
                print('Removing ',cd)
            anns_upd.delete(ind)
            Nbads += 1
            remtype += [cd]
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
    print('removeAnnsByDescr: Overall with basestr={} removed {} annotations: {}'.
          format( anns_descr_to_remove, Nbads, set(remtype) ) )

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
            ttl = f'{tiname}:{fbname}\n{timin:.1f},  {timax:.1f}  logscale={logscale}'
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
        if ch['ch_name'].find('LFPL') >= 0:
            side = 'left'
        elif ch['ch_name'].find('LFPR') >= 0:
            side = 'right'
        else:
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

def findRawArtifacts(raw , thr_mult = 2.5, thr_use_mean=0, show_max_always=0, data_mod = 'MEG' ):
    '''
    I was initially using it for filtered ([1-100] Hz bandpassed) raw. But maybe it can be used before as well
    '''
    import utils_tSNE as utsne
    import mne

    raw_only_mod = raw.copy()
    if data_mod == 'MEG':
        raw_only_mod.pick_types(meg=True)
        artif_prefix = 'BAD_MEG'
    elif data_mod == 'LFP':
        chns = np.array(raw.ch_names)[ mne.pick_channels_regexp(raw.ch_names,'LFP*') ]
        raw_only_mod.pick_channels(chns)
        artif_prefix = 'BAD_LFP'

    assert len(raw_only_mod.info['bads']) == 0, 'There are bad channels!'

    #chnames_perside_mod, chis_perside_mod = collectChnamesBySide(raw.info)
    chnames_perside_mod, chis_perside_mod = collectChnamesBySide(raw_only_mod.info)

    fig,axs = plt.subplots(2,1, figsize=(14,7), sharex='col')

    sides = sorted(chnames_perside_mod.keys())
    anns = mne.Annotations([],[],[])
    cvl_per_side = {}
    for sidei,side in enumerate(sides ):
        chnames_curside = chnames_perside_mod[side]
        moddat, times = raw_only_mod[chnames_curside]
        #moddat = raw_only_mod.get_data()
        me, mn,mx = utsne.robustMean(moddat, axis=1, per_dim =1, ret_aux=1, q = .25)
        if np.min(mx-mn) <= 1e-15:
            raise ValueError('mx == mn for side {}'.format(side) )
        moddat_scaled = ( moddat - me[:,None] ) / (mx-mn)[:,None]
        moddat_sum = np.sum(np.abs(moddat_scaled),axis=0)
        me_s, mn_s,mx_s = utsne.robustMean(moddat_sum, axis=None, per_dim =1, ret_aux=1, pos = 1)

        if thr_use_mean:
            moddat_sum_mod = moddat_sum/ me_s
        else:
            moddat_sum_mod = moddat_sum/ mx_s
        mask= moddat_sum_mod > thr_mult
        cvl,ivals_mod_artif = getIntervals(np.where(mask)[0] ,\
            include_short_spikes=1, endbin=len(mask), minlen=2, thr=0.001, inc=1,\
            extFactorL=0.1, extFactorR=0.1 )
        cvl_per_side[side] = cvl

        print('{} artifact intervals found (bins) {}'.format(data_mod ,ivals_mod_artif) )
        #import ipdb; ipdb.set_trace()

        ax = axs[sidei]
        ax.plot(raw.times,moddat_sum_mod)
        #ax.axhline( me_s , c='r', ls=':', label='mean_s')
        #ax.axhline( me_s * thr_mult , c='r', ls='--', label = 'mean_s * thr_mult' )
        ax.axhline( thr_mult , c='r', ls='--', label = 'mean_s * thr_mult' )

        if show_max_always or not thr_use_mean:
            #ax.axhline( mx_s , c='purple', ls=':', label='max_s')
            #ax.axhline( mx_s * thr_mult , c='purple', ls='--', label = 'mx_s * thr_mult')
            ax.axhline( mx_s / me_s * thr_mult , c='purple', ls='--', label = 'mx_s * thr_mult')
        ax.set_title('{} {} artif'.format(data_mod,side) )

        for ivl in ivals_mod_artif:
            b0,b1 = ivl
            #b0t,b1t = raw.times[b0], raw.times[b1]
            #anns.append([b0t],[b1t-b0t], ['BAD_MEG{}'.format( side[0].upper() ) ]  )
            ax.axvline( raw.times[b0] , c='r', ls=':')
            ax.axvline( raw.times[b1] , c='r', ls=':')

        anns = intervals2anns(ivals_mod_artif,  '{}{}'.format(artif_prefix, side[0].upper() ), raw.times )

        ax.set_xlim(raw.times[0], raw.times[-1]  )
        ax.legend(loc='upper right')

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

# Old and wrong
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

def removeOpsideAnns(anns,main_side):
    ms_letter = main_side[0].upper()
    badstr = '_' + getOppositeSideStr(ms_letter)
    anns_upd = removeAnnsByDescr(anns, [badstr],printLog=0)
    return anns_upd

def getEMGside(chname):
    from globvars import gp
    corresp =gp.EMG_per_hand_base
    for side in  corresp:
        for chn_base in corresp[side]:
            if chname.startswith(  chn_base):
                return side
    return None

def getEMGperHand(rectconvraw):
    import globvars as gv
    EMG_per_hand = gv.EMG_per_hand
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

    return getRawPerSide( raw_lfponly, templ_str = 'LFP', remove_anns = 'ipsi', key=key)

    #EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
#    import mne
#    raw_lfponly.load_data()
#
#    raws_lfp_perside = {}
#    for side in ['left', 'right' ]:
#        sidelet = side[0].upper()
#        if key == 'str':
#            sidekey = side
#        elif key == 'letter':
#            sidekey = sidelet
#        raws_lfp_perside[sidekey] = raw_lfponly.copy()
#        chis = mne.pick_channels_regexp(raw_lfponly.ch_names, 'LFP{}.*'.format(sidelet))
#        chnames_lfp = [raw_lfponly.ch_names[chi] for chi in chis]
#        raws_lfp_perside[sidekey].pick_channels(   chnames_lfp  )
#
#    for sidekey in raws_lfp_perside:
#        # first we remove tremor annotations from other side. Since it is
#        # brain, other side means ipsilateral
#        sidelet = sidekey[0].upper()
#        badstr = '_' + sidelet
#        anns_upd = removeAnnsByDescr(raws_lfp_perside[sidekey].annotations, [badstr])
#
#        # now we remove artifcats for other side of the brain
#        badstr = 'LFP' + getOppositeSideStr(sidelet)
#        anns_upd = removeAnnsByDescr(anns_upd, [badstr])
#
#        # set result
#        raws_lfp_perside[sidekey].set_annotations(anns_upd)
#
#    return  raws_lfp_perside

def getRawPerSide(raw, templ_str = 'msrc', key='letter', remove_anns = 'ipsi',
                  switch_sides = False,
                  switch_sides_chinfo = False):
    '''
    Also filter annotations
    if sort_sides is False, just seelcts using regexp
    returns dict
    '''

    if switch_sides_chinfo:
        assert switch_sides
    # note that I should NOT remove ipsi MEGartif annotations if I use msrc with ipsilateral CB

    #r = raws_lfp_perside[side]
    #info = copy.deepcopy(r.info)
    #for chni,chinfo in enumerate(info['chs'] ):
    #    sideind = 3
    #    sidelet = info['name'][sideind]
    #    opsidelet = GetOppositeSideStr(sidelet)
    #    info['name'].replace('LFP{}'.format(sidelet), 'LFP{}'.format(opsidelet) )
    #mne.RawArray(d.data, r.info )

    #EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
    import mne, copy
    raw.load_data()

    #templ_str = 'LFP'
    #templ_str = 'msrc'

    raws_perside = {}
    for side in ['left', 'right' ]:
        sidelet = side[0].upper()
        if key == 'str':
            sidekey = side
        elif key == 'letter':
            sidekey = sidelet
        r = raw.copy()
        chis = mne.pick_channels_regexp(raw.ch_names, '{}{}.*'.format(templ_str,sidelet))
        chnames = [raw.ch_names[chi] for chi in chis]
        r.pick_channels(   chnames  )

        if switch_sides_chinfo:
            #opsidelet = getOppositeSideStr(sidelet)
            #info = copy.deepcopy( r.info )
            #newchns = []
            #for chni in range(len(info['chs']) ):
            #    oldchn = info['chs'][chni]['ch_name']
            #    if oldchn.find('msrc') == 0:
            #        info['chs'][chni]['loc'][0] = -info['chs'][chni]['loc'][0]
            #    newchn = oldchn
            #    newchn.replace(templ_str + sidelet , templ_str + opsidelet)
            #    info['chs'][chni]['ch_name'] = newchn
            #rnew = mne.RawArray(r.get_data() ,info)
            rnew = changeRawInfoSides(r)
        else:
            rnew = r

        if switch_sides:
            sidekey = getOppositeSideStr(sidekey)
        raws_perside[sidekey] = rnew

    for sidekey in raws_perside:
        # first we remove tremor annotations from other side. Since it is
        # brain, other side means ipsilateral
        if remove_anns == 'ipsi':
            sidelet = sidekey[0].upper()
        elif remove_anns == 'contra':
            sidelet = getOppositeSideStr(sidekey)[0].upper()
        else:
            continue
        badstr = '_' + sidelet
        anns_upd = removeAnnsByDescr(raws_perside[sidekey].annotations, [badstr])

        # now we remove artifcats for other side of the brain
        badstr = templ_str + getOppositeSideStr(sidelet)
        anns_upd = removeAnnsByDescr(anns_upd, [badstr])

        # set result
        raws_perside[sidekey].set_annotations(anns_upd)

    return  raws_perside

def changeRawInfoSides(raw,roi_labels=None, srcgrouping_names_sorted=None):
    '''
    it does a bit more than just renaming channels
    '''
    import copy
    info = copy.deepcopy( raw.info )

    #EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
    #EMG_per_hand_base = {'right':['EMG061', 'EMG062'], 'left':['EMG063', 'EMG064' ] }

    newchns = []
    for chni in range(len(raw.info['chs']) ):
        oldchn = raw.info['chs'][chni]['ch_name']
        newchn = oldchn[:]
        if oldchn.find('msrc') == 0:
            assert roi_labels is not None
            info['chs'][chni]['loc'][0] = -raw.info['chs'][chni]['loc'][0]
            templ_str = 'msrc'
            #sidelet = oldchn[ len(templ_str)]

            sidelet,srcgroup_ind, ind, subind = parseMEGsrcChnameShort(oldchn)
            opsidelet = getOppositeSideStr(sidelet)

            rl = roi_labels[srcgrouping_names_sorted[srcgroup_ind] ]
            lab = rl[ind]
            lab_l = list(lab)
            assert lab_l[-1] == sidelet
            lab_l[-1] = opsidelet
            oplab = ''.join(lab_l)
            indop = rl.index(oplab)

            newchn = genMEGsrcChnameShort(opsidelet, srcgroup_ind, indop, subind)

            #newchn = newchn.replace(templ_str + sidelet , templ_str + opsidelet)
        elif oldchn.find('MEG') == 0:
            # TODO probably there are smarter ways of doing it, changing more
            # things than just location. Maybe reassigning names as well
            # somehow
            info['chs'][chni]['loc'][0] = -raw.info['chs'][chni]['loc'][0]
            templ = None
            print('changeRawInfoSides Warning: too naive change of sides for ',oldchn)
        elif oldchn.find('LFP') == 0:  # use that left and right channel have names only dimmering by one letter
            templ_str = 'LFP'
            sidelet = oldchn[ len(templ_str)]
            opsidelet = getOppositeSideStr(sidelet)
            newchn =newchn.replace(templ_str + sidelet , templ_str + opsidelet)
        elif oldchn.find('EOG') == 0:
            print('changeRawInfoSides Warning: not changing side for ',oldchn)
        elif oldchn.find('EMG') == 0:
            for side in gv.EMG_per_hand_base:
                opside = getOppositeSideStr(side)
                for chnbasei,chnbase in enumerate(gv.EMG_per_hand_base[side]):
                    if oldchn.find(chnbase) >= 0:
                        newchnbase = gv.EMG_per_hand_base[opside][chnbasei]
                        newchn = newchn.replace(chnbase,newchnbase)

        info['chs'][chni]['ch_name'] = newchn
        info['ch_names'][chni] = newchn
        #print('changeRawInfoSides:',oldchn,newchn)

    # do NOT change the data order, only channel names
    rnew = mne.io.RawArray(raw.get_data() ,info)
    return rnew


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

def getIntervalsTotalLens(ann, include_unlabeled = False, times=None, interval=None):
    # interval is bascially crop range
    if isinstance(ann,dict):
        ivalis = ann
    else:
        ivalis = ann2ivalDict(ann)

    if interval is not None:
        assert len(interval) == 2 and max(interval) <= times[-1] \
            and min(interval)>=0 and interval[1] > interval[0]
    if times is not None:
        assert times[-1] > times[0]

    lens = {}
    totlen_labeled = {'_L':0, '_R':0 }
    for it in ivalis:
        lens[it] = 0
        for ival in ivalis[it]:
            a,b,it_ = ival
            if interval is None:
                a_upd = a
                b_upd = b
            else:
                a_upd = max(a,interval[0] )
                a_upd = min(a_upd,b)

                b_upd = min(b,interval[1] )
                b_upd = max(b_upd,a)


            len_cur = b_upd-a_upd
            lens[it] += len_cur
            for sidestr in totlen_labeled:
                if it.find(sidestr) >= 0:
                    totlen_labeled[sidestr] += len_cur

    if include_unlabeled:
        assert times is not None
        begtime = times[0]
        endtime = times[-1]
        if interval is not None:
            begtime = interval[0]
            endtime = interval[1]
        for sidestr in totlen_labeled:
            lens['nolabel'+sidestr] = endtime - begtime - totlen_labeled[sidestr]

    return lens

# take times, dataset bounds and windows bound and annotaions
# output annotations intersecting windows
def getWindowIndicesFromIntervals(wbd,ivalis,sfreq, ret_type='bins_list',
                                  wbd_type='contig', ret_indices_type = 'window_inds',
                                  nbins_total=None, nwins_total=None):
    '''
    wbd are bin indices bounds in original timebins array
    ivalis count from zero, time bounds
    wbd_type = contig means that windows overlap each other and it makes sense to return just bounds
    wbd[1] is right bound, NON INCLUSIVE [a,b)
    return indices of windows
    '''
    # bins_contig means to each interval corresponds one big array of indices
    assert ret_type in ['bins_list', 'bins_contig',
                        'bin_bounds_list', 'time_bounds_list']
    assert wbd_type in ['contig', 'holes_allowed']
    if wbd_type == 'holes_allowed':  # whether wbd are spaced with equal steps
        assert ret_type == 'bins_contig'
        #otherwise I only get windows intersecting beginning and end of the interval
        assert ret_indices_type == 'bin_inds'
    #else:
    #    d = np.diff(wbd[0] )
    #    assert np.max(d) - np.min(d) == 0  # no, that's too much to ask from a
    #    merged wbd

    if isinstance(wbd,list) and isinstance(wbd[0], np.ndarray):
        rs = []
        for subwbd in wbd:
            r = getWindowIndicesFromIntervals(subwbd,ivalis,sfreq,ret_type,wbd_type)
            rs += [r]
        return rs


    if nbins_total is None:
        nbins_total = np.max(wbd)

    wbd = np.minimum(wbd,nbins_total)

    assert ret_indices_type in ['window_inds','bin_inds']
    if ret_indices_type == 'window_inds':
        #assert skip is not None
        assert ret_type in ['bins_list', 'bins_contig', 'bin_bounds_list']

    bins = {}
    bin_bounds = {}
    time_bounds = {}
    for it in ivalis:
        total_bindins_found = 0
        intervals = ivalis[it]
        binscur       = []
        bin_boundscur = []
        time_boundscur = []
        empty = False
        for s,e,it_ in intervals:
            # I don't care if interval is intirely contained
            # in the window or just intersects it
            cond1 = (s*sfreq >= wbd[0,:] ) * (s*sfreq < wbd[1,:])
            cond2 = (e*sfreq >= wbd[0,:] ) * (e*sfreq < wbd[1,:])
            cond = np.logical_or(cond1,cond2)

            # maybe also interval is entirely contained
            #cond3 = (s*sfreq >= wbd[0,:] ) * (e*sfreq < wbd[1,:])
            #cond = np.logical_or(cond,cond3)

            inds = np.where(cond)[0]   # indices of windows
            total_bindins_found += len(inds)
            #print(it,s,e, len(inds) )

            if len(inds) == 0:
                print('getWindowIndicesFromIntervals: EMPTY {}, start {}, end {}'.format(it_,s,e))
                raise ValueError(f's,e,it_={s,e,it_}')
                #import pdb; pdb.set_trace()
                empty = True
                continue
            ws,we = inds[0],inds[-1]
            #print(it,'inds = ',list(inds))
            if wbd_type == 'contig':
                if ret_indices_type == 'bin_inds':
                    news,newe = wbd[0,ws],wbd[1,we]   # indices of (original) timebins
                    bin_boundscur += [(news, newe,it)]
                    if ret_type == 'time_bounds_list':
                        time_boundscur += [(news/sfreq, newe/sfreq,it)]
                    binscur += [ list( range(news,newe) ) ]
                else:
                    bin_boundscur += [(ws, we,it)]
                    binscur += [ list( range(ws,we+1) ) ]  # +1 because where includes last
            else:  # if holes are allowed
                if ret_indices_type == 'bin_inds':
                    for ind in inds:
                        news,newe = wbd[0,ind],wbd[1,ind]
                        binscur += [ list(range(news,newe) ) ]
                #else:  # if window inds
                #    binscur += [ list(inds ) ]

        if ret_type == 'bins_contig':
            b = []
            for bb in binscur:
                assert isinstance(bb,list), 'type {}'.format(type(bb) ) # for summation
                b += bb
            binscur = b

        if not empty:
            bins[it] = binscur
            bin_bounds[it] = bin_boundscur
            time_bounds[it] = time_boundscur

        if total_bindins_found == 0:
            print('getWindowIndicesFromIntervals: EMPTY COMPLETELY {}'.format(it_))

    if ret_type == 'time_bounds_list':
        r = time_bounds
    elif ret_type == 'bin_bounds_list':
        r = bin_bounds
    elif ret_type in ['bins_list', 'bins_contig']:
        r = bins
    return r

def setArtifNaN(X, ivalis_artif_tb_indarrays_merged, feat_names, ignore_shape_warning=False, set_val=np.nan, in_place=False):
    '''
    ivalis -- dict, not necessarily containing only aritfacts
    copies input array
    feat_names can be None
    '''
    if feat_names is None:
        mode_all_chans = True
        feat_names = ['*&rAnDoM*&'] * X.shape[1]
    else:
        mode_all_chans = False
        assert isinstance(feat_names[0], str)
    assert X.shape[1] == len(feat_names), (X.shape[1], len(feat_names) )
    assert isinstance(X,np.ndarray)
    if not ignore_shape_warning:
        assert X.shape[0] > X.shape[1]
    assert isinstance(ivalis_artif_tb_indarrays_merged, dict)
    if in_place:
        Xout = X
    else:
        Xout = X.copy()

    #nums =  [0] * len(feat_names)
    num = 0
    for interval_name in ivalis_artif_tb_indarrays_merged:
        r = parseIntervalName(interval_name)
        assert r['interval_type'] == 'artif'
        #templ = '^BAD_(.+)'
        #matchres = re.match(templ,interval_name).groups()
        #assert matchres is not None and len(matchres) > 0
        #artif_type = matchres[0]

        #r['artifact_modality'] = 'LFP'

        mode_MEG_artif = False
        mode_LFP_artif = False
        if   r['artifact_modality'] == 'MEG':
            affected_chn_mod = 'msrc'
            #mode_MEG_artif = True
            #print('MEG',artif_type)
        elif r['artifact_modality'] == 'LFP':
            affected_chn_mod = 'LFP'
        else:
            affected_chn_mod = None
            #mode_LFP_artif = True
            #artif_chn = artif_type
            #print('LFP',artif_type)
        interval_bins = ivalis_artif_tb_indarrays_merged[interval_name]
        num += len(interval_bins)

        brain_side = r['artifact_brain_side']
        # over all features
        for feati,featn in enumerate(feat_names):
            do_set = False
            if mode_all_chans:
                do_set = True
                print('-----------1')
            elif r['artifact_chname'] is not None:
                do_set = featn.find( r['artifact_chname'] ) >= 0
                print('-----------2 ', do_set)
            elif featn.find(affected_chn_mod) >= 0:
                if brain_side is not None:
                    do_set = (featn.find(affected_chn_mod+ brain_side) >= 0 )
                # if no brain_side is found then we should have BAD_LFP
                    #import pdb; pdb.set_trace()
                    print(f'-----------3  {featn} {interval_name} {affected_chn_mod}', do_set, r)
                else:
                    do_set = True
                    print('-----------4')

            print('my imptuer ',interval_name, featn, do_set)
            if do_set:
                Xout[interval_bins,feati] = set_val
                #print('fd')
            print(f'--in {featn} ----- set {num} bins to {set_val}')

    return Xout

def getArtifForFiltering(chn,ann_aritf):
    # creates a new annotation tailored specifically for this channel using different levels
    onsets = []
    durations = []
    descrs = []
    for a in ann_aritf:
        descr = a['description']
        r = re.match('^BAD_(LFP|MEG)(.?)(.*)',descr)
        mod,side,chn_sub_id = r.groups()
        if mod == 'MEG':
            mod_chn = 'msrc'
        else:
            mod_chn = mod
        t = mod_chn +side+ chn_sub_id
        print(r.groups(),descr, t)
        if chn.find( t) >= 0:
            onsets += [a['onset']]
            durations += [a['duration']]
            if mod == 'LFP':
                descrs += [f'BAD_{chn}']
            else:
                descrs += [f'BAD_{mod}{side}']
            print('badom')
    return mne.Annotations(onsets,durations,descrs)

def parseIntervalName(interval_name):
    r = {}

    matchres = re.match('^BAD_(LFP|MEG)(.?)(.*)',interval_name)
    if matchres is not None:
        mod,side,chn_sub_id = matchres.groups()
        r['interval_type'] = 'artif'
        r['artifact_modality'] = mod
        r['artifact_brain_side'] = side
        if (chn_sub_id is not None) and len(chn_sub_id) > 0:
            r['artifact_chname'] = mod+side+chn_sub_id
        else:
            r['artifact_chname'] = None

    #templ = r'^BAD_(LFP.*)'
    #matchres = re.match(templ,interval_name)
    #if matchres is not None:
    #    chn_hint = matchres.groups()[0]
    #    chn = None
    #    side = None
    #    if len(chn_hint) > 3:
    #        side = chn_hint[3]
    #    if len(chn_hint) > 4:
    #        chn = chn_hint

    #    r['interval_type'] = 'artif'
    #    r['artifact_modality'] = 'LFP'
    #    r['artifact_brain_side'] = side
    #    r['artifact_chname'] = chn

    #templ = r'^BAD_(MEG.*)'
    #matchres = re.match(templ,interval_name)
    #if matchres is not None:
    #    chn_hint = matchres.groups()[0]
    #    chn = None
    #    side = None
    #    if len(chn_hint) > 3:
    #        side = chn_hint[3]
    #    if len(chn_hint) > 4:
    #        chn = chn_hint

    #    r['interval_type'] = 'artif'
    #    r['artifact_modality'] = 'LFP'
    #    r['artifact_brain_side'] = side
    #    r['artifact_chname'] = chn

    if len(r) == 0:
        from globvars import gp
        for itp in gp.int_types_basic:
            templ = '^' + itp + '_(.)'
            matchres = re.match(templ,interval_name)
            if matchres is not None:
                side = matchres.groups()[0]
                r['interval_type'] = 'beh_state'
                r['beh_state_type'] = itp
                r['body_side'] = side
                r['brain_side'] = getOppositeSide(side )

    return r


def imputeConstArtif(X, anns_artif, featnames, wbd_merged=None, nbins_total=None, sfreq=256, set_val=0, in_place=False):
    if wbd_merged is None:
        wbd_merged= np.vstack( [np.arange(X.shape[0]),np.arange(X.shape[0]) + 1 ] )
        #print(wbd_merged.shape)
    ivalis_artif = ann2ivalDict(anns_artif)
    ivalis_artif_tb_indarrays_merged = \
        getWindowIndicesFromIntervals(wbd_merged,ivalis_artif,
                                    sfreq,ret_type='bins_contig',
                                    ret_indices_type = 'window_inds',
                                        nbins_total=nbins_total )

    X_imputed  = setArtifNaN(X, ivalis_artif_tb_indarrays_merged, featnames,
                                       ignore_shape_warning=False, set_val =set_val, in_place=in_place)
    return X_imputed

def imputeInterpArtif(X, anns_artif, featnames, wbd_merged=None, nbins_total=None, sfreq=256, in_place=False):
    # X: nbins x nchans
    if wbd_merged is None:
        wbd_merged= np.vstack( [np.arange(X.shape[0]),np.arange(X.shape[0]) + 1 ] )
        #print(wbd_merged.shape)
    ivalis_artif = ann2ivalDict(anns_artif)
    ivalis_artif_tb_indarrays_merged = \
        getWindowIndicesFromIntervals(wbd_merged,ivalis_artif,
                                    sfreq,ret_type='bins_contig',
                                    ret_indices_type = 'window_inds',
                                        nbins_total=nbins_total )

    X_artif_nan  = setArtifNaN(X, ivalis_artif_tb_indarrays_merged, featnames,
                                       ignore_shape_warning=False, in_place=in_place)

    import pandas as pd
    df = pd.DataFrame(X_artif_nan)
    df = df.interpolate(axis=0)
    return df.to_numpy()
    #return df

def indHoleToNoHole(binind, dataset_bounds_uncut_bins, nedge_bins):
    '''
        binind starts from zero
        dataset_bounds_bins for data without cutting of edge bins
    '''
    for db_i,(dbs,dbe) in enumerate(dataset_bounds_uncut_bins):
        if binind >= dbs - nedge_bins and binind < dbe - nedge_bins:
            newind = binind + nedge_bins * (db_i + 1)
            break

    newind = 0
    return newind

def makeRawFromFeats(X, feat_names, skip, sfreq=256, namelen = 15):
    '''
    first dimension of X is time
    '''

    feat_names_to_use = []  #ordered
    pattern = '([a-zA-Z]*)_'
    feat_types_cooresp_dict = {}
    for fni,fn in enumerate(feat_names):
        if len(fn) > namelen:
            r = re.match(pattern,fn)
            feat_type = r.groups()[0]
            newname = '{}_{}'.format(feat_type,fni)
        else:
            newname = fn
        #if feat_type not in feat_types_cooresp_dict:
        #    feat_types_cooresp_dict[feat_type]
        feat_types_cooresp_dict[newname] = fn
        feat_names_to_use += [newname]
    #feat_names_to_use = list(feat_names)

    info = mne.create_info(feat_names_to_use,sfreq/skip)

    print(len(feat_names_to_use))
    r = mne.io.RawArray(X.T,info)
    return r, feat_types_cooresp_dict

def makeSimpleRaw(dat,ch_names=None,sfreq=256,rescale=True,l=10,
                  force_trunc_renum=False, verbose=False, copy=False):
    # does rescaling to one
    assert dat.ndim == 2
    if ch_names is None:
        ch_names = list(  map(str, np.arange( len(dat)) ) )
    elif force_trunc_renum or np.max( [len(chn) for chn in ch_names ] ) > 15:
        ch_names = list(ch_names)[:]
        for chni in range(len(ch_names) ):
            ch_names[chni] = ch_names[chni][:l] + '_{}'.format(chni)

    info_ = mne.create_info( ch_names=list(ch_names), ch_types= ['csd']*len(ch_names), sfreq=sfreq)

    if copy:
        dat = dat.copy()

    if rescale:
        me = np.mean(dat, axis=1)
        std = np.std(dat, axis=1)

        dat = (dat-me[:,None])
        if np.min( np.abs(std) ) >= 1e-19:
            dat /= std[:,None]

    dispraw = mne.io.RawArray(dat,info_,verbose=verbose)
    return dispraw

def prepareSourceGroups(labels,srcgroups_all):
    from globvars import gp
    ############  left and right CB vs left and right rest
    labels_dict = {}
    srcgroups_dict = {}

    srcgroups_dict['all'] = srcgroups_all
    labels_dict['all'] = labels[:]

    srcgroups_dict['all_raw'] = srcgroups_all
    labels_dict['all_raw'] = labels[:]


    #------------  merged
    labels_merged = ['brain_B', 'unlabeled']

    srcgroups_merged = srcgroups_all.copy()
    unlabi = labels_merged.index('unlabeled');
    srcgroups_merged[srcgroups_merged == unlabi] = 1
    srcgroups_merged[srcgroups_merged != unlabi] = 0

    assert len(set(srcgroups_merged) ) == 2, set(srcgroups_merged)

    labels_dict['merged'] = labels_merged
    srcgroups_dict['merged'] = srcgroups_merged

    #------------   merged_by_side

    labels_merged_by_side = ['hemisphere_L',
                            'hemisphere_R', 'unlabeled']
    srcgroups_merged_by_side = srcgroups_all.copy()
    for labi, label in enumerate(labels):
        if label.endswith('_L'):
            srcgroups_merged_by_side[srcgroups_all == labi] = 0
        elif label.endswith('_R'):
            srcgroups_merged_by_side[srcgroups_all == labi] = 1
        else:
            srcgroups_merged_by_side[srcgroups_all == labi] = 2

    assert len(set(srcgroups_merged_by_side) ) == 3

    labels_dict['merged_by_side'] = labels_merged_by_side
    srcgroups_dict['merged_by_side'] = srcgroups_merged_by_side

    #----------------


    #which new labels do we want to have
    # should be understood as (not cerebellum)_L instead of not (cerebellum_L)
    labels_cb_vs_rest = ['Cerebellum_L', 'Cerebellum_R',
                        'notCerebellum_L', 'notCerebellum_R', 'unlabeled']

    srcgroups_cb_vs_rest = srcgroups_all.copy() # copy needed

    # indices of CB in the original original label list
    cbinds = [i for i in range(len(labels) ) if labels[i].find('Cerebellum') >= 0 ]
    assert len(cbinds )  == 2

    # mark those that are not cerebellum
    b = [True] * len(srcgroups_cb_vs_rest)  # first include all
    for i in cbinds:
        b = np.logical_and( srcgroups_cb_vs_rest != i, b)

    #addCBinds = True

    for j in range(len(srcgroups_cb_vs_rest)):
        curi = srcgroups_cb_vs_rest[j]
        if b[j]:  # rename not-cerebellum
            parcel_name = labels[curi]
            if parcel_name == 'unlabeled':
                #get index of this label in the list of new labels
                newlabi = labels_cb_vs_rest.index(parcel_name)
            else:
                sidelet = parcel_name[-1]
                newlabi = labels_cb_vs_rest.index('notCerebellum_' + sidelet)
            srcgroups_cb_vs_rest[srcgroups_all == curi] = newlabi
    #     else if srcgroups[j] in cbinds  and addCBinds:
    #         srcgroups_cb_vs_rest

    set_CB_unlab = False #make sense if we compute Cerbellum for other srcgroups
    # and then want to reuse it -- saves computation time but complicates code

    # look at new cerebellum labels
    for newlabi,newlab in enumerate(labels_cb_vs_rest[:2] ):
        #is it 0 or 1?
        orig_ind = labels.index(newlab)
        #print(orig_ind,newlab,newlabi)
        if set_CB_unlab:
            newlabi = labels_cb_vs_rest.index('unlabeled')
        #srcgroups_cb_vs_rest[srcgroups_all == cbinds[oldind] ] = newlabi  # its on purpose that I use 'old' srcgroups
        # we set cerebellum indices to be 0 or 1 depending whether it is left or right
        srcgroups_cb_vs_rest[srcgroups_all == orig_ind ] = newlabi

    #dispSrcGroupInfo(srcgroups_cb_vs_rest)
    #print(srcgroups_cb_vs_rest)
    labels_dict['CB_vs_rest'] = labels_cb_vs_rest
    srcgroups_dict['CB_vs_rest'] = srcgroups_cb_vs_rest

    #-----------   CBmerged_vs_rest

    srcgroups_cbm_vs_rest = srcgroups_cb_vs_rest.copy()

    lind = labels_cb_vs_rest.index('Cerebellum_L')
    rind = labels_cb_vs_rest.index('Cerebellum_R')
    # here we use that lind == 0
    assert lind == 0 and rind == 1
    srcgroups_cbm_vs_rest[srcgroups_cbm_vs_rest == rind] = lind
    srcgroups_cbm_vs_rest[srcgroups_cbm_vs_rest > 1] -= 1

    labels_cbm_vs_rest = labels_cb_vs_rest[:] #copy
    del labels_cbm_vs_rest[0]
    labels_cbm_vs_rest[0] = 'Cerebellum_B'

    labels_dict['CBmerged_vs_rest'] = labels_cbm_vs_rest
    srcgroups_dict['CBmerged_vs_rest'] = srcgroups_cbm_vs_rest


    #--------------

    motorlike_both_sides = [0] * len(gp.areas_list_aal_my_guess)*2
    for i in range(len(motorlike_both_sides)):
        side_let = ['_L','_R'][i % 2]
        motorlike_both_sides[i] = gp.areas_list_aal_my_guess[ i // 2 ] + side_let
    #motorlike_both_sides


    #-------------  merged my regions and other ones
    labels_my_intuition = ['motor-related_L','motor-related_R',
                                'Cerebellum_L', 'Cerebellum_R',
                                    'rest_L', 'rest_R', 'unlabeled']

    srcgroups_my_intuition = srcgroups_all.copy() # copy needed

    #for labi, label in enumerate(labels):
    for sgi,sg in enumerate(srcgroups_my_intuition):
        orig_label = labels[sg]
        side_letter = orig_label[-1]
        if orig_label != 'unlabeled':
            assert side_letter in ['L','R'], (orig_label)
        if orig_label in motorlike_both_sides:
            if orig_label.startswith('Cerebellum'):
                newlabi = labels_my_intuition.index(orig_label)
            else:
                newlabi = labels_my_intuition.index('motor-related_'+side_letter)
        elif orig_label == 'unlabeled':
            newlabi = labels_my_intuition.index(orig_label)
        else:
            newlabi = labels_my_intuition.index('rest_' + side_letter)
        srcgroups_my_intuition[sgi] = newlabi

    labels_dict['motor-related_vs_CB_vs_rest'] = labels_my_intuition
    srcgroups_dict['motor-related_vs_CB_vs_rest'] = srcgroups_my_intuition

    #-------------  merged my regions and other ones (CB merged)
    labels_my_intuition_cb_merged = ['motor-related_L','motor-related_R',
                                'Cerebellum_B', 'rest_L', 'rest_R', 'unlabeled']

    srcgroups_my_intuition_cb_merged = srcgroups_all.copy() # copy needed

    #for labi, label in enumerate(labels):
    for sgi,sg in enumerate(srcgroups_my_intuition_cb_merged):
        orig_label = labels[sg]
        side_letter = orig_label[-1]
        if orig_label != 'unlabeled':
            assert side_letter in ['L','R'], (orig_label)
        if orig_label in motorlike_both_sides:
            if orig_label.startswith('Cerebellum'):
                newlabi = labels_my_intuition_cb_merged.index('Cerebellum_B')
            else:
                newlabi = labels_my_intuition_cb_merged.index('motor-related_'+side_letter)
        elif orig_label == 'unlabeled':
            newlabi = labels_my_intuition_cb_merged.index(orig_label)
        else:
            newlabi = labels_my_intuition_cb_merged.index('rest_' + side_letter)
        srcgroups_my_intuition_cb_merged[sgi] = newlabi


    labels_dict['motor-related_vs_CBmerged_vs_rest'] = labels_my_intuition_cb_merged
    srcgroups_dict['motor-related_vs_CBmerged_vs_rest'] = srcgroups_my_intuition_cb_merged

    #-------------  merged my regions (including corresponding CB side) and other ones
    labels_my_intuition_incCB = ['motor-related_incCB_L',
                                            'motor-related_incCB_R' ,
                                            'rest_L', 'rest_R',
                                            'unlabeled']

    srcgroups_my_intuition_incCB = srcgroups_all.copy() # copy needed

    #for labi, label in enumerate(labels):
    for sgi,sg in enumerate(srcgroups_my_intuition_incCB):
        orig_label = labels[sg]
        side_letter = orig_label[-1]
        if orig_label != 'unlabeled':
            assert side_letter in ['L','R'], (orig_label)
        if orig_label in motorlike_both_sides:
            newlabi = labels_my_intuition_incCB.index('motor-related_incCB_'+side_letter)
        elif orig_label == 'unlabeled':
            newlabi = labels_my_intuition_incCB.index(orig_label)
        else:
            newlabi = labels_my_intuition_incCB.index('rest_' + side_letter)
        srcgroups_my_intuition_incCB[sgi] = newlabi

    labels_dict['motor-related_incCB_vs_rest'] = labels_my_intuition_incCB
    srcgroups_dict['motor-related_incCB_vs_rest'] = srcgroups_my_intuition_incCB

    #-------------  merged my regions and other ones (all merged across sides)
    labels_my_intuition_merge_across = ['motor-related_B', 'Cerebellum_B', 'rest_B', 'unlabeled']

    srcgroups_my_intuition_merge_across = srcgroups_all.copy() # copy needed

    #for labi, label in enumerate(labels):
    for sgi,sg in enumerate(srcgroups_my_intuition_merge_across):
        orig_label = labels[sg]
        side_letter = orig_label[-1]
        if orig_label != 'unlabeled':
            assert side_letter in ['L','R'], (orig_label)
        if orig_label in motorlike_both_sides:
            if orig_label.startswith('Cerebellum'):
                newlabi = labels_my_intuition_merge_across.index('Cerebellum_B')
            else:
                newlabi = labels_my_intuition_merge_across.index('motor-related_B')
        elif orig_label == 'unlabeled':
            newlabi = labels_my_intuition_merge_across.index(orig_label)
        else:
            newlabi = labels_my_intuition_merge_across.index('rest_B')
        srcgroups_my_intuition_merge_across[sgi] = newlabi

    labels_dict['motor-related_vs_CB_vs_rest_merge_across_sides'] = labels_my_intuition_merge_across
    srcgroups_dict['motor-related_vs_CB_vs_rest_merge_across_sides'] = srcgroups_my_intuition_merge_across

    #-------------  my regions without any merging
    labels_my_intuition_only = motorlike_both_sides + [ 'unlabeled']

    srcgroups_my_intuition_only = srcgroups_all.copy() # copy needed

    #for labi, label in enumerate(labels):
    for sgi,sg in enumerate(srcgroups_my_intuition_only):
        orig_label = labels[sg]
        if orig_label not in motorlike_both_sides:
            newlabi = labels_my_intuition_only.index('unlabeled')
        else:
            newlabi = labels_my_intuition_only.index(orig_label)

        srcgroups_my_intuition_only[sgi] = newlabi

    srcgroups_dict['motor-related_only'] = srcgroups_my_intuition_only
    labels_dict['motor-related_only'] = labels_my_intuition_only

    #TODO my code is not adjusted to working with both-sides sources (i.e. I'll have to assign some side perhaps)
    #------------------
    #TODO: add CBmerged_vs_rest_merged_by_side

    labels_test = ['Supp_Motor_Area_L','Supp_Motor_Area_R', 'Precentral_L', 'Precentral_R', 'unlabeled']
    srcgroups_test = srcgroups_all.copy() # copy needed
    #for labi, label in enumerate(labels):
    for sgi,sg in enumerate(srcgroups_test):
        orig_label = labels[sg]
        if orig_label not in labels_test[:-1]:
            newlabi = labels_test.index('unlabeled')
        else:
            newlabi = labels_test.index(orig_label)

        srcgroups_test[sgi] = newlabi

    srcgroups_dict['test'] = srcgroups_all
    labels_dict['test'] =   labels_test


    assert set(labels_dict.keys() ) == set( gp.src_grouping_names_order )
    return labels_dict, srcgroups_dict

def dispSrcGroupInfo(grps):
    '''
    grps is a list of ints
    '''
    ugrps = np.unique(grps)
    print('min {}, max {}; ulen {}, umin {}, umax {}'.
          format(min(grps), max(grps), len(ugrps), min(ugrps), max(ugrps)) )

def addSideToParcels(parcel_labels_no_side,body_side):
    # accorindg to hemisphere
    res = []
    for p in parcel_labels_no_side:
        assert not (p.endswith('_L') or p.endswith('_R') or p.endswith('_B') )
        if p == 'Cerebellum':
            rp = p + '_' + body_side[0].upper()
        else:
            opside = getOppositeSideStr(body_side)
            rp = p+ '_' + opside[0].upper()

        res += [rp]


def vizGroup(sind_str,positions,labels,srcgroups, show=True, labels_to_skip = ['unlabeled'],
             show_legend=True):
    '''
    visualize src group
    positions -- 3D coords of ALL sources (not just subgroups)
    srcgroups -- for each coord index an index from the list of an item in the labels list
    '''
    import pymatreader as pymr
    import os
    import mayavi.mlab as mam
    import globvars as gv


    #print('    ', sgdn)

    headfile = pymr.read_mat(os.path.join(gv.data_dir,
                                        'headmodel_grid_{}_surf.mat'.format(sind_str)) )
    tris = headfile['hdm']['bnd']['tri'].astype(int)
    rr_mm = headfile['hdm']['bnd']['pos']
    rr_mm -= 1
    mam.triangular_mesh(rr_mm[:,0], rr_mm[:,1], rr_mm[:,2],
                        tris-1, color=(0.5,0.5,0.5),
                        opacity = 0.3, transparent=False)


    # create random colors for various labels
    labels_cur = labels#_dict[sgdn]
    clrs = np.random.uniform(low=0.3,size=(len(labels_cur),3))
    # for clri in range(len(clrs) ):
    #     colcur= clrs[clri]
    #     ii = np.argmin(colcur)
    #     colcur[ii] = max(colcur[ii], 0.7 )

    for grpi in range(len(labels_cur)):
        lab = labels_cur[grpi]
        print(lab)
        #inds = np.where(srcgroups_dict[sgdn] == grpi)
        inds = np.where(srcgroups == grpi)
        if lab in labels_to_skip:
            print('  skipping {} {} points'.format(len(inds),lab) )
            continue
        x,y,z = positions[inds].T
        mam.points3d(x,y,z, scale_factor=0.5, color = tuple(clrs[grpi]) )

    if show_legend:
        import matplotlib as mpl
        legels = []
        s = 12
        for i in range(len(labels)):
            legel = mpl.lines.Line2D([0], [0], marker='o', color='w', label=labels[i],
                                            markerfacecolor=clrs[i], markersize=s)
            legels += [legel]
        plt.legend(handles = legels)

    if show:
        mam.show()
    #mam.close()"
    return clrs

def vizGroup2(sind_str, positions, labels, srcgroups,
              show=True, labels_to_skip = ['unlabeled'],
             show_legend=True, alpha_surf=0.1, seed=None, figsize_mult = 1, msz=15,
              printLog=0, color_grouping = None,
             color_group_labels = None, sizes=None, def_alpha = 0.9,
              msz_mult=0.25, roi_labels_all = None):
    '''
    visualize src group. Slower than vizGroup, but allows to save things normally and legends as well
    positions -- 3D coords of ALL sources (not just subgroups)
    srcgroups -- for each coord index an index from the list of an item in the labels list
    [!] labels -- which labels we actually plot
    '''
    import pymatreader as pymr
    import os
    #import mayavi.mlab as mam
    import globvars as gv
    import matplotlib as mpl

    assert len(srcgroups) == len(positions)
    if color_grouping is not None:
        assert len(color_grouping) == len(labels), (len(color_grouping) , len(labels) )



    nc = 2
    if show_legend:
        nc += 1
    nr = 1
    fig,axs = plt.subplots(nr,nc,
                           figsize=(14*figsize_mult,5*figsize_mult),
                           subplot_kw={'projection':'3d', 'proj_type':'ortho'})
    plt.subplots_adjust(wspace=0.01)
    ax_top  = axs[0]
    ax_side = axs[1]
    if show_legend:
        ax_leg  = axs[-1]


    # if I want to add a 2D plot as well
    #ax__ = axs[2]
    #ax__.plot([0,1],[0,1] )
    #ax__.view_init(0,0)

    #print('    ', sgdn)
    #fig = plt.figure(figsize=(14*figsize_mult,5*figsize_mult))
    #main_shape = 120
    #if show_legend:
    #    main_shape = 130
    #ax_top = fig.add_subplot(main_shape+1,projection='3d', proj_type='ortho')
    #ax_side = fig.add_subplot(main_shape+2,projection='3d', proj_type='ortho')
    #if show_legend:
    #    ax_leg = fig.add_subplot(main_shape+3,projection='3d', proj_type='ortho')


    headfile = pymr.read_mat(os.path.join(gv.data_dir,
                                        'headmodel_grid_{}_surf.mat'.format(sind_str)) )
    tris = headfile['hdm']['bnd']['tri'].astype(int)
    rr_mm = headfile['hdm']['bnd']['pos']
#     mam.triangular_mesh(rr_mm[:,0], rr_mm[:,1], rr_mm[:,2],
#                         tris-1, color=(0.5,0.5,0.5),
#                         opacity = 0.3, transparent=False)
    print(tris.shape)
    #x,y,z = positions[inds].T
    ax_top.view_init(90,-90)
    ax_top.plot_trisurf(rr_mm[:,0], rr_mm[:,1], rr_mm[:,2], triangles=tris-1,
                    alpha=alpha_surf)
    ax_top.set_zticks([])
    #
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    #ax_top.set_top_view()

    ax_side.view_init(0,0)
    ax_side.plot_trisurf(rr_mm[:,0], rr_mm[:,1], rr_mm[:,2], triangles=tris-1,
                    alpha=alpha_surf)
    ax_side.set_xticks([])
    #
    ax_side.set_yticks([])
    ax_side.set_zticks([])
    #ax_side.set_top_view()


    if seed is not None:
        np.random.seed(seed)
    # create random colors for various labels
    labels_cur = labels#_dict[sgdn]

    if color_group_labels is not None:
        clrs = np.random.uniform(low=0.3,size=(len(color_group_labels),3) )
        alphas = def_alpha * np.ones( (len(color_group_labels),1) )

        clrs = np.hstack([clrs,alphas ]   )
    else:
        clrs = np.random.uniform(low=0.3,size=(len(labels_cur),3) )
    # for clri in range(len(clrs) ):
    #     colcur= clrs[clri]
    #     ii = np.argmin(colcur)
    #     colcur[ii] = max(colcur[ii], 0.7 )

    xs = []
    ys = []
    zs = []

    if sizes is not None:
        assert len(sizes) == len(labels)

    msz = msz * figsize_mult
    color_list = []
    marker_sizes = []
    # plot positions acoording to scrgroups
    for grpi,lab in enumerate(labels_cur):
        if printLog:
            print(lab)
        #inds = np.where(srcgroups_dict[sgdn] == grpi)
        # indices in srcgroups

        if roi_labels_all is None:
            group_code = grpi
        else:
            group_code = list(roi_labels_all).index(lab)
        inds = np.where(srcgroups == group_code)
        if lab in labels_to_skip:
            print('  skipping {} {} points'.format(len(inds),lab) )
            continue
        x,y,z = positions[inds].T
        #mam.points3d(x,y,z, scale_factor=0.5, color = tuple(clrs[grpi]) )
        #ax_top.scatter(xs,ys,zs,color=tuple(clrs[grpi]), s=msz)
        #ax_side.scatter(xs,ys,zs,color=tuple(clrs[grpi]), s=msz)
        xs += [x]
        ys += [y]
        zs += [z]

        if sizes is not None:
            marker_sizes +=  [ msz + sizes[grpi]  * (msz * msz_mult) ] * len(z)

        if color_group_labels is None:
            color_list += [  tuple(clrs[grpi]) ] * len(z)
        else:
            cur_col = np.zeros(4)
            for ci in color_grouping[grpi]:
                cur_col += clrs[ ci ]
            cur_col  /=  len(color_grouping[grpi] )
            color_list += [  tuple(cur_col) ] * len(z)
    #if color_group_labels is not None:
    #    color_list = np.array(clrs)[ color_grouping ]
    xs = np.hstack(xs)
    ys = np.hstack(ys)
    zs = np.hstack(zs)

    if sizes is None:
        marker_sizes = [msz] * len(xs)

    #print(color_list)
    #print(marker_sizes)

    ax_top.scatter(xs,ys,zs,color=color_list,  s=marker_sizes)
    ax_side.scatter(xs,ys,zs,color=color_list, s=marker_sizes)

    if show_legend:
        legels = []
        s = 12
        if color_group_labels is None:
            for labi,lab in enumerate(labels):
                if lab in labels_to_skip:
                    continue
                legel = mpl.lines.Line2D([0], [0], marker='o', color='w', label=lab,
                                                markerfacecolor=clrs[labi], markersize=s)
                legels += [legel]
                #print(lab)
        else:
            for i in range(len(color_group_labels)):
                legel = mpl.lines.Line2D([0], [0], marker='o', color='w',
                                         label=color_group_labels[i],
                                         markerfacecolor=clrs[i], markersize=s)
                legels += [legel]
        ax_leg.legend(handles = legels,loc='center', prop={'size':13} )
        ax_leg.set_xticks([])
        ax_leg.set_yticks([])
        ax_leg.set_zticks([])
        ax_leg.view_init(90,-90)
        ax_leg.grid(0)

    return axs, clrs

def parseMEGsrcChnamesShortList(chnames):
    sides = []
    groupis = []
    parcelis = []
    compis = []
    for chn in chnames:
        if chn.startswith('LFP'):
            side, groupi, parceli, compi = chn[3], np.nan, np.nan, np.nan
        else:
            side, groupi, parceli, compi = parseMEGsrcChnameShort(chn)
        sides += [side]
        groupis += [groupi]
        parcelis += [parceli]
        compis += [compi]
    return sides,groupis,parcelis,compis

def genRecInfoFn(rawname_,sources_type,src_file_grouping_ind):
    #if fntype == 'rec_info':
    r = '{}_{}_grp{}_src_rec_info.npz'.format(rawname_,sources_type,src_file_grouping_ind)
    return r

def genPrepDatFn(rawn, new_main_side, data_modalities, use_main_LFP_chan, src_file_grouping_ind,
                 src_grouping):
    fn_suffix_dat = 'dat_{}_newms{}_mainLFP{}_grp{}-{}.npz'.format(','.join(data_modalities),
                                                                   new_main_side[0].upper(),
                                                                use_main_LFP_chan,
                                                                src_file_grouping_ind, src_grouping)

    fname = '{}_'.format(rawn) + fn_suffix_dat
    return fname

def genStatsFn(rawnames,
               new_main_side, data_modalities, use_main_LFP_chan, src_file_grouping_ind,
                 src_grouping, prefix=None):
    fn_suffix_dat = 'dat_{}_newms{}_mainLFP{}_grp{}-{}.npz'.format(','.join(data_modalities),
                                                                   new_main_side[0].upper(),
                                                                use_main_LFP_chan,
                                                                src_file_grouping_ind, src_grouping)

    if rawnames is None and prefix is not None:
        fname_stats = prefix + fn_suffix_dat
    else:
        nr = len(rawnames)
        sind_str_list_sorted = list( sorted( set([rawn[0:3] for rawn in rawnames] ) ))
        inds_str = ','.join(sind_str_list_sorted )
        fname_stats = 'stats_{}_{}_'.format(inds_str,nr)  + fn_suffix_dat
    return fname_stats

def genStatsMultiBandFn(rawnames,
               new_main_side, data_modalities, use_main_LFP_chan, src_file_grouping_ind,
                 src_grouping, bands_precision, prefix=None):
    fn_suffix_dat = 'dat_{}_newms{}_mainLFP{}_grp{}-{}.npz'.format(','.join(data_modalities),
                                                                   new_main_side[0].upper(),
                                                                use_main_LFP_chan,
                                                                src_file_grouping_ind, src_grouping)

    if rawnames is None and prefix is not None:
        fname_stats = prefix + fn_suffix_dat
    else:
        nr = len(rawnames)
        l = list( sorted( set([rawn[0:3] for rawn in rawnames] ) ))
        inds_str = ','.join(l )
        fname_stats = 'stats_{}_{}_{}_'.format(inds_str,nr,bands_precision)  + fn_suffix_dat
    return fname_stats

def genMLresFn(rawnames, sources_type, src_file_grouping_ind, src_grouping,
            prefix, n_channels, nfeats_used,
                pcadim, skip, windowsz,use_main_LFP_chan,
               grouping_key,int_types_key, nr=None, regex_mode=False):

    if nr is None:
        nr = len(rawnames)
    sind_str_list_sorted = list( sorted( set([rawn[0:3] for rawn in rawnames] ) ))
    sind_join_str = ','.join(sind_str_list_sorted )
    out_name_templ = '_{}_grp{}-{}_{}ML_nr{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}'
    out_name = (out_name_templ ).\
        format(sources_type, src_file_grouping_ind, src_grouping,
            prefix, nr,
            n_channels, nfeats_used,
            pcadim, skip, windowsz)

    if use_main_LFP_chan:
        out_name += '_mainLFP'

    a = grouping_key is not None
    b = int_types_key is not None
    if a or b:
        assert a and b
        if regex_mode:
            out_name += "__\({},{}\)".format( grouping_key,int_types_key )
        else:
            out_name += "__({},{})".format( grouping_key,int_types_key )

    out_name = '_{}{}.npz'.format(sind_join_str,out_name)
    return out_name

def collectFeatTypeInfo(feature_names, keep_sides=0, ext_info = True):
    import re
    ftypes = []
    fbands = []
    fbands_first, fbands_second = [],[]
    fband_pairs = []
    fband_per_ftype = {}

    bichan_feat_info = {'bpcorr':None, 'rbcorr':None }
    for bfik in bichan_feat_info:
        bichan_feat_info[bfik] = {'band_LHS':[], 'band_RHS':[] , 'mod_LHS':[],
                                  'mod_RHS':[], 'band_mod_LHS':[],
                                  'band_mod_RHS':[] }
        for k in bichan_feat_info[bfik]:
            bichan_feat_info[bfik][k] = []

    crop_LFP = 3
    crop_msrc = 4
    if keep_sides:
        crop_LFP += 1
        crop_msrc += 1

    bpcorr_left  = []
    bpcorr_right = []
    rbcorr_left  = []
    rbcorr_right = []

    bpcorr_left_mod  = []
    bpcorr_right_mod = []
    rbcorr_left_mod  = []
    rbcorr_right_mod = []
    for fn in feature_names:
        Hmode = False
        rH = re.match('(H_[a-z]{1,5})_',fn)
        if rH is None:
            r = re.match('([a-zA-Z0-9]+)_',fn)
            Hmode = True
        else:
            r = rH
        ftype = r.groups()[0]
        ftypes += [ftype]

        r = re.match(ftype + '_([a-zA-Z0-9]+)_',fn)
        if r is not None and rH is None:
            fb = r.groups()[0]
            fbands += [fb]
            fband_per_ftype[ftype] = fband_per_ftype.get(ftype,[]) + [fb]

        if ext_info:
            if ftype in bichan_feat_info:
                r = re.match(ftype + '_([a-zA-Z0-9]+)_.*,([a-zA-Z0-9]+)_',fn)
                fb1 = r.groups()[0]
                fb2 = r.groups()[1]
                fband_pairs += [(fb1,fb2)]
                fbands_first  += [fb1]
                fbands_second += [fb2]

                fband_per_ftype[ftype] = fband_per_ftype.get(ftype,[] ) + [fb2]  # fb1 is aleady there


                r = re.match(ftype + '_([a-zA-Z0-9]+)_(.+)*,([a-zA-Z0-9]+)_(.+)$',fn)
                fb1_ = r.groups()[0]
                mod1 = r.groups()[1]
                fb2_ = r.groups()[2]
                mod2 = r.groups()[3]
                assert fb1_ == fb1
                assert fb2_ == fb2

                if mod1.startswith('LFP'):
                    mod1 = mod1[:crop_LFP]
                if mod2.startswith('LFP'):
                    mod2 = mod2[:crop_LFP]
                if mod1.startswith('msrc'):
                    mod1 = mod1[:crop_msrc]
                if mod2.startswith('msrc'):
                    mod2 = mod2[:crop_msrc]

                bichan_feat_info[ftype]['band_LHS']     += [fb1 ]
                bichan_feat_info[ftype]['band_RHS']     += [fb2 ]
                bichan_feat_info[ftype]['mod_LHS']      += [mod1]
                bichan_feat_info[ftype]['mod_RHS']      += [mod2]
                bichan_feat_info[ftype]['band_mod_LHS'] += [(fb1,mod1)]
                bichan_feat_info[ftype]['band_mod_RHS'] += [(fb2,mod2)]

    for k1,v1 in bichan_feat_info.items():
        for k2,v2 in v1.items():
            #print(k1,k2,len(v2) )
            bichan_feat_info[k1][k2] = list( set(v2) )

    for k1,v1 in fband_per_ftype.items():
        fband_per_ftype[k1] = list( sorted( set(v1) ))




    info = {}
    info['ftypes'] = list(sorted( set(ftypes)) )
    info['fbands'] = list(sorted( set(fbands + fbands_second)) )
    info['fbands_first'] = list(sorted( set(fbands_first)) )
    info['fbands_second'] = list(sorted(set(fbands_second)) )
    info['fband_pairs']= list(set(fband_pairs))
    info['fband_per_ftype']= fband_per_ftype

    # remove those that were not found
    for ftype in set( bichan_feat_info.keys() ) - set(ftypes):
        del bichan_feat_info[ftype]
    info['bichan_feat_info'] = bichan_feat_info

    return info

def getMainSide(s, main_type = 'trem'):
    if len(s) < 5:
        subj = s
    else:
        subj,medcond_,task_ = getParamsFromRawname(s)
    import globvars as gv
    side = None
    if main_type == 'trem':
        side = gv.gen_subj_info[subj]['tremor_side']
    elif main_type == 'move':
        side = gv.gen_subj_info[subj].get('move_side',None)
    if side is None:
        print('{}: {} is None'.format(s, main_type))

    return side[0].upper()
