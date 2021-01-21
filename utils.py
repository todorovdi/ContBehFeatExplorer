import numpy as np
import udus_dataproc as mdp # main data proc
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

def nicenMEGsrc_chnames(chns, roi_labels=None, keyorder=None, prefix = ''):
    assert isinstance(roi_labels,dict)
    # nor MEGsrc chnames left unchanged
    if len(chns):
        assert isinstance(chns[0],str)
    else:
        return []
    chns_nicened = [0] * len(chns)
    for i,chn in enumerate(chns):
        fi = chn.find('src')
        chn_nice = chn
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
              freqres = [1 , 2 ,4 ], frmults = [1, 1/2, 1/4] , sfreq=None ):
    # gamma_bound is just bound when we change freq res to a coarser one
    # by default we have differnt freq steps for different bands and also different
    # window sizes (rusulting from different ratios of n_cycles and freq)
    # if we set all frmults to be equal, we get constant window size
    # (dpss windos would be of sfreq size then)

    pi5 = 2*np.pi / 5    # 5 is used in MNE as multiplier of std of gaussian used for Wavelet

    assert max_freq > gamma_bound, 'we want {} > {} '.format(max_freq, gamma_bound)
    assert min_freq < gamma_bound, 'we want {} < {} '.format(min_freq, gamma_bound)
    fbands = [ [min_freq,gamma_bound], [gamma_bound,HFO_bound], [HFO_bound,max_freq]  ]
    #freqres = np.array( [ 1, 2, 4  ] ) * min_freqres
    #frmults = frmult_scale * np.array( [pi5, pi5/2, pi5/4] )
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

    assert mode in ['valid','same']

    #dat has shape n_chans x ntimes // decim
    #returns n_chans x freqs x dat.shape[-1] // decim
    assert dat.ndim == 2
    assert len(freqs) == len(n_cycles)
    if abs(sfreq - int(sfreq) ) > 1e-5:
        raise ValueError('Integer sfreq is required')
    sfreq = int(sfreq)

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

    #wsz =
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
        first_ind = (wsz // 2) // decim
        last_offset = (wsz // 2) // decim
        tfrres = tfrres[:,:,first_ind:-last_offset]
        window_boundaries_st =  np.arange(0,nbins_orig - wsz, decim ) # we start from zero if wsz divide 2 and decim well
        window_boundaries_end = window_boundaries_st + wsz
        window_boundaries = np.vstack( [ window_boundaries_st, window_boundaries_end] )

    assert window_boundaries.shape[-1] == tfrres.shape[-1]

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
    if isinstance(anns,mne.Annotations):
        anns = [anns]

    nbins = int(duration * sfreq)
    bins = np.zeros(nbins)

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
                bins[sl] = 1
    return bins

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

# found on stackoverflow by someone called NaN
def stride(a, win=(3, 3), stepby=(1, 1)):
    """Provide a 2D sliding/moving view of an array.
    There is no edge correction for outputs. Use the `pad_` function first."""
    err = """Array shape, window and/or step size error.
    Use win=(3,) with stepby=(1,) for 1D array
    or win=(3,3) with stepby=(1,1) for 2D array
    or win=(1,3,3) with stepby=(1,1,1) for 3D
    ----    a.ndim != len(win) != len(stepby) ----
    cuts away last edge bins

    if for 2-D array given win(1,wsz) then return array of shape nchans x nwindows x wsz
    """
    from numpy.lib.stride_tricks import as_strided
    a_ndim = a.ndim
    if isinstance(win, int):
        win = (win,) * a_ndim
    if isinstance(stepby, int):
        stepby = (stepby,) * a_ndim
    assert (a_ndim == len(win)) and (len(win) == len(stepby)), err
    shp = np.array(a.shape)    # array shape (r, c) or (d, r, c)
    win_shp = np.array(win)    # window      (3, 3) or (1, 3, 3)
    ss = np.array(stepby)      # step by     (1, 1) or (1, 1, 1)
    newshape = tuple(((shp - win_shp) // ss) + 1) + tuple(win_shp)
    newstrides = tuple(np.array(a.strides) * ss) + a.strides
    a_s = as_strided(a, shape=newshape, strides=newstrides, subok=True).squeeze()
    return a_s

def H_difactmob(dat,dt, windowsz = None, dif = None, skip=None, stride_ver = True):
    import pandas as pd
    # last axis is time axis
    if dif is None:
        dif = np.diff(dat,axis=-1, prepend=dat[:,0][:,None] ) / dt
    if windowsz is None:
        activity = np.var(dat, axis=-1)
        vardif = np.var(dif, axis=-1)
    else:
        #raise ValueError('not implemented yet')
        if dat.ndim > 1:
            if stride_ver:
                win = (1,windowsz)
                step = (1,skip)
                stride_view_dat = stride(dat, win=win, stepby=step )
                activity = np.var(stride_view_dat,axis=-1, ddof=1)  #ddof=1 to agree with pandas version

                #print(dat.shape, activity.shape, win, step)

                stride_view_dif = stride(dif, win=win, stepby=step )
                vardif = np.var(stride_view_dif,axis=-1, ddof=1)
            else:
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
            raise ValueError('wrong ndim, shape = {}'.format(dat.shape) )

    bad_act = np.abs(activity) < 1e-10
    activity_bettered = activity[:]
    activity_bettered[bad_act] = 1e-10
    mobility = np.sqrt( vardif / activity_bettered )
    mobility[bad_act] = 0
    if (skip is not None) and not stride_ver:
        #dif       = dif[:,::skip]  # DON'T touch it!!
        activity  = activity[:,::skip]
        mobility  = mobility[:,::skip]

    return dif,activity, mobility

def Hjorth(dat, dt, windowsz = None, skip=None,stride_ver=True, remove_invalid = False):
    if isinstance(dat,list):
        acts, mobs, compls, wbds = [],[],[], []
        for subdat in dat:
            a,m,c,wbd = Hjorth(subdat,dt,windowsz,skip,stride_ver, remove_invalid=remove_invalid)
            acts += [a]
            mobs += [m]
            compls += [c]
            wbds += [wbd]
        #acts = np.concatenate(acts,axis=-1)
        #mobs = np.concatenate(mobs,axis=-1)
        #compls = np.concatenate(compls,axis=-1)
        return acts, mobs, compls, wbds
    elif not isinstance(dat,np.ndarray):
        raise ValueError('Wrong type {}'.format(type(dat) ) )

    #print(dat.shape)

    ndb = dat.shape[-1]
    padlen = windowsz-skip
    #if  int(ndb / windowsz) * windowsz - ndb == 0:  # dirty hack to agree with pandas version
    #    padlen += 1

    if stride_ver:
        #print('Using Hjorth stride ver for dat type {}'.format( type(dat) ) )
        assert dat.ndim == 2, dat.shape
        dat = np.pad(dat, [(0,0), (padlen,0) ], mode='edge' )
    else:
        raise ValueError('lost validity')
    dif = np.diff(dat,axis=-1, prepend=dat[:,0][:,None] ) / dt
    dif2 = np.diff(dif,axis=-1, prepend=dif[:,0][:,None] ) / dt

    dif, activity, mobility = H_difactmob(dat,dt, windowsz=windowsz, dif=dif, skip=skip, stride_ver=stride_ver)
    import gc;gc.collect()
    dif2, act2, mob2 = H_difactmob(dif,dt, windowsz=windowsz, dif=dif2, skip=skip, stride_ver=stride_ver)
    del dif
    del dif2
    del act2

    bad_mob = np.abs(mobility) < 1e-10
    mobility_bettered = mobility[:]
    mobility_bettered[bad_mob] = 1e-10
    complexity = mob2 / mobility_bettered
    complexity[bad_mob] = 0
    #complexity = mob2 / mobility

    n = windowsz // skip
    #n = padlen-1
    if stride_ver and not remove_invalid:               # dirty hack to agree with pandas version
        activity[:,:n] = np.nan
        mobility[:,:n] = np.nan
        complexity[:,:n] = np.nan

    if not stride_ver and remove_invalid:
        activity   = activity[:,n:]
        mobility   = mobility[:,n:]
        complexity = complexity[:,n:]

    nbins_orig = ndb
    wsz = windowsz
    decim = skip

    pred = np.arange(ndb + padlen)
    wnds = stride(pred, (windowsz,), (decim,) )
    window_boundaries = np.vstack( [ wnds[:,0], wnds[:,-1] + 1 ] )
    #print(window_boundaries)

    if remove_invalid:
        #window_boundaries_st =  np.arange(0,nbins_orig - wsz, decim )
        #window_boundaries_end = window_boundaries_st + wsz
        #window_boundaries = np.vstack( [ window_boundaries_st, window_boundaries_end] )


        #mobility = mobility[    :,0:-windowsz//decim]
        #activity = activity[    :,0:-windowsz//decim]
        #complexity = complexity[:,0:-windowsz//decim]
        sl = slice(windowsz//decim - 1,None,None)
        mobility = mobility[    :,sl]
        activity = activity[    :,sl]
        complexity = complexity[:,sl]
        window_boundaries = window_boundaries[:,sl] - padlen
        #print(window_boundaries)
    #else:
    #    strt = 0
    #    window_boundaries_st =  np.arange(strt - wsz + 1,nbins_orig, decim ) # we start from zero if wsz divide 2 and decim well
    #    #window_boundaries_st = np.maximum( window_boundaries_st - wsz, 0)
    #    window_boundaries_end = window_boundaries_st + wsz
    #    window_boundaries_end = np.minimum( window_boundaries_st, nbins_orig)
    #    window_boundaries = np.vstack( [ window_boundaries_st, window_boundaries_end] )

    if window_boundaries.shape[-1] != activity.shape[-1]:
        s = 'dat {}, padlen {} act {}, wbd {}'.format(dat.shape, padlen, activity.shape, window_boundaries.shape)
        raise ValueError(s)

    return activity, mobility, complexity, window_boundaries


def selectIndPairs(chnames_nice, chnames_short ,include_pairs,upper_diag=True,inc_same=True,
                   LFP2LFP_only_self=True, cross_within_parcel=False):
    '''
    include_pairs -- pairs of regex
    chnames_nice -- msrc_<roi_label>_c<component ind>
    inc_same -- whether we include self couplings (only makes sense if upper_diag is True)

    returns
    ind_pairs -- list of lists of channel indices coupled to a given channel ind
    ind_pairs_parcelis -- dict (key=parcel ind) of dictionaries (keys=channel ind) of channel inds
    ind_pairs_parcelsLFP -- dict (key=LFP chname) of lists of src channel inds
    parcel_couplings -- dict (key=pair of parcel inds) of lists of pairs of channel inds
    LFP2parcel_couplings -- dict (key=pair LFP chname, parcel ind) of lists of pairs (LFP channel ind, source ind)
    '''
    assert len(chnames_nice) == len(chnames_short)
    N = len(chnames_nice)
    ind_pairs = [0] * N
    parcel_couplings = {}
    LFP2parcel_couplings = {}
    LFP2LFP_couplings = {}

    sides_,groupis_,parcelis_,compis_ = parseMEGsrcChnamesShortList(chnames_short)

    ind_pairs_parcelsLFP = {}
    #len( set(parcelis_) )
    ind_pairs_parcelis = {}
    # look at all upper diag (or entire matrix) combinations
    for i in range(N):
        chn1 = chnames_nice[i]
        inds = []
        #ind_of_ind = 0
        if upper_diag:
            if inc_same:
                sl = range(i,N)
            else:
                sl = range(i+1,N)
        else:
            sl = range(0,N)
        # cycle over channel indices
        for j in sl:
            chn2 = chnames_nice[j]
            wasSome = False
            if ( ('LFP_self','LFP_self') in include_pairs) and \
                    (chn1.startswith('LFP') and chn2.startswith('LFP')):
                if j == i and LFP2LFP_only_self:
                    wasSome = True

                    pair = (chn1,chn2)
                    if pair not in LFP2LFP_couplings:
                        LFP2LFP_couplings[pair] = [ ]
                    #LFP2parcel_couplings[pair] += [ (i,ind_of_ind) ]
                    LFP2LFP_couplings[pair] += [ (i,j) ]
            else:
                # check if given upper diag combination is good (belongs to any
                # of the type crossings
                for pairing_ind,(s1,s2) in enumerate(include_pairs):
                    # first we decide if we process this pair or skip it
                    parcel_ind1 = parcelis_[i]
                    parcel_ind2 = parcelis_[j]

                    r1,r2 =  None,None
                    if s1 == 'msrc_self' and s2 == 'msrc_self':
                        cond_chans_same = (chn1==chn2)
                        cond_parcels_same = (parcel_ind1==parcel_ind2)
                        if cross_within_parcel:
                            cond = cond_parcels_same
                        else:
                            cond = cond_chans_same
                        if chn1.startswith('msrc') and cond:
                            r1,r2 = 1,1 #anything that is not None
                    else:
                        r1 = re.match(s1,chn1)
                        r2 = re.match(s2,chn2)
                    #if j == N - 1 and i == 1 and s1.find('LFP') < 0:
                    #    print(r1,r2)
                    #    print(chn1, chn2, s1, s2)

                    if r1 is None or r2 is None:
                        continue


                    #print(s1,s2,chn1,chn2,parcel_ind1,parcel_ind1)
                    #print(pairing_ind,chn1,chn2,parcel_ind1,parcel_ind2)

                    wasSome = True
                    if chn1.startswith('msrc') and chn2.startswith('msrc'):
                        #print(pairing_ind,chn1,chn2,parcel_ind1,parcel_ind2)

                        #side1, gi1, parcel_ind1, si1  = parseMEGsrcChnameShort(chnames_short[i])
                        #side2, gi2, parcel_ind2, si2  = parseMEGsrcChnameShort(chnames_short[j])
                        pair = (parcel_ind1,parcel_ind2)

                        if pair not in parcel_couplings:
                            parcel_couplings[pair] = []
                        #parcel_couplings[pair] += [ (i,ind_of_ind) ]
                        parcel_couplings[pair] += [ (i,j) ]

                        if parcel_ind1 not in ind_pairs_parcelis:
                            dd =  {i:[] }
                            ind_pairs_parcelis[parcel_ind1] = dd
                        if i not in ind_pairs_parcelis[parcel_ind1]:
                            ind_pairs_parcelis[parcel_ind1][i] = []
                        ind_pairs_parcelis[parcel_ind1][i] += [j]
                    else:
                        parcel_ind_valid = None
                        lfp_chn = None
                        msrc_chi = None
                        lfp_chi  = None
                        if chn1.startswith('LFP') and chn2.startswith('msrc'):
                            #side2, gi2, parcel_ind2, si2  = parseMEGsrcChnameShort(chnames_short[j])
                            lfp_chn = chn1
                            parcel_ind_valid = parcel_ind2
                            msrc_chi = j
                            lfp_chi  = i
                        elif chn1.startswith('msrc') and chn2.startswith('LFP'):
                            #side1, gi1, parcel_ind1, si1  = parseMEGsrcChnameShort(chnames_short[i])
                            lfp_chn = chn2
                            parcel_ind_valid = parcel_ind1
                            msrc_chi = i
                            lfp_chi  = j
                        else:
                            raise ValueError('Weird channel names {} {}'.format(chn1,chn2) )
                        pair = (lfp_chn,parcel_ind_valid)
                        assert pair[0] is not None and pair[1] is not None, '2 Weird channel names {} {}'.format(chn1,chn2)

                        #print('pair {}, lfp_chi {} ,msrc_chi {}, i {},j {},chn1 {},chn2 {}'.
                        #      format(pair, lfp_chi,msrc_chi, i,j,chn1,chn2) )

                        if pair not in LFP2parcel_couplings:
                            LFP2parcel_couplings[pair] = []
                        #LFP2parcel_couplings[pair] += [ (i,ind_of_ind) ]
                        #LFP2parcel_couplings[pair] += [ ( i,j) ]
                        LFP2parcel_couplings[pair] += [ (lfp_chi,msrc_chi) ]

                        if lfp_chn not in ind_pairs_parcelsLFP:
                            ind_pairs_parcelsLFP[lfp_chn] = []
                        #if i not in ind_pairs_parcelsLFP[lfp_chn]:
                        #    ind_pairs_parcelsLFP[lfp_chn][lfp_chi] = []
                        ind_pairs_parcelsLFP[lfp_chn] += [msrc_chi]

                        #break
            if wasSome:
                inds += [j]
                #ind_of_ind += 1
        ind_pairs[i] = inds
    return ind_pairs, ind_pairs_parcelis, ind_pairs_parcelsLFP, \
        parcel_couplings, LFP2parcel_couplings, LFP2LFP_couplings

def tfr2csd(dat, sfreq, returnOrder = False, skip_same = [], ind_pairs = None,
            parcel_couplings=None, LFP2parcel_couplings=None, LFP2LFP_couplings=None,
            oldchns = None, newchns = None,
            normalize = False, res_group_id=9, log=False):

    #ind_pairs_parcels = None,  ind_pairs_parcelsLFP = None,
    ''' csd has dimensions Nchan x nfreax x nbins

    sum over couplings of same kind
    for couplings of parcels it computes absolute value of imag coherence
    for LFP-parcel couplings, just abs of CSD

    returns n x (n+1) / 2  x nbins array
    skip same = indices of channels for which we won't compute i->j correl (usually LFP->LFP)
      note that it ruins getCsdVals
    ind_paris -- list of pairs (index, list of indces)


    parcel_couplings -- dict pair of indices of parcels -> list of pairs
    '''
    assert dat.ndim == 3
    n_channels = dat.shape[0]
    csds = []
    order  = []

    # get info corresponding to every source
    sides_,groupis_,parcelis_,compis_ = parseMEGsrcChnamesShortList(oldchns)

    fast_ver = False
    if fast_ver:
        raise ValueError('Needs debugging!')

    if ind_pairs is not None:
        assert len(ind_pairs) == n_channels
        for chi in range(n_channels):
            good_sec_inds = ind_pairs[chi]
            if len(good_sec_inds):
                r = np.conj ( dat[[chi]] ) *  ( dat[good_sec_inds] )    # upper diagonal elements only, same freq cross-channels
                if normalize:
                    norm = np.abs(r)
                    r /= norm
                csds += [r  ]

                secarg = np.array( good_sec_inds, dtype=int )
                firstarg = np.ones(len(secarg), dtype=int ) * chi
                curindPairs = np.vstack( [firstarg,secarg] )
                order += [curindPairs]

    else:
        if parcel_couplings is None:
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

                if normalize:
                    norm = np.abs(r)
                    r /= norm
                #print(r.shape)
                csds += [r  ]
        else:
            for pc in parcel_couplings:
                (pi1,pi2) = pc
                firstarg = []
                secarg = []
                chn1,chn2 = None,None
                r = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=np.complex )
                rimabs = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=np.complex )

                if fast_ver:
                    # which sources belong to the parcel with given index
                    chis_cur_parcel = np.where(parcelis_ == pi1)[0]
                    ntot =0
                    for chi in chis_cur_parcel:
                        good_sec_inds_ = ind_pairs[chi]  # channel indices
                        good_sec_inds = good_sec_inds_(parcelis_[ good_sec_inds_  ] == pi2 ) #selecting a subset
                        ntot += len(good_sec_inds)
                        if len(good_sec_inds):
                            rtmp = np.conj ( dat[[chi]] ) *  ( dat[good_sec_inds] )    # upper diagonal elements only, same freq cross-channels
                            if normalize:
                                norm = np.abs(rtmp)
                                rtmp /= norm
                            r      += np.sum(np.abs(rtmp),      axis=0) #TODO: NOOO! this way we get 1 here always
                            rimabs += np.sum(np.abs(rtmp.imag), axis=0)

                            #secarg = np.array( good_sec_inds, dtype=int )
                            #firstarg = np.ones(len(secarg), dtype=int ) * chi
                            #curindPairs = np.vstack( [firstarg,secarg] )
                            #order += [curindPairs]
                        side1 =  sides_[chi]
                        side2 =  sides_[good_sec_inds[0] ]
                    r /= ntot
                    rimabs /= ntot
                else:
                    ind_pairs = parcel_couplings[pc]
                    #ip_from = [ ip[0] for ip in ind_pairs ]
                    #ip_to =   [ ip[1] for ip in ind_pairs ]

                    #print('starting parcel pair ',pc)
                    r = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=np.complex )
                    rimabs = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=np.complex )

                    #for ifrom from ip_from:
                    #    rtmp = np.conj ( dat[[ifrom]] ) *  ( dat[ip_to] )  # upper diagonal elements only, same freq cross-channels
                    ninds_counted = 0

                    for (i,j) in ind_pairs:
                        rtmp = np.conj ( dat[[i]] ) *  ( dat[[j]] )    # upper diagonal elements only, same freq cross-channels

                        if normalize:
                            norm = np.abs(rtmp)
                            rtmp /= norm

                        r_cur = rtmp / len(ind_pairs)
                        rimabs_cur = np.abs(rtmp.imag) / len(ind_pairs)
                        if log:
                            r_cur = np.log(r_cur)
                            if pi1 != pi2:  # otherwise we won't use it anyway
                                m = np.min(rimabs_cur)
                                if m < 1e-14:
                                    numsmall = np.sum(rimabs_cur <  1e-14)
                                    print('tfr2csd Warning: in rimabs_cur min is {}, total num of small bins {}={:.2f}%. Using mixumim'.
                                          format(m,numsmall, (numsmall/rimabs_cur.size) * 100) )
                                    rimabs_cur = np.maximum(1e-14, rimabs_cur)
                                    #import ipdb; ipdb.set_trace()
                                rimabs_cur = np.log(rimabs_cur)
                        r +=  r_cur
                        rimabs += rimabs_cur
                        ninds_counted += 1

                        # DEBUG
                        #print('pc={} pair={}; mins dat[i]={:.4f}, dat[j]={:.4f}, rtmp={:.8f}, rtmpi={}'.format(
                        #    pc,(i,j), np.min(np.abs(dat[i]) ), np.min(np.abs(dat[j]) ),
                        #    np.min(np.abs(rtmp) ), np.min(np.abs(rtmp.imag) ) ) )

                        # we don't really care which particular sources for
                        # chnames, we only need parcel indices to know which
                        # new channel it corresponds to
                        chn1 = oldchns[i]
                        chn2 = oldchns[j]

                    assert ninds_counted > 0

                    # we don't need parcel inds (we know them) but we need
                    # sides
                    side1, gi1, parcel_ind1, si1  = parseMEGsrcChnameShort(chn1)
                    side2, gi2, parcel_ind2, si2  = parseMEGsrcChnameShort(chn2)
                    assert pi1 == parcel_ind1 and pi2 == parcel_ind2

                newchn1 = 'msrc{}_{}_{}_c{}'.format(side1,res_group_id,pi1,0)
                newchn2 = 'msrc{}_{}_{}_c{}'.format(side2,res_group_id,pi2,0)

                newi = newchns.index(newchn1)
                newj = newchns.index(newchn2)

                firstarg += [newi]
                secarg   += [newj]

                curindPairs = np.vstack( [firstarg,secarg] )
                order += [curindPairs]

                #print(r.shape)
                if pi1 == pi2:
                    csds += [r  ]
                else:
                    csds += [rimabs  ]

                if np.max(np.abs(csds[-1].imag)) > 1e-10:
                    print('nonzero imag for parcel pair ',pc)

            for pc in LFP2parcel_couplings:
                (chn1,pi2) = pc
                ind_pairs = LFP2parcel_couplings[pc]

                firstarg = []
                secarg = []
                chn2 = None

                r = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=np.complex )
                #rimabs = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=np.complex )

                if fast_ver:
                    chi = np.where(oldchns == chn1)[0][0]
                    ntot =0
                    good_sec_inds_ = ind_pairs[chi]  # channel indices
                    good_sec_inds = good_sec_inds_(parcelis_[ good_sec_inds_  ] == pi2 ) #selecting a subset
                    ntot += len(good_sec_inds)
                    if len(good_sec_inds):
                        rtmp = np.conj ( dat[[chi]] ) *  ( dat[good_sec_inds] )    # upper diagonal elements only, same freq cross-channels
                        if normalize:
                            norm = np.abs(rtmp)
                            rtmp /= norm
                        r      += np.sum(np.abs(rtmp),      axis=0)
                        #rimabs += np.sum(np.abs(rtmp.imag), axis=0)

                        #secarg = np.array( good_sec_inds, dtype=int )
                        #firstarg = np.ones(len(secarg), dtype=int ) * chi
                        #curindPairs = np.vstack( [firstarg,secarg] )
                        #order += [curindPairs]
                    side2 =  sides_[good_sec_inds[0] ]
                    r /= ntot
                    #rimabs /= ntot
                else:
                    ninds_counted = 0
                    for (i,j) in ind_pairs:
                        rtmp = np.conj ( dat[[i]] ) *  ( dat[[j]] )    # upper diagonal elements only, same freq cross-channels

                        if normalize:
                            norm = np.abs(rtmp)
                            rtmp /= norm

                        r_cur = rtmp / len(ind_pairs)
                        if log:
                            r_cur = np.log(r_cur)
                        r += r_cur
                        ninds_counted += 1

                        # we don't really care which particular sources, we only need parcel indices
                        chn2 = oldchns[j]
                    #print(i,j,chn1,chn2)
                    side2, gi2, parcel_ind2, si2  = parseMEGsrcChnameShort(chn2)

                    assert ninds_counted > 0

                newchn1 = chn1
                newchn2 = 'msrc{}_{}_{}_c{}'.format(side2,res_group_id,parcel_ind2,0)

                newi = newchns.index(newchn1)
                newj = newchns.index(newchn2)

                firstarg += [newi]
                secarg   += [newj]

                curindPairs = np.vstack( [firstarg,secarg] )
                order += [curindPairs]

                #print(r.shape)
                csds += [r  ]
            for pc in LFP2LFP_couplings:
                (chn1,chn2) = pc
                ind_pairs = LFP2LFP_couplings[pc]

                firstarg = []
                secarg = []

                ninds_counted = 0
                r = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=np.complex )
                for (i,j) in ind_pairs:
                    rtmp = np.conj ( dat[[i]] ) *  ( dat[[j]] )    # upper diagonal elements only, same freq cross-channels

                    if normalize:
                        norm = np.abs(rtmp)
                        rtmp /= norm

                    r_cur = rtmp / len(ind_pairs)
                    if log:
                        r_cur = np.log(r_cur)

                    r += r_cur
                    ninds_counted += 1

                    # we don't really care which particular sources, we only need parcel indices
                assert ninds_counted > 0
                newchn1 = chn1
                newchn2 = chn2

                newi = newchns.index(newchn1)
                newj = newchns.index(newchn2)

                firstarg += [newi]
                secarg   += [newj]

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

def removeAnnsByDescr(anns, anns_descr_to_remove, printLog=True):
    '''
    anns_descr_to_remove -- list of SUBstrings annotation names (NOT regexs)
    decide by using find (so can be both sides of just one)
    full equality is searched for
    '''
    #anns_descr_to_remove = ['endseg', 'initseg', 'middle', 'incPost' ]
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
    print('Overall removed {} annotations: {}'.format(Nbads, set(remtype) ) )

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

            sidelet,srcgroup_ind, ind, subind = parseMEGsrcChnameShort(chn)
            opsidelet = getOppositeSideStr(sidelet)

            rl = roi_labels[srcgrouping_names_sorted[srcgroup_ind] ]
            lab = rl[ind]
            lab_l = list(lab)
            assert labl_l[-1] == sidelet
            lab_l[-1] = opsidelet
            oplab = str(lab_l)
            indop = rl.index(oplab)

            newchn = genMEGsrcChnameShort(opsidelet, grouping_ind, opind, subind)

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
        print(oldchn,newchn)

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
                                  #skip = None):
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
            if len(inds) == 0:
                print('getWindowIndicesFromIntervals: EMPTY {},{},{}'.format(it_,s,e))
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

    if ret_type == 'time_bounds_list':
        r = time_bounds
    elif ret_type == 'bin_bounds_list':
        r = bin_bounds
    elif ret_type in ['bins_list', 'bins_contig']:
        r = bins
    return r

def setArtifNaN(X, ivalis_artif_tb_indarrays_merged, feat_names, ignore_shape_warning=False):
    '''
    copies input array
    '''
    assert X.shape[1] == len(feat_names), (X.shape[1], len(feat_names) )
    assert isinstance(X,np.ndarray)
    if not ignore_shape_warning:
        assert X.shape[0] > X.shape[1]
    assert isinstance(ivalis_artif_tb_indarrays_merged, dict)
    if feat_names is None:
        mode_all_chans = True
        feat_names = ['*&rAnDoM*&'] * X.shape[1]
    else:
        mode_all_chans = False
        assert isinstance(feat_names[0], str)
    Xout = X.copy()
    for interval_name in ivalis_artif_tb_indarrays_merged:
        templ = 'BAD_(.+)'
        matchres = re.match(templ,interval_name).groups()
        assert matchres is not None and len(matchres) > 0
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

        # over all features
        for feati,featn in enumerate(feat_names):
            #print(featn)
            if mode_all_chans:
                Xout[interval_bins,feati] = np.nan
            elif mode_LFP_artif and featn.find(artif_chn) >= 0:
                Xout[interval_bins,feati] = np.nan
                #print('fd')
            elif mode_MEG_artif and featn.find('msrc') >= 0:
                Xout[interval_bins,feati] = np.nan
                #print('fd')

    return Xout

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

def makeSimpleRaw(dat_,ch_names=None,sfreq=256,rescale=True,l=10,force_trunc_renum=False):
    # does rescaling to one
    assert dat_.ndim == 2
    if ch_names is None:
        ch_names = list(  map(str, np.arange( len(dat_)) ) )
    elif force_trunc_renum or np.max( [len(chn) for chn in ch_names ] ) > 15:
        ch_names = list(ch_names)[:]
        for chni in range(len(ch_names) ):
            ch_names[chni] = ch_names[chni][:l] + '_{}'.format(chni)

    info_ = mne.create_info( ch_names=ch_names, ch_types= ['csd']*len(ch_names), sfreq=sfreq)

    if rescale:
        me = np.mean(dat_, axis=1)
        std = np.std(dat_, axis=1)

        dat_ = (dat_-me[:,None])/std[:,None]

    dispraw = mne.io.RawArray(dat_,info_)
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

def vizGroup(sind_str,positions,labels,srcgroups, show=True, labels_to_skip = ['unlabeled'],
             show_legend=True):
    '''
    visualize src group
    positions
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


def bandAverage(freqs,freqs_inc_HFO,csd_pri,csdord_pri,csdord_LFP_HFO_pri,
               csd_LFP_HFO_pri, fbands,fband_names, fband_names_inc_HFO,
               newchnames, subfeature_order_lfp_highres, log_before_bandaver = True):
    '''
    csd -- num csds x num freqs x time dim OR  list of such things
    csdord -- 2 x num csds (  csdord[0,csdi] -- index of first channel in newchnames
    \<csdord_bandwise\> -- concat of csdord with corresp band
    '''
    print('Averaging over freqs within bands')
    #if bands_only in ['fine', 'crude']:
    #    if bands_only == 'fine':
    #        fband_names = fband_names_fine
    #    else:
    #        fband_names = fband_names_crude

    bpow_abscsd_pri = []
    csdord_strs_pri = []

    if isinstance(csd_pri,np.ndarray):
        csd_pri = [csd_pri]
    if isinstance(csd_LFP_HFO_pri,np.ndarray):
        csd_LFP_HFO_pri = [csd_LFP_HFO_pri]


    for dati  in range(len(csd_pri)):
        csdord_strs = []
        csd_ = csd_pri[dati]
        csdord = csdord_pri[dati]
        bpow_abscsd_curband = []
        for bandi,bandname in enumerate(fband_names):
            # get necessary freq indices
            low,high = fbands[bandname]
            freqis = np.where( (freqs >= low) * (freqs <= high) )[0]
            assert len(freqis) > 0, bandname

            #for dati  in range(len(csd_pri)):
            # average over these freqs
            csdcur = csd_[:,freqis,:]
            vals_to_aver = np.abs(csdcur  )
            if log_before_bandaver:
                assert np.min( vals_to_aver ) > 1e-15
                vals_to_aver = np.log(vals_to_aver)
            bandpow = np.mean(vals_to_aver   , axis=1 )

            # put into resulting list
            bpow_abscsd_curband += [bandpow[:,None,:]]


            #csdord_bandwise += [ np.concatenate( [csdord.T,  np.ones(csd.shape[0], \
            #    dtype=int)[:,None]*bandi  ] , axis=-1) [:,None,:] ]

            # currently it does not make sense because I already put in csd abs of
            # imag CSD for parcel 2 parcel couplings
            #bandaver_imagcoh = np.mean(   csdcur.imag  , axis=1 )
            #bpow_imagcsd +=  [ bandaver_imagcoh [:,None,:] ]

            for csdi in range(bandpow.shape[0]):
                k1,k2 = csdord[:,csdi]
                k1 = int(k1); k2=int(k2)
                s = '{}_{},{}'.format( bandname, newchnames[k1] , newchnames[k2] )
                csdord_strs += [s]

            #bandpow2 = np.concatenate(bpow_abscsd_curband, axis=-1  )  # over time
            #bpow_abscsd += [bandpow2]
        bpow_abscsd = np.concatenate(bpow_abscsd_curband, axis=1)  # over bands
        csdord_strs_pri += [csdord_strs]
        bpow_abscsd_pri += [bpow_abscsd]


    del csd_
    del vals_to_aver
    del bandpow
    del bpow_abscsd_curband
    import gc; gc.collect()
    #bpow_imagcsd = np.concatenate(bpow_imagcsd, axis=1)


    #for bandi,bandname in enumerate(fband_names):
    #    csdord_bandwise += [ np.concatenate( [csdord.T,  np.ones(csd.shape[0], \
    #        dtype=int)[:,None]*bandi  ] , axis=-1) [:,None,:] ]

    # last dimension is index of band
    #csdord_bandwise = np.concatenate(csdord_bandwise,axis=1)
    #csdord_bandwise.shape

    use_lfp_HFO = csd_LFP_HFO_pri is not None
    csdord_strs_HFO_pri = []
    bpow_abscsd_LFP_HFO_pri = []
    if use_lfp_HFO:
        #if bands_only in ['fine', 'crude']:
        assert fband_names_inc_HFO is not None
        fband_names_HFO = fband_names_inc_HFO[len(fband_names):]  # that HFO names go after
        #bpow_abscsd_LFP_HFO = []
        freqs_HFO = freqs_inc_HFO[ len(freqs): ]

        for dati in range(len(csd_LFP_HFO_pri)):

            csdord_strs_HFO = []
            csd_LFP_HFO_ = csd_LFP_HFO_pri[dati]
            csdord_LFP_HFO = csdord_LFP_HFO_pri[dati]
            bpow_abscsd_curband = []
            for bandi,bandname in enumerate(fband_names_HFO):
                low,high = fbands[bandname]
                freqis = np.where( (freqs_HFO >= low) * (freqs_HFO <= high) )[0]
                assert len(freqis) > 0, bandname

                csdcur = csd_LFP_HFO_[:,freqis,:]
                vals_to_aver = np.abs(csdcur  )
                if log_before_bandaver:
                    assert np.min( vals_to_aver ) > 1e-15
                    vals_to_aver = np.log(vals_to_aver)

                bandpow = np.mean( vals_to_aver  , axis=1 )
                bpow_abscsd_curband += [bandpow[:,None,:]]

                #bandpow2 = np.concatenate(bpow_abscsd_curband, axis=-1  )  # over time
                #bpow_abscsd_LFP_HFO += [bandpow2]

                for csdi in range(bandpow.shape[0]):
                    k1,k2 = csdord_LFP_HFO[:,csdi]
                    k1 = int(k1); k2=int(k2)
                    s = '{}_{},{}'.format( bandname, subfeature_order_lfp_highres[k1] ,
                                        subfeature_order_lfp_highres[k2] )
                    csdord_strs_HFO += [s]


                #bpow_abscsd_LFP_HFO += [bpow_abscsd_curband]
            bpow_abscsd_LFP_HFO = np.concatenate(bpow_abscsd_curband, axis=1) # over bands

            csdord_strs_HFO_pri += [csdord_strs_HFO]
            bpow_abscsd_LFP_HFO_pri += [ bpow_abscsd_LFP_HFO ]

        #csdord_bandwise_LFP_HFO = []
        #for bandi,bandname in enumerate(fband_names_HFO):
        #    csdord_bandwise_LFP_HFO += [ np.concatenate( [csdord_LFP_HFO.T,  np.ones(csd_LFP_HFO.shape[0], dtype=int)[:,None]*bandi  ] , axis=-1) [:,None,:] ]

        #csdord_bandwise_LFP_HFO = np.concatenate(csdord_bandwise_LFP_HFO,axis=1)
        #csdord_bandwise_LFP_HFO.shape

    ###################################

    #print('Preparing csdord_strs')
    #csdord_strs = []
    ##for csdord_cur in csdords:
    #csdord_cur = csdord
    #for bandi in range(csdord_bandwise.shape[1] ):
    #    for i in range(csdord_cur.shape[1]):
    #        k1,k2 = csdord_cur[:,i]
    #        k1 = int(k1); k2=int(k2)
    #        s = '{}_{},{}'.format( fband_names[bandi], newchnames[k1] , newchnames[k2] )
    #        csdord_strs += [s]


    #if use_lfp_HFO:
    #    csdord_cur = csdord_LFP_HFO
    #    for bandi in range(csdord_bandwise_LFP_HFO.shape[1] ):
    #        for i in range(csdord_cur.shape[1]):
    #            k1,k2 = csdord_cur[:,i]
    #            k1 = int(k1); k2=int(k2)
    #            s = '{}_{},{}'.format( fband_names_HFO[bandi],
    #                                subfeature_order_lfp_highres[k1] , subfeature_order_lfp_highres[k2] )
    #            csdord_strs += [s]

    bpow_imagcsd = None
    return bpow_abscsd_pri, bpow_imagcsd, csdord_strs_pri, csdord_strs_HFO_pri, bpow_abscsd_LFP_HFO_pri

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

def collectDataFromMultiRaws(rawnames, raws_permod_both_sides, sources_type,
                             src_file_grouping_ind, src_grouping, use_main_LFP_chan,
                             side_to_use, new_main_side, data_modalities,
                             only_load_data,crop_start,crop_end,msrc_inds,
                             mainLFPchans_pri=None, mainLFPchan_newname=None):
    '''
    side_to_use can be 'tremor_side', 'move_side' , 'both', 'left' , 'right'
    rawnames are important to have because they give ordering
    '''

    assert new_main_side in ['left','right']
    dat_pri = []
    times_pri = []
    times_hires_pri = []
    dat_lfp_hires_pri = []
    ivalis_pri = []

    extdat_pri = []
    anns_pri = []
    raws_permod_pri = []
    rec_info_pri = []
    subfeature_order_pri = []
    subfeature_order_lfp_hires_pri = []

    use_ipsilat_CB = True

    data_dir = gv.data_dir
    aux_info_per_raw = {}
    for rawind in range(len(rawnames) ):
        gen_subj_info = gv.gen_subj_info
        subj,medcond,task  = getParamsFromRawname(rawnames[rawind])
        maintremside = gen_subj_info[subj]['tremor_side']
        mainmoveside = gen_subj_info[subj].get('move_side',None)
        if mainLFPchans_pri is not None:
            mainLFPchan =  mainLFPchans_pri[rawind]
        else:
            mainLFPchan =  gen_subj_info[subj].get('lfpchan_used_in_paper',None)

        rawname_ = rawnames[rawind]

        src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rawname_,sources_type,src_file_grouping_ind)
        src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
        rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
        rec_info_pri += [rec_info]

        roi_labels = rec_info['label_groups_dict'][()]      # dict of (orderd) lists
        srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]  # order of grouping names
        assert set(roi_labels.keys() ) == set(srcgrouping_names_sorted )
        assert len(roi_labels) == 1, 'several groupings in single run -- not implmemented'
        # assuming we have only one grouping present
        roi_labels_cur = roi_labels[srcgrouping_names_sorted[src_grouping ]  ]



        anns_fn = rawname_ + '_anns.txt'
        anns_fn_full = os.path.join(data_dir, anns_fn)
        anns = mne.read_annotations(anns_fn_full)
        if crop_end is not None:
            anns.crop(crop_start,crop_end)
        anns_pri += [anns]

        ivalis_pri += [ ann2ivalDict(anns) ]

        #############################################################

        raws = raws_permod_both_sides[rawname_]

        if side_to_use == 'tremor_side':
            main_body_side = maintremside
        elif side_to_use == 'move_side':
            main_body_side = mainmoveside
        elif side_to_use in ['left', 'right']:
            main_body_side = side_to_use
        elif side_to_use == 'both':
            main_body_side = ['left', 'right']
        else:
            raise ValueError('wrong side name')

        if main_body_side != new_main_side:
            side_switch_needed = True
            print('WE WILL BE SWITCHING SIDES for {}'.format(rawname_) )
        else:
            side_switch_needed = False

        aux_info_per_raw[rawname_] = {}
        aux_info_per_raw[rawname_]['side_switched'] = side_switch_needed
        aux_info_per_raw[rawname_]['main_body_side'] = main_body_side


        import copy

        assert isinstance(main_body_side,str)
        ms_letter = main_body_side[0].upper()


        raw_lfponly       = raws['LFP']
        raw_lfp_hires     = raws.get('LFP_hires',None)
        raw_srconly       = raws['src']
        raw_emg_rectconv  = raws['EMG']

        raw_lfponly.load_data()
        raw_srconly.load_data()
        raw_lfp_hires.load_data()

        # first we separate channels by sides (we will select one side only later)
        brain_sides = ['L', 'R'] # brain sides   # this is just for construction of data, we will restrict later
        hand_sides_all = ['L', 'R']  # hand sides

        # first create copies
        raws_lfp_perside = {'L': raw_lfponly.copy(), 'R': raw_lfponly.copy() }
        raws_srconly_perside = {'L': raw_srconly.copy(), 'R': raw_srconly.copy() }
        if raw_lfp_hires is not None:
            raws_lfp_hires_perside = {'L': raw_lfp_hires.copy(), 'R': raw_lfp_hires.copy() }

        # then pick channels from corresponding side
        for side in brain_sides:
            chis = mne.pick_channels_regexp(raw_lfponly.ch_names, 'LFP{}.*'.format(side))
            chnames_lfp = [raw_lfponly.ch_names[chi] for chi in chis]
            if use_main_LFP_chan:
                assert mainLFPchan in chnames_lfp
                chnames_lfp_to_use = [mainLFPchan]
            else:
                chnames_lfp_to_use = chnames_lfp

            raws_lfp_perside[side].pick_channels(   chnames_lfp_to_use )
            if raw_lfp_hires is not None:
                raws_lfp_hires_perside[side].pick_channels(   chnames_lfp_to_use  )



            chis =  mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_all_*'.format(side)  )
            if len(chis) == 0:
                chis =  mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_{}_[0-9]+_c[0-9]+'.
                                                format(side, src_grouping)  )

            if use_ipsilat_CB:
                CB_contrahand_parcel_ind = roi_labels_cur.index('Cerebellum_{}'.format(side))
                hand_side = getOppositeSideStr(side)
                CB_ipsihand_parcel_ind = roi_labels_cur.index('Cerebellum_{}'.format(hand_side) )
                CB_contrahand_inds = mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_{}_{}_c[0-9]+'.
                                                            format(side,src_grouping,CB_contrahand_parcel_ind)  )
                CB_ipsihand_inds = mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_{}_{}_c[0-9]+'.
                                                        format(hand_side,src_grouping,CB_ipsihand_parcel_ind)  )
                chis = np.setdiff1d(chis, CB_contrahand_inds)
                chis = np.hstack( [chis, CB_ipsihand_inds])
                #TODO: remove Cerbellum sources, add Cerebellum sources from the other side

            assert len(chis) > 0
            chnames_src = [raw_srconly.ch_names[chi] for chi in chis]
            raws_srconly_perside[side].pick_channels(   chnames_src  )



            print('{} side,  {} sources'.format(side, len(chis) ) )


        if side_switch_needed:
            raws_srconly_perside_new = {}
            raws_lfp_perside_new = {}
            raws_lfp_hires_perside_new = {}
            for sidelet in brain_sides:
                # we want to reassign the channels names (changing side),
                # keeping the data in place. And we want to have the correct
                # side assignment acoording to new channel naming
                opsidelet = getOppositeSideStr(sidelet)
                raws_srconly_perside_new[opsidelet] = \
                    changeRawInfoSides(raws_srconly_perside[sidelet],roi_labels,
                                       srcgrouping_names_sorted)
                raws_lfp_perside_new[opsidelet] = changeRawInfoSides(raws_lfp_perside[sidelet])
                if raw_lfp_hires is not None:
                    raws_lfp_hires_perside_new[opsidelet] = changeRawInfoSides(raws_lfp_hires_perside[sidelet])

            raw_emg_rectconv = changeRawInfoSides(raw_emg_rectconv)
        print(raws_lfp_perside[side].ch_names)


        ####################  Load emg

        EMG_per_hand = gv.EMG_per_hand
        if isinstance(main_body_side,str):
            chnames_emg = EMG_per_hand[main_body_side]
        else:
            chnames_emg = raw_emg_rectconv.ch_names

        print(rawname_,chnames_emg)

        rectconv_emg, ts_ = raw_emg_rectconv[chnames_emg]
        chnames_emg = [chn+'_rectconv' for chn in chnames_emg]


        ############# Concatenate data
        raws_permod = {'LFP' : raws_lfp_perside, 'msrc' : raws_srconly_perside }
        if only_load_data:
            raws_permod_pri += [raws_permod]
        if isinstance(main_body_side,str):
            hand_sides = [ms_letter ]
        elif isinstance(main_body_side,list) and isinstance(main_body_side[0], str):
            hand_sides = main_body_side
        else:
            raise ValueError('Wrong main_body_side',main_body_side)
        print('main_body_side {}, hand_sides to construct features '.format(main_body_side) ,hand_sides)

        if sources_type == 'HirschPt2011':
            allowd_srcis_subregex = '[{}]'.format( ','.join( map(str, msrc_inds ) ))
        #else:
        #    allowd_srcis_subregex = '[{}]'.format( ','.join( map(str, msrc_inds ) ))
        subfeature_order = []
        dats = []
        for side_hand in hand_sides:
            for mod in data_modalities:
                #sd = hand_sides_all[1-hand_sides.index(side_hand) ]  #
                opside= getOppositeSideStr(side_hand)
                #if mod in ['src','msrc']:  No! They are both in the brain, so both contralat!

                curraw = raws_permod[mod][opside]

                if mod == 'msrc':
                    chns = curraw.ch_names
                    # note that we want to allow all sides in regex here
                    # because we might have ipsilateral structures as well (as
                    # selected by siding previsouly)
                    if sources_type == 'HirschPt2011':
                        inds = mne.pick_channels_regexp(  chns  , 'msrc.*_{}'.format(allowd_srcis_subregex) )
                    else:
                        inds = mne.pick_channels_regexp(  chns  , 'msrc._{}.*'.format(src_grouping) )
                    assert len(inds) > 0
                    chns_selected = list( np.array(chns)[inds]  )
                    curdat, times = curraw[chns_selected]
                    #msrc_inds
                    chnames_added = chns_selected
                else:
                    curdat = curraw.get_data()
                    chnames_added = curraw.ch_names
                dats += [ curdat ]

                subfeature_order += chnames_added
                print(mod,opside)

        if mainLFPchan_newname is not None:
            mainLFPchan_ind = subfeature_order.index(mainLFPchan)
            subfeature_order[mainLFPchan_ind] = mainLFPchan_newname

        subfeature_order_pri += [subfeature_order]
        #dats = {'lfp': dat_lfp, 'msrc':dat_src}
        dat = np.vstack(dats)
        times = raw_srconly.times

        dat_pri += [dat]
        times_pri += [times]
        times_hires_pri += [raw_lfp_hires.times]

        if raw_lfp_hires is not None:
            dats_lfp_hires = []
            subfeature_order_lfp_hires = []
            for side_hand in hand_sides:
                opside= getOppositeSideStr(side_hand)
                #sd = hand_sides_all[1-hand_sides.index(side_hand) ]  #
                curraw = raws_lfp_hires_perside[opside]
                curdat  = curraw.get_data()
                chnames_added = curraw.ch_names
                subfeature_order_lfp_hires += chnames_added
                dats_lfp_hires += [ curdat]


            if mainLFPchan_newname is not None:
                mainLFPchan_ind = subfeature_order_lfp_hires.index(mainLFPchan)
                subfeature_order_lfp_hires[mainLFPchan_ind] = mainLFPchan_newname
            subfeature_order_lfp_hires_pri += [subfeature_order_lfp_hires]

            dat_lfp_hires = np.vstack(dats_lfp_hires)
            dat_lfp_hires_pri += [dat_lfp_hires]


        ecg_fname = os.path.join(data_dir, '{}_ica_ecg.npz'.format(rawname_) )
        if os.path.exists( ecg_fname ):
            f = np.load( ecg_fname )
            ecg = f['ecg']
            #ecg_normalized = (ecg - np.min(ecg) )/( np.max(ecg) - np.min(ecg) )
            ecg_normalized = (ecg - np.mean(ecg) )/( np.quantile(ecg,0.93) - np.quantile(ecg,0.01) )
        else:
            ecg = np.zeros( len(times) )
            ecg_normalized = ecg

        extdat_pri += [ np.vstack( [ecg, ecg_normalized] ) ]

        #TODO: rename mainLFPchans to  mainLFPchan_newname if its is not None


    extnames = ['ecg'] + chnames_emg



    return dat_pri, dat_lfp_hires_pri, extdat_pri, anns_pri, ivalis_pri, rec_info_pri, times_pri,\
    times_hires_pri, subfeature_order_pri, subfeature_order_lfp_hires_pri, aux_info_per_raw



def _smoothData1D_proxy(arg):
    ind, data,Tp,data_estim,state_noise_std,ic_mean=arg
    smoothend = smoothData1D(data,Tp,data_estim,state_noise_std,ic_mean)
    print('  {} smoothing finished'.format(ind) )
    return  ind,smoothend

def smoothData1D(data,Tp,data_estim=None,state_noise_std=None,ic_mean=None):
    # data estim to be used as IC and also to get variance size
    import simdkalman  # can work with NaNs
    assert (data_estim is not None) or (state_noise_std is not None)

    #Tp -- pred itnerval. Lower means smoother.
    # Probably I can say that it is the time interval
    # I want the true state to be approx linear in
    if state_noise_std is None and data_estim is not None:
        state_noise_std = np.var(data_estim)
    meas_noise_std = 1


    #kalman gain depends on the ratio of these two values
    kf = simdkalman.KalmanFilter(
        state_transition = [[1,Tp],[0,1]],        # matrix A
        process_noise = state_noise_std * np.array( [[ Tp**3/3, Tp**2/2 ],
                                 [Tp**2/2, Tp]]),    # Q
        observation_model = np.array([[1,0]]),   # H
        observation_noise = meas_noise_std)                 # R


    # smoothed = kf.smooth(chd_test,
    #                      initial_value = [np.mean(estim),0],
    #                      initial_covariance = np.diag([np.std(estim), np.std(estim)]))

    if ic_mean is None:
        ic_mean = np.mean(data_estim)

    #print(data_estim.shape,state_noise_std, ic_mean)
    smoothed = kf.smooth(data,
                         initial_value = [ic_mean,0],
                         initial_covariance = np.diag([0,0]))

    return smoothed

#TODO: make paralel
#NOTE: it won't be so slow because I will apply it to windows
# not to the original data with many samples
def smoothData(data,Tp,data_estim=None,state_noise_std=None,ic_mean=None,
              n_jobs = 6):

    import multiprocessing as mpr
    assert data.ndim == data_estim.ndim
    if data.ndim == 1:
        data = data[None,:]
        data_estim = data_estim[None,:]
    assert data.ndim == 2

    args = []

    N = len(data)
    r = [0]*N
    rstates = [0]*N
    for dim in range(N):
        curdat = data[dim]

        args += [(dim,curdat,Tp,data_estim[dim],None,None)]

    print(n_jobs)
    if n_jobs == 1:
        for arg in args:
            dim,curdat,Tp,data_estim_cur,_,_ = arg
            cursmooth = smoothData1D(curdat,Tp,data_estim_cur)
            r[dim] = cursmooth
            rstates[dim] = cursmooth.states.mean[:,0]
    else:
        pool = mpr.Pool(n_jobs)
        print('smoothData:  Sending {} tasks to {} cores'.format(len(args), mpr.cpu_count()))
        res = pool.map(_smoothData1D_proxy, args)

        for dim,cursmooth in res:
            r[dim] = cursmooth
            rstates[dim] = cursmooth.states.mean[:,0]

        pool.close()
        pool.join()
    return np.vstack(rstates)

