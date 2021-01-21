import mne
import os
import numpy as np
import gc
import utils

def getFileAge(fname_full, ret_hours=True):
    import os, datetime
    if not os.path.exists(fname_full):
        return np.inf
    created = os.stat( fname_full ).st_ctime
    dt = datetime.datetime.fromtimestamp(created)
    today = datetime.datetime.today()
    tdelta = (today - dt)
    r = tdelta
    if ret_hours:
        nh = tdelta.seconds / ( 60 * 60 )
        r = nh
    return nh

def getRaw(rawname_naked, rawname = None ):
    if os.environ.get('DATA_DUSS') is not None:
        data_dir = os.path.expandvars('$DATA_DUSS')
    else:
        data_dir = '/home/demitau/data'

    if rawname is None:
        rawname = rawname_naked + '_resample_raw.fif'
    fname_full = os.path.join(data_dir, rawname)
    # read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
    raw = mne.io.read_raw_fif(fname_full, None)

    return raw

def getSubRaw(rawname_naked, picks = ['EMG.*old'], raw=None, rawname = None):
    '''
    no preproc is done, just load
    picks -- list of regexes
    creates a copy, so no damaging input
    '''
    if isinstance(picks, str):
        picks = [picks]
    assert isinstance(picks,list) and isinstance(picks[0],str)

    if raw is None:
        raw = getRaw(rawname_naked, rawname)

    subraw = raw.copy()
    subraw.load_data()
    chis = []
    for pick in picks:
        chis_cur = mne.pick_channels_regexp(subraw.ch_names, pick )
        chis += chis_cur
    restr_names = np.array(subraw.ch_names)[chis]
    restr_names = restr_names.tolist()

    subraw.pick_channels(restr_names)
    gc.collect()

    return subraw

def loadBadChannelList(rawname_,ch_names):
    import json
    import utils
    sind_str,medstate,task = utils.getParamsFromRawname(rawname_)
    # get info about bad MEG channels (from separate file)
    with open('subj_info.json') as info_json:
        #raise TypeError

        #json.dumps({'value': numpy.int64(42)}, default=convert)
        gen_subj_info = json.load(info_json)

    #subj,medcond,task  = utils.getParamsFromRawname(rawname_)
    subjind = int(sind_str[1:])
    if subjind > 7:
        badchlist_ = gen_subj_info[sind_str]['bad_channels'][medstate]['hold']
    else:
        badchlist_ = gen_subj_info[sind_str]['bad_channels'][medstate][task]
    badchlist= []
    for chname in badchlist_:
        if chname.find('EMG') >= 0 and ( (chname.find('_kil') < 0) and (chname.find('_old') < 0) ):
            badchlist += [chname + '_old', chname + '_kil']
        else:
            if chname not in ch_names:
                print('Warning: channel {} not found in {}'.format(chname,rawname_) )
                continue
            else:
                badchlist += [chname]
    return badchlist
def getRawnameListStructure(rawnames):
    subjs_analyzed = {}  # keys -- subjs,  vals -- arrays of keys in raws
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

    # I use that medcond, tasks don't intersect with keys I use
    for ki,k in enumerate(rawnames):
        #f = raws[k]
        sind_str,medcond,task = utils.getParamsFromRawname(k)

        cursubj = {}
        if sind_str in subjs_analyzed:
            cursubj = subjs_analyzed[sind_str]

        if 'datasets' not in cursubj:
            cursubj['datasets'] = []
        cursubj['datasets'] += [k]

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
                if 'datasets' not in cursubj[medcond]:
                    cursubj[medcond]['datasets'] = [k]
                else:
                    cursubj[medcond]['datasets'] += [k]
        else:
            cursubj[medcond] = { task: k}
            if 'datasets' not in cursubj[medcond]:
                cursubj[medcond]['datasets'] = [k]
            else:
                cursubj[medcond]['datasets'] += [k]

        subjs_analyzed[sind_str] =  cursubj
    return subjs_analyzed

def rescaleFeats(rawnames, X_pri, featnames, wbd_pri,
                 sfreq, times_pri, int_type_pri, main_side_pri=None,
                 minlen_bins = 5 * 256 / 32, combine_within='no'):
    '''
    usually notrem_<sidelet>
    modifies raws in place. Rescales to zero mean, unit std
    '''
    if int_type_pri is None:
        int_type_pri = [ 'entire' ] * len(rawnames)
    for int_type in int_type_pri:
        assert int_type.find('{}') < 0  # it should not be a template


    assert len(featnames) == X_pri[0].shape[1],  ( len(featnames),  X_pri[0].shape[0] )
    assert len(rawnames) == len(X_pri)
    assert len(rawnames) == len(wbd_pri)
    assert len(rawnames) == len(times_pri)
    assert len(rawnames) == len(int_type_pri)
    assert combine_within in ['subject', 'medcond', 'no']
    mods = ['msrc' , 'LFP']

    if main_side_pri is None:
        main_side_pri = [ it[-1].upper() for it in int_type_pri ]

    #print('Start raws rescaling for modality {} based on interval type {}'.format(mod,int_type_templ) )
    import utils_tSNE as utsne
    rwnstr = ','.join(rawnames)

    #if mod == 'src':
    #    chn_name_side_ind = 4
    #if mod in ['LFP', 'LFP_hires']:
    #    chn_name_side_ind = 3

    if combine_within == 'no':
        indsets = [ np.arange(len(rawnames) ) ]
    else:
        subjs_analyzed = getRawnameListStructure(rawnames)
        if combine_within == 'subj':
            indsets = []
            for subj in subjs_analyzed:
                subj_sub = subjs_analyzed[subj]
                indset_cur = []
                dsets = subj_sub['datasets']
                for rn in dsets:
                    indset_cur += [rawnames.index(rn) ]
                indsets += [indset_cur]
            indsets += [ np.arange(len(rawnames) ) ]
        elif combine_within == 'medcond':
            indsets = []
            for subj in subjs_analyzed:
                subj_sub = subjs_analyzed[subj]
                for medcond in subj_sub['medconds']:
                    indset_cur = []
                    dsets = subj_sub['medconds']['datasets']
                    for rn in dsets:
                        indset_cur += [rawnames.index(rn) ]
                    indsets += [indset_cur]
            indsets += [ np.arange(len(rawnames) ) ]
    #for

    ib_MEG_perit_perraw = {}
    ib_LFP_perit_perraw = {}
    ib_mvt_perit_perraw = {}
    for rawi,rn in enumerate(rawnames):
        #ib_MEG_perit = getCleanIntervalBins(rn,sfreq, times,['_ann_MEGartif'] )
        #ib_LFP_perit = getCleanIntervalBins(rn,sfreq, times,['_ann_LFPartif'] )
        main_side = main_side_pri[rawi]
        wrong_brain_sidelet = main_side[0].upper()

        wbd = wbd_pri[rawi]
        times = times_pri[rawi]

        anns_mvt, anns_artif_pri, times2, dataset_bounds = \
        utsne.concatAnns([rn],[times] )
        #ivalis_mvt = utils.ann2ivalDict(anns_mvt)
        ib_mvt_perit_merged = \
        utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_mvt) ,
                                            sfreq,ret_type='bins_contig',
                                            wbd_type='contig',
                                            ret_indices_type =
                                                'window_inds', nbins_total=len(times) )

        anns_MEGartif, anns_artif_pri, times2, dataset_bounds = \
            utsne.concatAnns([rn],[times],['_ann_MEGartif'] )
        # here I don't want to remove artifacts from "wrong" brain side because
        # we use ipsilateral CB
        ib_MEG_perit_merged = \
            utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_MEGartif) ,
                                            sfreq,ret_type='bins_contig',
                                            wbd_type='contig',
                                            ret_indices_type =
                                                'window_inds', nbins_total=len(times) )

        anns_LFPartif, anns_artif_pri, times2, dataset_bounds = \
            utsne.concatAnns([rn],[times],['_ann_LFPartif'] )
        anns_LFPartif = utils.removeAnnsByDescr(anns_LFPartif, ['artif_LFP{}'.format(wrong_brain_sidelet) ])
        ib_LFP_perit_merged = \
            utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_LFPartif) ,
                                            sfreq,ret_type='bins_contig',
                                            wbd_type='contig',
                                            ret_indices_type =
                                                'window_inds', nbins_total=len(times) )

        #import pdb; pdb.set_trace()

        ib_MEG_perit_perraw[rn] = ib_MEG_perit_merged
        ib_LFP_perit_perraw[rn] = ib_LFP_perit_merged
        ib_mvt_perit_perraw[rn] = ib_mvt_perit_merged

        it = int_type_pri[rawi]
        print('rescaleFeats: Rescaling features for raw {} accodring to data in interval {}'.format(rn,it ) )

    rescale_separately = True

    for rawindseti_cur,indset_cur in enumerate(indsets):
        stat_perchan = {}
        for feati,featn in enumerate(featnames):
            dats_forstat = []
            # for each dataset separtely we rescale features accodring to an
            # interval
            for rawi in indset_cur:
                #if mod == 'src':
                #    chnames_nicened = utils.nicenMEGsrc_chnames(chnames, roi_labels, srcgrouping_names_sorted,
                #                                    prefix='msrc_')
                rn = rawnames[rawi]
                ib_MEG_perit =  ib_MEG_perit_perraw[rn]
                ib_LFP_perit =  ib_LFP_perit_perraw[rn]
                ib_mvt_perit =  ib_mvt_perit_perraw[rn]

                dat =  X_pri[rawi][:,feati]
                l = len(dat)

                it = int_type_pri[rawi]
                if it == 'entire':
                    dat_forstat = dat
                else:
                    #ib_MEG = ib_MEG_perit[it]
                    #ib_LFP = ib_LFP_perit[it]
                    ib_mvt = ib_mvt_perit[it]

                    mask = np.zeros(l, dtype=bool)
                    mask[ib_mvt] = 1

                    if featn.find('LFP' ) >= 0:
                        for bins in ib_LFP_perit.values():
                            #print('LFP artif nbins ',len(bins))
                            mask[bins] = 0
                    if featn.find('msrc' ) >= 0:
                        for bins in ib_MEG_perit.values():
                            #print('MEG artif nbins ',len(bins))
                            mask[bins] = 0

                    n = np.sum(mask)
                    assert n  > minlen_bins, (n, n/ mask.size)

                    dat_forstat = dat[mask]
                    dats_forstat += [dat_forstat]

                mn,std = utsne.robustMean(dat_forstat,ret_std=1)
                assert abs(std) > 1e-20

                if rescale_separately:
                    X_pri[rawi][:,feati] -= mn
                    X_pri[rawi][:,feati] /= std


            # here I would for normalization stats gather from all
            # participating datasets from the group
            if not rescale_separately:
                dats_forstat = np.hstack(dats_forstat)
                mn,std = utsne.robustMean(dats_forstat,ret_std=1)
                assert abs(std) > 1e-20
                stat_perchan[featn] = (mn,std)

                for rawi in indset_cur:
                    X_pri[rawi][:,feati] -= mn
                    X_pri[rawi][:,feati] /= std

    return X_pri
    #fname_stats = rwnstr + '_stats.npz'
    #np.savez(fname_stats, dat_forstat_perchan=dat_forstat_perchan,
    #         combine_within_medcond=combine_within_medcond,
    #         subjs_analyzed=subjs_analyzed)

def rescaleRaws(raws_permod_both_sides, mod='LFP',
                int_type_templ = 'notrem_{}', minlen_sec = 5, combine_within_medcond=True,
                roi_labels=None, srcgrouping_names_sorted = None, src_labels_ipsi = ['Cerebellum']):
    '''
    modifies raws in place. Rescales to zero mean, unit std
    roi_labels are there only for ipsilateral cerebellum essentially
    '''
    if mod == 'src':
        assert roi_labels is not None
        assert srcgrouping_names_sorted is not None
# intbins_per_raw = {}

#     for rawname_ in raws_permod_both_sides:
#         raw = raws_permod_both_sides[rawname_][mod]
#         ib = upre.getCleanIntervalBins(rn,raws_permod_both_sides[rn]['LFP'])

#     #     side = None
#     #     if side_to_use == 'main_trem':
#     #         side = gv.gen_subj_info[subj]['tremor_side']
#     #     elif side_to_use == 'main_move':
#     #         side = gv.gen_subj_info[subj].get('move_side',None)
#     #     if side is None:
#     #         print('{}: {} is None'.format(rawname_, side_to_use))
#     #     side_letter = side[0].upper()

#     #     sidelet_per_raw[rawname_] = side_letter

#         for sidelets in ['R', 'L']:
#             it = int_type_templ.format(side_letter)
#             assert len(ib[it]) / sfreq > minlen_sec

#         intbins_per_raw[rawname_] = ib

    # defrn = rawnames[0]
    # defraw = raws_permod_both_sides[defrn][mod]
    # defchn = 'LFPL01'
    # defdat,_ = defraw[defchn]
    # ib = intbins_per_raw[defrn][int_type_templ.format('L')]
    # defdat_forstat = defdat[0,ib]
    # def_mn,def_std = utsne.robustMean(defdat_forstat,ret_std=1)
    print('Start raws rescaling for modality {} based on interval type {}'.format(mod,int_type_templ) )
    import utils_tSNE as utsne
    rawnames = list( sorted(raws_permod_both_sides.keys() ) )
    rwnstr = ','.join(rawnames)

    if mod == 'src':
        chn_name_side_ind = 4
    if mod in ['LFP', 'LFP_hires']:
        chn_name_side_ind = 3


    subjs_analyzed = getRawnameListStructure(rawnames)
    dat_forstat_perchan = {}
    if combine_within_medcond:
        #print('subjs_analyzed = ',subjs_analyzed)
        for subj in subjs_analyzed:
            subj_sub = subjs_analyzed[subj]
            tasks = subj_sub['tasks']
            for medcond in subj_sub['medconds']:
                rn0 = subj_sub[medcond][tasks[0]]
                chnames = raws_permod_both_sides[rn0][mod].ch_names

                if mod == 'src':
                    chnames_nicened = utils.nicenMEGsrc_chnames(chnames, roi_labels, srcgrouping_names_sorted,
                                                    prefix='msrc_')
                for chni,chn in enumerate(chnames):
                    dats_forstat = []
                    sidelet = chn[chn_name_side_ind]
                    opsidelet = utils.getOppositeSideStr(sidelet)
                    for task in subj_sub[medcond]:
                        rn = subj_sub[medcond][task]


                        raw = raws_permod_both_sides[rn][mod]
                        sfreq = raw.info['sfreq']

                        if mod in ['LFP', 'LFP_hires']:
                            suffixes = ['_ann_LFPartif']
                        elif mod == 'src':
                            suffixes = ['_ann_MEGartif']
                        ib_perit = getCleanIntervalBins(rn, raw.info['sfreq'], raw.times,suffixes)

                        dat,_ = raw[chn]

                        if mod == 'src' and chnames_nicened[chni].find('Cerebellum') >= 0:
                            it = int_type_templ.format(sidelet)
                        else:
                            it = int_type_templ.format(opsidelet)
                        assert len(ib_perit[it]) / sfreq > minlen_sec

                        ib = ib_perit[it]
                        dat_forstat = dat[0,ib]
                        dats_forstat += [dat_forstat]

                    dats_forstat = np.hstack(dats_forstat)

                    mn,std = utsne.robustMean(dats_forstat,ret_std=1)
                    assert abs(std) > 1e-20

                    dat_forstat_perchan[chn] = (mn,std)

                    # rescale each raw individually based on common stats
                    fun = lambda x: (x-mn) /std
                    for task in subj_sub[medcond]:
                        rn = subj_sub[medcond][task]
                        raw = raws_permod_both_sides[rn][mod]
                        raw.load_data()
                        raw.apply_function(fun,picks=[chn])
    else:
        for rawname_ in raws_permod_both_sides:
            raw = raws_permod_both_sides[rawname_][mod]
            sfreq = raw.info['sfreq']

            if mod in ['LFP', 'LFP_hires']:
                suffixes = ['_ann_LFPartif']
            elif mod == 'src':
                suffixes = ['_ann_MEGartif']
            ib_perit = getCleanIntervalBins(rawname_, raw.info['sfreq'], raw.times,suffixes)

            if mod == 'src':
                chnames_nicened = utils.nicenMEGsrc_chnames(raw.ch_names, roi_labels,
                                                            srcgrouping_names_sorted, prefix='msrc_')
            for chni,chn in enumerate(raw.ch_names):
                sidelet = chn[chn_name_side_ind]
                opsidelet = utils.getOppositeSideStr(sidelet)
                #if rawname_ == defrn and chn == defchn:
                #    continue
                dat,_ = raw[chn]

                if mod == 'src' and chnames_nicened[chni].find('Cerebellum') >= 0:
                    it = int_type_templ.format(sidelet)
                else:
                    it = int_type_templ.format(opsidelet)
                assert len(ib_perit[it]) / sfreq > minlen_sec

                ib = ib_perit[it]
                dat_forstat = dat[0,ib]
                mn,std = utsne.robustMean(dat_forstat,ret_std=1)
                assert abs(std) > 1e-20
                #fun = lambda x: (x-mn) * (def_std/std) + def_mn
                fun = lambda x: (x-mn) /std
                raw.load_data()
                raw.apply_function(fun,picks=[chn])

                # to check
                dat,_ = raw[chn]
                dat_forstat = dat[0,ib]
                mn,std = utsne.robustMean(dat_forstat,ret_std=1)
                dat_forstat_perchan[chn] = (mn,std)

                #print(mn,std)
                assert abs(mn-0) < 1e-10
                assert abs(std-1) < 1e-10

    fname_stats = rwnstr + '_stats.npz'
    np.savez(fname_stats, dat_forstat_perchan=dat_forstat_perchan,
             combine_within_medcond=combine_within_medcond,
             subjs_analyzed=subjs_analyzed)


def getCleanIntervalBins(rawname_,sfreq, times, suffixes = ['_ann_LFPartif'],verbose=False):
    #returns bins belonging to annotations, without ANY artifacs
    #from given suffix (regardless of channel name), as boolean array
    import utils_tSNE as utsne
    import utils
    sfreq = int(sfreq)

    anns_mvt, anns_artif_pri, times2, dataset_bounds = \
    utsne.concatAnns([rawname_],[times] )
    ivalis_mvt = utils.ann2ivalDict(anns_mvt)
    ivalis_mvt_tb, ivalis_mvt_tb_indarrays = utsne.getAnnBins(ivalis_mvt, times,
                                                                0, sfreq, 1, 1,
                                                                  dataset_bounds)
    ivalis_mvt_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_mvt_tb_indarrays)


    anns_artif, anns_artif_pri, times2, dataset_bounds = \
    utsne.concatAnns([rawname_],[times],suffixes )
    ivalis_artif = utils.ann2ivalDict(anns_artif)
    ivalis_artif_tb, ivalis_artif_tb_indarrays = utsne.getAnnBins(ivalis_artif, times,
                                                                0, sfreq, 1, 1,
                                                                  dataset_bounds)
    ivalis_artif_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)

    for it in ivalis_mvt_tb_indarrays_merged:
        ivalbins = ivalis_mvt_tb_indarrays_merged[it]
        mask = np.zeros( len(times), dtype=bool)
        mask[ivalbins] = True

        for artif_type in ivalis_artif_tb_indarrays_merged:
            artif_bins_cur = ivalis_artif_tb_indarrays_merged[artif_type]
            mbefore = np.sum(mask)
            mask[artif_bins_cur] = False
            mafter = np.sum(mask)
            ndiscard = mbefore - mafter
            if ndiscard > 0 and verbose:
                print('{}:{}, {} in {} artifact bins (={:5.2f}s) discarded'.\
                      format(rawname_,it,artif_type,ndiscard,ndiscard/sfreq))

        ivalbins = np.where(mask)[0]
        ivalis_mvt_tb_indarrays_merged[it] = ivalbins
        if len(ivalbins) == 0:
            print('getCleanIntervalBins:  Warning: len(ivalbins) == 0')
    return ivalis_mvt_tb_indarrays_merged


def loadRaws(rawnames,mods_to_load, sources_type = None, src_type_to_use=None,
             src_file_grouping_ind=None, use_saved = True, highpass_lfreq = None):
    '''
    use_saved means using previously done preproc
    '''
    import globvars as gv
    data_dir = gv.data_dir

    raws_permod_both_sides = {}
    for rawname_ in rawnames:
        raw_permod_both_sides_cur = {}
        print('!!!--Loading raws--!!! current rawname --- ',rawname_)


        if 'FTraw' in mods_to_load:
            rawname_FT = rawname_ + '.mat'
            rawname_FT_full = os.path.join( data_dir, rawname_FT )
            f = read_raw_fieldtrip(rawname_FT_full, None)

            badchlist = loadBadChannelList(rawname_,f.ch_names)
            f.info['bads'] = badchlist

            for i,chn in enumerate(f.ch_names):
                #chn = f.ch_names[chi]
                show = 0
                if chn.find('_old') >= 0:
                    f.set_channel_types({chn:'emg'}); show = 1
                elif chn.find('_kil') >= 0:
                    f.set_channel_types({chn:'misc'}); show = 1
                elif chn.find('LFP') >= 0:
                    f.set_channel_types({chn:'bio'}); show = 1  # or stim, ecog, eeg

                if show:
                    print(i, chn )

            raw_permod_both_sides_cur['FTraw'] = f

        if 'resample' in mods_to_load or (not use_saved and 'EMG' in mods_to_load or "MEG" in mods_to_load):
            rawname_resample = rawname_ + '_resample_raw.fif'
            rawname_resample_full = os.path.join(data_dir, rawname_resample)
            raw_resample = mne.io.read_raw_fif(rawname_resample_full)
            if 'resample' in mods_to_load:
                raw_permod_both_sides_cur['resample'] = raw_resample

        if 'LFP' in mods_to_load:
            if use_saved:
                rawname_LFPonly = rawname_ + '_LFPonly'+ '.fif'
                rawname_LFPonly_full = os.path.join( data_dir, rawname_LFPonly )
                raw_lfponly = mne.io.read_raw_fif(rawname_LFPonly_full, None)
            else:
                raw_lfponly = getSubRaw(rawname_, raw=raw_resample, picks = ['LFP.*'])

            raw_permod_both_sides_cur['LFP'] = raw_lfponly
        if 'LFP_hires' in mods_to_load:
            raw_lfp_highres = saveLFP(rawname_, skip_if_exist = 1, sfreq=1024)
            if raw_lfp_highres is None:
                lfp_fname_full = os.path.join(data_dir, '{}_LFP_1kHz.fif'.format(rawname_) )
                raw_lfp_highres = mne.io.read_raw_fif(lfp_fname_full)
            raw_permod_both_sides_cur['LFP_hires'] = raw_lfp_highres

        if 'src' in mods_to_load:
            assert sources_type is not None
            assert src_type_to_use is not None
            assert src_file_grouping_ind is not None
            src_fname_noext = 'srcd_{}_{}_grp{}'.format(rawname_,sources_type,src_file_grouping_ind)
            if src_type_to_use == 'center':
                newsrc_fname_full = os.path.join( data_dir, 'cnt_' + src_fname_noext + '.fif' )
            elif src_type_to_use == 'mean_td':
                newsrc_fname_full = os.path.join( data_dir, 'av_' + src_fname_noext + '.fif' )
            elif src_type_to_use == 'parcel_ICA':
                newsrc_fname_full = os.path.join( data_dir, 'pcica_' + src_fname_noext + '.fif' )
            else:
                raise ValueError('Wrong src_type_to_use {}'.format(src_type_to_use) )
            raw_srconly =  mne.io.read_raw_fif(newsrc_fname_full, None)
            raw_permod_both_sides_cur['src'] = raw_srconly

        if 'EMG' in mods_to_load:
            if use_saved:
                saveRectConv(rawname_, skip_if_exist = 1)
                rectconv_fname_full = os.path.join(data_dir, '{}_emg_rectconv.fif'.format(rawname_) )
                raw_emg = mne.io.read_raw_fif(rectconv_fname_full)
                assert 4 == len(raw_emg.ch_names), rawname_
            else:
                raw_emg = getSubRaw(rawname_, raw=raw_resample, picks = ['EMG.*old'])
            raw_permod_both_sides_cur['EMG'] = raw_emg

        if 'MEG' in mods_to_load:
            if use_saved:
                raise ValueError('not possible')
            else:
                raw_meg = getSubRaw(rawname_, raw=raw_resample, picks = ['MEG.*'])
            raw_permod_both_sides_cur['MEG'] = raw_meg

        if 'afterICA' in mods_to_load:
            rawname_afterICA = rawname_ + '_resample_afterICA_raw.fif'
            rawname_afterICA_full = os.path.join(data_dir, rawname_afterICA)
            raw_afterICA = mne.io.read_raw_fif(rawname_afterICA_full)
            raw_permod_both_sides_cur['afterICA'] = raw_afterICA

        if 'SSS' in mods_to_load:
            rawname_SSS = rawname_ + '_notch_SSS_raw.fif'
            rawname_SSS_full = os.path.join(data_dir, rawname_SSS)
            raw_SSS = mne.io.read_raw_fif(rawname_SSS_full)
            raw_permod_both_sides_cur['SSS'] = raw_SSS

        if highpass_lfreq is not None:
            for mod in raw_permod_both_sides_cur:
                raw_permod_both_sides_cur[mod].\
                filter(l_freq=highpass_lfreq, h_freq=None,picks='all',
                       n_jobs = 6)


        #raw_permod_both_sides_cur['afterICA']

        raws_permod_both_sides[rawname_] = raw_permod_both_sides_cur
    return raws_permod_both_sides


def saveLFP(rawname_naked, f_highpass = 2, skip_if_exist = 1,
                         n_free_cores = 2, ret_if_exist = 0, notch=1, highpass=1,
            raw_FT=None, sfreq=1024, filter_artif_care=1, save_with_anns = 0):
    import globvars as gv
    import multiprocessing as mpr
    lowest_freq_to_preserve = 1.

    data_dir = gv.data_dir

    lfp_fname_full = os.path.join(data_dir, '{}_LFP_1kHz.fif'.format(rawname_naked) )
    if os.path.exists(lfp_fname_full) :
        #subraw = mne.io.read_raw_fif(lfp_fname_full, None)
        #return subraw
        print('{} already exists!'.format(lfp_fname_full) )
        if ret_if_exist:
            raw =  mne.io.read_raw_fif(lfp_fname_full, None)
            if int(raw.info['sfreq'] ) == int(sfreq):
                return raw
            else:
                print('Existing has differe sfreq {} recomputing'.format(raw.info['sfreq'] ) )
        elif skip_if_exist:
            return None
        else:
            print('saveLFP: starting to prepare new LFP file sfreq={}'.format(sfreq) )

    if raw_FT is not None:
        assert raw_FT.info['sfreq'] > 1500
        raw = raw_FT
    else:
        fname = rawname_naked + '.mat'
        fname_full = os.path.join(data_dir,fname)
        if not os.path.exists(fname_full):
            raise ValueError('wrong naked name' + rawname_naked )
        #raw = mne.io.read_raw_fieldtrip(fname_full, None)
        raw = read_raw_fieldtrip(fname_full, None)
        print('Orig sfreq is {}'.format(raw.info['sfreq'] ) )


    subraw = getSubRaw(rawname_naked, picks = ['LFP.*'], raw=raw )
    del raw
    import gc; gc.collect()


    y = {}
    for chname in subraw.ch_names:
        y[chname] = 'eeg'
    subraw.set_channel_types(y)


    num_cores = mpr.cpu_count() - 1
    nj = max(1, num_cores-n_free_cores)
    subraw.resample(sfreq, n_jobs= nj )

    artif_fname = os.path.join(data_dir , '{}_ann_LFPartif.txt')
    if os.path.exists(artif_fname ) and filter_artif_care:
        anns = mne.read_annotations(artif_fname)
        subraw.set_annotations(artif_fname)
    else:
        print('saveLFP: {} does not exist'.format(artif_fname) )

    if notch:
        freqsToKill = np.arange(50, sfreq//2, 50)  # harmonics of 50
        subraw.notch_filter(freqsToKill,  n_jobs= nj)

    if highpass:
        subraw.filter(l_freq=lowest_freq_to_preserve, h_freq=None,skip_by_annotation='BAD_LFP', n_jobs= nj,
                      pad='symmetric')

    if not save_with_anns:
        subraw.set_annotations(mne.Annotations([],[],[]) )

    subraw.save(lfp_fname_full, overwrite=1)
    return subraw

def saveRectConv(rawname_naked, raw=None, rawname = None, maxtremfreq=9, skip_if_exist = 0,
                 lowfreq=10):
    if os.environ.get('DATA_DUSS') is not None:
        data_dir = os.path.expandvars('$DATA_DUSS')
    else:
        data_dir = '/home/demitau/data'

    rectconv_fname_full = os.path.join(data_dir, '{}_emg_rectconv.fif'.format(rawname_naked) )
    if os.path.exists(rectconv_fname_full) and skip_if_exist:
        return None

    #if raw is None:
    #    if rawname is None:
    #        rawname = rawname_naked + '_resample_raw.fif'
    #    fname_full = os.path.join(data_dir, rawname)
    #    # read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
    #    raw = mne.io.read_raw_fif(fname_full, None)

    #emgonly = raw.copy()
    #emgonly.load_data()
    #chis = mne.pick_channels_regexp(emgonly.ch_names, 'EMG.*old')
    emgonly = getSubRaw(rawname_naked, picks = ['EMG.*old'], raw=raw,
                        rawname = rawname)
    assert len(emgonly.ch_names) == 4
    chdata = emgonly.get_data()

    y = {}
    for chname in emgonly.ch_names:
        y[chname] = 'eeg'
    emgonly.set_channel_types(y)

    emgonly.filter(l_freq=lowfreq, h_freq=None, pad='symmetric')

    windowsz = int(emgonly.info['sfreq'] / maxtremfreq)
    print('wind size is {} s = {} bins'.
          format(windowsz/emgonly.info['sfreq'], windowsz))

    # help(emgonly.pick_channels)

    # high passed filtered
    rectconvraw = emgonly  #.copy
    # hilbraw.plot(duration=2)

    rectconvraw.apply_function(np.abs)
    rectconvraw.apply_function(lambda x: np.convolve(x,  np.ones(windowsz),  mode='same') )
    rectconvraw.apply_function(lambda x: x / np.quantile(x,0.75) )

    rectconvraw.save(rectconv_fname_full, overwrite=1)

    return rectconvraw

def read_raw_fieldtrip(fname, info, data_name='data'):
    """Load continuous (raw) data from a FieldTrip preprocessing structure.

    This function expects to find single trial raw data (FT_DATATYPE_RAW) in
    the structure data_name is pointing at.

    .. warning:: FieldTrip does not normally store the original information
                 concerning channel location, orientation, type etc. It is
                 therefore **highly recommended** to provide the info field.
                 This can be obtained by reading the original raw data file
                 with MNE functions (without preload). The returned object
                 contains the necessary info field.

    Parameters
    ----------
    fname : str
        Path and filename of the .mat file containing the data.
    info : dict or None
        The info dict of the raw data file corresponding to the data to import.
        If this is set to None, limited information is extracted from the
        FieldTrip structure.
    data_name : str
        Name of heading dict/ variable name under which the data was originally
        saved in MATLAB.

    Returns
    -------
    raw : instance of RawArray
        A Raw Object containing the loaded data.
    """
    #from ...externals.pymatreader.pymatreader import read_mat
    from mne.externals.pymatreader.pymatreader import read_mat
    from mne.io.fieldtrip.utils import _validate_ft_struct, _create_info

    ft_struct = read_mat(fname,
                         ignore_fields=['previous'],
                         variable_names=[data_name])

    _validate_ft_struct(ft_struct)
    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    info = _create_info(ft_struct, info)  # create info structure
    trial_struct = ft_struct['trial']
    if isinstance(trial_struct, list) and len(trial_struct) > 1:
        data = np.hstack( trial_struct)
    else:
        data = np.array(ft_struct['trial'])  # create the main data array

    if data.ndim > 2:
        data = np.squeeze(data)

    if data.ndim == 1:
        data = data[np.newaxis, ...]

    if data.ndim != 2:
        raise RuntimeError('The data you are trying to load does not seem to '
                           'be raw data')

    raw = mne.io.RawArray(data, info)  # create an MNE RawArray
    return raw

def getCompInfl(ica,sources, comp_inds = None):
    if comp_inds is None:
        comp_inds = np.arange(ica.n_components_)

    sel = comp_inds

    # unmix_large = np.eye(ica.pca_components_.shape[0])
    # unmix_large[:ica.n_components_, :ica.n_components_]  = ica.unmixing_matrix_
    # unmix_appl = np.dot(unmix_large,   ica.pca_components_ ) [sel, :]

    mix_large = np.eye(ica.pca_components_.shape[0])
    mix_large[:ica.n_components_, :ica.n_components_]  = ica.mixing_matrix_
    mix_appl = ica.pca_components_.T @ mix_large  #np.dot(mix_large,   ica.pca_components_ ) [ sel]
    mix_appl = mix_appl [:, sel]

    #print(mix_appl.shape, unmix_appl.shape)
    print(mix_appl.shape, sources[comp_inds].shape)


    assert ica.noise_cov is None
    # if not none
    #inved = linalg.pinv(self.pre_whitener_, cond=1e-14)

    # influence of component on every channel
    infl = []
    for curi in sel:
        r = np.dot(mix_appl[ :,[curi]] , sources[[curi]])
        r += ica.pca_mean_[curi]
        print(curi, r.shape)

        r *= ica.pre_whitener_[curi]


        infl += [r ]
    return infl #np.vstack( infl )

def readInfo(rawname, raw, sis=[1,2], check_info_diff = 1, bandpass_info=0 ):
    import globvars as gv
    data_dir = gv.data_dir

    import pymatreader
    infos = {}
    for si in sis:
        info_name = rawname + '{}_info.mat'.format(si)
        fn = os.path.join(data_dir,info_name)
        if not os.path.exists(fn):
            continue
        rr  =pymatreader.read_mat(fn )
        print( rr['info']['chs'].keys() )
        print( len( rr['info']['chs']['loc'] ) )
        info_Jan = rr['info']
        chs_info_Jan = info_Jan['chs']

        infos[si] = info_Jan

    assert len(infos) > 0

    if len(infos) > 1 and check_info_diff:
        from deepdiff import DeepDiff
        dif = DeepDiff(infos[1],infos[2])
        dif_ch = DeepDiff(infos[1]['chs'],infos[2]['chs'])
        print('Dif betwenn infos is ',dif)
        assert len(dif_ch) == 0


    import copy
    unmod_info = raw.info
    mod_info  = copy.deepcopy(unmod_info)
    fields = ['loc', 'coord_frame', 'unit', 'unit_mul', 'range',
              'scanno', 'cal', 'logno', 'coil_type', 'kind' ]
    #fields += ['coil_trans']  #first I had it then I had to remove it becuse of MNE complaints
    for ch in mod_info['chs']:
        chn = ch['ch_name']
        if chn.find('MEG') < 0:
            continue
        ind = chs_info_Jan['ch_name'].index(chn)
        #for i,ch_Jan in enumerate(info_Jan['ch_name']):
        for field in fields:
            ch[field] = chs_info_Jan[field][ind]
        #ch['coord_frame'] = chs_info_Jan['coord_frame'][ind]

    digs = info_Jan['dig']
    fields = digs.keys()
    digpts = []
    for digi in range(len(digs['kind'])):
        curdig = {}
        for key in fields:
            curdig[key] = digs[key][digi]
        curdig_ = mne.io._digitization.DigPoint(curdig)
        digpts.append( curdig_)
    #     digs['kind'][digi]
    #     digs['ident'][digi]
    #     digs['coord_frame']
    #     digs['r']
    mod_info['dig'] = digpts

    # if we load it to use in conjunction with already processed file, maybe we
    # don't want it to be saved. Same with number of channels
    if bandpass_info:
        fields_outer = ['highpass', 'lowpass']
        for field in fields_outer:
            mod_info[field] = info_Jan[field]

    d = info_Jan['dev_head_t']
    mod_info['dev_head_t'] =  mne.transforms.Transform(d['from'],d['to'], d['trans'])


    prj = infos[1]['projs']

    projs = []
    for i in range(len(prj)):
        p = {}
        for k in prj:
            p[k] = prj[k][i]

    #     proj_cur = prj[i]
        if len(p['data']['row_names']) == 0:
            p['row_names'] = None

        if p['data']['data'].ndim == 1:
            p['data']['data'] =  p['data']['data'][None,:]
        one = mne.Projection(kind=p['kind'], active=p['active'], desc=p['desc'],
                        data=p['data'],explained_var=None)

    #     one = Projection(kind=p['kind'], active=p['active'], desc=p['desc'],
    #                      data=dict(nrow=nvec, ncol=nchan, row_names=None,
    #                                col_names=names, data=data),
    #                      explained_var=explained_var)

        projs.append(one)

    mod_info['projs'] = projs

    mne.channels.fix_mag_coil_types(mod_info)

    return mod_info, infos

def extractEMGData(raw, rawname_=None, skip_if_exist = 1, tremfreq = 9):
    # highpass and convolve
    import globvars as gv
    raw.info['bads'] = []

    chis = mne.pick_channels_regexp(raw.ch_names, 'EMG.*old')
    restr_names = np.array( raw.ch_names )[chis]

    emgonly = raw.copy()
    emgonly.load_data()
    emgonly.pick_channels(restr_names.tolist())
    emgonly_unfilt = emgonly.copy()
    print(emgonly.ch_names)
    #help(emgonly.filter)

    y = {}
    for chname in emgonly.ch_names:
        y[chname] = 'eeg'
    emgonly.set_channel_types(y)

    emgonly.filter(l_freq=10, h_freq=None, picks='all',pad='symmetric')

    sfreq = raw.info['sfreq']
    windowsz = int( sfreq / tremfreq )
    print( 'wind size is {} s = {} bins'.format(windowsz/emgonly.info['sfreq'], windowsz ))

    rectconvraw = emgonly.copy()
    #hilbraw.plot(duration=2)

    rectconvraw.apply_function( np.abs)
    rectconvraw.apply_function( lambda x: np.convolve(x,  np.ones(windowsz),  mode='same') )
    #rectconvraw.apply_function( lambda x: x / np.quantile(x,0.75) )

    rectconvraw.apply_function( lambda x: x / 100 ) # 100 is just empirical so that I don't have to scale the plot

    if rawname_ is not None:
        rectconv_fname_full = os.path.join(gv.data_dir, '{}_emg_rectconv.fif'.format(rawname_) )
        if not (skip_if_exist and os.path.exists(rectconv_fname_full) ):
            print('EMG raw saved to ',rectconv_fname_full)
            rectconvraw.save(rectconv_fname_full, overwrite=1)

    return rectconvraw

def getECGindsICAcomp(icacomp, mult = 1.25, ncomp_test_for_ecg = 6):
    '''
    smaller mult gives stricter rule
    '''
    import utils
    sfreq = int(icacomp.info['sfreq'])
    normal_hr  = [55,105]  # heart rate bounds, Mayo clinic says 60 to 100
    ecg_compinds = []
    ecg_ratio_thr = 6
    rmax = 0
    ratios = []
    for i in range(len(icacomp.ch_names)):
        comp_ecg_test,times = icacomp[i]
        #r_ecg_ica_test = mne.preprocessing.ica_find_ecg_events(filt_raw,comp_ecg_test)
        da = np.abs(comp_ecg_test[0])
        thr = (normal_hr[1]/60) * mult
        qq = np.percentile(da, [ thr, 100-thr, 50 ] )
        mask = da > qq[1]
        bis = np.where(mask)[0]
        pl = False
        r = (qq[1] - qq[2]) / qq[2]
        ratios += [r]
        rmax = max(rmax, r)
        if r < ecg_ratio_thr:
            continue

    strog_ratio_inds = np.where( ratios > ( np.max(ratios) + np.min(ratios) )  /2  )[0]
    nstrong_ratios = len(strog_ratio_inds)
    print('nstrong_ratios = ', nstrong_ratios)

    ecg_evts_all = []
    for i in np.argsort(ratios)[::-1][:ncomp_test_for_ecg]:
        comp_ecg_test,times = icacomp[i]
        #r_ecg_ica_test = mne.preprocessing.ica_find_ecg_events(filt_raw,comp_ecg_test)
        da = np.abs(comp_ecg_test[0])
        thr = (normal_hr[1]/60) * mult
        qq = np.percentile(da, [ thr, 100-thr, 50 ] )
        mask = da > qq[1]
        bis = np.where(mask)[0]

        if i > 8:
            pl = 0
        cvl, ecg_evts  = utils.getIntervals(bis, width=5, thr=1e-5, percentthr=0.95,
                                      inc=5, minlen=2,
                           extFactorL=1e-2, extFactorR=1e-2, endbin = len(mask),
                           include_short_spikes=1, min_dist_between=50, printLog=pl,
                                     percent_check_window_width = sfreq//10)

        nevents = len( ecg_evts )
        #nevents = r_ecg_ica_test.shape[0]

        event_rate_min = 60 * nevents / (icacomp.times[-1] - icacomp.times[0])
        print('ICA comp inds {:2}, ratio={:.2f} event rate {:.2f}'.format(i,ratios[i],event_rate_min) )
        if  event_rate_min >= normal_hr[0]  and event_rate_min <= normal_hr[1]:
            ecg_compinds += [i]
            ecg_evts_all += [ecg_evts]
    return ecg_compinds, ratios, ecg_evts_all


def checkStatsSimilarity(raws):
    '''
    compute some measures (e.g. means and stds), put them in a vector for each
    dataset and then see how large the dispersion is around mean. If it's
    comparable to the mean, then there is a problem
    '''
    return

def concatRaws(raws,rescale=True,interval_for_stats = (0,300) ):
    '''
    concat with rescaling
    '''
    import copy, mne
    import utils_tSNE as utsne

    raws_ = copy.deepcopy(raws)  #otherwise it does stupid damage to the first raw

    means_pri = []
    qs_pri  = []
    for rawi,raw in enumerate(raws):
        means = []
        qs    = []
        # TODO: treat all channels at once to boost speed
        for chi in range(len(raw.ch_names) ):
            if interval_for_stats is not None:
                tbs,tbe = raw.time_as_index( interval_for_stats )
            else:
                tbs, tbe = 0, len(raw.times)-1
            dat, times = raw[chi,tbs:tbe]
            me, mn,mx = utsne.robustMean(dat, axis=1, per_dim =1, ret_aux=1, q = .05)
            #q = np.percentile(dat, [5,95] )
            means += [ me  ]
            qs += [ (mn,mx)  ]
        means_pri += [means]
        qs_pri += [qs]

    if rescale:
        for rawi,raw in enumerate(raws):
            raws_[rawi].load_data()
            for chi in range(len(raw.ch_names) ):
                me0 = means_pri[0][chi]
                me = means_pri[rawi][chi]

                qs0 = qs_pri[0][chi]
                rng0 = qs0[1]-qs0[0]

                qs = qs_pri[rawi][chi]
                rng = qs[1]-qs[0]
                raws_[rawi].apply_function(lambda x: (x-me)/rng * rng0 + me0,  picks = [chi] )

    merged = mne.concatenate_raws(raws_)

    return merged

    #rectconvraw_perside[side] = tmp


def getGenIntervalInfoFromRawname(rawname_, interval=None):
    import utils
    import globvars as gv

    subj,medcond,task  = utils.getParamsFromRawname(rawname_)

    maintremside = gv.gen_subj_info[subj]['tremor_side']
    moveside = gv.gen_subj_info[subj].get('move_side','UNDEF')
    tremfreq = gv.gen_subj_info[subj]['tremfreq']


    nonmaintremside = utils.getOppositeSideStr(maintremside)
    mts_letter = maintremside[0].upper()
    #print(rawname_,'Main trem side ' ,maintremside,mts_letter)
    print('----{}\n{} is maintremside, tremfreq={} move side={}'.format(rawname_, mts_letter,tremfreq,
                                                                          moveside) )
    print(r'^ is tremor, * is main tremor side')

    rawname = rawname_ + '_resample_raw.fif'
    fname_full = os.path.join(gv.data_dir,rawname)

    # read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
    raw = mne.io.read_raw_fif(fname_full, None, verbose=0)

    times = raw.times
    if interval is None:
        interval = (times[0],times[-1])
    else:
        assert len(interval) == 2 and max(interval) <= times[-1] \
        and min(interval)>=0 and interval[1] > interval[0]

    begtime = max(times[0], interval[0] )
    endtime = min(times[-1], interval[1] )
    ##########

    ots_letter = utils.getOppositeSideStr(mts_letter)
    mts_trem_str = 'trem_{}'.format(mts_letter)
    mts_notrem_str = 'notrem_{}'.format(mts_letter)
    mts_task_str = '{}_{}'.format(task,mts_letter)
    ots_task_str = '{}_{}'.format(task,ots_letter)

    ########

    anns_fn = rawname_ + '_anns.txt'
    anns_fn_full = os.path.join(gv.data_dir, anns_fn)
    if os.path.exists(anns_fn_full):
        anns = mne.read_annotations(anns_fn_full)
        #raw.set_annotations(anns)
        anns_descr_to_remove = ['endseg', 'initseg', 'middle', 'incPost' ]
        anns_upd = utils.removeAnnsByDescr(anns, anns_descr_to_remove)
        anns_upd = utils.renameAnnDescr(anns, {'mvt':'hold', 'no_tremor':'notrem'})
    else:
        print(anns_fn_full, ' does not exist')
        anns_upd = None



    #ann_dict = {'Jan':anns_cnv_Jan, 'prev_me':anns_cnv, 'new_me':anns_upd}
    ann_dict = { 'new_me':anns_upd}



    ann_len_dict = {}
    meaningful_totlens = {}
    for ann_name in ann_dict:
        #print('{} interval lengths'.format(ann_name))
        #display(anns_cnv_Jan.description)
        anns = ann_dict[ann_name]
        if anns is None:
            continue
        lens = utils.getIntervalsTotalLens(anns, True, times=raw.times,
                                           interval=interval)

        lens_keys = list(sorted(lens.keys()) )
        for lk in lens_keys:
            lcur = lens[lk]
            lk_toshow = lk
            if lk.find('trem') == 0:  #lk, not lk_toshow!
                lk_toshow = '^' + lk_toshow
            if lk.find('_' + mts_letter) >= 0:
                lk_toshow = '*' + lk_toshow
            print('{:10}: {:6.2f} = {:6.3f}% of total {:.2f}s'.
                  format(lk_toshow, lcur,  lcur/(endtime-begtime) * 100, endtime-begtime))
        #display(lens  )
        #lens_cnv_Jan = utils.getIntervalsTotalLens(anns_cnv_Jan, True, times=raw.times)
        #display(lens_cnv_Jan  )
        if mts_trem_str not in anns.description:
            print('!! There is no tremor, accdording to {}'.format(ann_name))

        meaningul_label_totlen = lens.get(mts_trem_str,0) + lens.get(mts_task_str,0)
        meaningful_totlens[ann_name] = meaningul_label_totlen
        if meaningul_label_totlen < 10:
            print('Too few meaningful labels {}'.format(ann_name))

        for it in lens:
            if it.find(mts_task_str) < 0 and it.find(ots_task_str) >= 0:
                print('{} has task {} which is opposite side to tremor {}'.format(
                    ann_name, ots_task_str, mts_task_str) )
            assert not( it.find(mts_task_str) >= 0 and it.find(ots_task_str) >= 0),\
                'task marked on both sides :('


        print('\n')

    return lens
    # print('\nmy prev interval lengths')
    # lens_cnv = utils.getIntervalsTotalLens(anns_cnv, True, times=raw.times)
    # display(lens_cnv )
    # #display(anns_cnv.description)
    # if mts_trem_str not in anns_cnv.description:
    #     print('!! There is no tremor, accdording to prev me')

#Coord frames:  1  -- device , 4 -- head,


#FIFF.FIFFV_POINT_CARDINAL = 1
#FIFF.FIFFV_POINT_HPI      = 2
#FIFF.FIFFV_POINT_EEG      = 3
#FIFF.FIFFV_POINT_ECG      = FIFF.FIFFV_POINT_EEG
#FIFF.FIFFV_POINT_EXTRA    = 4
#FIFF.FIFFV_POINT_HEAD     = 5  # Point on the surface of the head
#
#
#
#_dig_kind_dict = {
#    'cardinal': FIFF.FIFFV_POINT_CARDINAL,
#    'hpi': FIFF.FIFFV_POINT_HPI,
#    'eeg': FIFF.FIFFV_POINT_EEG,
#    'extra': FIFF.FIFFV_POINT_EXTRA,
#}
#
#
#_cardinal_kind_rev = {1: 'LPA', 2: 'Nasion', 3: 'RPA', 4: 'Inion'}
#
#    kind : int
#        The kind of channel,
#        e.g. ``FIFFV_POINT_EEG``, ``FIFFV_POINT_CARDINAL``.
#    r : array, shape (3,)
#        3D position in m. and coord_frame.
#    ident : int
#        Number specifying the identity of the point.
#        e.g.  ``FIFFV_POINT_NASION`` if kind is ``FIFFV_POINT_CARDINAL``,
#        or 42 if kind is ``FIFFV_POINT_EEG``.
#    coord_frame : int
#        The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.
