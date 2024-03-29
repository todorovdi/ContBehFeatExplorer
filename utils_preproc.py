import mne
import os
import numpy as np
import gc
import utils
import utils_tSNE as utsne
import globvars as gv
from os.path import join as pjoin

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

def recalcMEGArtifacts(rawname, raw,
                        thr_mult, use_mean,
                        flt_thr_mult, flt_use_mean,
                        ICA_thr_mult = 2.3, ICA_use_mean = False,
                       lowest_freq_to_keep=1.5, threshold_muscle = 5,
                       n_jobs=-1, raw_flt = None, raw_notched=None, savedir=None,
                       ICA_force_recalc=0, n_ICA_comp=3, notch_freqsToKill=None,
                      force_ICA_recalc = True, pdf=None ):
    '''
    takes raw (unresamled) un maybe filtered raw raw_flt (if not it does itself)
    threshold_muscle is  z-score, threshold is data depenent
    and computed artif -- normal and after filtering
    '''
    import matplotlib.pyplot as plt

    if len(raw.info['bads'] ):
        raw = raw.copy()
        raw.drop_channels(raw.info['bads'])

    # assuming bad channels were set already

    #n_components_ICA = 0.95
    n_components_ICA = 10  # if I keep it to 0.95, it takes a lot of time
    # on NOT filtered data
    icafname_full = pjoin(savedir, f'{rawname}_artif_det-ica.fif.gz')
    from mne.preprocessing import read_ica,ICA
    if os.path.exists(icafname_full) and not force_ICA_recalc:
        ica = read_ica(icafname_full)
    else:
        ica = ICA(n_components = n_components_ICA, random_state=0).fit(raw)
        ica.save(icafname_full, overwrite=True)
        print(f'ICA saved to {icafname_full}')

    icacomp = ica.get_sources(raw)
    anns_icaMEG_artif, cvl_ica_per_side = utils.findRawArtifacts(icacomp ,
        thr_mult = ICA_thr_mult,
        thr_use_mean = ICA_use_mean, plot_name='ICA', sided=False,
            n_ICA_comp = n_ICA_comp)

    if pdf is not None:
        pdf.savefig(plt.gcf())


    if len(anns_icaMEG_artif) > 0:
        print('Artif found in ICA {}, maxlen {:.3f} totlen {:.3f}'.
                format(anns_icaMEG_artif, np.max(anns_icaMEG_artif.duration),
                        np.sum(anns_icaMEG_artif.duration) ) )
    else:
        print('Artif found in ICA {} is NONE'.  format(anns_icaMEG_artif) )

    if not os.path.exists(savedir ):
        os.makedirs(savedir)
    if savedir is not None:
        fnf = os.path.join(savedir,
            '{}_ann_MEGartif_ICA.txt'.format(rawname) )
        anns_icaMEG_artif.save(fnf, overwrite=True )
        print('Saved ',fnf)

    ###########################################

    if raw_flt is None:
        raw_flt = raw.copy()
        raw_flt.load_data()
        raw_flt.filter(l_freq=lowest_freq_to_keep,
                        h_freq=None, n_jobs=n_jobs) #, skip_by_annotation='BAD_MEG')


    anns_MEG_artif, cvl_per_side = utils.findRawArtifacts(raw ,
        thr_mult = thr_mult,
        thr_use_mean = use_mean, plot_name='')

    if pdf is not None:
        pdf.savefig(plt.gcf())

    if len(anns_MEG_artif) > 0:
        print('Artif found in UNfilt {}, maxlen {:.3f} totlen {:.3f}'.
                format(anns_MEG_artif, np.max(anns_MEG_artif.duration),
                        np.sum(anns_MEG_artif.duration) ) )
    else:
        print('Artif found in UNfilt {} is NONE'.  format(anns_MEG_artif) )

    if not os.path.exists(savedir ):
        os.makedirs(savedir)
    if savedir is not None:
        fnf = os.path.join(savedir, '{}_ann_MEGartif.txt'.format(rawname) )
        anns_MEG_artif.save(fnf, overwrite=True )
        print('Saved ',fnf)


    ######################################

    if raw_notched is None:
        raw_notched = raw.copy()
        raw_notched.notch_filter(notch_freqsToKill, n_jobs=n_jobs)

    # this has to be done after notching
    from mne.preprocessing import annotate_muscle_zscore
    anns_muscle, scores_muscle = annotate_muscle_zscore(
        raw_notched, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
        filter_freq=[110, 140])

    if len( anns_muscle ) > 0:
        print('Artif found muscles {}, maxlen {:.3f} totlen {:.3f}'.
            format(anns_muscle, np.max(anns_muscle.duration),
                    np.sum(anns_muscle.duration) ) )
    else:
        print('Artif found in muscles {} is NONE'.  format(anns_muscle) )

    if savedir is not None:
        anns_muscle.save(os.path.join(savedir, '{}_ann_MEGartif_muscle.txt'.format(rawname) ), overwrite=True )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14,4) )
    ax.plot(raw_notched.times, scores_muscle)
    ax.axhline(y=threshold_muscle, color='r')
    ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity')
    ax.set_xlim( 0, np.max(raw_notched.times ) )

    for an in anns_muscle:
        b0 = an['onset']
        dur = an['duration']
        ax.fill_betweenx( [np.min(scores_muscle), np.max(scores_muscle)],
            b0, b0 + dur, color='red', alpha=0.3)

    if pdf is not None:
        pdf.savefig(fig)

    #################################

    anns_MEG_artif_flt, cvl_per_side = findRawArtifacts(raw_flt ,
        thr_mult = flt_thr_mult,
        thr_use_mean = flt_use_mean,
        plot_name=f'low pass {lowest_freq_to_keep:.1f}')

    if pdf is not None:
        pdf.savefig(plt.gcf())


    if len( anns_MEG_artif_flt ) > 0:
        print('Artif found in filtered {}, maxlen {:.3f} totlen {:.3f}'.
            format(anns_MEG_artif_flt, np.max(anns_MEG_artif_flt.duration),
                    np.sum(anns_MEG_artif_flt.duration) ) )
    else:
        print('Artif found in filtered {} is NONE'.  format(anns_MEG_artif_flt) )

    if savedir is not None:
        anns_MEG_artif_flt.save(os.path.join(savedir, '{}_ann_MEGartif_flt.txt'.format(rawname) ), overwrite=True )
    return anns_MEG_artif, anns_MEG_artif_flt, anns_icaMEG_artif, anns_muscle

def findRawArtifacts(raw , thr_mult = 2.5, thr_use_mean=0,
                     show_max_always=0, data_mod = 'MEG',
                     plot_name = '', sided= True,
                     n_ICA_comp = None):
    '''
    I was initially using it for filtered ([1-100] Hz bandpassed) raw. But maybe it can be used before as well
    '''
    import matplotlib.pyplot as plt
    import utils_tSNE as utsne
    import mne
    from utils import collectChnamesBySide,getIntervals,intervals2anns

    raw_only_mod = raw.copy()
    if data_mod == 'MEG':
        if n_ICA_comp is None:
            raw_only_mod.pick_types(meg=True)
        artif_prefix = 'BAD_MEG'
    elif data_mod == 'LFP':
        if n_ICA_comp is None:
            chns = np.array(raw.ch_names)[ mne.pick_channels_regexp(raw.ch_names,'LFP*') ]
            raw_only_mod.pick_channels(chns)
        artif_prefix = 'BAD_LFP'

    if n_ICA_comp is None:
        raw_only_mod = raw

    assert len(raw_only_mod.info['bads']) == 0, 'There are bad channels!'



    nr = 2
    if n_ICA_comp is not None:
        nr = n_ICA_comp
    fig,axs = plt.subplots(nr,1, figsize=(14,7), sharex='col')

    if sided:
        #chnames_perside_mod, chis_perside_mod = collectChnamesBySide(raw.info)
        chnames_perside_mod, chis_perside_mod = collectChnamesBySide(raw_only_mod.info)
        sides = sorted(chnames_perside_mod.keys())
    else:
        if n_ICA_comp is not None:
            #zz = [ (i, raw_only_mod[chn][0][0] ) for i,chn in\
            #      enumerate(raw_only_mod.ch_names[:n_ICA_comp] )  ]
            zz = list( enumerate(raw_only_mod.ch_names[:n_ICA_comp] ) )
            chnames_perside_mod = dict(zz)
            #chnames_perside_mod =  {sides[0]: raw.ch_names }
            sides = list( chnames_perside_mod.keys() )
        else:
            sides = ['both']
            chnames_perside_mod =  {sides[0]: raw.ch_names }

    anns = mne.Annotations([],[],[])
    cvl_per_side = {}
    for sidei,side in enumerate(sides ):
        chnames_curside = chnames_perside_mod[side]
        moddat, times = raw_only_mod[chnames_curside]
        #moddat = raw_only_mod.get_data()

        if n_ICA_comp is not None:
            pos = True
        else:
            pos = False
        # first rescale using only 50% of the data. We take so little compute
        # the data range
        me, mn,mx = utsne.robustMean(moddat, axis=1, per_dim =1, ret_aux=1,
                                     q = .25, pos = pos)
        if np.min(mx-mn) <= 1e-15:
            raise ValueError('mx == mn for side {}'.format(side) )
        moddat_scaled = ( moddat - me[:,None] ) / (mx-mn)[:,None]

        # then sum of absolute values
        moddat_sum = np.sum(np.abs(moddat_scaled),axis=0)
        # use 90% of the data, here pos should always be True, because it is
        # sum of abs values
        me_s, mn_s,mx_s = utsne.robustMean(moddat_sum, axis=None, per_dim =1,
                                           ret_aux=1, pos = 1, q=0.05)

        # divide by max or by mean?
        if thr_use_mean:
            moddat_sum_mod = moddat_sum/ me_s
            lbl = '/me_s'
        else:
            moddat_sum_mod = moddat_sum/ mx_s
            lbl = '/mx_s'
        mask= moddat_sum_mod > thr_mult
        cvl,ivals_mod_artif = getIntervals(np.where(mask)[0] ,\
            include_short_spikes=1, endbin=len(mask), minlen=2, thr=0.001, inc=1,\
            extFactorL=0.1, extFactorR=0.1 )
        cvl_per_side[side] = cvl

        print('{} artifact intervals found (bins) {}'.format(data_mod ,ivals_mod_artif) )
        #import ipdb; ipdb.set_trace()

        mx_plot = np.max( moddat_sum_mod)
        mn_plot = np.min( moddat_sum_mod)

        ax = axs[sidei]
        ax.plot(raw.times,moddat_sum_mod, label= 'moddat_sum_mod' + lbl)
        #ax.axhline( me_s , c='r', ls=':', label='mean_s')
        #ax.axhline( me_s * thr_mult , c='r', ls='--', label = 'mean_s * thr_mult' )
        ax.axhline( thr_mult , c='r', ls='--', label = 'thr_mult' )

        if show_max_always or not thr_use_mean:
            #ax.axhline( mx_s , c='purple', ls=':', label='max_s')
            #ax.axhline( mx_s * thr_mult , c='purple', ls='--', label = 'mx_s * thr_mult')
            ax.axhline( mx_s / me_s * thr_mult , c='purple', ls='--', label = 'mx_s / me_s * thr_mult')
        ax.set_title('{} {} {} artif'.format(plot_name, data_mod,side) )

        for ivl in ivals_mod_artif:
            b0,b1 = ivl
            #b0t,b1t = raw.times[b0], raw.times[b1]
            #anns.append([b0t],[b1t-b0t], ['BAD_MEG{}'.format( side[0].upper() ) ]  )
            #ax.axvline( raw.times[b0] , c='r', ls=':')
            #ax.axvline( raw.times[b1] , c='r', ls=':')

            ax.fill_betweenx( [mn_plot,mx_plot],
                raw.times[b0], raw.times[b1], color='red', alpha=0.3)

        if sided:
            descr =  '{}{}'.format(artif_prefix, side[0].upper() )
        else:
            descr =  '{}'.format(artif_prefix )
        anns_cur_side = intervals2anns(ivals_mod_artif, descr , raw.times )
        anns += anns_cur_side

        ax.set_xlim(raw.times[0], raw.times[-1]  )
        ax.legend(loc='upper right')

        #ax.set_ylim(0,

    return anns, cvl_per_side

def getRaw(rawname_naked, rawname = None ):
    #if os.environ.get('DATA_DUSS') is not None:
    #    data_dir = os.path.expandvars('$DATA_DUSS')
    #else:
    #    data_dir = '/home/demitau/data'

    if rawname is None:
        rawname = rawname_naked + '_resample_raw.fif'
    fname_full = os.path.join(gv.data_dir, rawname)
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
    from globvars import gen_subj_info

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
def getRawnameListStructure(rawnames, change_rest_to_hold=False, ret_glob=False):
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

    subjs_analyzed_glob = {}
    subjs_analyzed_glob['per_medcond'] = {}
    subjs_analyzed_glob['per_task'] = {}

    subj_list = []
    # I use that medcond, tasks don't intersect with keys I use
    for ki,k in enumerate(rawnames):
        #f = raws[k]
        sind_str,medcond,task = utils.getParamsFromRawname(k)
        if change_rest_to_hold and task == 'rest':
            task = 'hold'

        subj_list += [sind_str]

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

        # per medcond within subj
        if medcond in cursubj:
            m = cursubj[medcond]
            if task in m:
                raise ValueError('Duplicate raw key!')
            else:
                m[task] = k
                if 'datasets' not in cursubj[medcond]:
                    cursubj[medcond]['datasets'] = [k]
                    cursubj[medcond]['tasks']   =  [task]
                else:
                    cursubj[medcond]['datasets'] += [k]
                    cursubj[medcond]['tasks']    += [task]
        else:
            cursubj[medcond] = { task: k}
            if 'datasets' not in cursubj[medcond]:
                cursubj[medcond]['datasets'] = [k]
                cursubj[medcond]['tasks']   =  [task]
            else:
                cursubj[medcond]['datasets'] += [k]
                cursubj[medcond]['tasks']    += [task]

        # per task within subj
        if task in cursubj:
            t = cursubj[task]
            if medcond in t:
                raise ValueError('Duplicate raw key!')
            else:
                t[medcond] = k
                if 'datasets' not in cursubj[task]:
                    cursubj[task]['datasets'] = [k]
                else:
                    cursubj[task]['datasets'] += [k]
        else:
            cursubj[task] = { medcond: k}
            if 'datasets' not in cursubj[task]:
                cursubj[task]['datasets'] = [k]
            else:
                cursubj[task]['datasets'] += [k]

        if medcond not in subjs_analyzed_glob['per_medcond']:
            d = { 'datasets': [] }
            subjs_analyzed_glob['per_medcond'][medcond] = d
        subjs_analyzed_glob['per_medcond'][medcond]['datasets'] += [k]

        if task not in subjs_analyzed_glob['per_task']:
            d = { 'datasets': [] }
            subjs_analyzed_glob['per_task'][task] = d
        subjs_analyzed_glob['per_task'][task]['datasets'] += [k]

        subjs_analyzed[sind_str] =  cursubj

    r = subjs_analyzed
    subjs_analyzed_glob['subject_list'] = list(sorted(set(subj_list )))
    if ret_glob:
        r = (subjs_analyzed, subjs_analyzed_glob)
    return r

def genCombineIndsets(rawnames, combine_within):
    '''
    returns list of ndarrays of indices or rawnames
    '''
    import globvars as gv
    assert combine_within in gv.rawnames_combine_types
    if combine_within == 'no':   # dont combine at all, do separately
        indsets = [ [i] for i in range(len(rawnames) ) ]
    else:
        subjs_analyzed, subjs_analyzed_glob = \
            getRawnameListStructure(rawnames, ret_glob=True)
        if combine_within == 'subj':
            indsets = []
            for subj in subjs_analyzed:
                subj_sub = subjs_analyzed[subj]
                indset_cur = []
                dsets = subj_sub['datasets']
                for rn in dsets:
                    indset_cur += [rawnames.index(rn) ]
                indsets += [indset_cur]
        elif combine_within == 'medcond':
            indsets = []
            # for every subject
            for subj,subj_sub in subjs_analyzed.items():
                # for every medcond for this subject
                for medcond in subj_sub['medconds']:
                    indset_cur = []
                    dsets = subj_sub[medcond]['datasets']
                    # putting all datasets in the same indset
                    for rn in dsets:
                        indset_cur += [rawnames.index(rn) ]
                    indsets += [indset_cur]
        elif combine_within == 'task':
            indsets = []
            for subj in subjs_analyzed:
                subj_sub = subjs_analyzed[subj]
                for task in subj_sub['tasks']:
                    indset_cur = []
                    dsets = subj_sub[task]['datasets']
                    for rn in dsets:
                        indset_cur += [rawnames.index(rn) ]
                    indsets += [indset_cur]
        elif combine_within == 'medcond_across_subj':
            indsets = []
            for medcond,pm in subjs_analyzed_glob['per_medcond'].items():
                indset_cur = []
                dsets = pm['datasets']
                for rn in dsets:
                    indset_cur += [rawnames.index(rn) ]
                indsets += [indset_cur]
        elif combine_within == 'task_across_subj':
            indsets = []
            for task,pt in subjs_analyzed_glob['per_task'].items():
                indset_cur = []
                dsets = pt['datasets']
                for rn in dsets:
                    indset_cur += [rawnames.index(rn) ]
                indsets += [indset_cur]
        elif combine_within == 'across_everything':
            indsets = [ np.arange( len(rawnames) ) ]
    return indsets

def plotFeatStatsScatter(rawnames,X_pri, int_types_to_stat,
                         feature_names_pri,sfreq,
                         rawtimes_pri,side_switch_happened_pri, wbd_pri, save_fig=True,
                        figname_prefix='', separate_by = '', artif_handling='reject',
                         combine_couple=('across_everything','no'),
                        bindict_per_rawn = None ):

    print('plotFeatStatsScatter: starting plotting preparation')

    assert artif_handling in ['reject', 'no', 'impute']

    #if not ( isinstance(feature_names_pri, list) and isinstance(feature_names_pri[0], str) ):
    if ( (isinstance(feature_names_pri, list) or isinstance(feature_names_pri,np.ndarray) )\
        and isinstance(feature_names_pri[0], str) ):
        feature_names_pri = len(rawnames) * [feature_names_pri]

    if wbd_pri is None:
        dts = np.diff( rawtimes_pri[0] )
        assert abs(np.mean(dts  ) - 1/sfreq) < 1e-10 and np.var(dts) < 1e-10  # should be times, not bins

        wbd_pri = []
        for times in rawtimes_pri:
            # wbd is bin indices, not times!
            temp = np.vstack([times*sfreq,times*sfreq] ).astype(int)
            temp[1] += 1
            wbd_pri += [temp]


    indsets, means_combine_all, stds_combine_all, stats_per_indset_combine_all = \
        gatherFeatStats(rawnames, X_pri, feature_names_pri,
                             wbd_pri, sfreq, rawtimes_pri,int_types_to_stat
                , side_rev_pri = side_switch_happened_pri,
                combine_within = combine_couple[0],require_intervals_present=False,
                        printLog=False, minlen_bins = 2,
                        artif_handling = artif_handling,
                        bindict_per_rawn=bindict_per_rawn)

    indsets, means_combine_no, stds_combine_no, stats_per_indset_combine_no = \
        gatherFeatStats(rawnames, X_pri, feature_names_pri,
                             wbd_pri, sfreq, rawtimes_pri,
                int_types_to_stat, side_rev_pri = side_switch_happened_pri,
                combine_within = combine_couple[1], require_intervals_present=False, printLog=False,
                        minlen_bins = 2, artif_handling = artif_handling,
                        bindict_per_rawn=bindict_per_rawn)

    #assert len(separate_by) <= 1

    if len(separate_by) > 0:
        if 'feat_type' == separate_by:
            p = "([a-zA-Z]+)_.*"
        elif 'mod' == separate_by:
            p = "(LFP|msrc)[LR].*"
        else:
            raise ValueError('not implemented for separate_by={}'.formate(separate_by) )
        import re
        featnames_per_feat_type = {}
        featnames_per_feat_type_perraw = {}
        for rawi,rawn in enumerate(rawnames) :
            featnames_per_feat_type_curraw = {}
            for fni,fn in enumerate(feature_names_pri[rawi] ):
                r = re.match(p,fn)
                #print(fn,r)
                if r is not None:
                    feat_type = r.groups()[0]
                else:
                    feat_type = 'undef_feat_type'

                if feat_type not in featnames_per_feat_type:
                    featnames_per_feat_type[feat_type] = [fn ]
                else:
                    featnames_per_feat_type[feat_type] += [fn ]

                if feat_type not in featnames_per_feat_type_curraw:
                    featnames_per_feat_type_curraw[feat_type] = [fn ]
                else:
                    featnames_per_feat_type_curraw[feat_type] += [fn ]
            featnames_per_feat_type_perraw[rawn]  = featnames_per_feat_type_curraw

    else:
        featnames_per_feat_type = {}
        featnames_per_feat_type_perraw = {}
        for rawi,rawn in enumerate(rawnames) :
            featnames_per_feat_type_curraw = {}
            #for fni,fn in enumerate(feature_names_pri[rawi] ):
            featnames_per_feat_type['undef_feat_type'] = feature_names_pri[rawi]
            featnames_per_feat_type_curraw['undef_feat_type'] = feature_names_pri[rawi]
            featnames_per_feat_type_perraw[rawn] = featnames_per_feat_type_curraw

    feat_types = list( sorted(featnames_per_feat_type.keys() ) )
    nfeat_types_glob = len(featnames_per_feat_type)

    subjs_analyzed = getRawnameListStructure(rawnames)
    sind_strs = list(sorted(subjs_analyzed.keys()))

    print('plotFeatStatsScatter: starting plotting {}'.format( feat_types ))

    int_types = list(sorted(means_combine_all[0].keys() ))
    nr = len(int_types)
    nc = nfeat_types_glob
    ww = 5
    hh = 4
    import matplotlib.pyplot as plt
    shx,shy = False,False  # 'col','col'
    fig,axs = plt.subplots(nr,nc,figsize=(ww*nc,hh*nr),sharex=shx,sharey=shy)
    axs = axs.reshape( (nr,nc) )
    #plt.subplots_adjust()
    for iti,int_type in enumerate(int_types):
        for rawi,rawn in enumerate(rawnames):
            means_curint_combine_all = means_combine_all[0][int_type]
            stds_curint_combine_all = stds_combine_all[0][int_type]

            if int_type not in means_combine_no[rawi]:
                continue
            stdgl = stds_curint_combine_all
            means_curint = np.nan * np.ones( X_pri[rawi].shape[1] )
            stds_curint  = np.nan * np.ones( X_pri[rawi].shape[1] )
            if stdgl is not None:
                cur_mean = means_combine_no[rawi][int_type]
                if cur_mean is not None:
                    means_curint = (cur_mean - means_curint_combine_all) / stdgl
                    stds_curint  = stds_combine_no[rawi][int_type] / stdgl


            # for debug only
            #stdgl = stds_combine_no[0][int_type]
            #means_curint = (means_combine_no[rawi][int_type] - means_combine_no[0][int_type]) / stdgl
            #stds_curint = stds_combine_no[rawi][int_type] / stdgl

            #pts = np.zeros( (2, len(means_curint)))
            for fti,feat_type in enumerate(feat_types):
                fns_cur = featnames_per_feat_type_perraw[rawn].get(feat_type,None)

                #import pdb; pdb.set_trace()
                if fns_cur is None:
                    print('skip ',feat_type)
                    continue
                pts0 = []
                pts1 = []
                for fn in fns_cur:
                    chi = list(feature_names_pri[rawi]).index(fn)
                #for chi in range(len(means_curint)):
                    mean_curch = means_curint[chi]
                    std_curch = stds_curint[chi]
                    pts0 += [ mean_curch ]
                    pts1 += [ std_curch ]

                pts = [ np.array(pts0), np.array(pts1) ]

                ax = axs[iti,fti]
                ax.scatter(pts[0],pts[1],label='{}: {}'.
                           format(rawnames[rawi], len(fns_cur) ), alpha=0.6)
                ax.set_title('{}: {}'.format(feat_type,int_type))

                ax.legend(loc='upper right')
                ax.set_xlabel('(mean - glob_mean) / glob std')
                ax.set_ylabel('std / glob std')

    for ax in axs.ravel():
        ax.axvline(x=0,ls=':')
        ax.axhline(y=1,ls=':')

    plt.tight_layout()
    if save_fig:
        figfn = '{}:{}_{}_Feat_stats_across.pdf'.format(','.join(sind_strs), figname_prefix, len(rawnames))
        print('Saving fig to',figfn)
        plt.savefig(figfn)


def getArtifForChnOrFeat(s,bindict,lendat):
    chnstarts = ['msrc', 'LFP', 'MEG', 'EMG', 'src']
    chn_mode = False
    for chns in chnstarts:
        if s.startswith(chns):
            chn_mode = True
    if chn_mode:
        return getArtifForChn(s, bindict,lendat)
    else:
        return getArtifForFeat(s, bindict,lendat)

def getArtifForFeat(featn,bindict,lendat):
    import featlist
    r = featlist.parseFeatName(featn)
    chns = [ r['ch1'], r['ch2'] ]
    mask = np.zeros(lendat, dtype=bool)
    res_artif = {}
    for chn in chns:
        if chn is None:
            continue
        res_artif_cur, artif_cur = getArtifForChn(chn,bindict,lendat)
        mask = mask | artif_cur
        res_artif.update(res_artif_cur)
    return res_artif,mask

def getArtifForChn(chn,bindict,lendat):
    mask = np.zeros(lendat, dtype=bool)

    sidelet = ''
    artif_basename = ''
    if chn.startswith('msrc'):
        artif = bindict['artif'].get('MEG',[])
        artif_basename = 'MEG'
        sidelet = chn[4]
    elif chn.startswith('MEG'):
        artif = bindict['artif'].get('MEG',[])
        artif_basename = 'MEG'
        # I need to find the side by looking at coordinates
        #sidelet = chn[3]
    elif chn.startswith('LFP'):
        artif = bindict['artif'].get('LFP',[])
        artif_basename = 'LFP'
        sidelet = chn[3]
    else:
        raise ValueError(f'wrong chn format {chn}')

    assert sidelet in ['L','R']

    res_artif = {}
    for artif_itname,bins in artif.items():
        good_to_put = False
        # it should be really '==', not just part of the string
        if artif_itname == f'BAD_{artif_basename}':
            good_to_put = True
        elif artif_itname == f'BAD_{artif_basename}{sidelet}':
            good_to_put = True
        elif artif_itname == f'BAD_{chn}':
            good_to_put = True

        if good_to_put:
            res_artif[artif_itname] = bins
            mask[bins] = True

    return  res_artif, mask

def collectAllMarkedIntervals( rn,times, main_side, side_rev, sfreq=256,
            ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt',
            printLog=True, allow_missing_files=False,
                              remove_nonmain_artif=True, 
                              verbose=0):
    '''
    * does reversal if needed
    * removes artifacts from the wrong side, but does not remove
    beh state anns from the other side (only if remove_nonmain_artif==True ),
    which should not be so when main side == both
    '''
    # main_side -- main side AFTER reversal

    anndict_per_intcat = {'artif':{}, 'beh_state':{} }
    wbd = np.vstack([times*sfreq,1 + times*sfreq] ).astype(int)
    wrong_brain_sidelet = main_side[0].upper()

    anns_mvt, anns_artif_pri, times2, dataset_bounds = \
    utsne.concatAnns([rn],[times],side_rev_pri=[side_rev],
                     allow_missing=allow_missing_files, verbose=verbose)
    #ivalis_mvt = utils.ann2ivalDict(anns_mvt)
    lens_mvt = utils.getIntervalsTotalLens(anns_mvt, include_unlabeled
                                            =False, times=times)
    if printLog:
        print(rn,lens_mvt)

    anndict_per_intcat['beh_state'] = anns_mvt


    anns_MEGartif, anns_artif_pri, times2, dataset_bounds = \
        utsne.concatAnns([rn],[times],[ann_MEGartif_prefix_to_use],
                         side_rev_pri=[side_rev],
                         allow_missing=allow_missing_files, verbose=verbose )

    lens_MEGartif = utils.getIntervalsTotalLens(anns_MEGartif, include_unlabeled =False,
                                            times=times)
    if printLog:
        print(rn,lens_MEGartif)
    # here I don't want to remove artifacts from "wrong" brain side because
    # we use ipsilateral CB
    anndict_per_intcat['artif']['MEG'] = anns_MEGartif

    anns_LFPartif, anns_artif_pri, times2, dataset_bounds = \
        utsne.concatAnns([rn],[times],['_ann_LFPartif'],
                         side_rev_pri=[side_rev],
                         allow_missing=allow_missing_files, verbose=verbose )
    lens_LFPartif = utils.getIntervalsTotalLens(anns_LFPartif, include_unlabeled =False,
                                            times=times)
    if printLog:
        print(rn,lens_LFPartif)

    if remove_nonmain_artif:
        # note that we do it after reversal so it should be fine
        anns_LFPartif = utils.removeAnnsByDescr(anns_LFPartif,\
                ['artif_LFP{}'.format(wrong_brain_sidelet) ])

    anndict_per_intcat['artif']['LFP'] = anns_LFPartif

    return anndict_per_intcat

def markedIntervals2Bins(anndict_per_intcat,times,sfreq,wbd=None):
    bindict_per_bintype = {'artif':{}, 'beh_state':{} }
    if wbd is None:
        wbd = np.vstack([times*sfreq,1 + times*sfreq] ).astype(int)


    #anns_mvt, anns_artif_pri, times2, dataset_bounds = \
    #utsne.concatAnns([rn],[times],side_rev_pri=[side_rev],
    #                 allow_missing=allow_missing_files)
    ##ivalis_mvt = utils.ann2ivalDict(anns_mvt)
    #lens_mvt = utils.getIntervalsTotalLens(anns_mvt, include_unlabeled
    #                                        =False, times=times)
    #if printLog:
    #    print(rn,lens_mvt)
    #import pdb; pdb.set_trace()
    anns_mvt = anndict_per_intcat['beh_state']

    ib_mvt_perit_merged = \
    utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_mvt) ,
                    sfreq,ret_type='bins_contig',
                    wbd_type='contig',
                    ret_indices_type =
                        'window_inds', nbins_total=len(times) )

    bindict_per_bintype['beh_state'] = ib_mvt_perit_merged


    #anns_MEGartif, anns_artif_pri, times2, dataset_bounds = \
    #    utsne.concatAnns([rn],[times],[ann_MEGartif_prefix_to_use],
    #                     side_rev_pri=[side_rev],
    #                     allow_missing=allow_missing_files )
    #lens_MEGartif = utils.getIntervalsTotalLens(anns_MEGartif, include_unlabeled =False,
    #                                        times=times)
    #if printLog:
    #    print(rn,lens_MEGartif)

    anns_MEGartif = anndict_per_intcat['artif']['MEG']
    # here I don't want to remove artifacts from "wrong" brain side because
    # we use ipsilateral CB
    ib_MEG_perit_merged = \
        utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_MEGartif) ,
                                        sfreq,ret_type='bins_contig',
                                        wbd_type='contig',
                                        ret_indices_type =
                                            'window_inds', nbins_total=len(times) )
    bindict_per_bintype['artif']['MEG'] = ib_MEG_perit_merged

    #anns_LFPartif, anns_artif_pri, times2, dataset_bounds = \
    #    utsne.concatAnns([rn],[times],['_ann_LFPartif'],
    #                     side_rev_pri=[side_rev],
    #                     allow_missing=allow_missing_files )
    #lens_LFPartif = utils.getIntervalsTotalLens(anns_LFPartif, include_unlabeled =False,
    #                                        times=times)
    #if printLog:
    #    print(rn,lens_LFPartif)
    #anns_LFPartif = utils.removeAnnsByDescr(anns_LFPartif, ['artif_LFP{}'.format(wrong_brain_sidelet) ])
    anns_LFPartif = anndict_per_intcat['artif']['LFP']

    ib_LFP_perit_merged = \
            utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_LFPartif) ,
                                            sfreq,ret_type='bins_contig',
                                            wbd_type='contig',
                                            ret_indices_type =
                                                'window_inds', nbins_total=len(times) )

    bindict_per_bintype['artif']['LFP'] = ib_LFP_perit_merged

    return bindict_per_bintype

def collectAllMarkedIntervalBins(rn,times,main_side, side_rev,
            sfreq=256, ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt',
                                 printLog=True,
                                 allow_missing_files=False,
                                remove_nonmain_artif=True, wbd=None,
                                verbose = 0):

    anndict_per_intcat = collectAllMarkedIntervals(rn,times,main_side,
        side_rev, sfreq=sfreq,
        ann_MEGartif_prefix_to_use = ann_MEGartif_prefix_to_use,
        printLog=printLog, allow_missing_files=allow_missing_files,
        remove_nonmain_artif=remove_nonmain_artif, verbose=verbose)

    return markedIntervals2Bins(anndict_per_intcat,times,sfreq,wbd=wbd)

def collecInfoForPlotHistAcrossDatasets(raws_permod_both_sides,
    aux_info_perraw=None, fnames_noext = None,  modalities = ['src', 'LFP'],
    qch_hist_xshift = 0.1, qmult = 1.15,
    side_to_use =  'main_move', int_types_templ=None,
     ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt'           ):
    '''
    qmult  # how much to multiply the qunatile span
    qch_hist_xshift
    '''
    if fnames_noext is None:
        fnames_noext = list(sorted(raws_permod_both_sides.keys() ))

    assert int_types_templ is not None



    xshifts_rel_perint_permod = {}  # set shifts for displaying so that histograms don't interesect
    dat_permod_perraw_perint = {}
    for mod in modalities:
        dat_permod_perraw_perint[mod] = {}
        xshifts_rel_perint = {}


        n_channels_all = [ len(raws_permod_both_sides[rn][mod].ch_names) for rn in fnames_noext ]
        nmn,nmx = np.min(n_channels_all), np.max(n_channels_all)
        assert nmn==nmx, (nmn,nmx)
        n_channels = n_channels_all[0]
        for int_type in int_types_templ:
            xshifts_rel = [0]*n_channels
            xshifts_rel_perint[int_type] = xshifts_rel

        # first gather info
        for i in range(len(fnames_noext)):
            rawind = i
            rawname_ = fnames_noext[rawind]
            dat_permod_perraw_perint[mod][rawname_] = {}
            subj,medcond,task  = utils.getParamsFromRawname(rawname_)
            #raw = subraws[mod ][rawind]
            raw = raws_permod_both_sides[rawname_][mod]
            sfreq = int(raw.info['sfreq'] )

            #fname_full_LFPartif = os.path.join(gv.data_dir, '{}_ann_LFPartif.txt'.format(rawname_) )
            #anns_LFP_artif = mne.read_annotations(fname_full_LFPartif)

            anns_mvt, anns_artif_pri, times2, dataset_bounds = \
            utsne.concatAnns([rawname_],[raw.times] )
            ivalis_mvt = utils.ann2ivalDict(anns_mvt)
            ivalis_mvt_tb, ivalis_mvt_tb_indarrays = utsne.getAnnBins(ivalis_mvt, raw.times,
                                                                        0, sfreq, 1, 1,
                                                                        dataset_bounds)
            ivalis_mvt_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_mvt_tb_indarrays)


            if mod == 'LFP':
                prefixes = ['_ann_LFPartif']
            elif mod in ['MEG','src']:
                prefixes = [ann_MEGartif_prefix_to_use]
            anns_artif, anns_artif_pri, times2, dataset_bounds = \
            utsne.concatAnns([rawname_],[raw.times],prefixes )
            ivalis_artif = utils.ann2ivalDict(anns_artif)
            ivalis_artif_tb, ivalis_artif_tb_indarrays = utsne.getAnnBins(ivalis_artif, raw.times,
                                                                        0, sfreq, 1, 1,
                                                                        dataset_bounds)
            ivalis_artif_tb_indarrays_merged = \
                utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)


            # I set it to something now just to avoid syntax errors
            meg_chis = None
            src_chis = None
            for j in range(len(int_types_templ)):
                if mod == 'MEG':
                    chdata,times = raw[meg_chis,:]
                    chnames = np.array(raw.ch_names)[meg_chis]
                elif mod == 'msrc':
                    chdata,times = raw[src_chis,:]
                    chnames = np.array(raw.ch_names)[src_chis]
                else:
                    chdata = raw.get_data()
                    chnames = raw.ch_names

                #side = None
                #if side_to_use == 'main_trem':
                #    side = gv.gen_subj_info[subj]['tremor_side']
                #elif side_to_use == 'main_move':
                #    side = gv.gen_subj_info[subj].get('move_side',None)
                #if side is None:
                #    print('{}: {} is None'.format(rawname_, side_to_use))
                #side_letter = side[0].upper()

                #side_rev = aux_info_perraw[rawname_]['side_switched']
                if aux_info_perraw is not None:
                    main_side =  aux_info_perraw[rawname_]['main_body_side']
                    side_letter = main_side[0].upper()
                else:
                    side_letter = utils.getMainSide(subj,side_to_use)

                itcur = int_types_templ[j]
                int_type_cur = itcur.format(side_letter)

                ivalbins = ivalis_mvt_tb_indarrays_merged.get(int_type_cur, None )
                if ivalbins is None:
                    print(f'{rawname_},{mod}: No artifacts found for {int_type_cur}')
                    continue
                mask = np.zeros(chdata.shape[1], dtype=bool)
                mask[ivalbins] = True
                dat_permod_perraw_perint[mod][rawname_][itcur] = [0]*len(chnames) #chds

                for chni,chn in enumerate(chnames):
                    artif_bins_cur = ivalis_artif_tb_indarrays_merged.\
                        get('BAD_{}'.format(chn),[])
                    mbefore = np.sum(mask)
                    mask[artif_bins_cur] = False
                    mafter = np.sum(mask)
                    ndiscard = mbefore - mafter
                    if ndiscard > 0:
                        print('{}:{} in {} {} artifact bins (={:5.2f}s) discarded'.\
                            format(rawname_,chn,int_type_cur,ndiscard,ndiscard/sfreq))
                    chd = chdata[chni,mask] # noe that it is not modified

                    dat_permod_perraw_perint[mod][rawname_][itcur][chni] = chd

                    #if chd.size < 10:
                    #chd = chdata[0,sl]

                    # compute spread
                    r = np.quantile(chd,1-qch_hist_xshift)- np.quantile(chd,qch_hist_xshift)
                    xshift = r * qmult
                    xshifts_rel_perint[itcur][chni] = max(xshifts_rel_perint[itcur][chni], xshift)
        xshifts_rel_perint_permod[mod] = xshifts_rel_perint

    print('\nStats gather finished')
    return dat_permod_perraw_perint, xshifts_rel_perint_permod

def plotHistAcrossDatasets(raws_permod_both_sides, dat_permod_perraw_perint,
        xshifts_rel_perint_permod, src_chis , aux_info_perraw,
        artifact_handling = 'no', stat_per_int = None,
        rawnames_to_show=None, modalities=['src', 'LFP'],
        show_std = False,
        int_types_templ = ['trem_{}', 'notrem_{}', 'hold_{}', 'move_{}'],
        cmap = None, ann_MEGartif_prefix_to_use ='_ann_MEGartif_flt',
        highpass_used = False, qsh = 5e-2, qsh_disp = 5e-3,
        nbins_hist = 100, alpha=0.7):
    '''
    qsh = 5e-2  # what will be used for limits computations
    qsh_disp = 5e-3 # what will be given to hist function
    data is NOT scaled, only shifted
    normally  dat_permod_perraw_perint  contain data with already rejected artif
    mean_dict -- dict (interval name -> means)
    '''

    import matplotlib.pyplot as plt

    if stat_per_int  is not None:
        assert isinstance(stat_per_int,dict)

    if int_types_templ is None:
        int_types_templ = ['trem_{}', 'notrem_{}', 'hold_{}', 'move_{}']
    if rawnames_to_show is None:
        rawnames_to_show = list(sorted(raws_permod_both_sides.keys() ) )

    if cmap is None:
        vals = np.linspace(0,1, 20)
        #np.random.shuffle(vals)
        cmap = plt.cm.colors.ListedColormap(plt.cm.tab20(vals))

    #timerange = 0,100
    #timerange = None
    nr = len(rawnames_to_show);
    nc = len(int_types_templ)
    #nr =2
    ww = 10; hh = 3


    subjinds = [ int( rn[1:3] ) for rn in rawnames_to_show]
    subjinds = list(sorted(set(subjinds) ) )
    subjindlist_str = ','.join(map(str,subjinds) )
    for mod in modalities:
        fig,axs = plt.subplots(nrows=nr, ncols=nc, figsize= (nc*ww,nr*hh), sharex='col')
        axs = axs.reshape( (nr,nc) )
        plt.subplots_adjust(left=0.03,right=0.99, bottom=0.02,top=0.97)
        mns = nc*[np.inf]
        mxs = nc*[-np.inf]

        for i,rawname_ in enumerate(rawnames_to_show):
            rawind = i
            rawname_ = rawnames_to_show[rawind]
            main_side =  aux_info_perraw[rawname_]['main_body_side']
            main_side = main_side[0].upper()
            side_rev = aux_info_perraw[rawname_]['side_switched']
            raw = raws_permod_both_sides[rawname_][mod ]
            sfreq = int( raw.info['sfreq'] )

            bindict_per_bintype = collectAllMarkedIntervalBins(rawname_,
                raw.times,main_side,
                side_rev, sfreq,
                ann_MEGartif_prefix_to_use = ann_MEGartif_prefix_to_use)

            for j in range(nc):
                ax = axs[i,j]
                itcur = int_types_templ[j]

                # it will be reset later if interval is indeed found
                ax.set_title('{} {} interval_type={}, 0s'.\
                            format(rawnames_to_show[rawind], mod, itcur) )


                chds = dat_permod_perraw_perint[mod][rawname_].get(itcur,None)
                if chds is None:
                    continue

                meg_chis = None
                if mod == 'MEG':
                    chnames = np.array(raw.ch_names)[meg_chis]
                elif mod == 'src':
                    chnames = np.array(raw.ch_names)[src_chis]
                else:
                    chnames = raw.ch_names
                for chni,chn in enumerate(chnames):
                    chdata, times = raw[chn]
                    clr = cmap(vals[chni ])

                    chd = chds[chni]

                    if artifact_handling == 'reject':
                        _,artif_mask = getArtifForChn(chn, bindict_per_bintype,
                                                    chd.shape[-1] )
                        adur = np.sum(artif_mask) / sfreq
                        print(f'{rawname_},{chn}: Rejecting {adur}s of artifacts')
                        chd = chd[~artif_mask]

                    # sum shifts over all prev channel indices
                    xshift = np.sum( xshifts_rel_perint_permod[mod][itcur][:chni] )
                    chd2 = chd + xshift
                    if chd2.size == 0:
                        print('fdf')
                        continue

                    # I have to do it per channel because chds is a list0.7 of
                    # arrays of varying length (because different channels have
                    # different artifacts)
                    q0 = np.quantile(chd2,qsh)
                    q1 = np.quantile(chd2,1-qsh)
                    q0_disp = np.quantile(chd2,qsh_disp)
                    q1_disp = np.quantile(chd2,1-qsh_disp)

                    lbl = f'{chn}:{chd2.size}'
                    ax.hist(chd2, bins=nbins_hist, label=lbl, alpha = alpha,
                            range=(q0_disp,q1_disp), color=clr )
                    if stat_per_int is None:
                        # normal mean of the data with (hand-marked)
                        # artifacts discarded
                        mean_cur =  np.mean(chd2)
                        if show_std:
                            std_cur =  np.std(chd2)
                    else:
                        a = stat_per_int[itcur.format(main_side) ]
                        if a is not None:
                            mean_cur,std_cur = a[chn]
                        else:
                            mean_cur =  np.mean(chd2)
                            if show_std:
                                std_cur =  np.std(chd2)
                        # robust mean of the data with (hand-marked)
                        # artifacts discarded
                        #mean_cur_ = mean_dict[itcur.format(main_side) ]
                        #if mean_cur_ is not None:
                        #    mean_cur = mean_cur_[chni] + xshift
                        #else:
                        #    mean_cur =  np.mean(chd2)
                    ax.axvline(x=mean_cur, c=clr, ls=':')
                    if show_std:
                        ax.axvline(x=mean_cur - std_cur, c=clr, ls=':')
                        ax.axvline(x=mean_cur + std_cur, c=clr, ls=':')

                    mns[j] = min(mns[j], q0)
                    mxs[j] = max(mxs[j], q1)

    #                 if chn == 'LFPL12':
    #                     print(mod,rawname_,itcur,'LFPL12',np.min(chd2),np.max(chd2))

                    #print('{} shift {}  q = {},{}'.format(chn,xshift,q0,q1) )


                ax.legend(loc='upper left')
                ax.grid()
                ax.set_title('{} {} interval_type={}, {:.2f}s'.
                            format(rawnames_to_show[rawind], mod, itcur, len(chd2)/raw.info['sfreq'] ) )

            print('  {} of {} finished'.format(mod, rawname_))
        for i in range(nr):
            for j in range(nc):
                if not np.any( np.isinf([mns[j], mxs[j] ] ) ):
                    axs[i,j].set_xlim(mns[j],mxs[j])

        fig_fname = '{}_stat_across_subj_highpass{}_{}_artif_{}_locmeans{}.pdf'.\
            format(subjindlist_str,highpass_used,mod,artifact_handling,
                   int(stat_per_int is None) )
        fig_fname_full = pjoin(gv.dir_fig,fig_fname)
        plt.savefig(fig_fname_full)
        plt.close()
        print('{} finished'.format(fig_fname))

def gatherFeatStats(rawnames, X_pri, featnames_pri, wbd_pri,
                 sfreq, times_pri, int_types, main_side=None,
                 side_rev_pri = None,
                 minlen_bins = 5 * 256 / 32, combine_within='no',
                    require_intervals_present = 1,
                    ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt',
                    printLog=True, artif_handling = 'reject',
                   bindict_per_rawn = None, verbose =0 ):
    '''
    it assumes that featnams are constant across datasets WITHIN indset.
        So I cannot apply it to raws from different subjects really. But from the same subject I can
    int_types should be same (after reversal) across all datasets
    main_side -- AFTER reversal (because I pass side_rev to concatAnns)
    usually notrem_<sidelet>

    bindict_per_rawn[rawn ]['beh_state'] and bindict_per_rawn[rawn ]['artif']['MEG' | 'LFP']
    when bindict_per_rawn is set, side_rev_pri has no effect

    returns indsets,means,stds each is a list of dicts (key = interval type)
    '''
    if isinstance(require_intervals_present, int) or isinstance(require_intervals_present, bool):
        require_all_intervals_present = require_intervals_present
        if require_all_intervals_present:
            intervals_required = int_types
        else:
            intervals_required = []
    elif isinstance(require_intervals_present, list):
        intervals_required = require_intervals_present
    else:
        raise ValueError(f"wrong type of require_intervals_present {type(require_intervals_present) }")

    if int_types is None:
        int_types = [ 'entire' ]
    for int_type in int_types:
        assert int_type.find('{}') < 0  # it should not be a template

    assert len(rawnames) == len(X_pri)
    assert len(rawnames) == len(side_rev_pri)

    assert artif_handling in ['reject', 'no', 'impute']

    if isinstance( X_pri[0], mne.io.BaseRaw):
        featnames_pri = len(X_pri) * [0]
        wbd_pri = len(X_pri) * [0]
        times_pri = len(X_pri) * [0]
        X_pri_new = len(X_pri) * [0]
        for rawi,raw in enumerate(X_pri):
            if sfreq is None:
                sfreq = int( raw.info['sfreq'] )
            assert int( raw.info['sfreq'] ) == int(sfreq)
            assert set( raw.ch_names ) == set(featnames_pri[0] )
            tmp = np.vstack( [raw.times,raw.times] )
            tmp[1] += 1 / sfreq
            wbd_pri[rawi]   = [ tmp ]
            times_pri[rawi] = [raw.times]
            X_pri_new[rawi] =  raw.get_data()
            featnames_pri[rawi] = raw.ch_names
    else:
        if (isinstance(featnames_pri,list) or isinstance(featnames_pri,np.ndarray) ) and\
                isinstance(featnames_pri[0],str):
            featnames_pri = [ featnames_pri ] * len(rawnames)
        assert len(rawnames) == len(featnames_pri)

        for i in range(len(X_pri) ):
            assert len(featnames_pri[i]) == X_pri[i].shape[1],  ( len(featnames_pri[i]),  X_pri[i].shape )

    import globvars as gv

    if wbd_pri is None:
        wbd_pri = []
        for times in times_pri:
            # wbd is bin indices, not times!
            temp = np.vstack([times*sfreq,times*sfreq] ).astype(int)
            temp[1] += 1
            wbd_pri += [temp]
    assert len(rawnames) == len(wbd_pri)

    assert len(rawnames) == len(wbd_pri)
    assert len(rawnames) == len(times_pri)
    assert combine_within in gv.rawnames_combine_types
    #['subject', 'medcond', 'task', 'no', 'medcond_across_subj', 'task_across_subj',
                              #'across_everything']

    if isinstance(int_types,str):
        int_types = [int_types]

    if main_side is None:
        main_side = int_types[0][-1].upper()

    # get indsets based on combination strategy
    indsets = genCombineIndsets(rawnames, combine_within)

    # collect annotations
    #ib_MEG_perit_perraw = {}
    #ib_LFP_perit_perraw = {}
    #ib_mvt_perit_perraw = {}
    if bindict_per_rawn is None:
        bindict_per_rawn = {}
        bindict_set = False
    else:
        bindict_set = True

    for rawi,rn in enumerate(rawnames):
        #ib_MEG_perit = getCleanIntervalBins(rn,sfreq, times,['_ann_MEGartif'] )
        #ib_LFP_perit = getCleanIntervalBins(rn,sfreq, times,['_ann_LFPartif'] )

        wbd = wbd_pri[rawi]
        times = times_pri[rawi]


        if not bindict_set :
            side_rev = side_rev_pri[rawi]
            bindict_per_bintype = \
            collectAllMarkedIntervalBins(rn,times,main_side, side_rev, sfreq,
                wbd = wbd, ann_MEGartif_prefix_to_use = ann_MEGartif_prefix_to_use,
                allow_missing_files=False, verbose=verbose)

            bindict_per_rawn[rn ] =bindict_per_bintype

        #anns_mvt, anns_artif_pri, times2, dataset_bounds = \
        #utsne.concatAnns([rn],[times],side_rev_pri=[side_rev])
        ##ivalis_mvt = utils.ann2ivalDict(anns_mvt)
        #lens_mvt = utils.getIntervalsTotalLens(anns_mvt, include_unlabeled
        #                                       =False, times=times_pri[rawi])
        #print(rn,lens_mvt)
        ##import pdb; pdb.set_trace()
        #ib_mvt_perit_merged = \
        #utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_mvt) ,
        #                sfreq,ret_type='bins_contig',
        #                wbd_type='contig',
        #                ret_indices_type =
        #                    'window_inds', nbins_total=len(times) )


        #anns_MEGartif, anns_artif_pri, times2, dataset_bounds = \
        #    utsne.concatAnns([rn],[times],[ann_MEGartif_prefix_to_use],
        #                     side_rev_pri=[side_rev] ) lens_MEGartif =
        #utils.getIntervalsTotalLens(anns_MEGartif, include_unlabeled =False,
        #                            times=times_pri[rawi])
        #print(rn,lens_MEGartif)
        ## here I don't want to remove artifacts from "wrong" brain side because
        ## we use ipsilateral CB
        #ib_MEG_perit_merged = \
        #    utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_MEGartif) ,
        #                                    sfreq,ret_type='bins_contig',
        #                                    wbd_type='contig',
        #                                    ret_indices_type =
        #                                        'window_inds', nbins_total=len(times) )

        #anns_LFPartif, anns_artif_pri, times2, dataset_bounds = \
        #    utsne.concatAnns([rn],[times],['_ann_LFPartif'], side_rev_pri=[side_rev] )
        #lens_LFPartif = utils.getIntervalsTotalLens(anns_LFPartif, include_unlabeled =False,
        #                                       times=times_pri[rawi])
        #print(rn,lens_LFPartif)
        #anns_LFPartif = utils.removeAnnsByDescr(anns_LFPartif, ['artif_LFP{}'.format(wrong_brain_sidelet) ])
        #ib_LFP_perit_merged = \
        #    utils.getWindowIndicesFromIntervals(wbd,utils.ann2ivalDict(anns_LFPartif) ,
        #                                    sfreq,ret_type='bins_contig',
        #                                    wbd_type='contig',
        #                                    ret_indices_type =
        #                                        'window_inds', nbins_total=len(times) )



        #import pdb; pdb.set_trace()

        #ib_MEG_perit_perraw[rn] = ib_MEG_perit_merged
        #ib_LFP_perit_perraw[rn] = ib_LFP_perit_merged
        #ib_mvt_perit_perraw[rn] = ib_mvt_perit_merged

        #it = int_type_pri[rawi]
        print('gatherFeatStats: Rescaling features for raw {}'
              ' accodring to data in interval {}, '
              'combining within {}'.format(rn, int_types,combine_within  ) )

    for int_type_cur in int_types:
        for rawindseti_cur,indset_cur in enumerate(indsets):
            int_found_within_indset = False
            for rawi in indset_cur:
                rawn = rawnames[rawi]
                if int_type_cur == 'entire':
                    int_found_within_indset = True
                    break
                elif int_type_cur in bindict_per_rawn[rawn ]['beh_state']:
                    int_found_within_indset = True
                    break
            if not int_found_within_indset:
                #s = "Warning, not data collected for interval {}".format(int_type_cur)
                s = "gatherFeatStats: in {} there is no intervals in the indeset {} for interval {}".\
                    format(rawn, rawindseti_cur, int_type_cur)
                if int_type_cur in intervals_required:
                    raise ValueError(s)
                else:
                    print('gatherFeatStats:  Warninig ',s)


    means_per_indset = []
    stds_per_indset = []
    stats_per_indset = []
    for rawindseti_cur,indset_cur in enumerate(indsets):
        mean_per_int_type = {}
        std_per_int_type  = {}
        stat_per_int_type = {}
        for int_type_cur in int_types:
            stat_perchan = {}
            featnames = featnames_pri[ indset_cur[0] ]  # they are supposed to be constant within indset
            me_perchan  = np.zeros( len(featnames) )
            std_perchan = np.zeros( len(featnames) )
            int_type_not_present = False
            # FIRST over features, THEN over datasets
            show_warn_featns = []
            for feati,featn in enumerate(featnames):
                stat_perchan[featn] = None
                dats_forstat = []
                # for each dataset separtely we collect features accodring to an
                # interval
                for rawi in indset_cur:
                    assert len(featnames) == len(featnames_pri[rawi] )
                    #if mod == 'src':
                    #    chnames_nicened = utils.nicenMEGsrc_chnames(chnames, roi_labels, srcgrouping_names_sorted,
                    #                                    prefix='msrc_')
                    rn = rawnames[rawi]
                    subj,medcond,task  = utils.getParamsFromRawname(rn)

                    ib_MEG_perit =  bindict_per_rawn[rn]['artif'].get('MEG',{})
                    ib_LFP_perit =  bindict_per_rawn[rn]['artif'].get('LFP',{})
                    ib_mvt_perit =  bindict_per_rawn[rn]['beh_state']
                    #print(rn, ib_mvt_perit.keys() )

                    if (int_type_cur != 'entire') and (int_type_cur not in ib_mvt_perit):
                        if task == int_type_cur[:-2]:
                            int_type_not_present = True
                        continue

                    dat =  X_pri[rawi][:,feati]
                    lendat= len(dat)

                    if int_type_cur == 'entire':
                        dat_forstat = dat
                    else:
                        ib_mvt = ib_mvt_perit[int_type_cur]

                        mask = np.zeros(lendat, dtype=bool)
                        mask[ib_mvt] = 1
                        nbinstot_mvt = np.sum(mask)

                        #nbinstot_LFP_artif = 0
                        #nbinstot_MEG_artif = 0
                        if artif_handling == 'reject':
                            artif,mask_artif = getArtifForChnOrFeat(featn,bindict_per_rawn[rn],lendat)
                            mask[mask_artif] = 0
                            n = np.sum(mask)
                            # TODO: I don't want to merge artifacts from all
                            # LFP channels together
                            #if featn.find('LFP' ) >= 0:
                            #    for bins in ib_LFP_perit.values():
                            #        #print('LFP artif nbins ',len(bins))
                            #        mask[bins] = 0
                            #    nbinstot_LFP_artif = nbinstot_mvt - np.sum(mask)
                            #if featn.find('msrc' ) >= 0:
                            #    for bins in ib_MEG_perit.values():
                            #        #print('MEG artif nbins ',len(bins))
                            #        mask[bins] = 0
                            #    nbinstot_MEG_artif = nbinstot_mvt - nbinstot_LFP_artif - np.sum(mask)
                            if gv.DEBUG_MODE:
                                print('{} {}, nremaining bins {}, '
                                    ' removed {} bins, '
                                    ' intlen/n {:.3f}, intlen {}'.
                                    format(int_type_cur,featn, n,  nbinstot_mvt - n,
                                        100*n/nbinstot_mvt, nbinstot_mvt,
                                            ) )
                        elif artif_handling == 'impute':
                            raise ValueError('not implemented')

                        n = np.sum(mask)

                        if (n  < minlen_bins) and (not gv.DEBUG_MODE):
                            #print('feature {}, nremaining bins {}, percentage {}, total in interval {} LFPaftif {} MEGartif'.
                            #    format(featn, n, n/mask.size, nbinstot_mvt,
                            #           nbinstot_LFP_artif, nbinstot_MEG_artif) )
                            raise ValueError(f'too few bins to compute stats {int_type_cur}: {n}<{minlen_bins}')

                        dat_forstat = dat[mask]
                    dats_forstat += [dat_forstat]

                if len(dats_forstat) > 0:
                    # here I would for normalization stats gather from all
                    # participating datasets from the group
                    dats_forstat = np.hstack(dats_forstat)  # over datasets
                    if artif_handling == 'reject':
                        me,std = utsne.robustMean(dats_forstat,ret_std=1)
                        if gv.DEBUG_MODE:
                            print('robust ',dats_forstat.shape, me,std)
                    else:
                        me = np.mean(dats_forstat, axis=0)
                        std = np.std(dats_forstat, axis=0)
                        if gv.DEBUG_MODE:
                            print('normal ',dats_forstat.shape, me,std)
                        if abs(std) <= 1e-14:
                            print(f'gatherFeatStats: WARNING: Std is small for {rawnames[rawi]}'
                            f' {featn} {int_type_cur}. std = {std}')
                            if not gv.DEBUG_MODE:
                                raise ValueError('too small std' )
                else:
                    for tk in list(set( [ utils.getParamsFromRawname(rawnames[rawi])[-1] for rawi in indset_cur ] )):
                        if tk == int_type_cur[:-2]:
                            show_warn_featns += [featn ]
                    me,std = np.nan, np.nan

                stat_perchan[featn] = (me,std)
                me_perchan [feati] = me
                std_perchan[feati] = std
                #print('-------------LENS ', len(me_perchan), len(stat_perchan) )
            ###### ENd of cycle over feats
            if printLog and len(show_warn_featns) :
                rn_str = ','.join( [rawnames[rawi]  for rawi in indset_cur ] )
                # since the orderding should be same
                if len(show_warn_featns) == len(featnames):
                    featns_str = 'all feats'
                else:
                    featns_str = ','.join(show_warn_featns)
                print('gatherFeatStats: Nothing found of {} for {} in raws {}'.format(int_type_cur,featns_str, rn_str) )

            if printLog and int_type_not_present:
                print('gatherFeatStats: interval {} is not present in {}'.format(int_type_cur,rn) )

            if np.all(np.isnan(me_perchan) ):
                mean_per_int_type[int_type_cur] = None
                std_per_int_type [int_type_cur] = None
                stat_per_int_type[int_type_cur] = None
            else:
                mean_per_int_type[int_type_cur] = me_perchan
                std_per_int_type [int_type_cur] = std_perchan
                stat_per_int_type[int_type_cur] = stat_perchan
            #print(mean_per_int_type.keys() )

        means_per_indset += [mean_per_int_type]
        stds_per_indset   += [std_per_int_type ]
        stats_per_indset += [stat_per_int_type]

    return indsets, means_per_indset, stds_per_indset, stats_per_indset

def rescaleFeats(rawnames, X_pri, featnames_pri, wbd_pri,
                 sfreq, times_pri, int_type, main_side=None,
                 side_rev_pri = None,
                 minlen_bins = 5 * 256 / 32, combine_within='no',
                 means=None, stds=None, indsets=None, stat_fname_full=None,
                 artif_handling_statcollect = 'reject', bindict_per_rawn=None ):
    '''
    rescales in-place
    usually notrem_<sidelet>
    modifies raws in place. Rescales to zero mean, unit std
    it does not do anything with artifacts, artif_handling_statcollect is just
        what controls statistics collection
    '''
    assert isinstance(int_type,str)
        #int_type_pri = [ 'entire' ] * len(rawnames)
    assert int_type.find('{}') < 0  # it should not be a template

    assert artif_handling_statcollect in ['reject', 'no', 'impute']

    assert len(rawnames) == len(featnames_pri)
    for i in range(len(X_pri) ):
        assert len(featnames_pri[i]) == X_pri[i].shape[1],  ( len(featnames_pri[i]),  X_pri[i].shape )

    assert len(rawnames) == len(X_pri)
    assert len(rawnames) == len(times_pri)
    assert len(rawnames) == len(side_rev_pri)

    if wbd_pri is None:
        wbd_pri = []
        for times in times_pri:
            # wbd is bin indices, not times!
            temp = np.vstack([times*sfreq,times*sfreq] ).astype(int)
            temp[1] += 1
            wbd_pri += [temp]
    assert len(rawnames) == len(wbd_pri)

    assert combine_within in gv.rawnames_combine_types

    if main_side is None:
        main_side = int_type[-1].upper()
    main_side = main_side[0].upper()

    #print('Start raws rescaling for modality {} based on interval type {}'.format(mod,int_type_templ) )
    #import utils_tSNE as utsne
    #rwnstr = ','.join(rawnames)

    if means is None or stds is None or indsets is None:
        if stat_fname_full is not None:
            f = np.load( stat_fname_full)
            means = f['means']
            stds = f['stds']
            indsets = f['indsets']
        else:
            indsets, means, stds, stats_per_indset = \
                gatherFeatStats(rawnames, X_pri, featnames_pri, wbd_pri, sfreq, times_pri,
                        int_type, side_rev_pri = side_rev_pri,
                        combine_within = combine_within, minlen_bins = minlen_bins,
                                artif_handling=artif_handling_statcollect,
                                bindict_per_rawn= bindict_per_rawn)
    else:
        assert len(means) == len(stds)
        assert len(means) == len(indsets)

    import copy
    means_rescaled = copy.deepcopy(means)
    stds_rescaled  = copy.deepcopy(stds)
    for rawindseti_cur,indset_cur in enumerate(indsets):
        # rescale everyone within indset according to the stats
        mn = means[rawindseti_cur][int_type]
        std = stds[rawindseti_cur][int_type]

        means_rescaled [rawindseti_cur][int_type]  -= mn
        bad_inds = np.where( std <= 1e-14 )[0]

        if len(bad_inds) :
            s = f'std is zero for some indices {bad_inds}'
            if gv.DEBUG_MODE:
                good_inds = np.setdiff1d(np.arange(X_pri[0].shape[1] ), bad_inds)
                stds_rescaled  [rawindseti_cur][int_type][good_inds]  /= std[good_inds]
            else:
                raise ValueError(s)
        else:
            stds_rescaled  [rawindseti_cur][int_type]  /= std

        for rawi in indset_cur:
            #rn = rawnames[rawi]
            X_pri[rawi] -= mn[None,:]
            if len(bad_inds) == 0:
                X_pri[rawi] /= std[None,:]
            else:
                X_pri[rawi][:,good_inds] /= std[None,good_inds]
            if gv.DEBUG_MODE:
                print(f'perform scaling, shift by {mn}, divide by {std}')

    return X_pri, indsets, means_rescaled, stds_rescaled
    #fname_stats = rwnstr + '_stats.npz'
    #np.savez(fname_stats, dat_forstat_perchan=dat_forstat_perchan,
    #         combine_within_medcond=combine_within_medcond,
    #         subjs_analyzed=subjs_analyzed)

def rescaleRaws(raws_permod_both_sides, mod='LFP',
                int_type_templ = 'notrem_{}', minlen_sec = 5, combine_within_medcond=True,
                roi_labels=None, srcgrouping_names_sorted = None, src_labels_ipsi = ['Cerebellum'],
                ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt' ):
    '''
    modifies raws in place. Rescales to zero mean, unit std
    roi_labels are there only for ipsilateral cerebellum essentially
    '''
    if mod == 'src':
        assert roi_labels is not None
        assert srcgrouping_names_sorted is not None

    print('Start raws rescaling for modality {} based on interval type {}'.format(mod,int_type_templ) )
    import utils_tSNE as utsne
    rawnames = list( sorted(raws_permod_both_sides.keys() ) )
    rwnstr = ','.join(rawnames)

    if mod == 'src':
        chn_name_side_ind = 4
    if mod in ['LFP', 'LFP_hires']:
        chn_name_side_ind = 3

    if combine_within_medcond:
        combine_within = 'medcond'
    else:
        combine_within = 'no'
    indsets = genCombineIndsets(rawnames, combine_within)

    assert combine_within in ['medcond','task','subj','no'] # we cannot do across subj


    for rawindseti_cur,indset_cur in enumerate(indsets):
        rn0 = rawnames[indset_cur[0] ]
        chnames = raws_permod_both_sides[rn0][mod].ch_names

        dat_forstat_perchan = {}

        if mod == 'src':
            chnames_nicened = utils.nicenMEGsrc_chnames(chnames, roi_labels, srcgrouping_names_sorted,
                                            prefix='msrc_')
        for chni,chn in enumerate(chnames):
            dats_forstat = []
            sidelet = chn[chn_name_side_ind]
            opsidelet = utils.getOppositeSideStr(sidelet)
            for rawi in indset_cur:
                rn = rawnames[rawi]
                raw = raws_permod_both_sides[rn][mod]
                sfreq = raw.info['sfreq']

                if mod in ['LFP', 'LFP_hires']:
                    suffixes = ['_ann_LFPartif']
                elif mod == 'src':
                    suffixes = [ ann_MEGartif_prefix_to_use ]
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

        for rawi in indset_cur:
            rn = rawnames[rawi]
            raw = raws_permod_both_sides[rn][mod]
            for chni,chn in enumerate(raw.ch_names):
                # rescale each raw individually based on common stats
                mn,std = dat_forstat_perchan[chn]
                fun = lambda x: (x-mn) /std
                raw = raws_permod_both_sides[rn][mod]
                raw.load_data()
                raw.apply_function(fun,picks=[chn])



    '''
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
    '''

    fname_stats = rwnstr + '_stats.npz'
    np.savez(fname_stats, dat_forstat_perchan=dat_forstat_perchan,
             combine_within_medcond=combine_within_medcond,
             rawnames=rawnames)


def getBaselineInt(subj_or_rawname, body_side_for_baseline_int, baseline_int_type='notrem'):
    '''
    based on gen subj info
    '''
    if subj_or_rawname.find('_') >= 0:
        subj_cur,medcond_cur,task_cur  = utils.getParamsFromRawname(subj_or_rawname)
    else:
        subj_cur = subj_or_rawname

    mainmoveside = gv.gen_subj_info[subj_cur].get('move_side',None)
    maintremside = gv.gen_subj_info[subj_cur].get('tremor_side',None)

    if body_side_for_baseline_int == 'body_move_side':
        main_side_let = mainmoveside[0].upper()
    elif body_side_for_baseline_int == 'body_tremor_side':
        main_side_let = maintremside[0].upper()
    elif body_side_for_baseline_int in ['left','right']:
        main_side_let = body_side_for_baseline_int[0].upper()

    if baseline_int_type != 'entire':
        baseline_int = '{}_{}'.format(baseline_int_type, main_side_let )
    else:
        baseline_int = baseline_int_type

    return baseline_int


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
             src_file_grouping_ind=None, use_saved = True, highpass_lfreq = None,
             input_subdir="", input_subdir_srcrec = "", 
             n_jobs=1, filter_phase = 'minimum', verbose=None  ):
    '''
    only loads data from different files, does not do any magic with sides
    use_saved means using previously done preproc
    filter_phase can be 'minimum' (gives causal) or 'zero'
    '''
    import globvars as gv
    data_dir = gv.data_dir

    print('Loading following raw types ',mods_to_load)

    raws_permod_both_sides = {}
    rawname2rec_info = {}
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
            #rawname_resample = rawname_ + '_resample_raw.fif'
            #if preproc_type in ['highpass', 'hipass']:
            #    rawname_resample = rawname_ + '_resample_notch_highpass.fif'
            #elif preproc_type in ['tSSS']:
            #    rawname_resample = rawname_ + '_tSSS_notch_highpass_resample.fif'
            #else:
            #    raise ValueError(f'Wrong preproc type {preproc_type}')
            rawname_resample = rawname_ + '_resample_notch_highpass.fif'
            rawname_resample_full = os.path.join(data_dir, rawname_resample)
            raw_resample = mne.io.read_raw_fif(rawname_resample_full, verbose=verbose)
            if 'resample' in mods_to_load:
                raw_permod_both_sides_cur['resample'] = raw_resample

        if 'LFP' in mods_to_load:
            if use_saved:
                rawname_LFPonly = rawname_ + '_LFPonly'+ '.fif'
                rawname_LFPonly_full = os.path.join( data_dir, rawname_LFPonly )
                raw_lfponly = mne.io.read_raw_fif(rawname_LFPonly_full, verbose=verbose)
            else:
                raw_lfponly = getSubRaw(rawname_, raw=raw_resample, picks = ['LFP.*'])

            raw_permod_both_sides_cur['LFP'] = raw_lfponly
        if 'LFP_hires' in mods_to_load:
            raw_lfp_highres = saveLFP(rawname_, skip_if_exist = 1, sfreq=1024)
            if raw_lfp_highres is None:
                lfp_fname_full = os.path.join(data_dir, '{}_LFP_1kHz.fif'.format(rawname_) )
                raw_lfp_highres = mne.io.read_raw_fif(lfp_fname_full, verbose=verbose)
            raw_permod_both_sides_cur['LFP_hires'] = raw_lfp_highres

        if 'src' in mods_to_load:
            assert sources_type is not None
            assert src_type_to_use is not None
            assert src_file_grouping_ind is not None


            src_rec_info_fn_full = utils.genRecInfoFn(rawname_,sources_type, src_file_grouping_ind, input_subdir_srcrec)
            rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)


            src_fname_noext = 'srcd_{}_{}_grp{}'.format(rawname_,sources_type,src_file_grouping_ind)
            if src_type_to_use == 'center':
                newsrc_fname_full = os.path.join( data_dir, input_subdir_srcrec, 'cnt_' + src_fname_noext + '.fif' )
            elif src_type_to_use == 'mean_td':
                newsrc_fname_full = os.path.join( data_dir, input_subdir_srcrec, 'av_' + src_fname_noext + '.fif' )
            elif src_type_to_use == 'parcel_ICA':
                newsrc_fname_full = os.path.join( data_dir, input_subdir_srcrec, 'pcica_' + src_fname_noext + '.fif' )
            else:
                raise ValueError('Wrong src_type_to_use {}'.format(src_type_to_use) )

            import datetime
            mtime = os.stat(newsrc_fname_full).st_mtime
            mtime = datetime.datetime.fromtimestamp(mtime)

            print(f'Loading reconstructed sources from {newsrc_fname_full}, mtime = {mtime}')
            raw_srconly =  mne.io.read_raw_fif(newsrc_fname_full, verbose=verbose)
            raw_permod_both_sides_cur['src'] = raw_srconly
        else:
            rec_info = None
        
        rawname2rec_info[rawname_] = rec_info

        if 'EMG' in mods_to_load:
            if use_saved:
                saveRectConv(rawname_, skip_if_exist = 1)
                rectconv_fname_full = os.path.join(data_dir, '{}_emg_rectconv.fif'.format(rawname_) )
                raw_emg = mne.io.read_raw_fif(rectconv_fname_full, verbose=verbose)
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
            rawname_afterICA_full = os.path.join(data_dir, input_subdir, rawname_afterICA)
            raw_afterICA = mne.io.read_raw_fif(rawname_afterICA_full, verbose=verbose)
            raw_permod_both_sides_cur['afterICA'] = raw_afterICA

        if 'tSSS' in mods_to_load:
            #rawname_SSS = rawname_ + '_notch_SSS_raw.fif'
            #rawname_SSS = rawname_ + '_notch_SSS_raw.fif'
            #rawname_SSS = rawname_ + '_SSS_notch_resample_raw.fif'
            #rawname_SSS = rawname_ + '_SSS_notch_highpass_resample_raw.fif'

            rawname_SSS = rawname_ + '_tSSS_notch_highpass_resample.fif'
            rawname_SSS_full = os.path.join(data_dir, input_subdir, rawname_SSS)
            raw_SSS = mne.io.read_raw_fif(rawname_SSS_full, verbose=verbose)
            raw_permod_both_sides_cur['tSSS'] = raw_SSS

        if highpass_lfreq is not None:
            for mod in raw_permod_both_sides_cur:
                if mod == "resample" and rawname_resample.find('highpass') >= 0:
                    continue
                if mod == "SSS" and rawname_SSS.find('highpass') >= 0:
                    continue
                raw_permod_both_sides_cur[mod].\
                filter(l_freq=highpass_lfreq, h_freq=None,picks='all',
                       n_jobs = n_jobs, phase=filter_phase)


        #raw_permod_both_sides_cur['afterICA']

        raws_permod_both_sides[rawname_] = raw_permod_both_sides_cur
    return raws_permod_both_sides
    #return raws_permod_both_sides, rawname2rec_info


def saveLFP(rawname_naked, f_highpass = 2, skip_if_exist = 1,
                         n_free_cores = 2, ret_if_exist = 0, notch=1,
            highpass=1, raw_FT=None, sfreq=1024, raw_resampled = None,
            filter_artif_care=1, save_with_anns = 0, n_jobs = 1):
    import globvars as gv
    import multiprocessing as mpr
    lowest_freq_to_preserve = f_highpass

    data_dir = gv.data_dir

    if sfreq in [1000, 1024]:
        lfp_fname_full = os.path.join(data_dir, '{}_LFP_1kHz.fif'.format(rawname_naked) )
    elif sfreq == 256:
        lfp_fname_full = os.path.join(data_dir, '{}_LFPonly.fif'.format(rawname_naked) )
    elif sfreq == -1:
        lfp_fname_full = os.path.join(data_dir, '{}_LFPonly_maxres.fif'.format(rawname_naked) )
    else:
        lfp_fname_full = os.path.join(data_dir, '{}_LFPonly_{}Hz.fif'.format(rawname_naked,sfreq) )
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
        if raw_resampled is not None and abs(raw_resampled.info['sfreq'] - sfreq) < 1e-10:
            raw = raw_resampled
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

    if sfreq < 0:
        sfreq = subraw.info['sfreq']

    set_channel_types = True # needed of for real but not test datasets
    if set_channel_types:
        y = {}
        for chname in subraw.ch_names:
            y[chname] = 'eeg'
        subraw.set_channel_types(y)



    #num_cores = mpr.cpu_count() - 1
    #nj = max(1, num_cores-n_free_cores)
    if abs(subraw.info['sfreq'] - sfreq) > 0.1:
        print('saveLFP: Resample {} to {}'.format(subraw.info['sfreq'],sfreq) )
        subraw.resample(sfreq, n_jobs= n_jobs )

    artif_fname = os.path.join(data_dir , '{}_ann_LFPartif.txt'.format(rawname_naked) )
    if os.path.exists(artif_fname ) and filter_artif_care:
        anns = mne.read_annotations(artif_fname)
        subraw.set_annotations(anns)
    else:
        print('saveLFP: {} does not exist'.format(artif_fname) )

    if notch:
        freqsToKill = np.arange(50, sfreq//2, 50)  # harmonics of 50
        print('saveLFP: Resample')
        subraw.notch_filter(freqsToKill,  n_jobs= n_jobs)

    if highpass:
        print('saveLFP: highpass')
        subraw.filter(l_freq=lowest_freq_to_preserve, h_freq=None,
                      skip_by_annotation='BAD_LFP', n_jobs= n_jobs,
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

    set_channel_types = True # needed of for real but not test datasets
    if set_channel_types:
        y = {}
        for chname in emgonly.ch_names:
            y[chname] = 'eeg'
        emgonly.set_channel_types(y)

    print('saveRectConv: highpass')
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
    from pymatreader import read_mat
    from mne.io.fieldtrip.utils import _validate_ft_struct, _create_info

    ft_struct = read_mat(fname,
                         ignore_fields=['previous'],
                         variable_names=[data_name])

    _validate_ft_struct(ft_struct)
    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    if info is None:
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

# component influence on every channel
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

def readInfo(rawname, raw, sis=[1,2],
             check_info_diff = 1, bandpass_info=0 ):
    import globvars as gv
    data_dir = gv.data_dir

    import pymatreader as pym
    infos = {}
    for si in sis:
        info_name = rawname + '{}_info.mat'.format(si)
        fn = os.path.join(data_dir,info_name)
        if not os.path.exists(fn):
            print(f'{fn} does not exist')
            continue
        rr  = pym.read_mat(fn )
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
    if raw is None:
        return None, infos
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
    with mod_info._unlock():
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

def extractEMGData(raw, rawname_=None, skip_if_exist = 1, tremfreq = 9, save_dir = None):
    # highpass and convolve
    # tremfreq should be max tremfreq found in the analyzed data (max over all
    # subjects)
    if save_dir is None:
        save_dir = gv.data_dir
    if rawname_ is not None:
        rectconv_fname_full = os.path.join(save_dir, '{}_emg_rectconv.fif'.format(rawname_) )
        if (skip_if_exist and os.path.exists(rectconv_fname_full) ):
            rectconvraw = mne.io.read_raw(rectconv_fname_full, verbose=0)
            return rectconvraw

    emgonly = raw.copy()
    emgonly.info['bads'] = []

    chis = mne.pick_channels_regexp(emgonly.ch_names, 'EMG.*old')
    if len(chis) == 0:
        print('WARNING: there are not EMG.*old channels ,trying to select EMG.*')
        chis = mne.pick_channels_regexp(emgonly.ch_names, 'EMG.*')
        if len(chis) == 0:
            raise ValueError('ERROR: there are not EMG channels at all')
    restr_names = np.array( emgonly.ch_names )[chis]

    emgonly.load_data()
    emgonly.pick_channels(restr_names.tolist())
    emgonly_unfilt = emgonly.copy()
    print(emgonly.ch_names)
    #help(emgonly.filter)

    set_channel_types = True # needed of for real but not test datasets
    if set_channel_types:
        y = {}
        for chname in emgonly.ch_names:
            y[chname] = 'eeg'
        emgonly.set_channel_types(y)
    highpass_freq = 10

    print('extractEMGData: {} Hz highpass'.format(highpass_freq) )
    emgonly.filter(l_freq=highpass_freq, h_freq=None, picks='all',pad='symmetric')

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
        if not (skip_if_exist and os.path.exists(rectconv_fname_full) ):
            print('EMG raw saved to ',rectconv_fname_full)
            rectconvraw.save(rectconv_fname_full, overwrite=1)

    return rectconvraw

def getECGindsICAcomp(icacomp, pct_thr = 2.2, ncomp_test_for_ecg = 6, ecg_ratio_thr = 6):
    '''
    icacomp -- output of MNEs ICA
    pct_thr -- determines how strict we are considering something heartbeat-like. Smaller pct_thr gives stricter rule

    returns:
        ecg_compinds, ratios, ecg_evts_all

        ecg_compinds: list of ints -- indices in icacomp
        ratios: list of floats of len=len(icacomp)
        ecg_evts_all: list of lists of tuples of ints (indices of timebins)
    '''
    from utils import getIntervals
    sfreq = int(icacomp.info['sfreq'])
    normal_hr  = [55,105]  # heart rate bounds in beats per min, Mayo clinic says 60 to 100
    ecg_compinds = []


    # Recall that sampling rate is high and hearbeats are realitvely slow
    # events and they are very sharp (i.e. short), so they won't contribute
    # too much to the distrubtion, they will be in the righmost part of the
    # tail

    rmax = 0
    ratios = []
    # cycle over ICA components and for each of them compute how much is
    # component (robust) maximum different from the median
    # this will be used to sort components later
    for i in range(len(icacomp.ch_names)):
        comp_ecg_test,times = icacomp[i]
        da = np.abs(comp_ecg_test[0]) # take absolute part of current component
        qq = np.percentile(da, [ pct_thr, 100-pct_thr, 50 ] ) # take bottom, top and median of the distribution
        r = (qq[1] - qq[2]) / qq[2]  # how much are top values different from the median compared to median itself
        ratios += [r]
        rmax = max(rmax, r)
    #    if r < ecg_ratio_thr:
    #        continue

    strog_ratio_inds = np.where( ratios > ( np.max(ratios) + np.min(ratios) )  /2  )[0]
    nstrong_ratios = len(strog_ratio_inds)
    print('nstrong_ratios = ', nstrong_ratios)

    ecg_evts_all = []
    # for the first every components, sorted by ratios, found above, find
    # number of reaches to the top of the distributio per second and compare it
    # with the normal heart rate range. If it is similar, we judge component to
    # be heartbeat-related and mark it for removal
    for i in np.argsort(ratios)[::-1][:ncomp_test_for_ecg]:
        comp_ecg_test,times = icacomp[i]
        da = np.abs(comp_ecg_test[0])
        qq = np.percentile(da, [ pct_thr, 100-pct_thr, 50 ] )
        # get indices of times where we are above the distribution robust top.
        mask = da > qq[1]
        bis = np.where(mask)[0]

        pl = 1
        if i > 8:  # we might want to look more carefully at what happens for components with lower ratios
            pl = 0
        # here we call a relatively slow function that converts a list of indices to a
        # list of continuous intervals, allowing some short holes to be present in
        # the list. Probably there exists faster versions of it
        # here we don't really care about the intervals themselves, only their number
        # in a sense it is my version of mne.preprocessing.ica_find_ecg_events(filt_raw,comp_ecg_test)
        _, ecg_evts  = getIntervals(bis, width=5, thr=1e-5, percentthr=0.95,
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


def getIntervalInfoFromRawname(rawname_, crop=None,
    collect_artif_info = True, 
    ann_MEGartif_prefix_to_use = [
        '_ann_MEGartif_flt', '_ann_MEGartif' , '_ann_MEGartif_ICA' ] ,
    print_empty = False, artif_thr_pct = 10, subdir = '',
    verbose=0):
    '''
    returns ann_len_dict, ann_dict
    '''
    # crop -- crop range in seconds
    import utils
    import globvars as gv

    subj,medcond,task  = utils.getParamsFromRawname(rawname_)

    maintremside = gv.gen_subj_info[subj]['tremor_side']
    move_side = gv.gen_subj_info[subj].get('move_side','UNDEF')
    tremfreq = gv.gen_subj_info[subj]['tremfreq']


    nonmaintremside = utils.getOppositeSideStr(maintremside)
    if move_side == 'UNDEF':
        main_side_let = maintremside[0].upper()
    else:
        main_side_let = move_side[0].upper()
    #print(rawname_,'Main trem side ' ,maintremside,main_side_let)
    print('----{}\n{} is maintremside, tremfreq={} move side={}'.\
          format(rawname_, maintremside,tremfreq, move_side) )
    print(r'^ is tremor, * is main tremor side')

    #rawname = rawname_ + '_resample_raw.fif'
    rawname = rawname_ + '_LFPonly.fif'
    fname_full = os.path.join(gv.data_dir,rawname)   #only needed to get times

    # read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_fif(fname_full, None, verbose=0, preload=0)

    times = raw.times
    if crop is None:
        crop = (times[0],times[-1])
    else:
        assert len(crop) == 2 and max(crop) <= times[-1] \
        and min(crop)>=0 and crop[1] > crop[0]

    begtime = max(times[0], crop[0] )
    endtime = min(times[-1], crop[1] )
    ##########

    ots_letter = utils.getOppositeSideStr(main_side_let)
    mts_trem_str = 'trem_{}'.format(main_side_let)
    mts_notrem_str = 'notrem_{}'.format(main_side_let)
    mts_task_str = '{}_{}'.format(task,main_side_let)
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
    ann_dict = { 'behav':anns_upd}

    ###########
    fnsuff2renameDict = {}
    #########################
    rd = {'BAD_MEGL':'BAD_MEGL', 'BAD_MEGR':'BAD_MEGR',
                     'BAD_MEG':'BAD_MEG'}
    fnsuff2renameDict['_ann_MEGartif_flt'] = rd
    #########################
    rd = {'BAD_MEGL':'BAD_rMEGL', 'BAD_MEGR':'BAD_rMEGR',
                     'BAD_MEG':'BAD_rMEG'}
    fnsuff2renameDict['_ann_MEGartif'] = rd
    #########################
    rd = {'BAD_MEGL':'BAD_icaMEGL', 'BAD_MEGR':'BAD_icaMEGR',
                     'BAD_MEG':'BAD_icaMEG'}
    fnsuff2renameDict['_ann_MEGartif_ICA'] = rd
    #########################
    rd = {'BAD_muscle':'BAD_muMEG'}
    fnsuff2renameDict['_ann_MEGartif_muscle'] = rd


    if collect_artif_info:
        suffixes = []
        data_modalities = ['LFP', 'msrc' ]
        if 'LFP' in data_modalities:
            suffixes += [ '_ann_LFPartif' ]

        if isinstance(ann_MEGartif_prefix_to_use, str):
            ann_MEGartif_prefix_to_use = [ann_MEGartif_prefix_to_use]

        if 'msrc' in data_modalities:
            suffixes += [ ann_MEGartif_prefix_to_use[0] ]
        anns_artif, anns_artif_pri, times_, dataset_bounds_ = \
            utsne.concatAnns([rawname_],[times], suffixes,crop=(crop[0],crop[1]),
                        allow_short_intervals=True,
                            side_rev_pri = [0],
                            wbd_pri = None, sfreq=int(raw.info['sfreq']),
                             subdir=subdir , verbose =  verbose)

        if len(ann_MEGartif_prefix_to_use) > 1:
            for ann_MEGartif_addprefix in ann_MEGartif_prefix_to_use[1:]:
                if verbose:
                    print(ann_MEGartif_addprefix)
                suffixes2 = [ann_MEGartif_addprefix]
                anns_artif2, anns_artif_pri, times_, dataset_bounds_ = \
                    utsne.concatAnns([rawname_],[times], suffixes2,crop=(crop[0],crop[1]),
                                allow_short_intervals=True,
                                    side_rev_pri = [0],
                                    wbd_pri = None, sfreq=int(raw.info['sfreq']),
                                     subdir=subdir, verbose =  verbose)
                # TODO: maybe concat here?
                rd = fnsuff2renameDict[ann_MEGartif_addprefix]
                anns_artif2 = utils.renameAnnDescr(anns_artif2,
                    rd, match_full_strings = 1)
                anns_artif += anns_artif2

        ann_dict['artif'] = anns_artif
    ############



    ann_len_dict = {}
    meaningful_totlens = {}
    for ann_name in ann_dict:
        #print('{} crop lengths'.format(ann_name))
        #display(anns_cnv_Jan.description)
        anns = ann_dict[ann_name]
        if anns is None:
            continue
        lens = utils.getIntervalsTotalLens(anns, True, times=raw.times,
                                           interval=crop)
        ann_len_dict[ann_name] = lens

        lens_keys = list(sorted(lens.keys()) )
        for lk in lens_keys:
            lcur = lens[lk]
            lk_toshow = lk

            if ann_name != 'artif':
                if lk.find('trem') == 0:  #lk, not lk_toshow!
                    lk_toshow = '^' + lk_toshow
                if lk.find('_' + main_side_let) >= 0:
                    lk_toshow = '*' + lk_toshow
            else:
                if lk.endswith(ots_letter) >= 0:
                    lk_toshow = '*' + lk_toshow

            if not (ann_name == 'artif' and lk.startswith('nolabel') ):
                print('{:12}: {:6.2f}s = {:6.3f}% of total {:.2f}s'.
                    format(lk_toshow, lcur,  lcur/(endtime-begtime) * 100, endtime-begtime))
        #display(lens  )
        #lens_cnv_Jan = utils.getIntervalsTotalLens(anns_cnv_Jan, True, times=raw.times)
        #display(lens_cnv_Jan  )
        if ann_name != 'artif':
            if mts_trem_str not in anns.description:
                print('!! There is no tremor, accdording to {}'.format(ann_name))

            meaningul_label_totlen = lens.get(mts_trem_str,0) + lens.get(mts_task_str,0)
            meaningful_totlens[ann_name] = meaningul_label_totlen
            if meaningul_label_totlen < 10:
                print('Too few meaningful labels {}'.format(ann_name))

        for it in lens:
            if ann_name != 'artif':
                if it.find(mts_task_str) < 0 and it.find(ots_task_str) >= 0:
                    print('{} has task {} which is opposite side to tremor {}'.format(
                        ann_name, ots_task_str, mts_task_str) )
                assert not( it.find(mts_task_str) >= 0 and it.find(ots_task_str) >= 0),\
                    'task marked on both sides :('
            #else:
            #    print('{} has task {} which is opposite side to tremor {}'.format(
            #        ann_name, ots_task_str, mts_task_str) )



        if ann_name == 'artif':
            del lens['nolabel_L']
            del lens['nolabel_R']


            artif_composite = {}


            for let in ['R','L','.']:
                pattern_list =  [ f'.*LFP{let}' ]
                if '_ann_MEGartif_flt' in ann_MEGartif_prefix_to_use:
                    pattern_list += [ f'.*(fltMEG{let}|LFP{let})',  f'.*fltMEG{let}' ]
                if '_ann_MEGartif_ICA' in ann_MEGartif_prefix_to_use:
                    pattern_list += [ f'.*(icaMEG{let}|LFP{let})',  f'.*fltMEG{let}' ]
                if '_ann_MEGartif' in ann_MEGartif_prefix_to_use:
                    pattern_list += [ f'.*(rMEG{let}|LFP{let})',  f'.*rMEG{let}' ]
                for pattern in pattern_list:
                    newlab = f'{pattern[2:]}, brain side = {let}'
                    artif_cur = utils.filterAnnDict(anns, sidelet=None,
                                                        artif_best_LFP_only=False,
                                                        pattern = pattern)

                    bins = utils.fillBinsFromAnns(artif_cur, times[-1],
                                                  raw.info['sfreq'] , [])
                    lcur = sum(bins) / raw.info['sfreq']
                    #artif_cur = utils.mergeAnns(artif_cur, times[-1],
                    #                            sfreq = int(raw.info['sfreq']),
                    #                            out_descr = newlab)
                    #lens_artif = utils.getIntervalsTotalLens(artif_cur, True, times=raw.times,
                    #                                interval=crop)
                    #if newlab not in lens_artif:
                    if lcur < 1e-10:
                        if print_empty:
                            print(pattern[2:],'NO ') #,lens_artif)
                    else:
                        #lcur = lens_artif[newlab]
                        pct_val = lcur/(endtime-begtime) * 100
                        if artif_thr_pct is not None and pct_val >= artif_thr_pct:
                            print('{:30}: {:6.2f}s = {:6.1f}% of total {:.2f}s'.
                                format(newlab, lcur,  lcur/(endtime-begtime) * 100,
                                endtime-begtime))

                    artif_composite[newlab] = lcur


            newlab = 'artif_all'
            #artif_cur = utils.mergeAnns(anns, times[-1],
            #                            sfreq = int(raw.info['sfreq']),
            #                            out_descr = newlab)
            #print(artif_cur)
            #lens_artif = utils.getIntervalsTotalLens(artif_cur, True, times=raw.times,
            #                                interval=crop)
            bins = utils.fillBinsFromAnns(anns, times[-1], raw.info['sfreq'] , [])
            lcur = sum(bins) / raw.info['sfreq']
            if lcur < 1e-10:
                if print_empty:
                    print(pattern[2:],'NO ') #,lens_artif)
                #lcur = lens_artif[newlab]
            else:
                pct_val = lcur/(endtime-begtime) * 100
                print('{:31}: {:6.2f}s = {:6.1f}% of total {:.2f}s'.
                    format(newlab, lcur,  pct_val,
                        endtime-begtime))

            artif_composite[newlab] = lcur

            ann_len_dict['artif_composite'] = artif_composite

            # TODO filter LFP
            # TODO filter LFP by side
            # TODO filter MEG
            # TODO filter MEG by side


        print('\n')

    return ann_len_dict, ann_dict
    # print('\nmy prev interval lengths')
    # lens_cnv = utils.getIntervalsTotalLens(anns_cnv, True, times=raw.times)
    # display(lens_cnv )
    # #display(anns_cnv.description)
    # if mts_trem_str not in anns_cnv.description:
    #     print('!! There is no tremor, accdording to prev me')


def getIndsetMask(indsets, allow_repeating=False, allow_holes = False):
    # indsets -- list of list of int
    # takes list of indsets, extrats all indices that were there and then returns a
    # (non-binary) mask, saying which index belongs to which indset
    assert isinstance(indsets,list)
    assert isinstance(indsets[0],list)
    allinds = []
    for iset in indsets:
        allinds += list(iset)
    allinds = sorted(allinds)
    allinds_s = set( allinds )
    if len(allinds_s) < len(allinds):
        warns = 'Warning indsets list contain repeating indices!' + str(indsets)
        print(warns)
        if not allow_repeating:
            raise ValueError(warns)

    if len(allinds) > 1:
        dai = np.diff(allinds)
        if np.min(dai) < 0:
            warns = 'Warning indsets contain indices with holes (some missign)!' + str(indsets)
            if not allow_holes:
                raise ValueError(warns)

    mask = -1 * np.ones( len(allinds), dtype=int)
    for ind in allinds:
        for iseti,iset in enumerate(indsets):
            if ind in iset:
                mask[ind] = iseti
    assert np.all(mask >=0 )
    return mask,allinds

def valsPerIndset2PerInd(indsets,vals):
    assert len(vals) == len(indsets)
    mask,allinds = getIndsetMask(indsets)
    r = [0] * len(allinds)
    for i in range(len(mask) ):
        #ind = allinds[i]
        iseti = mask[i]
        r[i] = vals[iseti]
    return r

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

def rncnv_my2h(rn, df):
    from globvars import code_dir
    import json
    with open( pjoin(code_dir,'subj_corresp.json'),'r') as f:
        sc = json.load(f)
    subj,mc,task = rn.split('_')
    subjh = sc['my2hilbert'][subj]
    import glob
    import pandas as pd
    from pathlib import Path

    dfc = df[ df['filename'].str.startswith(f'{subj}_{mc}') ]
    fns_full = glob.glob(  pjoin(gv.data_dir, f'raws_from_Hilbert/{subjh}_{mc.upper()}*') )
    #print(subj,mc,subjh,fns_full)

    r = {}
    for fnf in fns_full:
        fn = Path(fnf).name
        #print(fn)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw(fnf,verbose=0,preload=0,on_split_missing='ignore')
        dur = raw.times[-1]
        dfc2 = dfc.query(f'duration >= {dur} - 1e-9 and duration <= {dur} + 1e-9')
        #display(dfc)
        dv = dfc['duration'].values
        fnv = list( dfc['filename'].values )
        if len(dfc2 ) != 1:
            print(f'{subj,mc,subjh}:  exc {fn}, len = {len(dfc2 )},  {dur}, { list(zip(fnv,dv) ) }')
            continue
        r[ dfc2['filename'].values[0] ] = fn
    return r
    #subjh = sc['my2hilbert'][subj]


'''
def cnv(rawnames):
    import pymatreader as pymr
    #r = pymr.read_mat('cortical_grid.mat')
    #template_grid_cm = r['cortical_grid'] * 100

    times_pri = len(rawnames) * [0]
    custom_raws_pri = len(rawnames) * [0]
    coords_pri = len(rawnames) * [0]
    for rawni,rawname_ in enumerate(rawnames):
        sind_str,medcond,task = utils.getParamsFromRawname(rawname_)

        # this one does not have to be from the subdir
        #rawname = rawname_ + '_resample_notch_highpass_raw.fif'
        rawname = rawname_ + '_LFPonly.fif'
        fname_full = os.path.join(data_dir,rawname)

        #rawname = rawname_ + '_resample_raw.fif'
        #fname_full = os.path.join(data_dir,rawname)

        ## read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
        raw = mne.io.read_raw_fif(fname_full, None)

        #reconst_name = rawname_ + '_resample_afterICA_raw.fif'
        #reconst_fname_full = os.path.join(data_dir,reconst_name)
        #reconst_raw = mne.io.read_raw_fif(reconst_fname_full, None)

        times_pri[rawni] = raw.times

        src_fname_noext = 'srcd_{}_{}'.format(rawname_, sources_type)

        src_fname = src_fname_noext + '.mat'
        src_fname_full = os.path.join(data_dir,input_subdir,src_fname)
        print(src_fname_full)

        src_ft = h5py.File(src_fname_full, 'r')
        ff = src_ft


        #timeinfo = os.stat(src_fname_full)
        #print('Last mod of src was ', time.ctime(timeinfo.st_mtime) )
        #timeinfo = os.stat(reconst_fname_full)
        #print('Last raw was        ', time.ctime(timeinfo.st_mtime) )

        f = ff[ ff['source_data'][0,0] ]

        #########
        srcCoords_fn = sind_str + '_modcoord_parcel_aal.mat'

        crdf = pymr.read_mat(srcCoords_fn)

        # here we do not have label 'unlabeled'
        labels = crdf['labels']  #indices of sources - labels_corresp_coordance
        #crdf = sio.loadmat(srcCoords_fn)
        #lbls = crdf['labels'][0]
        #labels = [  lbls[i][0] for i in range(len(lbls)) ]

        coords = crdf['coords_Jan_actual']
        srcgroups_ = crdf['point_ind_corresp']

        coords_pri += [coords]

        # here we shift by 1 to add label 'unlabeled' in the beginning
        if np.min(srcgroups_) == 0:
            srcgroups_ #+= 1  # we don't need to add one because we have Matlab notation starting from 1
            labels = ['unlabeled'] + labels

        print(labels)
        #crdf['pointlabel']


        #scrgroups_dict = {}
        stcs = []
        pos_ = f['source_data']['pos'][:,:].T
        ##########

        defSideFromLabel = True
        for srcdi in range(ff['source_data'].shape[0] ):
            bandname = bandnames[srcdi]
            f = ff[ ff['source_data'][srcdi,0] ]

            freqBand = f['bpfreq'][:].flatten()

            t0 = f['source_data']['time'][0,0]
            tstep = np.diff( f['source_data']['time'][:10,0] ) [0]

            mom = f['source_data']['avg']['mom']
            # extractring unsorted data
            srcData_= mom[:,:].T

            assert pos_.shape[0] == srcData_.shape[0]

            numsrc_total = len(srcData_)
            allinds = np.arange(numsrc_total)
            pos = pos_[allinds]

            labels_deford = np.array(labels)[srcgroups_[allinds ] ]  # because 0 is unlabeled
            ldo = enumerate(labels_deford)
            Lchnis = [labi for labi,lab in ldo if lab.endswith('_L') ]
            Rchnis = [labi for labi,lab in ldo if not lab.endswith('_L') ]

            ####  Create my
            if sources_type == 'parcel_aal' and defSideFromLabel:
                lhi = map(str, Lchnis )
                rhi = map(str, Rchnis )

                concat = Lchnis + Rchnis
                vertices = [Lchnis, Rchnis]
            else:  #define side from coordinate
                leftInds_coord = np.where(pos[:,0]<= 0)[0]
                rightInds_coord = np.where(pos[:,0] > 0)[0]
                vertices = [leftInds_coord, rightInds_coord]

                lhi = map(str, list( vertices[0] ) )
                rhi = map(str, list( vertices[1] ) )

                concat = np.concatenate((leftInds_coord, rightInds_coord))

                #print(labels)

            srcData = srcData_[ allinds [concat]  ]   # with special ordering

            stc = mne.SourceEstimate(data = srcData, tmin = t0, tstep= tstep  ,
                                    subject = sind_str , vertices=vertices)
            stcs += [stc]

'''
