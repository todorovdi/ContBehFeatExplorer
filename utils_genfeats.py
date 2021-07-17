import numpy as np
import re
import utils

import globvars as gv
import scipy.signal as sig
import matplotlib.pyplot as plt
import globvars as gv
import os
import mne
import multiprocessing as mpr
from joblib import Parallel, delayed

from utils import getOppositeSideStr
from utils import parseMEGsrcChnameShort
from utils import parseMEGsrcChnamesShortList
from utils_tSNE import robustMeanPos, robustMean

def collectDataFromMultiRaws(rawnames, raws_permod_both_sides, sources_type,
                             src_file_grouping_ind, src_grouping, use_main_LFP_chan,
                             side_to_use, new_main_side, data_modalities,
                             crop_start,crop_end,msrc_inds, rec_info_pri,
                             mainLFPchans_pri=None, mainLFPchan_newname=None):
    '''
    uses loaded data in different modalities and converts is to numpy arrays,
    performing side switching if necessary

    side_to_use can be 'tremor_side', 'move_side' , 'both', 'left' , 'right'
    rawnames are important to have because they give ordering
    \<new_main_side\> is what will appear as main in the output

    usually mainLFPchan_newname wil not be used because I will better rename in run_PCA
    '''
    import copy
    from utils import changeRawInfoSides
    from utils_tSNE import revAnnSides

    assert new_main_side in ['left','right']
    dat_pri = []
    times_pri = []
    times_hires_pri = []
    dat_lfp_hires_pri = []

    extdat_pri = []
    anns_pri = []
    anndict_per_intcat_per_rawn = {}
    #rec_info_pri = []
    subfeature_order_pri = []
    subfeature_order_lfp_hires_pri = []

    use_ipsilat_CB = True

    aux_info_per_raw = {}
    for rawind in range(len(rawnames) ):
        gen_subj_info = gv.gen_subj_info
        subj,medcond,task  = utils.getParamsFromRawname(rawnames[rawind])
        maintremside = gen_subj_info[subj]['tremor_side']
        mainmoveside = gen_subj_info[subj].get('move_side',None)
        # here is just set this, I will not necessarily use it later, unless
        # use_main_LFP_chan is set
        if mainLFPchans_pri is not None:
            mainLFPchan =  mainLFPchans_pri[rawind]
        else:
            mainLFPchan =  gen_subj_info[subj].get('lfpchan_used_in_paper',None)

        rawname_ = rawnames[rawind]

        #src_rec_info_fn = genRecInfoFn(rawname_,sources_type,src_file_grouping_ind)
        #src_rec_info_fn_full = os.path.join(gv.data_dir, input_subdir,
        #                                    src_rec_info_fn)
        #rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
        #rec_info_pri += [rec_info]

        rec_info = rec_info_pri[rawind]

        roi_labels = rec_info['label_groups_dict'][()]      # dict of (orderd) lists
        srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]  # order of grouping names
        assert set(roi_labels.keys() ) == set(srcgrouping_names_sorted )
        assert len(roi_labels) == 1, 'several groupings in single run -- not implmemented'
        # assuming we have only one grouping present
        roi_labels_cur = roi_labels[srcgrouping_names_sorted[src_grouping ]  ]


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



        #assert isinstance(main_body_side,str)


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

            if len(chnames_lfp_to_use) == 0:
                raws_lfp_perside[side] = None
                if raw_lfp_hires is not None:
                    raws_lfp_hires_perside[side] = None
            else:
                raws_lfp_perside[side].pick_channels(   chnames_lfp_to_use )
                print('LFFFFP ',side, chnames_lfp_to_use, raws_lfp_perside[side].ch_names)
                if raw_lfp_hires is not None:
                    raws_lfp_hires_perside[side].pick_channels(   chnames_lfp_to_use  )



            # old ver
            chis =  mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_all_*'.format(side)  )
            if len(chis) == 0:
                chis =  mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_{}_[0-9]+_c[0-9]+'.
                                                format(side, src_grouping)  )
            nchis_before_CB_manip = len(chis)

            print('brain side ',side,chis)
            if use_ipsilat_CB:
                CB_contrahand_parcel_ind = roi_labels_cur.index('Cerebellum_{}'.format(side))
                hand_side = getOppositeSideStr(side)
                CB_ipsihand_parcel_ind = roi_labels_cur.index('Cerebellum_{}'.format(hand_side) )
                CB_contrahand_inds = mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_{}_{}_c[0-9]+'.
                                                            format(side,src_grouping,CB_contrahand_parcel_ind)  )
                CB_ipsihand_inds = mne.pick_channels_regexp(raw_srconly.ch_names, 'msrc{}_{}_{}_c[0-9]+'.
                                                        format(hand_side,src_grouping,CB_ipsihand_parcel_ind)  )
                print(chis, CB_contrahand_inds)
                chis = np.setdiff1d(chis, CB_contrahand_inds)
                print(chis)
                chis = np.hstack( [chis, CB_ipsihand_inds]).astype(int)  # for some reason hstack converts to float
                print(chis)
                #TODO: remove Cerbellum sources, add Cerebellum sources from the other side

            assert len(chis) > 0, f'no source channels found at least on {side} side, before CB manip was {nchis_before_CB_manip}'
            chnames_src = [raw_srconly.ch_names[chi] for chi in chis]
            raws_srconly_perside[side].pick_channels(   chnames_src  )



            print( raws_srconly_perside[side].ch_names)
            print('{} brain side,  {} sources'.format(side, len(chis) ) )



        import utils_preproc as upre
        sfreq = int( raw_lfponly.info['sfreq'] )
        anndict_per_intcat = upre.collectAllMarkedIntervals( rawname_, raw_lfponly.times,
            new_main_side, side_switch_needed, sfreq=sfreq,
            ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt',
            printLog=False, allow_missing_files=False,
            remove_nonmain_artif= side_to_use != 'both' )
        anndict_per_intcat_per_rawn[rawname_] = anndict_per_intcat

        anns_fn = rawname_ + '_anns.txt'
        anns_fn_full = os.path.join(gv.data_dir, anns_fn)
        anns = mne.read_annotations(anns_fn_full)
        if crop_end is not None:
            anns.crop(crop_start,crop_end)


        if side_switch_needed:
            print('collectDataFromMultiRaws: Performing switching sides')
            anns = revAnnSides(anns)      # revert anns

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


            raws_srconly_perside   = raws_srconly_perside_new
            raws_lfp_perside       = raws_lfp_perside_new
            raws_lfp_hires_perside = raws_lfp_hires_perside_new

            if isinstance(main_body_side, str):
                main_body_side = new_main_side

        if isinstance(main_body_side, str):
            test_side_let =  main_body_side[0].upper()
        else:
            test_side_let =  main_body_side[0][0].upper()
        print(test_side_let, raws_lfp_perside[test_side_let].ch_names)

        anns_pri += [anns]

        ####################  Load emg

        EMG_per_hand = gv.EMG_per_hand
        if isinstance(main_body_side,str):
            chnames_emg = EMG_per_hand[main_body_side]
        else:
            chnames_emg = raw_emg_rectconv.ch_names

        print(rawname_,chnames_emg)

        rectconv_emg, ts_ = raw_emg_rectconv[chnames_emg]
        chnames_emg = [chn+'_rectconv' for chn in chnames_emg]

        # select side to be outputted
        raws_permod = {'LFP' : raws_lfp_perside, 'msrc' : raws_srconly_perside }
        if isinstance(main_body_side,str):
            hand_sides = [main_body_side[0].upper() ]
        elif isinstance(main_body_side,list) and isinstance(main_body_side[0], str):
            hand_sides = main_body_side
        else:
            raise ValueError('Wrong main_body_side',main_body_side)
        print('main_body_side {}, hand_sides to construct features '.format(main_body_side) ,hand_sides)

        if sources_type == 'HirschPt2011':
            allowd_srcis_subregex = '[{}]'.format( ','.join( map(str, msrc_inds ) ))
        #else:
        #    allowd_srcis_subregex = '[{}]'.format( ','.join( map(str, msrc_inds ) ))

        ############# Concatenate data
        subfeature_order = []
        dats = []
        for side_hand in hand_sides:
            for mod in data_modalities:
                #sd = hand_sides_all[1-hand_sides.index(side_hand) ]  #
                opside= getOppositeSideStr(side_hand)
                #if mod in ['src','msrc']:  No! They are both in the brain, so both contralat!
                opside = opside[0].upper()

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
                    print(curraw.ch_names)
                    #import pdb; pdb.set_trace()

                    curdat = curraw.get_data()
                    chnames_added = curraw.ch_names
                dats += [ curdat ]

                subfeature_order += chnames_added
                print(f'--  side_hand {side_hand}, opside {opside} modality {mod}')
                print(chnames_added)

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
                opside = opside[0].upper()
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


        ecg_fname = os.path.join(gv.data_dir, '{}_ica_ecg.npz'.format(rawname_) )
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



    return dat_pri, dat_lfp_hires_pri, extdat_pri, anns_pri, anndict_per_intcat_per_rawn, times_pri,\
    times_hires_pri, subfeature_order_pri, subfeature_order_lfp_hires_pri, aux_info_per_raw

# found on stackoverflow by someone called NaN. Some black magic I don't quite
# understand
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

    if len(win) == 2 and win[0] == 1 and len(stepby) == 2 and stepby[0] == 1:
        if a.ndim == 2 and a_s.ndim == 2:
            a_s = a_s[:,None]
    return a_s

def H_difactmob(dat,dt, windowsz = None, dif = None, skip=None, stride_ver = True):
    import pandas as pd
    # last axis is time axis
    if dif is None:
        dif = np.diff(dat,axis=-1, prepend=dat[:,0][:,None] ) / dt
    if windowsz is None:
        print('AA')
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

                # there is some bug in stride function for shape=(1,N) array that I correct this way
                if activity.ndim == 1:
                    assert dat.shape[0] == 1
                    activity = activity[None,:]

                #print(dat.shape, activity.shape, win, step)

                stride_view_dif = stride(dif, win=win, stepby=step )
                vardif = np.var(stride_view_dif,axis=-1, ddof=1)

                if vardif.ndim == 1:
                    assert dat.shape[0] == 1
                    vardif = vardif[None,:]
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

    #import pdb;pdb.set_trace()

    #  we don't want to have activity equal exactly to zero because we will
    #  divide next. This way if vardif == 0, then we get 0 as asnwer still
    eps = 1e-14
    bad_act = np.abs(activity) < eps
    activity_bettered = activity.copy()
    activity_bettered[bad_act] = eps

    bad_vardif = np.abs(vardif) < eps

    # DEBUG
    #c0 = np.max( activity_bettered[0]  ) / np.max(vardif[0])
    #plt.figure(figsize=(15,4) ); plt.plot(vardif[0], label='vardif'); plt.legend()
    #plt.plot(activity[0] / c0 , label='activity', ls='--'); plt.legend()
    #plt.vlines( np.where(bad_act[0] )[0],    0,100, label=  'bad_act', color='purple' ); plt.legend()
    #plt.vlines( np.where(bad_vardif[0] )[0],-10,0,  label='bad_vardif', color='red' ); plt.legend()
    #print('np.sum(bad_vardif) = ',np.sum(bad_vardif) )
    #print('np.sum(bad_act) = ',np.sum(bad_act) )

    #print(activity[0])
    #print(bad_act[0])

    mobility = np.sqrt( vardif / activity_bettered )
    #plt.figure(figsize=(15,4) ); plt.plot(mobility[0], label='mobility_pre', ls=':'); plt.legend()

    #bad_act_and_vardif = bad_act & ( np.abs(vardif) < 1e-10 )
    #mobility[bad_act_and_vardif ] = 0
    mobility[bad_vardif ] = 0
    mobility[bad_act ] = 0  # "political" decision!

    #c = np.max( activity_bettered[0]  ) / np.max(mobility[0])
    #plt.figure(figsize=(15,4) ); plt.plot(mobility[0], label='mobility'); plt.legend()
    #plt.plot(activity_bettered[0] / c , label='activity_bettered', ls='--'); plt.legend()

    # we need to do something when vardif is nonzero and activity is zero (it
    # does happen e.g. when there is a step happening) -- I don't want to get
    # huge mobility because I replace eps. Mobility is not very well defined
    # but maybe I can just postulate it to be zero

    if (skip is not None) and not stride_ver:
        #dif       = dif[:,::skip]  # DON'T touch it!!
        activity  = activity[:,::skip]
        mobility  = mobility[:,::skip]

    return dif,activity, mobility

def Hjorth(dat, dt=1., windowsz = 1, skip=1,stride_ver=True, remove_invalid = False, pad=True):
    # first dim is channel number, second is time
    if isinstance(dat,list):
        acts, mobs, compls, wbds = [],[],[], []
        for subdat in dat:
            a,m,c,wbd = Hjorth(subdat,dt,windowsz,skip,stride_ver=stride_ver,
                               remove_invalid=remove_invalid)
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
    if pad:
        padlen = windowsz-skip
        #if  int(ndb / windowsz) * windowsz - ndb == 0:  # dirty hack to agree with pandas version
        #    padlen += 1

        if stride_ver:
            #print('Using Hjorth stride ver for dat type {}'.format( type(dat) ) )
            assert dat.ndim == 2, dat.shape
            dat = np.pad(dat, [(0,0), (padlen,0) ], mode='edge' )
        else:
            raise ValueError('lost validity')
    else:
        padlen = 0

    dif = np.diff(dat,axis=-1, prepend=dat[:,0][:,None] ) / dt
    dif2 = np.diff(dif,axis=-1, prepend=dif[:,0][:,None] ) / dt



    dif, activity, mobility = H_difactmob(dat,dt, windowsz=windowsz,
        dif=dif, skip=skip, stride_ver=stride_ver)
    import gc;gc.collect()
    dif2, act2, mob2 = H_difactmob(dif,dt, windowsz=windowsz,
        dif=dif2, skip=skip, stride_ver=stride_ver)

    #plt.figure(); plt.plot(dif.T, label='dif'); plt.legend()
    #plt.figure(); plt.plot(dif2.T, label='dif2'); plt.legend()

    #plt.figure(); plt.plot(activity.T, label='activity'); plt.legend()
    #plt.figure(); plt.plot(mobility.T, label='mobility'); plt.legend()

    #del dif
    #del dif2
    #del act2


    eps = 1e-14
    bad_mob = np.abs(mobility) < eps
    mobility_bettered = mobility.copy()
    mobility_bettered[bad_mob] = eps
    complexity = mob2 / mobility_bettered
    complexity[ np.abs(mob2) < eps ] = 0
    # political decision, otherwise compleixty becomes huge at the step moment
    complexity[ bad_mob ] = 0

    #plt.figure()
    #plt.plot(mobility[0], label='mobility')
    #plt.plot(mobility_bettered[0], label='mobility_bettered')
    #plt.plot(mob2[0], label='mob2')
    #plt.plot(complexity[0], label='complexity')
    #print(complexity[0] )
    ##plt.ylim(0,250)
    #plt.legend()

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
    I need chnames_nice because I will use them to select AAL parcels using reg expressions
    I also need short names because I will parse them. Of course they need to match

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
                        # here i and j are indices in chnames_nice
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

    order of newchns is not important becasue I will run newchns.index every time

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

    eps = 1e-14

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
                r = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=complex )
                rimabs = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=complex )

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
                    r = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=complex )
                    rimabs = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=complex )

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
                            r_cur_abs = np.abs(r_cur)
                            m = np.min(r_cur_abs)
                            if m < eps:
                                numsmall = np.sum(r_cur) <  eps
                                if numsmall/r_cur.size > 0.005:
                                    print('tfr2csd Warning: in r_cur parcel2parcel min is {}, total num of small bins {}={:.2f}%. Using maximum'.
                                        format(m,numsmall, (numsmall/r_cur.size) * 100) )
                                r_cur[r_cur_abs<eps] = eps
                            r_cur = np.log(r_cur)
                            if pi1 != pi2:  # otherwise we won't use it anyway, we'll use modulus of the whole

                                m = np.min(rimabs_cur)
                                if m < eps:
                                    numsmall = np.sum(rimabs_cur <  eps)
                                    if numsmall/rimabs_cur.size > 0.005:
                                        print('tfr2csd Warning: in rimabs_cur min is {}, total num of small bins {}={:.2f}%. Using maximum'.
                                            format(m,numsmall, (numsmall/rimabs_cur.size) * 100) )
                                    rimabs_cur = np.maximum(eps, rimabs_cur)
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
                    assert pi1 == parcel_ind1 and pi2 == parcel_ind2, (pi1,chn1,pi2,chn2)

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

                r = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=complex )
                #rimabs = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=complex )

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
                            r_cur_abs = np.abs(r_cur)
                            m = np.min(r_cur_abs)
                            if m < eps:
                                numsmall = np.sum(r_cur_abs) <  eps
                                if numsmall/r_cur.size > 0.005:
                                    print('tfr2csd Warning: in r_cur LFP2parcel min is {}, total num of small bins {}={:.2f}%. Using maximum'.
                                        format(m,numsmall, (numsmall/r_cur.size) * 100) )
                                r_cur[r_cur_abs<eps] = eps
                            r_cur = np.log(r_cur)
                        r += r_cur
                        ninds_counted += 1

                        # we don't really care which particular sources, we only need parcel indices
                        chn2 = oldchns[j]
                    #print(i,j,chn1,chn2)
                    side2, gi2, parcel_ind2, si2  = parseMEGsrcChnameShort(chn2)
                    assert pi2 == parcel_ind2, (pi2,chn2)

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
                r = np.zeros( (1, dat.shape[1], dat.shape[2] ), dtype=complex )
                for (i,j) in ind_pairs:
                    rtmp = np.conj ( dat[[i]] ) *  ( dat[[j]] )    # upper diagonal elements only, same freq cross-channels
                    #print(rtmp)

                    if normalize:
                        norm = np.abs(rtmp)
                        rtmp /= norm

                    r_cur = rtmp / len(ind_pairs)
                    if log:
                        r_cur_abs = np.abs(r_cur)
                        m = np.min(r_cur_abs)
                        if m < eps:
                            numsmall = np.sum(r_cur) <  eps
                            if numsmall/r_cur.size > 0.005:
                                print('tfr2csd Warning: in r_cur LFP2LFP min is {}, total num of small bins {}={:.2f}%. Using maximum'.
                                    format(m,numsmall, (numsmall/r_cur.size) * 100) )
                            r_cur[r_cur_abs<eps] = eps
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


def _feat_correl3(arg):
    resname,bn_from,bn_to,pc,fromi,toi,dfrom,dto,mfrom,mto, \
        windowsz,skip,oper,pos,local_means,pad,verbose = arg
    # fromi and toi --
    if verbose == 2:
        print('    _feat_correl3 debug output')
        for varn,var in locals().items():
            if varn not in ['dfrom', 'dto', 'arg','varn','var']:
                print(varn, '=',var)
        print('  end print arg')
    elif verbose == 3:
        print('    _feat_correl3 debug output')
        for varn,var in locals().items():
            if varn not in [ 'arg','varn','var']:
                print(varn, '=',var)
        print('  end print arg')
    #if verbose >= 3:
    #    print(arg)

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

    #print(mfrom,mto)

    #if gv.DEBUG_MODE:
    #    print(f'padding = {pad}')
    if pad == 'pre_left':
        # to agree with Hjorth
        ndb = len(dfrom)
        padlen = windowsz-skip
        #if  int(ndb / windowsz) * windowsz - ndb == 0:  # dirty hack to agree with pandas version
        #    padlen += 1
        # padding after right edge
        dfrom = np.pad(dfrom, [(0), (padlen) ], mode='edge' )
        dto   = np.pad(dto, [(0), (padlen) ], mode='edge' )
    elif pad == 'pre_right':
        ndb = len(dfrom)
        padlen = windowsz-skip
        dfrom = np.pad(dfrom, [ (padlen), (0) ], mode='edge' )
        dto   = np.pad(dto, [ (padlen), (0) ], mode='edge' )
    elif pad not in ['no','post_right','post_left']:
        raise ValueError(f'wrong pad value {pad}')

    win = (windowsz,)
    step = (skip,)
    stride_view_dfrom = stride(dfrom, win=win, stepby=step )
    stride_view_dto   = stride(dto, win=win, stepby=step )
    if oper == 'corr':
        if local_means:
            mfrom_cur = np.mean(stride_view_dfrom, axis=-1)[:,None]
            mto_cur   = np.mean(stride_view_dto,   axis=-1)[:,None]
        else:
            mfrom_cur = mfrom
            mto_cur = mto
        rr = np.mean( (stride_view_dfrom - mfrom_cur ) * (stride_view_dto - mto_cur ) , axis=-1)
    elif oper == 'div':
        assert pos, "One cannot divide if one may have zeros in the denominator!"
        rr = stride_view_dfrom / stride_view_dto
    else:
        raise ValueError('wrong oper {}'.format(oper) )
    #import ipdb; ipdb.set_trace()

    #if gv.DEBUG_MODE:
    #    plt.figure()
    #    plt.plot( dfrom, label ='from', ls='--' )
    #    plt.plot( dto, label='to', ls = ':')
    #    plt.plot( np.arange(len(dfrom) )[::skip][:len(rr) ] , rr, label=f'{bn_from} -- {bn_to}')
    #    plt.title(resname)
    #    plt.legend()

    if verbose >= 3:
        print('rr = ',rr)
    # removing invalid due to padding
    #rr = rr[0:-windowsz//skip]

    if pad == 'pre_left':
        sl = slice(None, - windowsz//skip + 1,None)
        if verbose >= 2:
            print(f'slice(windowsz//skip - 1,None,None) = {sl} ')
        rr = rr[sl]
    elif pad == 'pre_right':
        sl = slice(windowsz//skip - 1,None,None)
        if verbose >= 2:
            print(f'slice(windowsz//skip - 1,None,None) = {sl} ')
        rr = rr[sl]

    #if pad == 'pre_left':
    #    sl = slice(windowsz//skip - 1,None,None)
    #    if verbose >= 2:
    #        print(f'slice(windowsz//skip - 1,None,None) = {sl} ')
    #    rr = rr[sl]
    #elif pad == 'pre_right':
    #    sl = slice(None, - windowsz//skip + 1,None)
    #    if verbose >= 2:
    #        print(f'slice(windowsz//skip - 1,None,None) = {sl} ')
    #    rr = rr[sl]

    #ndb = len(dfrom)
    #padlen = windowsz-skip
    #if  int(ndb / windowsz) * windowsz - ndb == 0:  # dirty hack to agree with pandas version
    #    padlen += 1

    del stride_view_dfrom
    del stride_view_dto
    del dfrom
    del dto

    #pred = np.arange(ndb + padlen)
    #wnds = stride(pred, (windowsz,), (skip,) )
    #wbd = np.vstack( [ wnds[:,0], wnds[:,-1] + 1 ] )
    #wbd = wbd[:,sl] - padlen

    #window_boundaries_st =  np.arange(0,ndb - windowsz, skip ) # use before padding
    #window_boundaries_end = window_boundaries_st + windowsz
    #wbd = np.vstack( [ window_boundaries_st, window_boundaries_end] )
    #print(dfrom.shape, stride_view_dto.shape)

    # actually returned fromi and toi are not used later
    # only the bn_From
    return rr,resname,bn_from,bn_to,pc,fromi,toi,oper,pos

def computeCorr(raws, chnames_per_band, defnames, skip, windowsz, band_pairs,
                    parcel_couplings, LFP2parcel_couplings, LFP2LFP_couplings,
                    res_group_id = 9, n_jobs=None, positive=1, templ = None,
                    roi_labels = None, sort_keys=None, printLog=False, means=None,
                    local_means=False, reverse=True, verbose=0,
                pad='pre_left'):
    '''
    dat is chans x timebins
    positive -- wheather inputs are postive
    windowz is in bins of Xtimes_full
    defnames -- only used to get LFP indices
    returns lists
    chnames_per_band: dict of lists of strings with (short) channel names.
        Before we assumed that ordering and indexing of channels is the same in all names
    '''
    #e.g  bandPairs = [('tremor','beta'), ('tremor','gamma'), ('beta','gamma') ]
    # compute Pearson corr coef between different band powers all chan-chan

    #TODO: the names I receive here are not pure channel names so it won't work. Or maybe I don't need it
    #sides_,groupis_,parcelis_,compis_ = utils.parseMEGsrcChnamesShortList(names)

    # this is if we want to really compute second order features. I don't use
    # it this way
    if templ is None:
        templ = r'con_{}.*:\s(.*),\1'

    locm = local_means

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
        names_from = chnames_per_band[bn_from]
        names_to   = chnames_per_band[bn_to]

        # effind -- used to take raws,means and names

        if bn_from.find('HFO') < 0 and bn_to.find('HFO') < 0:
            for pc in parcel_couplings:
                (pi1,pi2) = pc
                ind_pairs = parcel_couplings[pc]  # ind_pairs won't work because it is indices not in band array
                for (i,j) in ind_pairs:
                    rev = (pi1 != pi2) and (bn_from != bn_to)
                    # indices in defnames
                    effind_from = names_from.index( defnames[i] )
                    effind_to   = names_to.index( defnames[j] )

                    dfrom = raws[bn_from][effind_from][0][0] # band from -> i
                    dto   = raws[bn_to][effind_to][0][0]     # band to   -> j
                    assert dfrom.size > 1
                    assert dto.size > 1

                    chn1 = names_from[effind_from]     # from[i]
                    chn2 = names_to[effind_to]         # to [j]
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn1)
                    side2, gi2, parcel_ind2, si2  = utils.parseMEGsrcChnameShort(chn2)

                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_from,side1,res_group_id,pi1,0)
                    newchn2 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,  side2,res_group_id,pi2,0)

                    resname = '{}_{},{}'.format(oper,newchn1,newchn2)
                    # we won't have final name before averaging
                    if means is not None:
                        #m1 = means[bn_from][chn1]
                        #m2 = means[bn_to][chn2]
                        m1 = means[bn_from][effind_from]
                        m2 = means[bn_to][effind_to]
                    else:
                        m1,m2=None,None
                    arg = resname,bn_from,bn_to,pc,i,j,\
                        dfrom,dto,m1,m2
                    args += [arg]
                    #print('1', resname)

                    # I had   <band_from> ch_1,  <band_to> ch_2,    now I want
                    # <band_to> ch_1 and <band_from> ch_2

                    if rev and reverse: # reversal of bands, not indices (they are symmetric except when division)
                        # change bands. So I need same channels (names and
                        # data) but with exchanged bands
                        effind_from = names_from.index( defnames[j] )
                        effind_to   = names_to.index( defnames[i] )

                        dfrom = raws[bn_to][effind_to][0][0]      # band to   -> i
                        dto   = raws[bn_from][effind_from][0][0]  # band from -> j

                        # this is about channel names, NOT bands
                        # here we should have names_to == names_from since both
                        # are for non-HFO bands
                        chn1 = names_to[effind_to]             # to[i]
                        chn2 = names_from[effind_from]         # from[j]
                        side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn1)
                        side2, gi2, parcel_ind2, si2  = utils.parseMEGsrcChnameShort(chn2)

                        # change bands
                        newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,    side1,res_group_id,pi1,0) # to ,    side of i , parcel ind 1
                        newchn2 = '{}_msrc{}_{}_{}_c{}'.format(bn_from,  side2,res_group_id,pi2,0) # from , side of j  , parcel ind 2

                        resname = '{}_{},{}'.format(oper,newchn1,newchn2)
                        # we won't have final name before averaging
                        if means is not None:
                            #m1 = means[bn_to][chn1]
                            #m2 = means[bn_from][chn2]
                            m1 = means[bn_to][effind_to]
                            m2 = means[bn_from][effind_from]
                        else:
                            m1,m2=None,None
                        arg = resname,bn_to,bn_from,pc,i,j,\
                            dfrom,dto,m1,m2
                        args += [arg]
                        #print('1rev', resname)

        #XOR instead of AND, allow one to has HFO
        #if bool(bn_from.find('HFO') < 0) ^ bool(bn_to.find('HFO') <= 0):
        for pc in LFP2parcel_couplings:
            (chn1,pi2) = pc
            ind_pairs = LFP2parcel_couplings[pc]
            # first in resname goes src

            # do I need beta LFP vs tremor src AND  beta src vs tremor LFP ?
            # maybe yes because if I duplicate paris in the input I'd have to
            # filter out it in LFP2LFP
            for (i,j) in ind_pairs:
                rev = False
                ind_lfp = i
                ind_src = j
                assert defnames[ind_lfp].startswith('LFP')
                assert defnames[ind_src].startswith('msrc')
                # I cannot reverse then
                # i -- always LFP, j -- alwyas src
                if bn_from.find('HFO') >= 0:
                    # effind_from index we'll use to access raws[bn_from]
                    effind_from = names_from.index( defnames[ind_lfp] )
                    effind_to   = names_to.index( defnames[ind_src] )

                    chn_src = names_to[ind_src]
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn_src)
                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,side1,res_group_id,parcel_ind1,0)

                    chn = names_from[effind_from]
                    newchn2 = '{}_{}'.format(bn_from, chn )
                # I cannot reverse then
                elif bn_to.find('HFO') >= 0:
                    # adds in wrong order but bothe will be there ultimately
                    effind_from = names_from.index( defnames[ind_src] )
                    effind_to   = names_to.index( defnames[ind_lfp] )

                    chn_src = names_from[ind_src]
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn_src)
                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_from,side1,res_group_id,parcel_ind1,0)

                    chn = names_to[effind_to]
                    newchn2 = '{}_{}'.format(bn_to, chn )
                elif bn_to.find('HFO') < 0 and  bn_to.find('HFO') < 0 :
                    effind_from = names_from.index( defnames[ind_src] )
                    effind_to   = names_to.index( defnames[ind_lfp] )

                    chn_src = names_from[effind_from]
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
                dto   = raws[bn_to]  [effind_to][0][0]
                assert dfrom.size > 1
                assert dto.size > 1

                #import pdb; pdb.set_trace()

                #name = '{}_{},{}'.format(oper,nfrom,nto)
                # we won't have final name before averaging
                if means is not None:
                    #m1 = means[bn_from][names_from[effind_from] ]
                    #m2 = means[bn_to][names_to[effind_to]]
                    m1 = means[bn_from][effind_from ]
                    m2 = means[bn_to][effind_to]
                else:
                    m1,m2=None,None
                arg = resname,bn_from,bn_to,pc,effind_from,effind_to,\
                    dfrom,dto,m1,m2
                args += [arg]
                #print('2', resname)

                ######################3
                if rev and reverse:
                    # change order
                    effind_from = names_from.index( defnames[ind_lfp] )
                    effind_to   = names_to.index( defnames[ind_src] )

                    chn_src = names_to[effind_to]
                    side1, gi1, parcel_ind1, si1  = utils.parseMEGsrcChnameShort(chn_src)
                    newchn1 = '{}_msrc{}_{}_{}_c{}'.format(bn_to,side1,res_group_id,parcel_ind1,0)

                    #2 corr_tremor_msrcR_9_2_c0,gamma_LFPR01
                    #2rev corr_gamma_msrcR_9_60_c0,tremor_msrcR_0_2_c1

                    chn = names_from[effind_from]
                    newchn2 = '{}_{}'.format(bn_from, chn )


                    resname_rev = '{}_{},{}'.format(oper,newchn1,newchn2)
                    #import pdb; pdb.set_trace()

                    resname = resname_rev
                    # not that we exchanged effinds
                    d1   = raws[bn_to]   [effind_to][0][0]
                    d2   = raws[bn_from] [effind_from][0][0]
                    #d1 = raws[bn_to][effind_from][0][0]
                    #d2   = raws[bn_from][effind_to][0][0]
                    assert d1.size > 1
                    assert d2.size > 1

                    #name = '{}_{},{}'.format(oper,nfrom,nto)
                    # we won't have final name before averaging
                    if means is not None:
                        #m1 = means[bn_to][names_to[effind_from] ]
                        #m2 = means[bn_from][names_from[effind_to]]
                        m1 = means[bn_to][effind_to ]
                        m2 = means[bn_from][effind_from]
                    else:
                        m1,m2=None
                    arg = resname,bn_to,bn_from,pc,effind_from,effind_to,\
                        d1,d2,m1,m2
                    args += [arg]
                    #print('2rev', resname)

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
                    effind_to   = names_to.index( defnames[j] )
                # I cannot reverse then
                elif bn_to.find('HFO') >= 0:
                    effind_from = names_from.index( defnames[i] )
                    effind_to   = names_to.index( defnames[j] )
                elif bn_to.find('HFO') < 0 and  bn_to.find('HFO') < 0 :
                    effind_from = names_from.index( defnames[i] )
                    effind_to   = names_to.index( defnames[j] )
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
                arg = resname,bn_from,bn_to,pc,effind_from,effind_to,\
                    dfrom,dto,m1,m2
                args += [arg]
                #print('3', resname)

                if rev and reverse:
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
                    arg = resname,bn_to,bn_from,pc,effind_from,effind_to,\
                        d1,d2,m1,m2
                    args += [arg]
                    #print('3rev', resname)

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

    common_arg_part = (windowsz,skip,oper,positive,local_means,pad,verbose)
    args = [ (*arg,*common_arg_part) for arg in args ]
    if ncores > 1:
        #if ncores > 1:
        print('high ord feats:  Sending {} tasks to {} cores'.format(len(args), ncores))
        #pool = mpr.Pool(ncores)
        #res = pool.map(_feat_correl3, args)
        #pool.close()
        #pool.join()

        res = Parallel(n_jobs=n_jobs)(delayed(_feat_correl3)( arg  ) for arg in args)
    else:
        res = []
        for arg in args:
            res += [ _feat_correl3(arg) ]

    dct = {}
    dct_nums = {}
    # wbd is same for all
    for r in res:
        rr,resname,bn_from,bn_to,pc,fromi,toi,oper,pos  = r
        # make averages

        #nfrom = pc[0]
        #nto = pc[1]
        #name = '{}_{},{}'.format(oper,nfrom,nto)
        #print('collect resname ',resname)
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
    wnds = stride(pred, (windowsz,), (skip,) )
    wbd = np.vstack( [ wnds[:,0], wnds[:,-1] + 1 ] )
    wbd = wbd[:,sl] - padlen

    #print( dct_nums )
    if verbose > 0:
        print('resname_min_num = ',resname_min_num, num)
    if verbose >= 2:
        print( f'dct_nums = {dct_nums}' )

    return cors,cor_names, dct_nums, wbd


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
        print('smoothData:  Sending {} tasks to {} cores'.format(len(args), mpr.cpu_count()))
        # I don't put any backend because later I will use "with"
        res = Parallel(n_jobs=n_jobs)(delayed(_smoothData1D_proxy)(arg) for arg in args)

        #pool = mpr.Pool(n_jobs)
        #res = pool.map(_smoothData1D_proxy, args)

        for dim,cursmooth in res:
            r[dim] = cursmooth
            rstates[dim] = cursmooth.states.mean[:,0]

        #pool.close()
        #pool.join()
    return np.vstack(rstates)

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

            #print(low,high,freqis)
            #plt.plot(csdcur[0][0], label=f'{bandname}')

            #plt.plot(bandpow[0], label=f'{bandname}')
            #plt.legend()

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


# I have to give allow_CUDA separately because some things I cannot run on CUDA
# (e.g. apply_function)
def bandFilter(rawnames, times_pri, main_sides_pri, side_switched_pri,
               sfreqs, skips, dat_pri_persfreq, fband_names_inc_HFO, fband_names_HFO_all,
               fbands, n_jobs_flt, allow_CUDA, chnames, chnames_hires,
               smoothen_bandpow = 0, ann_MEGartif_prefix_to_use="_ann_MEGartif_flt",
               artif_handling = 'reject', anns_MEGartif=None, artif_LFPartif=None,
              filter_phase='minimum',  anndict_per_intcat_per_rawn = None,
               artif_before_bandpow = 'impute_const', return_imputed_flt = True ):
    import utils_tSNE as utsne
    sfreq = sfreqs[0]
    raw_perband_flt_pri      = [0] * len(rawnames)
    raw_perband_bp_pri       = [0] * len(rawnames)
    chnames_perband_flt_pri  = [0] * len(rawnames)
    chnames_perband_bp_pri   = [0] * len(rawnames)
    means_perband_flt_pri    = [dict()] * len(rawnames)
    means_perband_bp_pri     = [dict()] * len(rawnames)

    filter_length_dict = {'tremor':'2s'}
    # maybe if I use sfreq >> 256 and care about really fast things I might
    # reconsider values here
    for fb in gv.fbands:
        if fb == 'tremor':
            continue
        filter_length_dict[fb] = '1s'

    n_jobs_maybe_cuda = n_jobs_flt
    if allow_CUDA and gv.CUDA_state == 'ok':
        n_jobs_maybe_cuda = 'cuda'

    for rawind,rn in enumerate(rawnames):
        raw_perband_flt = {}
        raw_perband_bp  = {}
        chnames_perband_flt = {}
        chnames_perband_bp  = {}
        #means_perband_flt = {}
        #means_perband_bp  = {}

        main_side_before_change = main_sides_pri[rawind]  # side of body
        opsidelet = utils.getOppositeSideStr(main_side_before_change[0].upper() ) # side of brain


        #fname_full_LFPartif = os.path.join(gv.data_dir, '{}_ann_LFPartif.txt'.format(rn) )
        #fname_full_MEGartif = os.path.join(gv.data_dir, '{}{}.txt'.format(rn,ann_MEGartif_prefix_to_use) )
        #anns_LFPartif = mne.read_annotations(fname_full_LFPartif)
        #anns_MEGartif = mne.read_annotations(fname_full_MEGartif)

        main_side_before_change = main_sides_pri[rawind]  # side of body
        opsidelet = utils.getOppositeSideStr(main_side_before_change[0].upper() ) # side of brain
        wrong_brain_sidelet = main_side_before_change[0].upper()


        if artif_handling == 'reject':
            if anndict_per_intcat_per_rawn is None:
                anns_MEGartif, _, _, _ = \
                    utsne.concatAnns([rn],[times_pri[rawind] ],[ann_MEGartif_prefix_to_use],
                                        side_rev_pri=[side_switched_pri[rawind] ] )

                anns_LFPartif, _, _, _ = \
                    utsne.concatAnns([rn],[times_pri[rawind] ],['_ann_LFPartif'],
                                        side_rev_pri=[side_switched_pri[rawind] ]  )
                #print(anns_MEGartif.onset)
                #print(anns_LFPartif.onset)
                anns_LFPartif = utils.removeAnnsByDescr(anns_LFPartif, ['artif_LFP{}'.format(wrong_brain_sidelet) ])
                #print(anns_LFPartif.onset)
            else:
                anns_MEGartif = anndict_per_intcat_per_rawn[rn ]['artif']['MEG']
                anns_LFPartif = anndict_per_intcat_per_rawn[rn ]['artif']['LFP']
        elif artif_handling == 'ignore':
            anns_MEGartif= mne.Annotations([],[],[])
            anns_LFPartif= mne.Annotations([],[],[])

        #anns_artif, anns_artif_pri, times2, dataset_bounds_ = \
        #    utsne.concatAnns(rawnames[rawind],times_pri[rawind], artif_mod_str,crop=(crop_start,crop_end),
        #                allow_short_intervals=True,
        #                    side_rev_pri = aux_info_perraw[rawnames[rawind]]['side_switched'],
        #                    wbd_pri = wbd_H_pri[rawind], sfreq=sfreq)



        for bandi,bandname in enumerate(fband_names_inc_HFO):
            means_perchan_flt = {}
            means_perchan_bp = {}
            for si,dat_pri_cur_sfreq in enumerate(dat_pri_persfreq):
                sfreq_cur = sfreqs[si]

                print(sfreq_cur,sfreq)
                # for hires we will only process HFO bands
                if sfreq_cur > sfreq + 1e-10 and (bandname not in fband_names_HFO_all):
                    continue
                if abs(sfreq-sfreq_cur) < 1e-10 and (bandname in fband_names_HFO_all):
                    continue

                if gv.DEBUG_MODE:
                    print('bandFilter ',si,bandname)

                dat_cur = dat_pri_cur_sfreq[rawind]
                low,high = fbands[bandname]
                # I don't want to put ch_namse because they can be too long
                r = utils.makeSimpleRaw(dat_cur, ch_names=None,
                                        sfreq=sfreq_cur, rescale=False,copy=True)
                filter_length = filter_length_dict[bandname]

                #anns_LFPartif.description

                # WARNING: this code does not check whether artifacs are
                # relevant. I.e. if I have artifact of kind LFPR<number>, it
                # will be counted REGARDLESS whether it is in the list of
                # channels names
                if bandname.find('HFO') < 0:
                    chnames_cur = chnames
                    #r.set_annotations(anns_MEGartif)
                    #r.set_annotations(anns_LFPartif)

                    chis_LFP  = mne.pick_channels_regexp(chnames_cur,'LFP.*' )
                    #chis_msrc = mne.pick_channels_regexp(chnames_cur,'msrc.*')

                    ann_toghether = mne.Annotations([],[],[])
                    for chni in chis_LFP:
                        temp_ann = utils.getArtifForFiltering(chnames_cur[chni], anns_LFPartif )
                        r.set_annotations(temp_ann)
                        ann_toghether = ann_toghether + temp_ann
                        r.filter(l_freq=low,h_freq=high, n_jobs=n_jobs_maybe_cuda,
                                skip_by_annotation='BAD_LFP{}'.format(opsidelet), pad='symmetric',
                                phase=filter_phase, filter_length=filter_length, picks=[chni] )

                    #r.set_annotations(anns_MEGartif)
                    for side in ['L','R']:
                        chis_msrc = mne.pick_channels_regexp(chnames_cur,f'msrc{side}.*')
                        temp_ann = utils.getArtifForFiltering(f'msrc{side}', anns_MEGartif )
                        r.set_annotations(temp_ann)
                        ann_toghether = ann_toghether + temp_ann
                        print(anns_MEGartif.__dict__)
                        print('LLLLLLLLLLLLLLLLLLLLLLll ',len(chis_msrc), side, len(temp_ann) )
                        r.filter(l_freq=low,h_freq=high, n_jobs=n_jobs_maybe_cuda,
                                skip_by_annotation=f'BAD_MEG{side}', pad='symmetric',
                                phase=filter_phase, filter_length=filter_length, picks=chis_msrc )
                    r.set_annotations(ann_toghether)
                else: # we have only LFP channels then
                    chnames_cur = chnames_hires
                    ann_toghether = mne.Annotations([],[],[])
                    for chni,chn in enumerate(chnames_cur):
                        temp_ann = utils.getArtifForFiltering(chnames_cur[chni], anns_LFPartif )
                        ann_toghether = ann_toghether + temp_ann
                        r.set_annotations(temp_ann)
                        r.filter(l_freq=low,h_freq=high, n_jobs=n_jobs_maybe_cuda,
                                skip_by_annotation='BAD_LFP{}'.format(opsidelet), pad='symmetric',
                                phase=filter_phase, filter_length=filter_length, picks=[chni] )

                    r.set_annotations(ann_toghether)

                print('TOGEYTEHR',r.annotations.__dict__)

                # Imputing is necessary becaue when bandpower receives something
                # with artifacats not removed, they have long influence, longer
                # than timewindow length
                # we give here annotations already filtered by side if necessary
                if artif_before_bandpow == 'impute_interp':
                    r_data_imputed = utils.imputeInterpArtif(r._data.T,  r.annotations, \
                                            chnames_cur, sfreq=sfreq_cur, in_place=return_imputed_flt)
                elif artif_before_bandpow == 'impute_const':
                    r_data_imputed = utils.imputeConstArtif(r._data.T,  r.annotations, \
                                            chnames_cur, sfreq=sfreq_cur, in_place=return_imputed_flt)

                r2 = utils.makeSimpleRaw(r_data_imputed.T, ch_names=None,
                                        sfreq=sfreq_cur, rescale=False,copy=True)
                r2.set_annotations(r.annotations)

                #r2 = r

                # for debug of some particular data
                #if np.max( np.abs( r.get_data() [:,-1] ) ) > 1e-10:
                #    import pdb; pdb.set_trace()

                if sfreq_cur <= sfreq + 1e-10:
                    rbp = r2.copy()
                    #chnames_perband_flt[bandname] = chnames_tfr
                    #chnames_perband_bp [bandname]  =chnames_tfr
                    chnames_perband_flt[bandname]  =chnames
                    chnames_perband_bp [bandname]  =chnames
                else:
                    rbp = r2  # we won't need filtered itself for hires
                    chnames_perband_bp [bandname] =chnames_hires
                rbp.apply_hilbert()
                # zeroth is convolution with 1 elemnt. After strideing it
                # should be like in tfr/Hjorth
                if smoothen_bandpow:
                    assert skips is not None
                    bpow_mav_wsz = skips[si] // 2
                    wnd_mav = np.ones( bpow_mav_wsz )
                    # or maybe for hires I can do it after resampling with
                    # a smaller window
                    fn = lambda x: np.convolve(wnd_mav, np.abs(x), mode='full' )[:dat_cur.shape[-1] ]
                else:
                    fn = np.abs
                rbp.apply_function(fn, n_jobs=n_jobs_flt, dtype=float)

                #raw_perband_flt += [ r  ]
                #raw_perband_bp  += [ rbp]
                if sfreq_cur <= sfreq + 1e-10:
                    assert bandname not in raw_perband_flt
                    raw_perband_flt[bandname] =  r
                else:
                    rbp.resample(sfreq=sfreq,n_jobs=n_jobs_maybe_cuda)

                assert bandname not in raw_perband_bp
                raw_perband_bp [bandname] =  rbp

                #q_to_include_mean_comp = 0.05
                # rejects bad.*
                #TODO maybe compute only over quiet periods
                #mflt = utsne.robustMean(r.get_data(reject_by_annotation='omit'),
                #                    q=q_to_include_mean_comp, per_dim=True, axis=-1)
                #mrb = utsne.robustMeanPos(rbp.get_data(reject_by_annotation='omit'),
                #                    q=q_to_include_mean_comp, per_dim=True,axis=-1)
                #means_perchan_flt = mflt
                #means_perchan_bp = mrb
            #means_perband_flt[bandname] = means_perchan_flt
            #means_perband_bp[bandname] = means_perchan_bp

        raw_perband_bp_pri  [rawind] = raw_perband_bp
        raw_perband_flt_pri [rawind] = raw_perband_flt

        chnames_perband_flt_pri [rawind] = chnames_perband_flt
        chnames_perband_bp_pri  [rawind] = chnames_perband_bp

        #means_perband_flt_pri += [means_perband_flt]
        #means_perband_bp_pri  += [means_perband_bp]


    return raw_perband_flt_pri, raw_perband_bp_pri, \
        chnames_perband_flt_pri, chnames_perband_bp_pri

def gatherMultiBandStats(rawnames,raw_perband_pri, times_pri, chnames_perband_pri,
                         side_switched_pri, sfreq,
                         baseline_int, scale_data_combine_type, artif_handling,
                         require_intervals_present, bindict_per_rawn=None):
    import utils_preproc as upre

    #means_perband_pri    = [ dict() ] * len(rawnames)
    #stds_perband_pri     = [ dict() ] * len(rawnames)

    means_per_indset_per_band = {}
    stds_per_indset_per_band = {}
    stats_per_indset_per_band = {}

    for bandname in raw_perband_pri[0].keys():
        #raws_pri_perband_[bandname] = [0]*len(rawnames)
        dats_band_T_pri = [0]*len(rawnames)
        for rawind in range(len(rawnames)):
            dats_band_T_pri[rawind] = raw_perband_pri[rawind][bandname].get_data().T

        chns_pri = [ chnames_perband_pri[rawind][bandname] for rawind in range(len(rawnames) ) ]
        indsets, means, stds, stats_per_indset = \
            upre.gatherFeatStats(rawnames, dats_band_T_pri,
                chns_pri, None, sfreq, times_pri,
                baseline_int, side_rev_pri = side_switched_pri,
                combine_within = scale_data_combine_type, minlen_bins = 5*sfreq,
                artif_handling=artif_handling,
                require_intervals_present=require_intervals_present,
                                 bindict_per_rawn= bindict_per_rawn)

        stats_per_indset_per_band[bandname] = stats_per_indset
        means_per_indset_per_band[bandname] = means
        stds_per_indset_per_band [bandname] = stds

        #means_curband_pri = upre.valsPerIndset2PerInd(indsets,means)
        #stds_curband_pri  = upre.valsPerIndset2PerInd(indsets,stds)

        #for rawind in range(len(rawnames)):
        #    means_perband_pri[rawind][bandname] = means_curband_pri[rawind]
        #    stds_perband_pri[rawind][bandname] = stds_curband_pri[rawind]




    #means_perband_flt_pri    = [ dict() ] * len(rawnames)

    #for bandname in raw_perband_flt_pri[0].keys():
    #    #raws_flt_pri_perband_[bandname] = [0]*len(rawnames)
    #    dats_band_T_pri = [0]*len(rawnames)
    #    for rawind in range(len(rawnames)):
    #        dats_band_T_pri[rawind] = raw_perband_flt_pri[rawind][bandname].get_data().T

    #    chns_pri = [ chnames_perband_flt_pri[rawind][bandname] for rawind in range(len(rawnames) ) ]
    #    indsets, means, stds = \
    #        upre.gatherFeatStats(rawnames, dats_band_T_pri,
    #                             chns_pri, None, sfreq, times_pri,
    #                baseline_int, side_rev_pri = side_switched_pri,
    #                combine_within = scale_data_combine_type, minlen_bins = 5*sfreq,
    #                        artif_handling=artif_handling)

    #    means_flt_curband_pri = upre.valsPerIndset2PerInd(indsets,means)

    #    for rawind in range(len(rawnames)):
    #        means_perband_flt_pri[rawind][bandname] = means_flt_curband_pri[rawind]

    #means_perband_bp_pri     = [ dict() ] * len(rawnames)
    #for bandname in raw_perband_bp_pri[rawind].keys():
    #    dats_band_T_pri = [0]*len(rawnames)
    #    for rawind in range(len(rawnames)):
    #        dats_band_T_pri[rawind] = raw_perband_bp_pri[rawind][bandname].get_data().T

    #    indsets, means, stds = \
    #        upre.gatherFeatStats(rawnames, dats_band_T_pri,
    #                             chnames_perband_bp[bandname], None, sfreq, times_pri,
    #                baseline_int, side_rev_pri = side_switched_pri,
    #                combine_within = scale_data_combine_type, minlen_bins = 5*sfreq,
    #                        artif_handling=artif_handling)

    #    means_bp_curband_pri = upre.valsPerIndset2PerInd(indsets,means)

    #    for rawind in range(len(rawnames)):
    #        means_perband_bp_pri[rawi][bandname] = means_bp_curband_pri[rawind]

    #return means_perband_pri, stds_perband_pri
    return indsets, means_per_indset_per_band, stds_per_indset_per_band  , stats_per_indset_per_band

