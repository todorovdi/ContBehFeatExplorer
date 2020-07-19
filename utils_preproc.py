import mne
import os
import numpy as np
import gc

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

def saveLFP_nonresampled(rawname_naked, f_highpass = 2, skip_if_exist = 1,
                         n_free_cores = 2, ret_if_exist = 0 ):
    if os.environ.get('DATA_DUSS') is not None:
        data_dir = os.path.expandvars('$DATA_DUSS')
    else:
        data_dir = '/home/demitau/data'

    lfp_fname_full = os.path.join(data_dir, '{}_LFP_1kHz.fif'.format(rawname_naked) )
    if os.path.exists(lfp_fname_full):
        #subraw = mne.io.read_raw_fif(lfp_fname_full, None)
        #return subraw
        print('{} already exists!'.format(lfp_fname_full) )
        if ret_if_exist:
            return mne.io.read_raw_fif(lfp_fname_full, None)
        else:
            return None

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

    sfreq_to_use = 1024

    import multiprocessing as mpr
    num_cores = mpr.cpu_count() - 1
    subraw.resample(sfreq_to_use, n_jobs= max(1, num_cores-n_free_cores) )

    subraw.filter(l_freq=1, h_freq=None)

    freqsToKill = np.arange(50, sfreq_to_use//2, 50)  # harmonics of 50
    subraw.notch_filter(freqsToKill)

    subraw.save(lfp_fname_full, overwrite=1)
    return subraw

def saveRectConv(rawname_naked, raw=None, rawname = None, maxtremfreq=9, skip_if_exist = 0):
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
    chdata = emgonly.get_data()

    y = {}
    for chname in emgonly.ch_names:
        y[chname] = 'eeg'
    emgonly.set_channel_types(y)

    emgonly.filter(l_freq=10, h_freq=None)

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
    if os.environ.get('DATA_DUSS') is not None:
        data_dir = os.path.expandvars('$DATA_DUSS')
    else:
        data_dir = '/home/demitau/data'

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
    fields += ['coil_trans']
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

    emgonly.filter(l_freq=10, h_freq=None, picks='all')

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


    #rectconvraw_perside[side] = tmp

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
