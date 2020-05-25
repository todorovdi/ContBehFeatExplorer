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
    raw = mne.io.read_raw_fieldtrip(fname_full, None)
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

    freqsToKill = np.arange(50, sfreq_to_use//2, 50) # harmonics of 50
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
    from ...externals.pymatreader.pymatreader import read_mat

    ft_struct = read_mat(fname,
                         ignore_fields=['previous'],
                         variable_names=[data_name])

    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    from mne.utils import _validate_ft_struct, _create_info
    _validate_ft_struct(ft_struct)
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

    raw = RawArray(data, info)  # create an MNE RawArray
    return raw
