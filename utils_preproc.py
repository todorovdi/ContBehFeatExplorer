import mne
import os
import numpy as np

def saveRectConv(rawname_naked, raw=None, rawname = None, maxtremfreq=9, skip_if_exist = 0):
    if os.environ.get('DATA_DUSS') is not None:
        data_dir = os.path.expandvars('$DATA_DUSS')
    else:
        data_dir = '/home/demitau/data'

    rectconv_fname_full = os.path.join(data_dir, '{}_emg_rectconv.fif'.format(rawname_naked) )
    if os.path.exists(rectconv_fname_full) and skip_if_exist:
        return None

    if raw is None:
        if rawname is None:
            rawname = rawname_naked + '_resample_raw.fif'
        fname_full = os.path.join(data_dir, rawname)
        # read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
        raw = mne.io.read_raw_fif(fname_full, None)

    emgonly = raw.copy()
    emgonly.load_data()
    chis = mne.pick_channels_regexp(emgonly.ch_names, 'EMG.*old')

    restr_names = np.array(emgonly.ch_names)[chis]
    restr_names = restr_names.tolist()
    chdata, times = emgonly[emgonly.ch_names]

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
