skip_existing_EMG = False
rectconv_filter_again = False
#14
#S037_PD
from os.path import join as pjoin
import utils_preprocess as upre
# TODO: put right data directories
data_dir_input = ''
data_dir_output = ''
rawname = 'S037_PD'
for rawname_ in rawnames:

#for subjind in subjinds:
#    sis = '{:02d}'.format(subjind)
#    for medstate in medstates:
#        for task in tasks:

    #fnames_noext = ['S{}_off_{}'.format(sis,task), 'S{}_on_{}'.format(sis,task)]
    #fnames_noext = ['S01_off_hold', 'S01_on_hold']
    #fnames_noext = ['S01_off_move', 'S01_on_move']
    #fnames_noext = ['S02_off_move', 'S02_on_move']
    #fname_noext = 'S{}_{}_{}'.format(sis,medstate,task)

    fname_noext = rawname_
    fname = fname_noext + '.mat'
    print('Reading {}'.format(fname) )

    addStr = ''
    print('--- Starting reading big 2kHz file!')
    fname_full = os.path.join(data_dir_input,fname)

    if not os.path.exists(fname_full):
        print('Warning: path does not exist!, skip! {}'.format(fname_full))
        continue

    f = upre.read_raw_fieldtrip(fname_full, None)
    rectconvraw = upre.extractEMGData(f,fname_noext, skip_if_exist = skip_existing_EMG,
                        save_dir = data_dir_output)  #saves emg_rectconv

#     mod_info, infos = upre.readInfo(fname_noext, f)
#     f.info = mod_info

#     raw_lfp = upre.saveLFP(fname_noext, skip_if_exist =
#                            skip_existing_LFP,sfreq=freqResample, raw_FT=f,n_jobs=n_jobs)
#     raw_lfp_highres = upre.saveLFP(fname_noext, skip_if_exist =
#                                    skip_existing_LFP,sfreq=freqResample_high,
#                                    raw_FT=f,n_jobs=n_jobs )

    rectconvraw.apply_function( lambda x: x / np.quantile(x,0.75) )
    hilbraw = rectconvraw.copy()
    if rectconv_filter_again:
        hilbraw.filter(l_freq=2,h_freq=10)
    hilbraw.apply_hilbert()

    # smoothness of hilb_freq depends heavilly on the band we use for filtering hilbraw
    hilb_amp = hilbraw.copy()
    hilb_amp.apply_function(np.abs)

    rectconv_env_fname_full = os.path.join(data_dir_output, '{}_emg_rectconv_envelope.fif'.format(rawname_) )
    if not (skip_existing_EMG and os.path.exists(rectconv_env_fname_full) ):
        print('EMG hilbert amp raw saved to ',rectconv_env_fname_full)
        hilb_amp.save(rectconv_env_fname_full, overwrite=1)






