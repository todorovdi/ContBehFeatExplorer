# resample data uing MNE

import os
import mne
from mne.preprocessing import ICA
import multiprocessing as mpr
import numpy as np
import utils
import json

subdir = ''
#subdir = '/new'

if os.environ.get('DATA_DUSS') is not None:
    data_dir_input = os.path.expandvars('$DATA_DUSS')   + subdir
    data_dir_output = os.path.expandvars('$DATA_DUSS')
else:
    data_dir_input = '/home/demitau/data'   + subdir
    data_dir_output = '/home/demitau/data' 

fine_cal_file  = os.path.join(data_dir_input, 'sss_cal.dat')
crosstalk_file = os.path.join(data_dir_input,  'ct_sparse.fif')


do_removeMEG = 0
do_resample = 1
overwrite_res = 1

do_SSS = 1
read_resampled = 0

do_ICA   = 1
do_notch = 1
    
num_cores = mpr.cpu_count() - 1

freqResample = 256
freqsToKill = np.arange(50, freqResample//2, 50)
#freqResample = 512
subjinds = [2,3,4,5,6,7]
#subjinds = [3,4,5,6,7]
subjinds = [1,8,9,10]
tasks = ['hold' , 'move', 'rest']

subjinds = [3,8,9,10]
subjinds = [2,3,4]
subjinds = [1]
tasks = ['hold' , 'move', 'rest']
medstates = ['on','off']

for subjind in subjinds:
    sis = '{:02d}'.format(subjind)
    for medstate in medstates:
        for task in tasks:

        #fnames_noext = ['S{}_off_{}'.format(sis,task), 'S{}_on_{}'.format(sis,task)]
        #fnames_noext = ['S01_off_hold', 'S01_on_hold']
        #fnames_noext = ['S01_off_move', 'S01_on_move']
        #fnames_noext = ['S02_off_move', 'S02_on_move']
            fname_noext = 'S{}_{}_{}'.format(sis,medstate,task)
            fname = fname_noext + '.mat'
            print('Reading {}'.format(fname) )

            addStr = ''
            if do_removeMEG:
                addStr += '_noMEG'
            if do_resample: 
                addStr += '_resample'
            if freqResample > 256:
                addStr += '_{:d}'.format(freqResample)
            if do_SSS:
                addStr += '_SSS'
            if do_ICA:
                addStr += '_notch'
            if do_ICA:
                addStr += '_ICA'

            if read_resampled:
                fname_full = os.path.join(data_dir_output,fname_noext + addStr + '_raw.fif')
            else:
                fname_full = os.path.join(data_dir_input,fname)

            if not os.path.exists(fname_full):
                print('Warning: path does not exist!, skip! {}'.format(fname_full))
                continue
            #else:
            #    print('Start processing {}'.format(fname_full) )

            resfn = os.path.join(data_dir_output,fname_noext + addStr + '_raw.fif') # always to out dir
            #fname = 'S01_off_hold.mat'
            if read_resampled:
                f = mne.io.read_raw_fif(fname_full, None)
                #resfn = os.path.join(data_dir_output,fname_noext + addStr + '_maxwell_raw.fif')
            else:
                #fname_full = os.path.join(data_dir_input,fname)

                if os.path.exists(resfn) and not overwrite_res:
                    continue
                f = mne.io.read_raw_fieldtrip(fname_full, None)
            
            #reshuffle channels types (by default LFP and EMG types are determined wronng)
            # set types for some misc channels
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
                    
                    
            bt = mne.io.pick.channel_indices_by_type(f.info)
            miscchans = bt['misc']
            gradchans = bt['grad']
            magchans = bt['mag']
            eogchans = bt['eog']
            emgchans = bt['emg']
            biochans = bt['bio']
            #mne.pick_channels(f,miscchans)

            print('miscchans', len(miscchans))
            print('gradchans', len(gradchans) )
            print('magchans', len(magchans))
            print('eogchans', len(eogchans))
            print('emgchans', len(emgchans))
            print('biochans', len(biochans))
            print( len(miscchans) + len(gradchans) + len(magchans) + len(eogchans) + len(emgchans) +
                  len(biochans), len(f.ch_names) )


            # get info about bad MEG channels (from separate file)
            with open('subj_info.json') as info_json:
                    #raise TypeError

                #json.dumps({'value': numpy.int64(42)}, default=convert)
                gen_subj_info = json.load(info_json)
                
            #subj,medcond,task  = utils.getParamsFromRawname(rawname_)
            badchlist = gen_subj_info['S'+sis]['bad_channels'][medstate][task]
            f.info['bads'] = badchlist
            print('bad channels are ',badchlist)

                
            #By default, MNE does not load data into main memory to conserve resources. adding, dropping, 
            # or reordering channels requires raw data to be loaded. 
            # Use preload=True (or string) in the constructor or raw.load_data().

            meg_chnames = [s for s in f.info['ch_names'] if 0 <= s.find('MEG') ]
            #print(meg_chnames)
            if do_removeMEG:
                f.drop_channels(meg_chnames)

            if do_resample and not read_resampled:
                print('Resampling starts')
                if f.info['sfreq'] > freqResample:
                    f.resample(freqResample, n_jobs=num_cores)
                    print(f.info['sfreq'])

            if not do_removeMEG and do_SSS:
                print('Start SSS for MEG!')
                frame = 'meg' # 'meg' or 'head'
                f_sss = mne.preprocessing.maxwell_filter(f , cross_talk=crosstalk_file, 
                        calibration=fine_cal_file, coord_frame=frame)
                f = f_sss

            if do_notch:
                print('Start notch filtering!')
                raw_sss.notch_filter(freqsToKill, n_jobs=num_cores)

            if do_ICA:
                filt_raw = f.copy()
                filt_raw.load_data().filter(l_freq=1., h_freq=None)
                if loadICA and os.path.exists(icaname_full):
                    ica = mne.preprocessing.read_ica(icaname_full)
                else:
                    ica = ICA(n_components = 0.95, random_state=0).fit(filt_raw)

                     #eog_inds, scores = ica.find_bads_eog(raw)
                    eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw)  # get single EOG trials
                    eog_inds, scores = ica.find_bads_eog(eog_epochs)
                    ica.exclude = eog_inds

                    ica.save(icaname_full)

                # find eog, mark first
                # need to find ECG component by hand

            f.save(resfn)

            del f

        import gc; gc.collect()
