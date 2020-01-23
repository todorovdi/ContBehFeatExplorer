# resample data uing MNE

import os
import mne

subdir = ''
#subdir = '/new'

if os.environ.get('DATA_DUSS') is not None:
    data_dir_input = os.path.expandvars('$DATA_DUSS')   + subdir
    data_dir_output = os.path.expandvars('$DATA_DUSS')
else:
    data_dir_input = '/home/demitau/data'   + subdir
    data_dir_output = '/home/demitau/data' 

fine_cal_file = os.path.join(data_dir_input, 'sss_cal.dat')
crosstalk_file = os.path.join(data_dir_input,  'ct_sparse.fif')


do_removeMEG = 0
do_resample = 1
overwrite_res = 1

do_SSS = 0
read_resampled = 0
    

freqResample = 256
#freqResample = 512
subjinds = [2,3,4,5,6,7]
#subjinds = [3,4,5,6,7]
subjinds = [1,8,9,10]
tasks = ['hold' , 'move', 'rest']

subjinds = [3,8,9,10]
subjinds = [2,3,4]
subjinds = [6]
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

            if read_resampled:
                fname_full = os.path.join(data_dir_output,fname_noext + addStr + '_raw.fif')
            else:
                fname_full = os.path.join(data_dir_input,fname)
            if not os.path.exists(fname_full):
                print('Warning: path does not exist!, skip! {}'.format(fname_full))
                continue
            #else:
            #    print('Start processing {}'.format(fname_full) )

            #fname = 'S01_off_hold.mat'
            if read_resampled:
                f = mne.io.read_raw_fif(fname_full, None)
                resfn = os.path.join(data_dir_output,fname_noext + addStr + '_maxwell_raw.fif')
            else:
                #fname_full = os.path.join(data_dir_input,fname)

                resfn = os.path.join(data_dir_output,fname_noext + addStr + '_raw.fif')
                if os.path.exists(resfn) and not overwrite_res:
                    continue
                f = mne.io.read_raw_fieldtrip(fname_full, None)
                
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
                    f.resample(freqResample)
                    print(f.info['sfreq'])

            if not do_removeMEG and do_SSS:

                print('Start SSS for MEG!')
                frame = 'meg' # 'meg' or 'head'
                f_sss = mne.preprocessing.maxwell_filter(f , cross_talk=crosstalk_file, calibration=fine_cal_file, coord_frame=frame)
                f = f_sss

            f.save(resfn)

            del f

        import gc; gc.collect()
