# resample data uing MNE

import os
import mne

subdir = ''
subdir = '/new'

if os.environ.get('DATA_DUSS') is not None:
    data_dir_input = os.path.expandvars('$DATA_DUSS')   + subdir
    data_dir_output = os.path.expandvars('$DATA_DUSS')
else:
    data_dir_input = '/home/demitau/data'   + subdir
    data_dir_output = '/home/demitau/data' 


do_removeMEG = 1
do_resample = 1
overwrite_res = 0
    

freqResample = 256
freqResample = 512
subjinds = [2,3,4,5,6,7]
#subjinds = [3,4,5,6,7]
subjinds = [1,8,9,10]
tasks = ['hold' , 'move', 'rest']

subjinds = [3,8,9,10]
subjinds = [1,2,3,4]
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
            fname_full = os.path.join(data_dir_input,fname)
            if not os.path.exists(fname_full):
                print('Warning: path does not exist!, skip! {}'.format(fname_full))
                continue
            #else:
            #    print('Start processing {}'.format(fname_full) )


            addStr = ''
            if do_removeMEG:
                addStr += '_noMEG'
            if do_resample: 
                addStr += '_resample'
            if freqResample > 256:
                addStr += '_{:d}'.format(freqResample)

            resfn = os.path.join(data_dir_output,fname_noext + addStr + '_raw.fif')
            if os.path.exists(resfn) and not overwrite_res:
                continue

            #fname = 'S01_off_hold.mat'
            fname_full = os.path.join(data_dir_input,fname)
            f = mne.io.read_raw_fieldtrip(fname_full, None)
                
            #By default, MNE does not load data into main memory to conserve resources. adding, dropping, 
            # or reordering channels requires raw data to be loaded. 
            # Use preload=True (or string) in the constructor or raw.load_data().

            meg_chnames = [s for s in f.info['ch_names'] if 0 <= s.find('MEG') ]
            #print(meg_chnames)
            if do_removeMEG:
                f.drop_channels(meg_chnames)

            if do_resample:
                print('Resampling starts')
                if f.info['sfreq'] > freqResample:
                    f.resample(freqResample)
                    print(f.info['sfreq'])

            f.save(resfn)

            del f

        import gc; gc.collect()
