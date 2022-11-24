#!/usr/bin/python3
# resample data uing MNE
import os
import mne
from mne.preprocessing import ICA
import multiprocessing as mpr
import numpy as np
import utils
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os.path import join as pjoin

import utils_preproc as upre
import matplotlib as mpl
import globvars as gv
from globvars import gp
import sys
import getopt
from globvars import dir_fig, data_dir
mpl.use('Agg')

input_subdir = ''
# input_subdir = '/new'
subdir_fig = '/preproc'

data_dir_input = pjoin(data_dir, input_subdir)
data_dir_output = data_dir

fine_cal_file  = os.path.join(data_dir_input, 'sss_cal.dat')
crosstalk_file = os.path.join(data_dir_input, 'ct_sparse.fif')

# resave read_type=[],  resample=1  notch=1
# resave read_type =resample,notch highpass=1.5

#read_type = ['resample', 'notch']
read_type = []
to_perform = ['resample', 'notch', 'highpass']


overwrite_res = 1

n_free_cores = gp.n_free_cores
n_jobs = max(1,mpr.cpu_count() - n_free_cores)

allow_CUDA_MNE = mne.utils.get_config('MNE_USE_CUDA')
allow_CUDA = True

#freqResample = 512
subjinds = [2,3,4,5,6,7]
#subjinds = [3,4,5,6,7]
subjinds = [1,8,9,10]
tasks = ['hold' , 'move', 'rest']

subjinds = [3,8,9,10]
subjinds = [1,2,3,4]

subjinds = [2,3,4]
subjinds = [4]

#subjinds = [2,3,4]
#subjinds = [5,6,7,8,9,10]
subjinds = [8]
subjinds = range(1,11)

subjinds = [4,5,7]
tasks = ['hold' , 'move', 'rest']
medstates = ['on','off']

subjinds = [8,9,10]
tasks = ['hold' , 'move', 'rest']
medstates = ['on','off']

subjinds = [1,2,3]
tasks = ['hold' , 'move', 'rest']
medstates = ['on','off']

subjinds = [1,2,4,5,7]
tasks = ['hold' , 'move', 'rest']
medstates = ['on','off']

subjinds = [1,2]
tasks = ['hold' ]
medstates = ['off']

#subjinds = [99]
#tasks = ['move']
#medstates = ['off']
#
#subjinds = [1]
#tasks = ['hold' , 'move', 'rest']
#medstates = ['off']

# there is a small MEG artifact in the middle here
#subjinds = [1]
#tasks = ['hold' ]
#medstates = ['off']

#subjinds = [1]
#tasks = ['move' ]
#medstates = ['on']

n_components_ICA = 0.95
tSSS_duration = 10  # default is 10s
frame_SSS = 'head'  # 'head'
do_ICA_only = 0

# use_mean=1, so mark many things as MEG artifacts
MEG_thr_use_mean = 1
MEG_artif_thr_mult = 2.5

# mark few things as MEG artifacts
MEG_flt_thr_use_mean = 0
MEG_flt_artif_thr_mult = 2.25 # before I used 2.5

lowest_freq_to_keep = 1.5
do_plot_helmet = 0

plot_ICA_prop = 0
ICA_exclude_EOG = 0
ICA_exclude_ECG = 1
#If no, use artif got from unfiltered (and unresampled). If yes, I may exclude
#too little. But on the other hand I apply it to the filtered thing...
use_MEGartif_flt_for_ICA = 1
do_recalc_MEGartif_flt = 1

freqResample = 256
freqResample_high = 1024
freqsToKill = np.arange(50, freqResample//2, 50)

#skip_existing_LFP = 1
skip_existing_LFP = 0
skip_existing_EMG = 0

badchans_SSS = 'recalc'  # or 'no', 'load'

output_subdir = ""
##############################

print('sys.argv is ',sys.argv)
effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

helpstr = 'Usage example\nresave.py --rawname <rawname_naked> '
opts, args = getopt.getopt(effargv,"hr:",
        ["rawname=", "to_perform=" , "read_type=", "tSSS_duration=",
         "frame_SSS=", "do_ICA_only=", "badchans_SSS=", "freq_resample=",
         "freq_resample_high=", "lowest_freq_to_keep=", "recalc_LFPEMG=",
         "output_subdir=" ])
print('opts is ',opts)
print('args is ',args)


for opt, arg in opts:
    print(opt,arg)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt in ('-r','--rawname'):
        if len(arg) < 5:
            print('Empty raw name provided, exiting')
            sys.exit(1)
        rawnames = arg.split(',')  #lfp of msrc
        for rn in rawnames:
            assert len(rn) > 3
        if len(rawnames) > 1:
            print('Using {} datasets at once'.format(len(rawnames) ) )
        #rawname_ = arg
    elif opt == '--output_subdir':
        output_subdir = arg
    elif opt == '--to_perform':
        to_perform = arg.split(',')
    elif opt == '--recalc_LFPEMG':
        skip_existing_LFP = not bool(arg)
        skip_existing_EMG = not bool(arg)
    elif opt == '--read_type':
        read_type = arg.split(',')
    elif opt == '--tSSS_duration':
        tSSS_duration = float(arg)
    elif opt == '--frame_SSS':
        frame_SSS = arg
    elif opt == '--do_ICA_only':  # it means not saving anything besides ICA
        do_ICA_only = int(arg)
    elif opt ==  '--badchans_SSS':
        badchans_SSS = arg
    elif opt ==  '--freq_resample':
        freqResample = int(arg)
    elif opt ==  '--freq_resample_high':
        freqResample_high = int(arg)
    elif opt ==  '--lowest_freq_to_keep':
        lowest_freq_to_keep = float(arg)
    else:
        s = 'Unknown option {},{}'.format(opt,arg)
        print(s)
        raise ValueError(s)


data_dir_output = os.path.join(data_dir_output, output_subdir)
if not os.path.exists(data_dir_output):
    print('Creating {}'.format(data_dir_output) )
    os.makedirs(data_dir_output)


print('read_type',read_type)
print('to_perform',to_perform)

if allow_CUDA and allow_CUDA_MNE:
    n_jobs = 'cuda'
    print('Using CUDA')

read_resampled = 'resample' in read_type

do_removeMEG = 0
do_resample = 'resample' in to_perform
do_notch =  'notch' in to_perform
do_tSSS = 'tSSS' in to_perform
do_SSP = 'SSP' in to_perform
do_ICA   = 'ICA' in to_perform
do_highpass = 'highpass' in to_perform  #highpass taking care of artifacts if annotation were saved before

if do_ICA_only:
    assert read_resampled #and to_perform

assert not (do_resample and read_resampled)
assert (not do_tSSS) or (not do_SSP)   # not both together!

tSSS_anns_type = 'MEG_flt'

do_highpass_after_SSS = 1

################ Start doing things

for rawname_ in rawnames:
    sis,medstate,task  = utils.getParamsFromRawname(rawname_)

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


    if read_resampled:
        # +addStr
        rtstr=''
        for s in read_type:
            rtstr += '_' + s
        effname = fname_noext + rtstr
        addStr = rtstr
    else:
        addStr = ''

    if do_removeMEG:
        addStr += '_noMEG'
    if do_resample:
        addStr += '_resample'
    if freqResample != 256:
        addStr += '_{:d}'.format(freqResample)
    if do_notch:
        addStr += '_notch'
    if do_tSSS:
        #addStr += '_SSS'
        #addStr += '_SSS_notch_resample'
        addStr += '_SSS_notch'
        if do_highpass_after_SSS:
            addStr += '_highpass'
        addStr += '_resample'
    if do_SSP:
        addStr += '_SSP'
    if do_highpass:
        addStr += '_highpass'
    #if do_ICA:  # do_ICA DOES NOT produce a new raw file, so no need to change the name
    #    addStr += '_ICA'

    print('addStr = ',addStr)

    if read_resampled:
        fname_full = os.path.join(data_dir_output,effname + '_raw.fif')
    else:
        print('--- Starting reading big 2kHz file!')
        fname_full = os.path.join(data_dir_input,fname)

    if not os.path.exists(fname_full):
        print('Warning: path does not exist!, skip! {}'.format(fname_full))
        continue
    #else:
    #    print('Start processing {}'.format(fname_full) )
    #continue

    resfn = os.path.join(data_dir_output,fname_noext + addStr + '_raw.fif') # always to out dir
    #fname = 'S01_off_hold.mat'
    if read_resampled:
        f = mne.io.read_raw_fif(fname_full, None)
        #resfn = os.path.join(data_dir_output,fname_noext + addStr + '_maxwell_raw.fif')
        f.load_data()
    else:
        #fname_full = os.path.join(data_dir_input,fname)
        if os.path.exists(resfn) and not overwrite_res:
            continue
        #f = mne.io.read_raw_fieldtrip(fname_full, None)
        f = upre.read_raw_fieldtrip(fname_full, None)

        mod_info, infos = upre.readInfo(fname_noext, f)
        f.info = mod_info

        raw_lfp = upre.saveLFP(fname_noext, skip_if_exist =
                               skip_existing_LFP,
                               sfreq=freqResample, raw_FT=f,n_jobs=n_jobs,
                               notch=1)
        raw_lfp_highres = upre.saveLFP(fname_noext, skip_if_exist =
                                       skip_existing_LFP,sfreq=freqResample_high,
                                       raw_FT=f,n_jobs=n_jobs,
                                       notch=1)

    upre.extractEMGData(f,fname_noext, skip_if_exist = skip_existing_EMG)  #saves emg_rectconv
    #continue

    assert np.std( np.diff(f.times) ) < 1e-10  #we really want times to be equally spaced

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


    badchlist = upre.loadBadChannelList(fname_noext, f.ch_names)
    f.info['bads'] = badchlist
    print('bad channels (provided by Jan) are ',badchlist)


    anns_MEG_artif, cvl_per_side = utils.findRawArtifacts(f , thr_mult = MEG_artif_thr_mult,
        thr_use_mean = MEG_thr_use_mean)
    anns_MEG_artif.save(os.path.join(data_dir_input, '{}_ann_MEGartif.txt'.format(fname_noext) ) )
    if len(anns_MEG_artif) > 0:
        print('Artif found in UNfilt {}, maxlen {:.3f} totlen {:.3f}'.
                format(anns_MEG_artif, np.max(anns_MEG_artif.duration),
                        np.sum(anns_MEG_artif.duration) ) )
    else:
        print('Artif found in UNfilt {} is NONE'.  format(anns_MEG_artif) )


    #By default, MNE does not load data into main memory to conserve resources. adding, dropping,
    # or reordering channels requires raw data to be loaded.
    # Use preload=True (or string) in the constructor or raw.load_data().

    meg_chnames = [s for s in f.info['ch_names'] if 0 <= s.find('MEG') ]
    #print(meg_chnames)
    if do_removeMEG:
        f.drop_channels(meg_chnames)

    if do_notch:
        # there is no way to say about artifacts explicitly in notch
        print('Start notch filtering!')
        f.notch_filter(freqsToKill, n_jobs=n_jobs)

    if do_highpass:
        import utils_tSNE as utsne
        anns_artif, anns_artif_pri, times2, dataset_bounds = utsne.concatArtif([fname_noext],[f.times],
                                                                                allow_missing=True)
        f.set_annotations(anns_artif)
        f.filter(l_freq=lowest_freq_to_keep,
                        h_freq=None, n_jobs=n_jobs, skip_by_annotation='BAD_', pad='symmetric')
        f.set_annotations(mne.Annotations([],[],[]) )

    if do_resample and not read_resampled:
        print('Resampling starts')
        if f.info['sfreq'] > freqResample:
            f.resample(freqResample, n_jobs=n_jobs)
            print(f.info['sfreq'])


    if do_recalc_MEGartif_flt:
        #assert f.info['sfreq'] < 1500
        filt_raw = f.copy()
        filt_raw.load_data()
        # if I don't filter raw artifcats, they will be found again
        # (probably) and then there is no sense and doing it again
        #filt_raw.set_annotations(anns_MEG_artif)
        filt_raw.filter(l_freq=lowest_freq_to_keep,
                        h_freq=None, n_jobs=n_jobs) #, skip_by_annotation='BAD_MEG')

        #############################  Plot
        anns_MEG_artif_flt, cvl_per_side = utils.findRawArtifacts(filt_raw , thr_mult = MEG_flt_artif_thr_mult,
            thr_use_mean = MEG_flt_thr_use_mean)
        anns_MEG_artif_flt.save(os.path.join(data_dir_input, '{}_ann_MEGartif_flt.txt'.format(fname_noext) ) )
        if len( anns_MEG_artif_flt ) > 0:
            print('Artif found in filtered {}, maxlen {:.3f} totlen {:.3f}'.
                format(anns_MEG_artif_flt, np.max(anns_MEG_artif_flt.duration),
                        np.sum(anns_MEG_artif_flt.duration) ) )
        else:
            print('Artif found in filtered {} is NONE'.  format(anns_MEG_artif_flt) )

    if not do_removeMEG and not do_ICA_only:
        mod_info, infos = upre.readInfo(fname_noext, f)
        radius, origin, _ = mne.bem.fit_sphere_to_headshape(mod_info, dig_kinds=('cardinal','hpi'))
        sphere = mne.make_sphere_model(info=mod_info, r0=origin, head_radius=radius)
        src = mne.setup_volume_source_space(sphere=sphere, pos=10.)


        if do_plot_helmet:
            import mayavi
            mne.viz.plot_alignment(
                mod_info, eeg='projected', bem=sphere, src=src, dig=True,
                surfaces=['brain', 'outer_skin'], coord_frame='meg', show_axes=True)
            mayavi.mlab.savefig(os.path.join(dir_fig,'{}_sensor_loc_vs_head.iv'.format(fname_noext) ) )
            mayavi.mlab.savefig(os.path.join(dir_fig,'{}_sensor_loc_vs_head.png'.format(fname_noext)) )
            mayavi.mlab.close()

        f.info = mod_info

        if do_tSSS:
            # apparently it is better to do tSSS before notching and perhaps
            # even before resampling
            assert ( len(read_type ) == 0 ) or (len(read_type) == 1 and len(read_type[0]) == 0 )
            assert ( len(to_perform) == 1 and to_perform[0] == 'tSSS'  ) or\
                (len(to_perform) == 2 and to_perform[0] == 'tSSS' and to_perform[1] == 'ICA')
            if frame_SSS == 'meg':
                origin = None

            fname_bads = '{}_MEGch_bads_upd.npz'.format(fname_noext)
            fname_bads_mat = '{}_MEGch_bads_upd.mat'.format(fname_noext)
            fname_bads_full = os.path.join( data_dir_input, fname_bads)
            fname_bads_mat_full = os.path.join( data_dir_input, fname_bads_mat)
            ex  = os.path.exists(fname_bads_full)
            if badchans_SSS == 'load' and ex:
                print('Reading additinal bad channes from ',fname_bads_full)
                badchlist_upd = list(np.load(fname_bads_full)['arr_0'] )
                print('Setting additional bad channels ', set(badchlist_upd)-set(badchlist))
                f.info['bads'] = badchlist_upd
            elif badchans_SSS == 'recalc':
                subraw = f.copy()
                subraw.info['bads'] = []
                subraw.load_data()
                subraw.pick_types(meg=True)

                r = mne.preprocessing.find_bad_channels_maxwell(
                    subraw, cross_talk=crosstalk_file, calibration=fine_cal_file,
                    verbose=True, coord_frame=frame_SSS, origin = origin)

                meg_chnames = np.array(f.ch_names)[ mne.pick_channels_regexp(f.ch_names, 'MEG*') ]

                # These are found after removing Jan's components
                bad_from_maxwell = r[0]+ r[1]
                susp = set(bad_from_maxwell) - set(badchlist)
                missed_by_alg =  set(badchlist) - set(bad_from_maxwell)
                print('Bad channels: missed by Jan {} missed by Maxwell alg {}'.format(susp, missed_by_alg) )
                f.info['bads'] = list( set(bad_from_maxwell + badchlist)  )

                np.savez(fname_bads_full, f.info['bads'])
                import scipy
                scipy.io.savemat(fname_bads_mat_full, {'bads':f.info['bads']} )
            elif badchans_SSS == 'no':
                f.info['bads'] = badchlist

            print('BAD MEG channels ', f.info['bads'] )


            print('Start tSSS for MEG!')
            # I will work only with highpassed data later
            # so makes sense to use those artifacts. I am not sure, maybe I'd
            # like tSSS to actually remove those artifacts
            if tSSS_anns_type == 'MEG_flt':
                f.set_annotations( anns_MEG_artif_flt)
            elif tSSS_anns_type == 'MEG':
                f.set_annotations( anns_MEG_artif)
            else:
                print('Not setting any artifact annotations for maxwell_filter')
            f_sss = mne.preprocessing.maxwell_filter(f , cross_talk=crosstalk_file,
                    calibration=fine_cal_file, coord_frame=frame_SSS,
                                                    origin=origin,
                                                    st_duration=tSSS_duration,
                                                        skip_by_annotation='BAD_MEG')

            f_sss.set_annotations( anns_MEG_artif_flt)
            f_sss.notch_filter(freqsToKill, n_jobs=n_jobs)

            if do_highpass_after_SSS:
                f_sss.filter(l_freq=lowest_freq_to_keep, h_freq=None,
                            n_jobs=n_jobs, skip_by_annotation='BAD_MEG',
                            pad='symmetric')

            f_sss.resample(freqResample, n_jobs=n_jobs)
            f_sss.set_annotations(mne.Annotations([],[],[]) )
            f = f_sss

        if do_SSP:
            print('Applying SSP')
            f.apply_proj()

    if not do_ICA_only:
        print('Saving ',resfn)
        f.save(resfn,overwrite=True)
    #else:
    #    f = mne.io.read_raw_fif(resfn, None)

    if do_ICA:  # we don't want to apply ICA right away, I have to look at it first
        #addstr_ica = ''
        #icafname = '{}_{}resampled-ica.fif.gz'.format(fname_noext,addstr_ica)
        addstr_ica = addStr
        icafname = '{}{}-ica.fif.gz'.format(fname_noext,addstr_ica)
        icafname_full = os.path.join(data_dir_output,icafname)

        filt_raw2 = f.copy()  # now do it again but filtering with skipping annotations
        if use_MEGartif_flt_for_ICA:
            filt_raw2.set_annotations(anns_MEG_artif_flt)
        else:
            filt_raw2.set_annotations(anns_MEG_artif)
        filt_raw2.load_data()
        if ('highpass' not in read_type) or ('tSSS' in to_perform and not do_highpass_after_SSS):
            filt_raw2.filter(l_freq=lowest_freq_to_keep,
                                h_freq=None, n_jobs=n_jobs, skip_by_annotation='BAD_MEG',
                                pad='symmetric')


        plt.savefig(os.path.join(dir_fig,('{}_MEG_deviation_before_ICA.png'.
                                            format(fname_noext) )), dpi=200)
        plt.close()
        #################################
        ica = ICA(n_components = n_components_ICA, random_state=0).fit(filt_raw2)
        ica.exclude = []
        if ICA_exclude_EOG:
            #eog_inds, scores = ica.find_bads_eog(raw)
            eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw2)  # get single EOG trials
            eog_inds, scores = ica.find_bads_eog(eog_epochs)
            ica.exclude += eog_inds
        # I don't want to save ECG here, better I select it by hand in jupyter
        if ICA_exclude_ECG:
            icacomp = ica.get_sources(filt_raw2)
            ecg_inds,ratios,ecg_intervals = upre.getECGindsICAcomp(icacomp)
            ica.exclude += ecg_inds

        ica.save(icafname_full)

        nonexcluded = list(  set( range( ica.get_components().shape[1] ) ) - set(ica.exclude) )
        #print( sorted(nonexcluded) )

        s = ','.join(map(str,ica.exclude) )
        exclStr = 'excl_' + s
        print(exclStr)

        compinds =  range( ica.get_components().shape[1] )  #all components
        #nr = len(compinds); nc = 2
        if plot_ICA_prop:
            print('Start plotting ICA properties')
            with PdfPages(os.path.join(dir_fig, '{}_ica_components_{}.pdf'.format(fname_noext,exclStr)) ) as pdf:
                filt_raw_tmp = filt_raw2.copy()
                filt_raw_tmp.set_annotations(mne.Annotations([],[],[]) )
                figs = mne.viz.plot_ica_properties(ica,filt_raw_tmp,compinds,
                                                show=0, psd_args={'fmax':75} )
                for fig in figs:
                    pdf.savefig(fig)
                    plt.close()
        plt.close('all')

        xlim = [filt_raw2.times[0],filt_raw2.times[-1]]
        utils.plotICAdamage(filt_raw2, fname_noext, ica, list(set(ica.exclude)), xlim)

        xlim = [80,100]
        utils.plotICAdamage(filt_raw2, fname_noext, ica, list(set(ica.exclude)), xlim)

        # find eog, mark first
        # need to find ECG component by hand


    del f

    import gc; gc.collect()
