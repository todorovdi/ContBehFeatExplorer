####!/usr/bin/python3  #no, we need a pyenv realted python
# resample data uing MNE
import sys, os
sys.path.append( os.path.expandvars('$OSCBAGDIS_DATAPROC_CODE') )

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
import getopt
from globvars import dir_fig, data_dir
mpl.use('Agg')

input_subdir = ''
# input_subdir = '/new'
subdir_fig = pjoin(dir_fig,'preproc')
if not os.path.exists(subdir_fig):
    os.makedirs(subdir_fig)

data_dir_input = pjoin(data_dir, input_subdir)
data_dir_output = data_dir

fine_cal_file  = pjoin(data_dir_input, 'sss_cal.dat')
crosstalk_file = pjoin(data_dir_input, 'ct_sparse.fif')

# resave read_type=[],  resample=1  notch=1
# resave read_type =resample,notch highpass=1.5

#read_type = ['resample', 'notch']
read_type = []
to_perform = ['resample', 'notch', 'highpass']


overwrite_res = 1

n_free_cores = gp.n_free_cores
n_jobs = max(1,mpr.cpu_count() - n_free_cores)


allow_CUDA_MNE = mne.utils.get_config('MNE_USE_CUDA')
#allow_CUDA = True
allow_CUDA = False

n_components_ICA = 0.95
tSSS_duration = 10  # default is 10s
frame_SSS = 'head'  # 'head'
do_ICA_only = 0
plot_ICA_damage = 1

# use_mean=1, so mark many things as MEG artifacts
MEG_thr_use_mean = 1

# =0 means mark few things as MEG artifacts
MEG_flt_thr_use_mean = 0
#MEG_flt_artif_thr_mult = 2.25 # before I used 2.5
artif_detection_params_def = {'ICA_thr_mult':2.3,
                              'thr_mult':2.5,
                              'threshold_muscle':5,
                              'flt_thr_mult':2.25}

#lowest_freq_to_keep = 1.5
lowest_freq_to_keep = 1
highest_freq_to_keep = 90
do_plot_helmet = 0

plot_ICA_prop = 1
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
recalc_artif = True
force_overwrite_hires = False
force_artif_ICA_recalc = True  # takes time

output_subdir = ""

exit_after = 'end'
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
         "output_subdir=", "exit_after=", "recalc_artif=" , "force_artif_ICA_recalc=" ])
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
    elif opt == "--recalc_artif":
        recalc_artif = int(arg)
    elif opt == '--to_perform':
        to_perform = arg.split(',')
    elif opt == '--exit_after':
        exit_after = arg
    elif opt == "--force_overwrite_hires":
        force_overwrite_hires = int(arg)
    elif opt == '--recalc_LFPEMG':
        recalc = bool(int(arg))
        skip_existing_LFP = not recalc
        skip_existing_EMG = not recalc
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
    elif opt == '--force_artif_ICA_recalc':
         force_artif_ICA_recalc = int(arg)
    elif opt ==  '--lowest_freq_to_keep':
        lowest_freq_to_keep = float(arg)
    else:
        s = 'Unknown option {},{}'.format(opt,arg)
        print(s)
        raise ValueError(s)


data_dir_output = pjoin(data_dir_output, output_subdir)
if not os.path.exists(data_dir_output):
    print('Creating {}'.format(data_dir_output) )
    os.makedirs(data_dir_output)

read_types_all = ['resample','notch','tSSS','SSP','highpass',
                  'hires-raw', 'fieldtrip_raw']

print('read_type',read_type)
print('to_perform',to_perform)

for rt in read_type:
    if len(rt):
        #print(rt, rt == '')
        assert rt in read_types_all, rt

if allow_CUDA and allow_CUDA_MNE:
    n_jobs = 'cuda'
    print('Using CUDA')



read_resampled = 'resample' in read_type
read_mat = (len(read_type) == 0) or \
    ( (len(read_type) == 1) and (read_type[0] == 'fieldtrip_raw') )
print('read_mat=',read_mat)

do_removeMEG = 0
do_resample = 'resample' in to_perform
do_notch =  'notch' in to_perform
do_tSSS = 'tSSS' in to_perform
do_SSP = 'SSP' in to_perform
do_ICA   = 'ICA' in to_perform
do_highpass = 'highpass' in to_perform  #highpass taking care of artifacts if annotation were saved before


# before it was to make it faster but now I don't care
#if do_ICA_only:
#    assert read_resampled #and to_perform

assert not (do_resample and read_resampled)
assert (not do_tSSS) or (not do_SSP)   # not both together!

tSSS_anns_type = 'MEG_flt'

do_highpass_after_SSS = 1

################ Start doing things

for rawname_ in rawnames:
    sis,medstate,task  = utils.getParamsFromRawname(rawname_)
    #if do_resample:
    #    raise ValueError('dr')

    fname_noext = rawname_
    read_hires = (('hires-raw' in read_type) or read_mat)
    if read_hires:
        if 'hires-raw' in read_type:
            fname_unproc = fname_noext + '_hires-raw.fif'
        elif read_mat:
            fname_unproc = fname_noext + '.mat'
        print('Reading {}'.format(fname_unproc) )


    if not read_hires:
        # +addStr_input_fn
        rtstr=''
        for s in read_type:
            rtstr += '_' + s
        effname = fname_noext + rtstr
        addStr_input_fn = rtstr
    else:
        addStr_input_fn = ''

    if do_removeMEG:
        addStr_input_fn += '_noMEG'
    if do_resample:
        addStr_input_fn += '_resample'
        if freqResample != 256:
            addStr_input_fn += '_{:d}'.format(freqResample)
    if do_notch:
        addStr_input_fn += '_notch'
    if do_tSSS:
        #addStr_input_fn += '_SSS'
        #addStr_input_fn += '_SSS_notch_resample'
        addStr_input_fn += '_SSS_notch'
        if do_highpass_after_SSS:
            addStr_input_fn += '_highpass'
        #addStr_input_fn += '_resample'
    if do_SSP:
        addStr_input_fn += '_SSP'
    if do_highpass:
        addStr_input_fn += '_highpass'
    #if do_ICA:  # do_ICA DOES NOT produce a new raw file, so no need to change the name
    #    addStr_input_fn += '_ICA'

    print('addStr_input_fn = ',addStr_input_fn)

    if not read_hires:
        fname_full = pjoin(data_dir_output,effname + '_raw.fif')
    else:
        print('--- Starting reading big 2kHz file!')
        fname_full = pjoin(data_dir_input,fname_unproc)

    if not os.path.exists(fname_full):
        print('Warning: path does not exist!, skip! {}'.format(fname_full))
        continue
    #else:
    #    print('Start processing {}'.format(fname_full) )
    #continue

    # resulting filename
    resfn = pjoin(data_dir_output,fname_noext + addStr_input_fn + '_raw.fif') # always to out dir
    if read_resampled:
        raw = mne.io.read_raw_fif(fname_full, None)
        #resfn = pjoin(data_dir_output,fname_noext + addStr_input_fn + '_maxwell_raw.fif')
        raw.load_data()
    else:
        if os.path.exists(resfn) and not overwrite_res:
            continue
        #raw = mne.io.read_raw_fieldtrip(fname_full, None)

        if not read_mat:
            raw = mne.io.read_raw_fif(fname_full, preload=1) #, on_split_missing=False)
        else:
            raw = upre.read_raw_fieldtrip(fname_full, None)
            mod_info, infos = upre.readInfo(fname_noext, raw)
            raw.info = mod_info

        freqResampleLFP = -1
        if do_resample:
            freqResampleLFP = freqResample
            raw_lfp = upre.saveLFP(fname_noext, skip_if_exist =
                                skip_existing_LFP,
                                sfreq=freqResampleLFP, raw_FT=raw,n_jobs=n_jobs,
                                notch=1)
        else:
            freqResample_high = -1
        raw_lfp_highres = upre.saveLFP(fname_noext, skip_if_exist =
                                       skip_existing_LFP,sfreq=freqResample_high,
                                       raw_FT=raw,n_jobs=n_jobs,
                                       notch=1)

    upre.extractEMGData(raw,fname_noext, skip_if_exist = skip_existing_EMG)  #saves emg_rectconv
    #continue

    assert np.std( np.diff(raw.times) ) < 1e-10  #we really want times to be equally spaced

    #reshuffle channels types (by default LFP and EMG types are determined wronng)
    # set types for some misc channels
    chnt = {}
    for i,chn in enumerate(raw.ch_names):
        #chn = raw.ch_names[chi]
        show = 0
        if chn.find('_old') >= 0:
            chnt[chn] = 'emg';  show = 1
        elif chn.find('_kil') >= 0:
            chnt[chn] = 'emg';  show = 1
        elif chn.find('LFP') >= 0:
            chnt[chn] = 'bio';  show = 1
            #raw.set_channel_types({chn:'bio'}); show = 1  # or stim, ecog, eeg

        if show:
            print(i, chn )

    raw.set_channel_types(chnt)


    bt = mne.io.pick.channel_indices_by_type(raw.info)
    miscchans = bt['misc']
    gradchans = bt['grad']
    magchans = bt['mag']
    eogchans = bt['eog']
    emgchans = bt['emg']
    biochans = bt['bio']
    #mne.pick_channels(raw,miscchans)

    print('miscchans', len(miscchans))
    print('gradchans', len(gradchans) )
    print('magchans', len(magchans))
    print('eogchans', len(eogchans))
    print('emgchans', len(emgchans))
    print('biochans', len(biochans))
    print( len(miscchans) + len(gradchans) + len(magchans) + len(eogchans) + len(emgchans) +
            len(biochans), len(raw.ch_names) )


    if 'hires-raw' not in read_type:
        badchlist = upre.loadBadChannelList(fname_noext, raw.ch_names)
        raw.info['bads'] = badchlist
    print('Bad channels (provided by Jan) are ',raw.info['bads'])

    fn_full = pjoin(gv.data_dir, fname_noext + '_hires-raw.fif' )
    if ('hires-raw' not in read_type) and \
        ( (not os.path.exists(fn_full) ) or (force_overwrite_hires) ):
        raw.save(fn_full, overwrite=True)



    #anns_MEG_artif, cvl_per_side = utils.findRawArtifacts(raw , thr_mult = MEG_artif_thr_mult,
    #    thr_use_mean = MEG_thr_use_mean)
    #if len(anns_MEG_artif) > 0:
    #    print('Artif found in UNfilt {}, maxlen {:.3f} totlen {:.3f}'.
    #            format(anns_MEG_artif, np.max(anns_MEG_artif.duration),
    #                    np.sum(anns_MEG_artif.duration) ) )
    #else:
    #    print('Artif found in UNfilt {} is NONE'.  format(anns_MEG_artif) )


    #anns_MEG_artif.save(pjoin(data_dir_input, '{}_ann_MEGartif.txt'.format(fname_noext) ) )

    #By default, MNE does not load data into main memory to conserve resources. adding, dropping,
    # or reordering channels requires raw data to be loaded.
    # Use preload=True (or string) in the constructor or raw.load_data().

    meg_chnames = [s for s in raw.info['ch_names'] if 0 <= s.find('MEG') ]
    #print(meg_chnames)
    if do_removeMEG:
        raw.drop_channels(meg_chnames)


    # if we recalc artif than we would like to recalc them after notching. But
    # if we do final notch, we want to remove artif before that
    if do_notch and recalc_artif:
        # there is no way to say about artifacts explicitly in notch
        print('Start notch filtering!')
        raw_notched = raw.copy()
        raw_notched.notch_filter(freqsToKill, n_jobs=n_jobs)
    else:
        raw_notched = None

    if recalc_artif or do_ICA:
        suff = ''
        if recalc_artif:
            suff += 'artif,'
        elif do_ICA:
            suff += 'ICAcomponents,'
        suff = suff[:-1]
        pdf = PdfPages(pjoin(subdir_fig,
            f'{fname_noext}_{suff}.pdf' ) )
    else:
        # otherwise we will overwrite pdf crated during artifact plotting
        pdf = None


    # this has to be done before any sort of filtering
    if recalc_artif:
        artif_detection_params = artif_detection_params_def
        # custom thresholds
        fnf_ct = pjoin(gv.data_dir,'artif_detction_params.json')
        if os.path.exists(fnf_ct):
            with open( fnf_ct ,'r') as f:
                artif_detection_params_allds = json.load(f)
            if fname_noext in artif_detection_params_allds:
                artif_detection_params = artif_detection_params_allds[fname_noext]


        r = upre.recalcMEGArtifacts(fname_noext,
            raw, use_mean=MEG_thr_use_mean,
            flt_use_mean=MEG_flt_thr_use_mean,
            lowest_freq_to_keep=lowest_freq_to_keep,
            n_jobs=n_jobs, raw_flt = None, raw_notched = raw_notched,
            savedir=data_dir_input, notch_freqsToKill=freqsToKill,
            force_ICA_recalc=force_artif_ICA_recalc, pdf=pdf, **artif_detection_params)
        anns_MEG_artif, anns_MEG_artif_flt, anns_icaMEG_artif, anns_MEG_artif_muscle  = r

    if exit_after == 'MEG_artif_calc':
        if pdf is not None:
            pdf.close()
        sys.exit(0)


    # this is purpose so
    if raw_notched is not None:
        del raw
        raw = raw_notched
    elif do_notch:
        # this is actually not necessary because MNE's notch filter cannot deal
        # with artifacts anyway. But I don't want to remove it so let it stay
        # here
        from utils_tSNE import concatAnns
        anns_artif, anns_artif_pri, times2, dataset_bounds =\
            concatAnns([fname_noext], [raw.times],
                    suffixes = ['_ann_MEGartif', '_ann_MEGartif_ICA',
                            '_ann_MEGartif_muscle' ], allow_missing=True )
        raw.set_annotations(anns_artif )

        raw_notched = raw.copy()
        raw_notched.notch_filter(freqsToKill, n_jobs=n_jobs)
        del raw
        raw = raw_notched


    if do_highpass:
        import utils_tSNE as utsne
        # loads artif from not filtered raw, we don't want to work with LFP
        # here anymore, so only MEG
        anns_artif, anns_artif_pri, times2, dataset_bounds =\
            utsne.concatAnns([fname_noext], [raw.times],
                             suffixes = ['_ann_MEGartif', '_ann_MEGartif_ICA',
                            '_ann_MEGartif_muscle' ], allow_missing=True )
        raw.set_annotations(anns_artif )
        raw.filter(l_freq=lowest_freq_to_keep,
                        h_freq=highest_freq_to_keep, n_jobs=n_jobs,
                   skip_by_annotation='BAD_', pad='symmetric')
        # reset anns
        raw.set_annotations(mne.Annotations([],[],[]) )

    if do_resample and not read_resampled:
        print('Resampling starts')
        if raw.info['sfreq'] > freqResample:
            # warning: this can propagate artifacts
            raw.resample(freqResample, n_jobs=n_jobs)
            print(f"Resampled to {raw.info['sfreq']}")



    if not do_removeMEG and not do_ICA_only:
        if ('hires-raw' in read_type) or (fname_noext.startswith('S9') ):
            mod_info = raw.info
        else:
            mod_info, infos = upre.readInfo(fname_noext, raw)
        radius, origin, _ = mne.bem.fit_sphere_to_headshape(mod_info, dig_kinds=('cardinal','hpi'))

        if do_plot_helmet:
            import mayavi
            sphere = mne.make_sphere_model(info=mod_info, r0=origin, head_radius=radius)
            src = mne.setup_volume_source_space(sphere=sphere, pos=10.)
            mne.viz.plot_alignment(
                mod_info, eeg='projected', bem=sphere, src=src, dig=True,
                surfaces=['brain', 'outer_skin'], coord_frame='meg', show_axes=True)
            mayavi.mlab.savefig(pjoin(subdir_fig,'{}_sensor_loc_vs_head.iv'.format(fname_noext) ) )
            mayavi.mlab.savefig(pjoin(subdir_fig,'{}_sensor_loc_vs_head.png'.format(fname_noext)) )
            mayavi.mlab.close()

        raw.info = mod_info

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
            fname_bads_full = pjoin( data_dir_input, fname_bads)
            fname_bads_mat_full = pjoin( data_dir_input, fname_bads_mat)
            ex  = os.path.exists(fname_bads_full)
            if badchans_SSS == 'load' and ex:
                print('Reading additinal bad channes from ',fname_bads_full)
                badchlist_upd = list(np.load(fname_bads_full)['arr_0'] )
                print('Setting additional bad channels ', set(badchlist_upd)-set(badchlist))
                raw.info['bads'] = badchlist_upd
            elif badchans_SSS == 'recalc':
                subraw = raw.copy()
                subraw.info['bads'] = []
                subraw.load_data()
                subraw.pick_types(meg=True)

                r = mne.preprocessing.find_bad_channels_maxwell(
                    subraw, cross_talk=crosstalk_file, calibration=fine_cal_file,
                    verbose=True, coord_frame=frame_SSS, origin = origin)

                meg_chnames = np.array(raw.ch_names)[ mne.pick_channels_regexp(raw.ch_names, 'MEG*') ]

                # These are found after removing Jan's components
                bad_from_maxwell = r[0]+ r[1]
                susp = set(bad_from_maxwell) - set(badchlist)
                missed_by_alg =  set(badchlist) - set(bad_from_maxwell)
                print('Bad channels: missed by Jan {} missed by Maxwell alg {}'.format(susp, missed_by_alg) )
                raw.info['bads'] = list( set(bad_from_maxwell + badchlist)  )

                np.savez(fname_bads_full, raw.info['bads'])
                import scipy
                scipy.io.savemat(fname_bads_mat_full, {'bads':raw.info['bads']} )
            elif badchans_SSS == 'no':
                raw.info['bads'] = badchlist

            print('BAD MEG channels ', raw.info['bads'] )


            print('Start tSSS for MEG!')
            # I will work only with highpassed data later
            # so makes sense to use those artifacts. I am not sure, maybe I'd
            # like tSSS to actually remove those artifacts
            if tSSS_anns_type == 'MEG_flt':
                raw.set_annotations( anns_MEG_artif_flt)
            elif tSSS_anns_type == 'MEG':
                raw.set_annotations( anns_MEG_artif)
            else:
                print('Not setting any artifact annotations for maxwell_filter')
            f_sss = mne.preprocessing.maxwell_filter(raw , cross_talk=crosstalk_file,
                    calibration=fine_cal_file, coord_frame=frame_SSS,
                                                    origin=origin,
                                                    st_duration=tSSS_duration,
                                                        skip_by_annotation='BAD_MEG')

            f_sss.set_annotations( anns_MEG_artif_flt)
            f_sss.notch_filter(freqsToKill, n_jobs=n_jobs)

            if do_highpass_after_SSS:
                f_sss.filter(l_freq=lowest_freq_to_keep,
                             h_freq=highest_freq_to_keep,
                             n_jobs=n_jobs, skip_by_annotation='BAD_MEG',
                             pad='symmetric')

            if do_resample:
                f_sss.resample(freqResample, n_jobs=n_jobs)
            f_sss.set_annotations(mne.Annotations([],[],[]) )
            raw = f_sss

        if do_SSP:
            print('Applying SSP')
            raw.apply_proj()

    if not do_ICA_only:
        print('Saving ',resfn)
        raw.save(resfn,overwrite=True)
    #else:
    #    raw = mne.io.read_raw_fif(resfn, None)

    # recalc ICA
    if do_ICA:  # we don't want to apply ICA right away, I have to look at it first
        #addstr_ica = ''
        #icafname = '{}_{}resampled-ica.fif.gz'.format(fname_noext,addstr_ica)
        addstr_ica = addStr_input_fn
        # no need for underscore here
        icafname = '{}{}'.format(fname_noext,addstr_ica)
        if do_tSSS:
            icafname += '_tSSS'
        if do_SSP:
            icafname += '_SSP'
        icafname += '-ica.fif.gz'
        icafname_full = pjoin(data_dir_output,icafname)

        filt_raw2 = raw.copy()
        # now do it again but filtering with skipping annotations
        # perhaps we want to include only the weakest artifacts so that the
        # rest enters ICA and has higher change to be related to blinks / ECG
        # (assuming that more agressive methods can just reject ECG / blinks
        # completely)
        if use_MEGartif_flt_for_ICA:
            filt_raw2.set_annotations(anns_MEG_artif_flt)
        else:
            filt_raw2.set_annotations(anns_MEG_artif)
        filt_raw2.load_data()
        if (not do_highpass) or ('tSSS' in to_perform and not do_highpass_after_SSS):
            filt_raw2.filter(l_freq=lowest_freq_to_keep,
                                h_freq=highest_freq_to_keep,
                                n_jobs=n_jobs, skip_by_annotation='BAD_MEG',
                                pad='symmetric')


        ttl = '{}_MEG_deviation_before_ICA.png'.format(fname_noext)
        plt.suptitle(ttl)
        #plt.savefig(pjoin(subdir_fig,(ttl)), dpi=200)
        pdf.savefig(plt.gcf() )
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

        ica.save(icafname_full, overwrite=True)

        comps = ica.get_components()
        ncomps = comps.shape[1]
        nonexcluded = list(  set( range( ncomps  ) ) - set(ica.exclude) )
        #print( sorted(nonexcluded) )

        s = ','.join(map(str,ica.exclude) )
        exclStr = 'excl_' + s
        print(exclStr)

        #compinds =  range( ica.get_components().shape[1] )  #all components
        maxi = min( np.max(ica.exclude) + 1 + 5, ncomps   )
        #set( range(maxi  ) ) -
        compinds = np.arange(maxi)    #all components
        #nr = len(compinds); nc = 2
        if plot_ICA_prop:
            print('Start plotting ICA properties for {maxi} compinds')
            filt_raw_tmp = filt_raw2.copy()
            filt_raw_tmp.set_annotations(mne.Annotations([],[],[]) )
            figs = mne.viz.plot_ica_properties(ica,filt_raw_tmp,compinds,
                show=0, psd_args={'fmax':75} )
            for fig in figs:
                fig.suptitle(f'Excl={exclStr}')
                pdf.savefig(fig)
                plt.close()

        plt.close('all')

        if plot_ICA_damage:
            xlim = [filt_raw2.times[0],filt_raw2.times[-1]]
            utils.plotICAdamage(filt_raw2, fname_noext, ica, list(set(ica.exclude)), xlim)

            xlim = [80,100]
            utils.plotICAdamage(filt_raw2, fname_noext, ica, list(set(ica.exclude)), xlim)

        # find eog, mark first
        # need to find ECG component by hand


    del raw
    if pdf is not None:
        pdf.close()

    import gc; gc.collect()

    print(f'Resave finished for {fname_noext}')
