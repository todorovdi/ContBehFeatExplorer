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

import utils_preproc as upre
import matplotlib as mpl
import globvars as gv
mpl.use('Agg')

subdir = ''
# subdir = '/new'
subdir_fig = '/preproc'

if os.environ.get('DATA_DUSS') is not None:
    data_dir_input = os.path.expandvars('$DATA_DUSS') + subdir
    data_dir_output = os.path.expandvars('$DATA_DUSS')
else:
    data_dir_input = '/home/demitau/data'   + subdir
    data_dir_output = '/home/demitau/data'

if os.environ.get('OUTPUT_OSCBAGDIS') is not None:
    dir_fig = os.path.expandvars('$OUTPUT_OSCBAGDIS')  + subdir_fig
else:
    dir_fig = '.' + subdir_fig


fine_cal_file  = os.path.join(data_dir_input, 'sss_cal.dat')
crosstalk_file = os.path.join(data_dir_input,  'ct_sparse.fif')


do_removeMEG = 0
do_resample = 0
overwrite_res = 1
read_resampled = 1

do_notch = 1
read_add_badchans = 1  # instead of recomputing
do_SSS = 1
do_SSP = 0
do_ICA   = 1
n_components_ICA = 0.95
tSSS_duration = 10  # default is 10s
frame_SSS = 'head'  # 'head'
do_ICA_only = 1

# mark few things as MEG artifacts
MEG_thr_use_mean = 0
MEG_artif_thr_mult = 2.25 # before I used 2.5

plot_ICA_prop = 1
ICA_exclude_EOG = 0
ICA_exclude_ECG = 1

# mark many things as MEG artifacts
#MEG_thr_use_mean = 1
#MEG_artif_thr_mult = 3
assert not (do_resample and read_resampled)
assert (not do_SSS) or (not do_SSP)   # not both together!

num_cores = mpr.cpu_count() - 1

freqResample = 256
freqsToKill = np.arange(50, freqResample//2, 50)
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

subjinds = [1,2,3]
tasks = ['hold' , 'move', 'rest']
medstates = ['on','off']

# there is a small MEG artifact in the middle here
#subjinds = [1]
#tasks = ['hold' ]
#medstates = ['off']

#subjinds = [1]
#tasks = ['move' ]
#medstates = ['on']

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
            if do_ICA:
                addStr += '_notch'
            if do_SSS:
                addStr += '_SSS'
            if do_SSP:
                addStr += '_SSP'
            #if do_ICA:
            #    addStr += '_ICA'

            if read_resampled:
                # +addStr
                fname_full = os.path.join(data_dir_output,fname_noext + '_resample' + '_raw.fif')
            else:
                fname_full = os.path.join(data_dir_input,fname)

            if not os.path.exists(fname_full):
                print('Warning: path does not exist!, skip! {}'.format(fname_full))
                continue
            #else:
            #    print('Start processing {}'.format(fname_full) )
            raw_lfp_highres = upre.saveLFP_nonresampled(fname_noext, skip_if_exist = 1)
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
                f = utils.read_raw_fieldtrip(fname_full, None)

            upre.extractEMGData(f,fname_noext, skip_if_exist = 1)  #saves emg_rectconv

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


            # get info about bad MEG channels (from separate file)
            with open('subj_info.json') as info_json:
                #raise TypeError

                #json.dumps({'value': numpy.int64(42)}, default=convert)
                gen_subj_info = json.load(info_json)

            #subj,medcond,task  = utils.getParamsFromRawname(rawname_)
            badchlist_ = gen_subj_info['S'+sis]['bad_channels'][medstate][task]
            badchlist= []
            for chname in badchlist_:
                if chname.find('EMG') >= 0 and ( (chname.find('_kil') < 0) and (chname.find('_old') < 0) ):
                    badchlist += [chname + '_old', chname + '_kil']
                else:
                    if chname not in f.ch_names:
                        print('Warning: channel {} not found in {}'.format(chname,fname_noext) )
                        continue
                    else:
                        badchlist += [chname]
            f.info['bads'] = badchlist
            print('bad channels (provided by Jan) are ',badchlist)


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

            if do_notch:
                print('Start notch filtering!')
                f.notch_filter(freqsToKill, n_jobs=num_cores)

            if not do_removeMEG and not do_ICA_only:
                import mayavi
                mod_info, infos = upre.readInfo(fname_noext, f)
                radius, origin, _ = mne.bem.fit_sphere_to_headshape(mod_info, dig_kinds=('cardinal','hpi'))
                sphere = mne.make_sphere_model(info=mod_info, r0=origin, head_radius=radius)
                src = mne.setup_volume_source_space(sphere=sphere, pos=10.)
                mne.viz.plot_alignment(
                    mod_info, eeg='projected', bem=sphere, src=src, dig=True,
                    surfaces=['brain', 'outer_skin'], coord_frame='meg', show_axes=True)
                mayavi.mlab.savefig(os.path.join(dir_fig,'{}_sensor_loc_vs_head.iv'.format(fname_noext) ) )
                mayavi.mlab.savefig(os.path.join(dir_fig,'{}_sensor_loc_vs_head.png'.format(fname_noext)) )
                mayavi.mlab.close()

                f.info = mod_info

                if do_SSS:
                    if frame_SSS == 'meg':
                        origin = None

                    fname_bads = '{}_MEGch_bads_upd.npz'.format(fname_noext)
                    fname_bads_full = os.path.join( data_dir_output, fname_bads)
                    ex  = os.path.exists(fname_bads_full)
                    if read_add_badchans and ex:
                        print('Reading additinal bad channes from ',fname_bads_full)
                        badchlist_upd = list(np.load(fname_bads_full)['arr_0'] )
                        print('Setting additional bad channels ', set(badchlist_upd)-set(badchlist))
                        f.info['bads'] = badchlist_upd
                    else:
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


                    print('Start tSSS for MEG!')
                    f_sss = mne.preprocessing.maxwell_filter(f , cross_talk=crosstalk_file,
                            calibration=fine_cal_file, coord_frame=frame_SSS,
                                                            origin=origin,
                                                            st_duration=tSSS_duration)
                    f = f_sss

                if do_SSP:
                    f.apply_proj()


            if do_ICA_only:
                #f.save(resfn,overwrite=True)
                f = mne.io.read_raw_fif(resfn, None)
            else:
                f.save(resfn,overwrite=True)

            if do_ICA:  # we don't want to apply ICA right away, I have to look at it first
                addstr_ica = ''
                icafname = '{}_{}resampled-ica.fif.gz'.format(fname_noext,addstr_ica)
                icafname_full = os.path.join(data_dir_output,icafname)


                filt_raw = f.copy()
                filt_raw.load_data().filter(l_freq=1., h_freq=None)

                #############################  Plot
                anns, cvl_per_side = utils.findMEGartifacts(filt_raw , thr_mult = MEG_artif_thr_mult,
                    thr_use_mean = MEG_thr_use_mean)
                filt_raw.set_annotations(anns)

                filt_raw.annotations.save(os.path.join(data_dir_output, '{}_ann_MEGartif.txt'.format(fname_noext) ) )
                print('Artif found ',anns)

                plt.savefig(os.path.join(dir_fig,('{}_MEG_deviation_before_ICA.png'.
                                                  format(fname_noext) )), dpi=200)
                plt.close()
                #################################
                ica = ICA(n_components = n_components_ICA, random_state=0).fit(filt_raw)
                ica.exclude = []
                if ICA_exclude_EOG:
                    #eog_inds, scores = ica.find_bads_eog(raw)
                    eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw)  # get single EOG trials
                    eog_inds, scores = ica.find_bads_eog(eog_epochs)
                    ica.exclude += eog_inds
                if ICA_exclude_ECG:
                    icacomp = ica.get_sources(filt_raw)
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
                    with PdfPages(os.path.join(dir_fig, '{}_ica_components_{}.pdf'.format(fname_noext,exclStr)) ) as pdf:
                        filt_raw_tmp = filt_raw.copy()
                        filt_raw_tmp.set_annotations(mne.Annotations([],[],[]) )
                        figs = mne.viz.plot_ica_properties(ica,filt_raw_tmp,compinds,
                                                        show=0, psd_args={'fmax':75} )
                        for fig in figs:
                            pdf.savefig(fig)
                            plt.close()
                plt.close('all')

                xlim = [filt_raw.times[0],filt_raw.times[-1]]
                utils.plotICAdamage(filt_raw, fname_noext, ica, list(set(ica.exclude)), xlim)

                xlim = [80,100]
                utils.plotICAdamage(filt_raw, fname_noext, ica, list(set(ica.exclude)), xlim)

                # find eog, mark first
                # need to find ECG component by hand


            del f

        import gc; gc.collect()
