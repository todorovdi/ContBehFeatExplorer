def test1():
    # test overall rescaling
    import numpy as np
    from utils_postprocess import printDict

    import globvars as gv
    import mne
    import matplotlib.pyplot as plt
    from os.path import join as pjoin


    defdgen = lambda sz :np.random.uniform(low=-1,high=1, size=sz)

    import utils_tSNE as utsne

    sfreq = 256
    sfreq_hires = 1024

    from IPython import get_ipython; ipython = get_ipython()
    ipython.magic('run -i ../test_data/gen_test_data1.py')

    special_chnis = {}
    for chn_descr,chn in special_chns.items():
        special_chnis[chn_descr] = sfo.index(chn)


    #rawnames = ['S01_off_move','S01_off_hold','S01_on_move']

    # prep_test_data

    ##############  put same data in all raws
    data_mult_per_rawi = [1] * len(rawnames)
    for rawi in range(len(rawnames)):
        if rawi == dati:
            continue
        # mulitply entire raws to check how cross-raw rescaling workgs
        mult = (rawi + 1) * 10
        data_mult_per_rawi[rawi] = mult
        dat_LFP_hires_pri[rawi] += dat_LFP_hires_pri[dati] * mult
        dat_pri[rawi]           += dat_pri[dati]           * mult

    dat_pri_gt           = dat_pri
    dat_LFP_hires_pri_gt = dat_LFP_hires_pri

    plt.plot(times,dat_pri[dati].T)
    plt.figure()
    plt.plot(times_hires,dat_LFP_hires_pri[dati].T, label=sfo_LFP)
    plt.legend()

    dat_EMG = defdgen( (2, nbins ) )

    # reversed if needed
    from utils import makeSimpleRaw
    from featlist import selFeatsRegexInds
    raws_permod_both_sides2 = {}
    for rawi,rawn in enumerate(rawnames):
        raws_permod_both_sides2[rawn] = {}
        chis_LFP = utsne.selFeatsRegexInds(sfo_pri[rawi], 'LFP.*' )
        dat_LFP = dat_pri[rawi][chis_LFP]
        raws_permod_both_sides2[rawn]['LFP'] = makeSimpleRaw(dat_LFP, np.array(sfo_pri[rawi])[chis_LFP], rescale=0 )

        chis_msrc = utsne.selFeatsRegexInds(sfo_pri[rawi], 'msrc.*' )
        dat_msrc = dat_pri[rawi][chis_msrc]
        raws_permod_both_sides2[rawn]['src'] = makeSimpleRaw(dat_msrc,
            np.array(sfo_pri[rawi])[chis_msrc] , rescale=0)

        raws_permod_both_sides2[rawn]['LFP_hires'] = makeSimpleRaw(dat_LFP_hires_pri[rawi],
            sfo_lfp_hires_pri[rawi], sfreq=sfreq_hires, rescale=0 )

        raws_permod_both_sides2[rawn]['EMG'] = makeSimpleRaw(dat_EMG,
            gv.EMG_per_hand[main_side], sfreq=sfreq, rescale=0)

    rawnames_str = ','.join(rawnames)


    ##########################################################################
    ##########################################################################

    import globvars as gv
    gv.DEBUG_MODE=1
    from IPython import get_ipython; ipython = get_ipython()
    #Once that's done, run a magic command like this:
    #%debug
    raws_permod_both_sides = raws_permod_both_sides2

    mstr = 'run -i ../run/run_prep_dat.py -r ' +   rawnames_str + \
            ' --param_file prep_dat_defparams.ini --calc_stats_multi_band 1' +\
            ' --save_dat 1 --save_stats 1 --use_preloaded_raws 1'
    ipython.magic(mstr)

    # %run -i ../run/run_prep_dat.py -r S99_off_move,S99_off_hold,S99_on_move \
    #         --param_file prep_dat_defparams.ini --calc_stats_multi_band 1 --save_dat 1 \
    #         --use_preloaded_raws 1 --save_stats 1

    gv.DEBUG_MODE = True
    from IPython import get_ipython; ipython = get_ipython()
    #%debug

    # import warnings
    # warnings.filterwarnings("error")
    # # with warnings.catch_warnings():
    # #     warnings.simplefilter('error')

    # try:

    # except ComplexWarning as e:
    #     pass

    rs = ('%run -i ../run/run_genfeats.py '
    ' --sources_type parcel_aal --bands crude'
    ' --src_grouping_fn 10'
    ' --src_grouping 0 '
    ' --raw {} '
    ' --feat_types con,H_act,H_mob,H_compl,rbcorr,bpcorr'
    ' --Kalman_smooth 0'
    ' --load_TFR 0'
    ' --load_CSD 0'
    ' --save_TFR 0'
    ' --save_CSD 0'
    ' --save_bpcorr 0'
    ' --save_rbcorr 0'
    ' --load_rbcorr 0'
    ' --use_existing_TFR 0'
    ' --use_preloaded_data 0'
    ' --allow_CUDA 0'
    ' --load_only 0'
    ' --show_plots 0'
    ' --plot_types ,'
    ' --prescale_data 1'
    ' --exit_after {}'
    ' --normalize_TFR separate'
    ' --scale_data_combine_type no'
    ' --baseline_int_type entire'
    ' --n_jobs 1'
    ' --save_feat 0'
    ' --feat_stats_artif_handling reject'
    ' --scale_data_combine_type no '
    ' --rbcorr_use_local_means 1'
    ' --output_subdir test'
    ' --stats_fn_prefix stats_{}_3_ '
    ' --param_file genfeats_defparams.ini ')

    exit_after = 'end'
    #exit_after = 'prescale_data'
    #exit_after = 'load'
    mstr2 = rs.format(rawnames_str,exit_after,rawnames_str[:3])
    ipython.magic(mstr2)



    ##########################################################################

    # Check the we have scaled correctly
    m0 = max( np.max( dat_pri_unscaled[0] - dat_pri_unscaled[1] ), np.max( dat_pri_unscaled[0] - dat_pri_unscaled[2] ) )
    m1 = max( np.max( dat_pri[0] - dat_pri[1] ), np.max( dat_pri[0] - dat_pri[2] ) )
    print(m0,m1)
    assert m1 < noise_size

    # Check the we have scaled correctly
    m0 = max( np.max( dat_lfp_hires_pri_unscaled[0] - dat_lfp_hires_pri_unscaled[1] ),
            np.max( dat_lfp_hires_pri_unscaled[0] - dat_lfp_hires_pri_unscaled[2] ) )
    m1 = max( np.max( dat_lfp_hires_pri[0] - dat_lfp_hires_pri[1] ),
            np.max( dat_lfp_hires_pri[0] - dat_lfp_hires_pri[2] ) )
    print(m0,m1)
    assert m1 < noise_size


def test2():
    # test feature independence for white noise dataset
    import numpy as np
    from utils_postprocess import printDict

    import globvars as gv
    import mne
    import matplotlib.pyplot as plt
    from os.path import join as pjoin


    defdgen = lambda sz :np.random.uniform(low=-1,high=1, size=sz)

    import utils_tSNE as utsne

    sfreq = 256
    sfreq_hires = 1024

    from IPython import get_ipython; ipython = get_ipython()
    ipython.magic('run -i ../test_data/gen_test_data2.py')

    special_chnis = {}
    for chn_descr,chn in special_chns.items():
        special_chnis[chn_descr] = sfo.index(chn)


    #rawnames = ['S01_off_move','S01_off_hold','S01_on_move']

    # prep_test_data

    ##############  put same data in all raws
    data_mult_per_rawi = [1] * len(rawnames)
    for rawi in range(len(rawnames)):
        if rawi == dati:
            continue
        # mulitply entire raws to check how cross-raw rescaling workgs
        mult = (rawi + 1) * 10
        data_mult_per_rawi[rawi] = mult
        dat_LFP_hires_pri[rawi] += dat_LFP_hires_pri[dati] * mult
        dat_pri[rawi]           += dat_pri[dati]           * mult

    dat_pri_gt           = dat_pri
    dat_LFP_hires_pri_gt = dat_LFP_hires_pri

    plt.plot(times,dat_pri[dati].T)
    plt.figure()
    plt.plot(times_hires,dat_LFP_hires_pri[dati].T, label=sfo_LFP)
    plt.legend()

    dat_EMG = defdgen( (2, nbins ) )

    # reversed if needed
    from utils import makeSimpleRaw
    from featlist import selFeatsRegexInds
    raws_permod_both_sides2 = {}
    for rawi,rawn in enumerate(rawnames):
        raws_permod_both_sides2[rawn] = {}
        chis_LFP = utsne.selFeatsRegexInds(sfo_pri[rawi], 'LFP.*' )
        dat_LFP = dat_pri[rawi][chis_LFP]
        raws_permod_both_sides2[rawn]['LFP'] = makeSimpleRaw(dat_LFP, np.array(sfo_pri[rawi])[chis_LFP], rescale=0 )

        chis_msrc = utsne.selFeatsRegexInds(sfo_pri[rawi], 'msrc.*' )
        dat_msrc = dat_pri[rawi][chis_msrc]
        raws_permod_both_sides2[rawn]['src'] = makeSimpleRaw(dat_msrc,
            np.array(sfo_pri[rawi])[chis_msrc] , rescale=0)

        raws_permod_both_sides2[rawn]['LFP_hires'] = makeSimpleRaw(dat_LFP_hires_pri[rawi],
            sfo_lfp_hires_pri[rawi], sfreq=sfreq_hires, rescale=0 )

        raws_permod_both_sides2[rawn]['EMG'] = makeSimpleRaw(dat_EMG,
            gv.EMG_per_hand[main_side], sfreq=sfreq, rescale=0)

    rawnames_str = ','.join(rawnames)


    ##########################################################################
    ##########################################################################

    import globvars as gv
    gv.DEBUG_MODE=1
    from IPython import get_ipython; ipython = get_ipython()
    #Once that's done, run a magic command like this:
    #%debug
    raws_permod_both_sides = raws_permod_both_sides2

    mstr = 'run -i ../run/run_prep_dat.py -r ' +   rawnames_str + \
            ' --param_file prep_dat_defparams.ini --calc_stats_multi_band 1' +\
            ' --save_dat 1 --save_stats 1 --use_preloaded_raws 1'
    ipython.magic(mstr)

    # %run -i ../run/run_prep_dat.py -r S99_off_move,S99_off_hold,S99_on_move \
    #         --param_file prep_dat_defparams.ini --calc_stats_multi_band 1 --save_dat 1 \
    #         --use_preloaded_raws 1 --save_stats 1

    gv.DEBUG_MODE = True
    from IPython import get_ipython; ipython = get_ipython()
    #%debug

    # import warnings
    # warnings.filterwarnings("error")
    # # with warnings.catch_warnings():
    # #     warnings.simplefilter('error')

    # try:

    # except ComplexWarning as e:
    #     pass

    rs = ('%run -i ../run/run_genfeats.py '
    ' --sources_type parcel_aal --bands crude'
    ' --src_grouping_fn 10'
    ' --src_grouping 0 '
    ' --raw {} '
    ' --feat_types con,H_act,H_mob,H_compl,rbcorr,bpcorr'
    ' --Kalman_smooth 0'
    ' --load_TFR 0'
    ' --load_CSD 0'
    ' --save_TFR 0'
    ' --save_CSD 0'
    ' --save_bpcorr 0'
    ' --save_rbcorr 0'
    ' --load_rbcorr 0'
    ' --use_existing_TFR 0'
    ' --use_preloaded_data 0'
    ' --allow_CUDA 0'
    ' --load_only 0'
    ' --show_plots 0'
    ' --plot_types ,'
    ' --prescale_data 1'
    ' --exit_after {}'
    ' --normalize_TFR separate'
    ' --scale_data_combine_type no'
    ' --baseline_int_type entire'
    ' --n_jobs 1'
    ' --save_feat 0'
    ' --feat_stats_artif_handling reject'
    ' --scale_data_combine_type no '
    ' --rbcorr_use_local_means 1'
    ' --output_subdir test'
    ' --stats_fn_prefix stats_{}_3_ '
    ' --param_file genfeats_defparams.ini ')

    exit_after = 'end'
    #exit_after = 'prescale_data'
    #exit_after = 'load'
    mstr2 = rs.format(rawnames_str,exit_after,rawnames_str[:3])
    ipython.magic(mstr2)



    ##########################################################################

    # Check the we have scaled correctly
    m0 = max( np.max( dat_pri_unscaled[0] - dat_pri_unscaled[1] ), np.max( dat_pri_unscaled[0] - dat_pri_unscaled[2] ) )
    m1 = max( np.max( dat_pri[0] - dat_pri[1] ), np.max( dat_pri[0] - dat_pri[2] ) )
    print(m0,m1)
    assert m1 < noise_size

    # Check the we have scaled correctly
    m0 = max( np.max( dat_lfp_hires_pri_unscaled[0] - dat_lfp_hires_pri_unscaled[1] ),
            np.max( dat_lfp_hires_pri_unscaled[0] - dat_lfp_hires_pri_unscaled[2] ) )
    m1 = max( np.max( dat_lfp_hires_pri[0] - dat_lfp_hires_pri[1] ),
            np.max( dat_lfp_hires_pri[0] - dat_lfp_hires_pri[2] ) )
    print(m0,m1)
    assert m1 < noise_size
