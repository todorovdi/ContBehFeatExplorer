import numpy as np
import utils_genfeats as ugf

def test_stride():
    U = np.vstack([np.arange(6 ),10+np.arange(6) ] )
    sv = ugf.stride( U , win=(1,3), stepby=(1,2) )
    tval = np.array( [
                [
                    [ 0,  1,  2],
                    [ 2,  3,  4]
                ],
                [
                    [10, 11, 12],
                    [12, 13, 14]
                ] ] )
    assert np.all(sv = tval)

def test2ch(bandPairs, ch1_band1,ch2_band1,ch1_band2,ch2_band2, chn1,chn2, windowsz, skip):
    import utils

    bp = bandPairs[0]

    # ch1_band1 = np.ones(dat_len)
    # #ch2_band1 = np.ones(dat_len)
    # ch2_band1 = np.random.uniform(size=dat_len)
    dat1 = np.array( [ch1_band1, ch2_band1] )

    # ch1_band2 = np.ones(dat_len)
    # ch2_band2 = np.ones(dat_len)
    dat2 = np.array( [ch1_band2, ch2_band2] )

    raw1 = utils.makeSimpleRaw(dat1,rescale=0)
    raw2 = utils.makeSimpleRaw(dat2,rescale=0)
    raw_perband = {}
    raw_perband[bp[0]]   = raw1
    raw_perband[bp[1]]   = raw2



    chnames_perband = {}
    # I cannot put just one band, it should have same bands in both modalities
    # unless the band is LFP
    # chnames_perband is one list for all modalities
    chnames_perband[bp[0]] = [chn1,chn2]
    chnames_perband[bp[1]] = [chn1,chn2]  # can be L as well if CB


    means_perband = {}
    means_perband[bp[0]] = np.zeros(dat1.shape[0])
    means_perband[bp[1]] = np.zeros(dat2.shape[0])

    #for i in range(2):
    #    display(bp[i], raw_perband[bp[i]].get_data())
    #display(chnames_perband)
    #display(means_perband)

    from utils import parseMEGsrcChnameShort


    srcgrouping_names_sorted = []

    # I think I want LFP to go first
    chnames_tfr = [chn1,chn2]
    print('chnames_tfr = ',chnames_tfr)

    parcel_couplings = []
    LFP2parcel_couplings = []
    LFP2LFP_couplings = []
    if chn1.startswith('LFP') and chn2.startswith('msrc'):
        side1, gi1, parcel_ind1, si1  = parseMEGsrcChnameShort(chn1)
        LFP2parcel_couplings = { (chn1, parcel_ind1): [(0,1)]  }
        #display(LFP2parcel_couplings)
    if chn1.startswith('msrc') and chn2.startswith('msrc'):
        side1, gi1, parcel_ind1, si1  = parseMEGsrcChnameShort(chn1)
        side2, gi2, parcel_ind2, si2  = parseMEGsrcChnameShort(chn2)

        # parcel indices -> chnames_tfr indices of chnames
        parcel_couplings = { (parcel_ind1, parcel_ind2):[(0,1)] }
        #display(parcel_couplings)
    if chn1.startswith('LFP') and chn2.startswith('LFP'):
        LFP2LFP_couplings = []
        #display(LFP2LFP_couplings)

    n_jobs = 1
    newchn_grouping_ind = 10


    # test differen simple imput combinations, see that correl indeed correls

    bpcorrs = []
    bpcor_names = []
    for bpcur in bandPairs:
        print('band pair ',bpcur)
        bpcorrs_curbp,bpcor_names_curbp,dct_nums,wbd_bpcorr =\
        ugf.computeCorr(raw_perband, names=chnames_perband,
                                defnames=chnames_tfr,
                                parcel_couplings=parcel_couplings,
                                LFP2parcel_couplings=LFP2parcel_couplings,
                                LFP2LFP_couplings=LFP2LFP_couplings,
                                res_group_id=newchn_grouping_ind,
                                skip=skip, windowsz = windowsz,
                                band_pairs = [bpcur], n_jobs=n_jobs,
                                positive=1, templ='{}_.*',
                                sort_keys=srcgrouping_names_sorted,
                                means=means_perband, reverse=1, verbose=0, pad=0)
        assert len(bpcor_names_curbp) > 0
        bpcorrs += bpcorrs_curbp
        bpcor_names += bpcor_names_curbp

    bpcorrs = np.vstack(bpcorrs)

    for feati in range(len(bpcor_names) ):
        bpcor_names[feati] = 'bp' + bpcor_names[feati]

    from numpy import array
    assert np.all( bpcorrs == array([
        [20., 10.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [20., 20., 20., 10.,  0.,  0.,  0.,  0.,  0.]])  )
    assert tuple(bpcor_names_curbp) == tuple( ['corr_tremor_msrcR_10_1_c0,beta_msrcR_10_2_c0',
    'corr_beta_msrcR_10_1_c0,tremor_msrcR_10_2_c0'] )
    assert np.all(wbd_bpcorr == array([[ 0,  2,  4,  6,  8, 10, 12, 14, 16],
        [ 4,  6,  8, 10, 12, 14, 16, 18, 20]])  )

    #print('----    Result is')
    #display(bpcorrs)
    #display(bpcor_names_curbp)
    #display(wbd_bpcorr)
    #display(dct_nums)

def test_2ch():
    bp = 'tremor','beta'
    bandPairs = [(bp[0],bp[1], 'corr')]

    windowsz = 4
    skip = 2
    dat_len = windowsz*5

    # ones will live in tremor band of channel 1
    ch1_band1 = np.ones(dat_len) * 1
    ch1_band1[windowsz:] = 0
    # twos will live in tremor band of channel 2
    ch2_band1 = np.ones(dat_len) * 2
    ch2_band1[windowsz*2:] = 0
    #ch2_band1 = np.random.uniform(size=dat_len)
    # tens will live in beta band of channel 1
    ch1_band2 = np.ones(dat_len) * 10
    ch1_band2[windowsz*3:] = 0
    #ch1_band2[windowsz:] = 0
    # twentys will live in beta band of channel 2
    ch2_band2 = np.ones(dat_len) * 20
    ch2_band2[windowsz*4:] = 0

    #chn1 = 'LFPR01'
    parcel_index1 = 1
    parcel_index2 = 2

    chn1 = f'msrcR_9_{parcel_index1}_c1'
    chn2 = f'msrcR_9_{parcel_index2}_c2'


    res = test2ch(bandPairs,ch1_band1,ch2_band1, ch1_band2, ch2_band2,chn1,chn2, windowsz,skip)



# test window alignments
def testTFRwindowAlign():
    import utils
    min_freq = 3
    sfreq = 256
    windowsz = 256

    m = windowsz / sfreq
    freqs,n_cycles, Ws, windowsz_max = utils.prepFreqs(min_freq = min_freq, max_freq = 90,
                                                    frmults=[2*m,m,m], sfreq=sfreq )

    n_jobs_tfr = 1

    fr = 12
    #m1 = 1.5
    m1 = 2; m2 = 9; m3 = 11
    tt = np.zeros((2,windowsz * m3 )  )
    sz = int(windowsz*(m2-m1) )
    tt[0,int(windowsz*m1):int(windowsz*m2)] = 1 * np.sin(fr * 2 * np.pi * np.arange( sz) / sz)
    tt[0,windowsz*m2:] = 0

    decim = 32
    #decim = 1
    tfrres_t,wbd_t = utils.tfr(tt, sfreq, freqs, n_cycles,
                            windowsz, decim = decim,
                            n_jobs=n_jobs_tfr, mode='valid')

    change_mode_freqi = 27
    #for freqSlice in [slice(None,change_mode_freqi), slice(change_mode_freqi,None)]:

    wbdi = 0
    b0 = wbd_t[0,wbdi]
    b1 = wbd_t[1,wbdi]

    start_segment = tt[0,b0:b1]
    start_segment2 = tt[0,b0 + windowsz: b1 + windowsz ]
    post_start_segment = tt[0,b0 + windowsz: b1 + windowsz + 2 ] # +1 still zero
    assert np.max( np.abs(start_segment) ) < 1e-14
    assert np.max( np.abs(start_segment2) ) < 1e-14
    assert np.max( np.abs(post_start_segment) ) > 1e-14
    assert np.max( np.abs(tfrres_t[:,change_mode_freqi:,wbdi:windowsz//decim + 1] ) ) < 1e-14
    assert np.max( np.abs(tfrres_t[:,change_mode_freqi:,wbdi:windowsz//decim + 2] ) ) > 1e-14


    wbdi = wbd_t.shape[1] - 1
    b0 = wbd_t[0,wbdi]
    b1 = wbd_t[1,wbdi]

    segment = tt[0,b0:b1]
    sh = (decim - 4) * (windowsz // decim)  # not the best formula to use
    #sh = (decim - 4) * (windowsz // decim)
    print(sh)
    segment2    = tt[0,b0 - sh    : b1 ]
    pre_segment = tt[0,b0 - sh -1 : b1 ] # +1 still zero
    # last window
    assert np.max( np.abs(segment) ) < 1e-14
    # first zero window after sine
    assert np.max( np.abs(segment2) ) < 1e-14
    # last nonzero window
    assert np.max( np.abs(pre_segment) ) > 1e-14
    assert np.max( np.abs(tfrres_t[:,change_mode_freqi:,wbdi- windowsz//decim + 1: wbdi] ) ) < 1e-14
    assert np.max( np.abs(tfrres_t[:,change_mode_freqi:,wbdi- windowsz//decim : wbdi]    ) ) > 1e-14



def test_align():
    #import IPython.ipapi
    #ipython = IPython.ipapi.get()
    #ipython.magic("timeit abs(-42)")
    %run -i ../run/run_genfeats.py --sources_type parcel_aal --bands crude\
    --src_grouping_fn 10\
    --src_grouping 0 \
    --raw S97_off_move \
    --feat_types con,H_act,H_mob,H_compl,rbcorr\
    --load_only 1\
    --show_plots 0\
    --plot_types ,\
    --scale_data_combine_type subj \
    --output_subdir test\
    --stats_fn_prefix stats_S97,S99_3_ \
    --param_file genfeats_defparams.ini
    ####################

    reclen = 4 # it is not enough because wavelet has len 2048
    nbins = windowsz * reclen
    nbins_hires = windowsz_hires * reclen
    #########
    times_pri = [np.arange(nbins ) * 1/sfreq]
    times_hires_pri = [np.arange( nbins_hires) * 1/sfreq_hires]
    extdat_pri = [np.zeros( (2,nbins))]
    subfeature_order_pri = [['LFPR01',
    'LFPR12',
    'msrcR_0_4_c5',
    'msrcR_0_8_c15',
    'msrcR_0_16_c8',
    'msrcR_0_32_c2',
    'msrcR_0_38_c9',
    'msrcR_0_44_c0',
    'msrcR_0_52_c9',
    'msrcR_0_58_c5',
    'msrcL_0_60_c24']]
    subfeature_order_lfp_hires_pri = [ ['LFPR01', 'LFPR12'] ]
    ivalis_pri = [{'notrem_L':(0.5,1,'notrem_L'),'trem_L':(1,1.5,'notrem_L') }]

    dat_pri = [np.zeros( (len(subfeature_order_pri[0]), nbins) )]
    dat_lfp_hires_pri = [np.zeros( (len(subfeature_order_lfp_hires_pri[0]), nbins_hires) )]

    offset_start = windowsz * 2
    for i in range(len(dat_pri[0])):
        dat_pri[0][i][offset_start:] = 1

    offset_hires_start = windowsz_hires* 2
    for i in range(len(dat_lfp_hires_pri[0])):
        dat_lfp_hires_pri[0][i][offset_hires_start:] = 1

    #####################
    gv.DEBUG_MODE = True
    #%debug

    # import warnings
    # warnings.filterwarnings("error")
    # # with warnings.catch_warnings():
    # #     warnings.simplefilter('error')

    # try:
    %run -i ../run/run_genfeats.py --sources_type parcel_aal --bands crude\
    --src_grouping_fn 10\
    --src_grouping 0 \
    --raw S97_off_move \
    --feat_types con,H_act,H_mob,H_compl,rbcorr,bpcorr\
    --Kalman_smooth 0\
    --load_TFR 0\
    --load_CSD 0\
    --save_TFR 0\
    --save_CSD 0\
    --save_bpcorr 0\
    --save_rbcorr 0\
    --load_rbcorr 0\
    --use_existing_TFR 0\
    --use_preloaded_data 1\
    --allow_CUDA 0\
    --load_only 0\
    --show_plots 0\
    --plot_types ,\
    --prescale_data 0\
    --normalize_TFR no\
    --n_jobs 1\
    --save_feat 0\
    --feat_stats_artif_handling no\
    --scale_data_combine_type subj \
    --rbcorr_use_local_means 1\
    --output_subdir test\
    --stats_fn_prefix stats_S97,S99_3_ \
    --param_file genfeats_defparams.ini
    # except ComplexWarning as e:
    #     pass

    ###################3
    N = dat_pri[0].shape[-1]
    print(tfrres.shape)
    nc = 11
    fig,axs = plt.subplots(nc,1,figsize=(12,2*nc), sharex='col')
    axind = 0
    ax = axs[axind]; axind += 1
    for chi in range(2,tfrres.shape[0]):
        for fi in range(27,tfrres.shape[1] ):
            ax.plot( np.abs( tfrres[chi][fi] ) )
    ax.plot( dat_pri[0][0][::skip] /8 )
    ax.fill_betweenx([0,.2],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)
    ax.set_title('SRC higher_freq')
    print(tfrres.shape)

    ax = axs[axind]; axind += 1
    for chi in range(2,tfrres.shape[0]):
        for fi in range(27 ):
            ax.plot( np.abs( tfrres[chi][fi] ) )
    ax.plot( dat_pri[0][0][::skip] /8 )
    ax.set_title('SRC low_freq')
    ax.fill_betweenx([0,2],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)

    ax = axs[axind]; axind += 1
    for feati in range(30,bpow_abscds_all_reshaped.shape[0]):
        ax.plot( np.abs( bpow_abscds_all_reshaped[feati] ) )
    ax.plot( dat_pri[0][0][::skip] / 10  )
    ax.set_title('abscds higher_freq')
    ax.fill_betweenx([0,0.2],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)


    # plot LFP
    #fig,axs = plt.subplots(2,1,figsize=(12,6))
    ax = axs[axind]; axind += 1
    for chi in range(2):
        for fi in range(27,tfrres.shape[1] ):
            ax.plot( np.abs( tfrres[chi][fi] ) )
    ax.plot( dat_pri[0][0][::skip] / 8  )
    ax.fill_betweenx([0,.2],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)
    ax.set_title('LFP higher_freq')

    ax = axs[axind]; axind += 1
    for chi in range(2):
        for fi in range(27 ):
            ax.plot( np.abs( tfrres[chi][fi] ) )
    ax.plot( dat_pri[0][0][::skip] *2  )
    ax.set_title('SRC lower_freq')
    ax.fill_betweenx([0,2],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)

    wbdi = (offset_start-windowsz)//skip
    s,e = tfrres_wbd_pri[0][:,wbdi] / skip
    ax.fill_betweenx([0,1],  s,e, color='yellow', alpha=0.15)

    #----

    ax = axs[axind]; axind += 1
    for chi in range(tfrres_LFP_HFO.shape[0]):
        for fi in range(tfrres_LFP_HFO.shape[1] ):
            ax.plot( np.abs( tfrres_LFP_HFO[chi][fi] ) )
    ax.plot( dat_pri[0][0][::skip] /8 )
    ax.set_title('LFP HFO')
    ax.fill_betweenx([0,0.1],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)


    #-------------------
    ax = axs[axind]; axind += 1
    d1 = act_pri[0][0]
    d2 = mob_pri[0][0]
    d3 = compl_pri[0][0]
    #plt.close('all')
    ax.plot( d1 / np.max(d1)  ,label='act', marker='*' )
    ax.plot( d2 / np.max(d2), label='mob' )
    ax.plot( d3 / np.max(d3)  , label='compl' )


    ax.plot( dat_pri[0][0][::skip] *2  )
    ax.fill_betweenx([0,2],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)
    ax.set_title('H')

    wbdi = (offset_start-windowsz)//skip
    s,e = wbd_H_pri[0][:,wbdi] / skip
    ax.fill_betweenx([0,1],  s,e, color='yellow', alpha=0.15)
    ax.legend()

    ax = axs[axind]; axind += 1
    for bn,r in raw_perband_flt_pri[0].items():
        d = r.get_data()
        print(bn, np.where( np.abs(d[:,-1] ) > 1e-10 )[0] )
        ax.plot(np.arange(N) / skip, d.T , label=bn)
    ax.plot(np.arange(N) / skip, dat_pri[0][0]/2 )


    ax = axs[axind]; axind += 1
    for bn,r in raw_perband_bp_pri[0].items():
        d = r.get_data()
        print(bn, np.where( np.abs(d[:,-1] ) > 1e-10 )[0] )
        ax.plot(np.arange(N) / skip,  d.T , label=bn)
    ax.plot( np.arange(N) / skip, dat_pri[0][0]/2 )


    ax = axs[axind]; axind += 1
    d = bpcorrs_pri[0][0:5]
    ax.plot( d.T, label='bp'  )

    d = rbcorrs_pri[0][0:5]
    ax.plot( d.T, label='rb'  )
    ax.plot( dat_pri[0][0][::skip] *0.02, ls=':'  )

    wbdi = (offset_start-windowsz)//skip
    s,e = wbd_rbcorr_pri[0][:,wbdi] / skip
    ax.fill_betweenx([0,0.02],  s,e, color='yellow', alpha=0.15)

    wbdi = (offset_start-windowsz)//skip
    s,e = wbd_bpcorr_pri[0][:,wbdi] / skip
    ax.fill_betweenx([-0.01,0],  s,e, color='purple', alpha=0.15)

    # TODO: test that beginning and end conside and there is somethign
    # interesting in the middel that happen right before the step in the
    # original data happens
