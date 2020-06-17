# compute diagonal terms
have_TFR = False
try:
    print('tfr (existing) num channels =', tfrres.shape[0] )
except NameError as e:
    have_TFR = False
else:
    have_TFR = True #and (tfrres.shape[0] == n_channels)

#if not have_TFR:
#    print('OOO')
#    sys.exit(0)

if not (use_existing_TFR and have_TFR):
    if load_TFR and os.path.exists( fname_tfr_full ):
        tfrres = np.load(fname_tfr_full)['tfrres']
    else:
        print('Starting TFR ')

        # if we use highres LFP raw, better to do entire TFR on LFP
        if use_lfp_HFO:
            dat_src,names_src = utsne.selFeatsRegex(dat_scaled, subfeature_order, ['msrc.*'])
            dat_for_tfr = dat_src
            chnames_tfr = names_src
        else:
            dat_for_tfr = dat_scaled
            chnames_tfr = subfeature_order

        # perhaps we want to ensure that wavelets intersect well. Then it's
        # better to use smaller skip and then downsample
        skip_div_TFR = 2
        assert ( skip - (skip // skip_div_TFR)  * skip_div_TFR ) < 1e-10

        tfrres_ = utils.tfr(dat_for_tfr, sfreq, freqs, n_cycles, decim = skip // skip_div_TFR)
        tfrres = utsne.downsample(tfrres_, skip_div_TFR, axis=-1)
        #tfrres = mne.time_frequency.tfr_array_morlet(dat_for_tfr, sfreq,
        #                                            freqs, freqs *
        #                                            freq2cycles_mult, n_jobs=10)
        #tfrres = tfrres[0]


        if use_lfp_HFO:

            dat_for_tfr = dat_lfp_highres_scaled

            tfrres_LFP_ = utils.tfr(dat_for_tfr, sfreq_highres, freqs_inc_HFO, n_cycles_inc_HFO,
                                   decim = skip_highres // skip_div_TFR)
            tfrres_LFP = utsne.downsample(tfrres_LFP_, skip_div_TFR, axis=-1)


            tfrres_LFP_LFO = tfrres_LFP[:,:len(freqs),:]
            tfrres_LFP_HFO = tfrres_LFP[:,len(freqs):,:]

            tfrres = np.concatenate( [tfrres, tfrres_LFP_LFO], axis=0 )
            chnames_tfr = chnames_tfr.tolist()  + subfeature_order_lfp_highres

            # I don't really need HFO csd across LFP contacts
            #csd_LFP, csdord_LFP = utils.tfr2csd(tfrres_LFP, sfreq_highres, returnOrder=1)  # csdord.shape = (2, csdsize)
            csd_LFP_HFO = tfrres_LFP_HFO * tfrres_LFP_HFO

            tmp = np.arange( tfrres_LFP.shape[0] )  # n_LFP_channels
            csdord_LFP_HFO = np.vstack([tmp,tmp] ) # same to same index, so just i->i

            #tfrres = mne.time_frequency.tfr_array_morlet(dat_for_tfr, sfreq,
            #                                            freqs, freqs *
            #                                            freq2cycles_mult, n_jobs=10)


            # no I have to do it later, I cannot vstack because it has differtn
            # freq count
            #assert tfrres.shape[-1] = tfrres_LFP.shape[-1]
            #tfrres = np.vstack( [tfrres, tfrres_LFP] )
            #subfeature_order = chnames_tfr + subfeature_order_lfp_highres

        if save_TFR:
            np.savez(fname_tfr_full, tfrres=tfrres)
            print('TFR saved to ',fname_tfr_full)

        csd, csdord = utils.tfr2csd(tfrres, sfreq, returnOrder=1)  # csdord.shape = (2, csdsize)


gc.collect()
ntimebins = tfrres.shape[-1]

############################# CSD

print('Averaging over freqs withing bands')
if bands_only in ['fine', 'crude']:
    if bands_only == 'fine':
        fband_names = fband_names_fine
    else:
        fband_names = fband_names_crude

    bpow_abscsd = []
    for bandi,bandname in enumerate(fband_names):
        low,high = fbands[bandname]
        freqis = np.where( (freqs >= low) * (freqs <= high) )[0]
        assert len(freqis) > 0, bandname
        bandpow = np.mean( np.abs(  csd[:,freqis,:])  , axis=1 )
        bpow_abscsd += [bandpow[:,None,:]]


    bpow_abscsd = np.concatenate(bpow_abscsd, axis=1)


csdord_bandwise = []
# last dimension is index of band
for bandi,bandname in enumerate(fband_names):
    csdord_bandwise += [ np.concatenate( [csdord.T,  np.ones(csd.shape[0], dtype=int)[:,None]*bandi  ] , axis=-1) [:,None,:] ]

csdord_bandwise = np.concatenate(csdord_bandwise,axis=1)
csdord_bandwise.shape

if use_lfp_HFO:
    if bands_only in ['fine', 'crude']:
        if bands_only == 'fine':
            fband_names_inc_HFO = fband_names_fine_inc_HFO
        else:
            fband_names_inc_HFO = fband_names_crude_inc_HFO

        bpow_abscsd_LFP_HFO = []
        fband_names_HFO = fband_names_inc_HFO[len(fband_names):]  # that HFO names go after
        freqs_HFO = freqs_inc_HFO[ len(freqs): ]
        for bandi,bandname in enumerate(fband_names_HFO):
            low,high = fbands[bandname]
            freqis = np.where( (freqs_HFO >= low) * (freqs_HFO <= high) )[0]
            assert len(freqis) > 0, bandname
            bandpow = np.mean( np.abs(  csd_LFP_HFO[:,freqis,:])  , axis=1 )
            bpow_abscsd_LFP_HFO += [bandpow[:,None,:]]

        bpow_abscsd_LFP_HFO = np.concatenate(bpow_abscsd_LFP_HFO, axis=1)

    csdord_bandwise_LFP_HFO = []
    for bandi,bandname in enumerate(fband_names_HFO):
        csdord_bandwise_LFP_HFO += [ np.concatenate( [csdord_LFP_HFO.T,  np.ones(csd_LFP_HFO.shape[0], dtype=int)[:,None]*bandi  ] , axis=-1) [:,None,:] ]

    csdord_bandwise_LFP_HFO = np.concatenate(csdord_bandwise_LFP_HFO,axis=1)
    csdord_bandwise_LFP_HFO.shape

###################################

print('Preparing csdord_strs')
csdord_strs = []
#for csdord_cur in csdords:
csdord_cur = csdord
for bandi in range(csdord_bandwise.shape[1] ):
    for i in range(csdord_cur.shape[1]):
        k1,k2 = csdord_cur[:,i]
        k1 = int(k1); k2=int(k2)
        s = '{}_{},{}'.format( fband_names[bandi], chnames_tfr[k1] , chnames_tfr[k2] )
        csdord_strs += [s]


if use_lfp_HFO:
    csdord_cur = csdord_LFP_HFO
    for bandi in range(csdord_bandwise_LFP_HFO.shape[1] ):
        for i in range(csdord_cur.shape[1]):
            k1,k2 = csdord_cur[:,i]
            k1 = int(k1); k2=int(k2)
            s = '{}_{},{}'.format( fband_names_HFO[bandi],
                                subfeature_order_lfp_highres[k1] , subfeature_order_lfp_highres[k2] )
            csdord_strs += [s]

################################  Plot CSD at some time point
if do_plot_CSD:
    for int_name in int_names:
        #nt_name = 'trem_{}'.format(maintremside)
        #intervals = ivalis[int_name]
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv
            ts,r, int_names = utsne.getIntervalSurround(start,end,extend, raw_srconly)

            #timebins = raw_lfponly.time_as_index
            utsne.plotCSD(csd, fband_names, chnames_tfr, list(r) , sfreq=sfreq, intervalMode=1,
                        int_names=int_names)
            pdf.savefig()
            plt.close()


############################## Hjorth

print('Computing Hjorth')
if use_lfp_HFO:
    dat_for_H = dat_src
else:
    dat_for_H = dat_scaled
act,mob,compl  = utils.Hjorth(dat_for_H, 1/sfreq, windowsz=windowsz)

if use_lfp_HFO:
    act_lfp,mob_lfp,compl_lfp  = utils.Hjorth(dat_lfp_highres_scaled, 1/sfreq_highres,
                                windowsz=int( (windowsz/sfreq)*sfreq_highres ) )
    act_lfp   =      act_lfp   [:,:: sfreq_highres//sfreq ]
    mob_lfp   =      mob_lfp   [:,:: sfreq_highres//sfreq ]
    compl_lfp =  compl_lfp [:,:: sfreq_highres//sfreq ]
    act = np.vstack( [act, act_lfp] )
    mob = np.vstack( [mob, mob_lfp] )
    compl = np.vstack( [compl, compl_lfp] )

if show_plots:
    fig=plt.figure()
    ax = plt.gca()
    ax.plot (  np.min(  act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='min')
    ax.plot (  np.max(  act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='max')
    ax.plot (  np.mean( act[:,nedgeBins:-nedgeBins], axis=-1 ) ,label='mean')
    ax.set_xlabel('channel')
    ax.legend()
    ax.set_xticks(range(n_channels))
    ax.set_xticklabels(subfeature_order,rotation=90)
    fig.suptitle('min_max of Hjorth params')

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #################################  Plot Hjorth

    fig,axs = plt.subplots(nrows=1,ncols=3, figsize = (15,4))
    axs = axs.reshape((1,3))

    for i in range(n_channels):
        axs[0,0].plot(times,act[i], label= subfeature_order[i])
        axs[0,0].set_title('activity')
        axs[0,1].plot(times,mob[i], label= subfeature_order[i])
        axs[0,1].set_title('mobility')
        axs[0,2].plot(times,compl[i], label= subfeature_order[i])
        axs[0,2].set_title('complexity')

    for ax in axs.reshape(axs.size):
        ax.legend(loc='upper right')

    pdf.savefig()


act = act[:,::skip]
mob = mob[:,::skip]
compl = compl[:,::skip]


#Xtimes_full = raw_srconly.times[nedgeBins:-nedgeBins]
#####################################################

if 'rbcorr' in features_to_use or 'bpcorr' in features_to_use:
    print('Filtering and Hilbert')
    names_flt = []
    dat_flt = []
    names_bpow = []
    dat_bpow = []

    bpow_mav_wsz = skip // 2
    wnd_mav = np.ones( bpow_mav_wsz )
    for bandi,bandname in enumerate(fband_names):
        low,high = fbands[bandname]
        for chi in range(n_channels):
            #dat_flt_cur = utils._flt(dat_scaled[chi,:], sfreq, low,high)
            ang, instfreq, instampl,dat_flt_cur = \
                utils.getBandHilbDat (dat_scaled[chi,:], sfreq, low,high,ret_flt=1)
        #return ang, instfreq, instampl, fltdata

            name = '{}_{}'.format(bandname, chnames_tfr[chi] )
            names_flt += [name]
            dat_flt += [dat_flt_cur[nedgeBins:-nedgeBins] ]

            name = '{}_{}'.format(bandname, chnames_tfr[chi] )
            names_bpow += [name]
            #instampl =
            dat_bpow += [ np.convolve(wnd_mav, instampl, mode='same' )[nedgeBins:-nedgeBins] ]

    bpow_mav_wsz = skip_highres // 2
    wnd_mav = np.ones( bpow_mav_wsz )

    names_flt_highres  = []
    dat_flt_highres  = []
    names_bpow_highres  = []
    dat_bpow_highres  = []
    for bandi,bandname in enumerate(fband_names_HFO):
        low,high = fbands[bandname]
        for chi in range(dat_lfp_highres_scaled.shape[0] ):
            #dat_flt_cur = utils._flt(dat_scaled[chi,:], sfreq, low,high)
            ang, instfreq, instampl,dat_flt_cur = \
            utils.getBandHilbDat(dat_lfp_highres_scaled[chi,:], sfreq_highres,
                                 low,high,ret_flt=1)
        #return ang, instfreq, instampl, fltdata

            name = '{}_{}'.format(bandname, subfeature_order_lfp_highres[chi] )
            names_flt_highres += [name]
            dat_flt_highres += [dat_flt_cur[nedgeBins:-nedgeBins] ]

            name = '{}_{}'.format(bandname, subfeature_order_lfp_highres[chi] )
            names_bpow_highres += [name]
            #instampl =
            dat_bpow_highres += [ np.convolve(wnd_mav, instampl, mode='same' )
                                 [nedgeBins_highres:-nedgeBins_highres: sfreq_highres//sfreq ] ]

    dat_flt = np.vstack(dat_flt)

    dat_flt_highres = np.vstack(dat_flt_highres)
    #dat_bpow_highres = np.vstack(dat_bpow_highres)

    dat_bpow = np.vstack(dat_bpow + dat_bpow_highres)
    names_bpow += names_bpow_highres

if 'rbcorr' in features_to_use:  #raw band corr
    #compute_raw_bands
    if bands_only == 'fine':
        fband_names = fband_names_fine
    else:
        fband_names = fband_names_crude



#def _flt(data,sfreq,lowcut,highcut,bandpass_order = 5):


    if bands_only == 'fine':
        bandPairs = [('tremor','tremor', 'corr'),
                     ('low_beta','low_beta', 'corr'), ('high_beta','high_beta', 'corr'),
                     ('low_gamma','low_gamma', 'corr') , ('high_gamma','high_gamma', 'corr') ]
    else:
        bandPairs = [('tremor','tremor', 'corr'), ('beta','beta', 'corr'), ('gamma','gamma', 'corr') ]


    rbcors,rbcor_names = utsne.computeFeatOrd2(dat_flt, names=names_flt,
                                            skip=skip, windowsz = windowsz, band_pairs = bandPairs,
                                            n_free_cores=2, positive=0, templ='{}_.*')

    # if use_lfp_HFO:  # we can compute time domain corr only between HFOs,
    # but they exist only at LFP and we don't want to compute corr of LFPs
    #    bandPairs = [ ('HFO','HFO', 'corr')  ]
    #    Xtimes_full_highres = raw_lfp_highres.times[nedgeBins_highres:-nedgeBins_highres]
    #    rbcors,rbcor_names = utsne.computeFeatOrd2(dat_flt_highres, Xtimes_full_highres,
    #                                               names=names_flt_highres,
    #                                        skip=skip_highres, windowsz = windowsz_highres,
    #                                               band_pairs = bandPairs,
    #                                        n_free_cores=2, positive=0, templ='{}_.*')

    assert len(rbcor_names) > 0

    rbcors = np.vstack(rbcors)

if 'bpcorr' in features_to_use:
    if bands_only == 'fine':
        bandPairs = [('tremor','low_beta', 'corr'), ('tremor','high_beta', 'corr'),
                     ('tremor','low_gamma', 'corr'), ('tremor','high_gamma', 'corr'),
                     ('low_beta','low_gamma', 'corr') , ('low_beta','high_gamma', 'corr') ,
                     ('high_beta','low_gamma', 'corr') , ('high_beta','high_gamma', 'corr') ]
    else:
        bandPairs = [('tremor','beta', 'corr'), ('tremor','gamma', 'corr'), ('beta','gamma', 'corr') ]
    if use_lfp_HFO:
        bandPairs += [ ('tremor','HFO', 'corr'), ('beta','HFO', 'corr'), ('gamma','HFO', 'corr') ]

        if bands_only == 'fine':
            bandPairs += [ ('HFO1','HFO2', 'div'), ('HFO1','HFO3', 'div'), ('HFO2','HFO3', 'div') ]

    bpcors,bpcor_names = utsne.computeFeatOrd2(dat_bpow, names=names_bpow,
                                            skip=skip, windowsz = windowsz,
                                               band_pairs = bandPairs,
                                            n_free_cores=2, positive=1, templ='{}_.*')
    assert len(bpcor_names) > 0

    bpcors = np.vstack(bpcors)


#####################################################

    #e.g  bandPairs = [('tremor','beta'), ('tremor','gamma'), ('beta','gamma') ]


#####################################################


# extract bandpowers only
#bpows = []
#for bandi in range(len(fband_names)):
#    bpow = np.vstack( [ utils.getCsdVals(bpow_abscsd[:,bandi,:],i,i, n_channels) for i in range (n_channels) ] )
#    bpows += [bpow[:,None,:]]
#bpows = np.concatenate( bpows, axis=1)
#bpows.shape

######################################################



defpct = (percentileOffset,100-percentileOffset)
center_spec_feats = spec_uselog
if spec_uselog:
    con_scale = defpct
else:
    con_scale = (0,100-percentileOffset)

feat_dict = { 'con':{'data':None, 'pct':con_scale, 'centering':center_spec_feats,
                     'names':csdord_strs },
             'H_act':{'data':act, 'pct':defpct, 'names':subfeature_order},
             'H_mob':{'data':mob, 'pct':defpct, 'names':subfeature_order},
             'H_compl':{'data':compl, 'pct':defpct, 'names':subfeature_order},
             'rbcorr':{'data':rbcors, 'pct':defpct, 'names':rbcor_names  },
             'bpcorr':{'data':bpcors, 'pct':defpct, 'names':bpcor_names  } }
 #'tfr':{'data':None, 'pct':con_scale, 'centering':center_spec_feats },

feat_dict['con']['centering'] = True

f = lambda x: x
if spec_uselog:
    f = lambda x: np.log(x)

if bands_only == 'no':
    tfres_ = tfrres.reshape( tfrres.size//ntimebins , ntimebins )
    #feat_dict['tfr']['data'] = f( np.abs( tfres_) )
    #feat_dict['con']['data'] = con.reshape( con.size//ntimebins , ntimebins )
    feat_dict['con']['data'] = f( np.abs( csd.reshape( csd.size//ntimebins , ntimebins ) ) )
else:
    #feat_dict['tfr']['data'] = f( bpows.reshape( bpows.size//ntimebins , ntimebins ) )
    tmp1 = bpow_abscsd.reshape( bpow_abscsd.size//ntimebins , ntimebins )
    if use_lfp_HFO:
        tmp2 = bpow_abscsd_LFP_HFO.reshape( bpow_abscsd_LFP_HFO.size//ntimebins , ntimebins )
        # add HFO to low freq
        tmp = np.vstack( [tmp1, tmp2])
        #TODO: note that csdord_strs by that moment already contains LFP HFO
        #names (see when csdord_strs i generated)
    else:
        tmp = tmp1

    assert len(tmp) == len(feat_dict['con']['names'] )

    if not use_LFP_to_LFP:
        templ_same_LFP = r'.*:\s(LFP.*),\1'
        inds_same_LFP = utsne.selFeatsRegexInds(csdord_strs, [templ_same_LFP], unique=1)
        templ_all_LFP = r'.*:\s(LFP.*),(LFP.*)'
        inds_all_LFP = utsne.selFeatsRegexInds(csdord_strs, [templ_all_LFP], unique=1)

        inds_notsame_LFP = np.setdiff1d( inds_all_LFP, inds_same_LFP)
        gi = np.setdiff1d( np.arange(len(csdord_strs) ) , inds_notsame_LFP)
        tmp = tmp[gi]

        feat_dict['con']['names'] = np.array(feat_dict['con']['names'])[gi]

    feat_dict['con']['data'] = f( tmp )

    #tmp_ord
    csdord1 = csdord_bandwise.reshape( (tmp1.shape[0], 3) )
    if use_lfp_HFO:
        csdord2 = csdord_bandwise_LFP_HFO.reshape( (tmp2.shape[0], 3) )
    #csdords = [csdord1, csdord2  ]
    #csdord = np.vstack(csdords  )


##########
for feat_name in feat_dict:
    if feat_name in ['rbcorr', 'bpcorr']:
        continue
    curfeat = feat_dict[feat_name]
    curfeat['data'] = curfeat['data'][:,nedgeBins//skip:-nedgeBins//skip]
    print(feat_name, curfeat['data'].shape)




#################################  Scale features
for feat_name in feat_dict:
    curfeat = feat_dict[feat_name]
    inp = curfeat['data']
    centering = curfeat.get('centering', True)
    pct = curfeat['pct']
    if pct is not None:
        scaler = RobustScaler(quantile_range=pct, with_centering=centering)
        scaler.fit(inp.T)
        outd = scaler.transform(inp.T)
        curfeat['data'] = outd.T
        cnt = scaler.center_
        scl = scaler.scale_
        if feat_name in ['tfr', 'con', 'rbcorr', 'bpcorr']:
            if cnt is not None:
                cnt = np.min(cnt), np.max(cnt), np.mean(cnt), np.std(cnt)
            if scl is not None:
                scl = np.min(scl), np.max(scl), np.mean(scl), np.std(scl)
        print(feat_name, curfeat['data'].shape, cnt,scl)



#csdord_strs = []
#for i in range(csdord.shape[1]):
#    k1,k2 = csdord[:,i]
#    k1 = int(k1); k2=int(k2)
#    s = '{},{}'.format( subfeature_order[k1] , subfeature_order[k2] )
#    csdord_strs += [s]
#print(csdord_strs)

#csdord_strs



if features_to_use == 'all':
    features_to_use = feat_dict.keys()
#features_to_use = [ 'tfr',  'H_act', 'H_mob', 'H_compl']
if len(features_to_use) < len(feat_dict.keys() ):
    print('  !!! Using only {} features out of {}'.format( len(features_to_use) , len(feat_dict.keys() ) ))


##########################  Construct full feature vector

X = []
feat_order = []
feature_names_all = []
for feat_name in features_to_use:
    #if feat_name == 'bandcorrel':  # this we add later
    #    continue
    curfeat = feat_dict[feat_name]
    cd = curfeat['data']
    print(feat_name, cd.shape)
    sfo = curfeat['names']
    #if feat_name == 'con':
    #    sfo = csdord_strs
    #else:
    #    sfo = chnames_tfr
    subfeats = ['{}_{}'.format(feat_name,sf) for sf in sfo]
    feature_names_all += subfeats

    assert len(subfeats) == len(cd)
    X += [ cd]
    feat_order += [feat_name]
X = np.vstack(X).T
print(feat_order, X.shape)


feature_names_all_ = []
for feat_name in feature_names_all:
    tmp = feat_name.replace('_allf','')
    feature_names_all_ += [tmp]
feature_names_all = feature_names_all_





##################################  Scale correl features
#for feat_name in feat_dict:
#    curfeat = feat_dict[feat_name]
#    inp = curfeat['data']
#    centering = curfeat.get('centering', True)
#    pct = curfeat['pct']
#    if pct is not None:
#        scaler = RobustScaler(quantile_range=pct, with_centering=centering)
#        scaler.fit(inp.T)
#        outd = scaler.transform(inp.T)
#        curfeat['data'] = outd.T
#        cnt = scaler.center_
#        scl = scaler.scale_
#        if feat_name in ['tfr', 'con']:
#            if cnt is not None:
#                cnt = np.min(cnt), np.max(cnt), np.mean(cnt), np.std(cnt)
#            if scl is not None:
#                scl = np.min(scl), np.max(scl), np.mean(scl), np.std(scl)
#        print(feat_name, curfeat['data'].shape, cnt,scl)

######################################

#if do_plot_feat_stats_full:
#    # Plot feature stats
#    utsne.plotBasicStatsMultiCh(Xfull.T, feature_names_all, printMeans = 0)
#
#    plt.tight_layout()
#    pdf.savefig()
#    plt.close()
#
#if do_plot_feat_timecourse_full:
#    print('Starting plotting timecourse of features' )
#    for int_name in int_names:
#        #nt_name = 'trem_{}'.format(maintremside)
#        #intervals = ivalis[int_name]
#        intervals = ivalis.get(int_name,[])
#        for iv in intervals:
#            start,end,_itp = iv
#
#            tt = utsne.plotIntervalData(Xfull.T,feature_names_all,iv,
#                                        times = Xtimes_full,
#                                        plot_types=['timecourse'],
#                                        dat_ext = extdat[:,nedgeBins:-nedgeBins],
#                                        extend=extend)
#
#            pdf.savefig()
#            plt.close()
#

# vec_features can have some NaNs since pandas fills invalid indices with them
assert np.any( np.isnan ( X ) ) == False
assert X.dtype == np.dtype('float64')



