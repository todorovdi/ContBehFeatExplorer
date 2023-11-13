def test_gatherFeatStats():
    dat_t = np.random.uniform(low=-1,high=1,size=(2,300))
    nbins_artif = 40
    dat_artif_t = np.random.uniform(low=10,high=40,size=(2,nbins_artif))
    dat_t[:,:nbins_artif] = dat_artif_t

    #print( dat_t[:,0:bins_t[-1]].shape )
    dat_t[0,:4] = 1e4

    dat_t_T_pri = [dat_t.T]
    sfo_t = ['LFPL01', 'LFPR22']
    times_t_pri = [np.arange(dat_t.shape[-1])/sfreq]
    rawnames_t = ['S05_on_move']
    bins_t = np.arange(0,int(0.5*sfreq) )
    bindict_per_rawn_t = {rawnames_t[0]:{'artif':{'LFP':{'BAD_LFPL01':np.arange(nbins_artif)}, 'msrc':{}},
        'beh_state':{'notrem_R':bins_t } }   }

    #display(dat_t.mean(axis=-1))
    #display(utsne.robustMean(dat_t, axis=-1, per_dim = 1 ) )

    #display('naive interval',dat_t[:,0:bins_t[-1]+1].mean(axis=-1))
    #display('robust interval',utsne.robustMean(dat_t[:,0:bins_t[-1]+1], axis=-1, per_dim = 1 ) )

    #display('naive interval noartif',dat_t[:,nbins_artif:bins_t[-1]+1].mean(axis=-1))
    #display('robust interval noartif',utsne.robustMean(dat_t[:,nbins_artif:bins_t[-1]+1], axis=-1, per_dim = 1 ) )

    int_types_to_gather_stats_t = ['notrem_R']

    #%debug
    combine_type = 'no'
    side_switched_pri_t = [True]
    artif_handling = 'reject'
    indsets, means, stds = \
            upre.gatherFeatStats(rawnames, dat_t_T_pri, sfo_t, None, sfreq, times_t_pri,
                    int_types_to_gather_stats_t, side_rev_pri = side_switched_pri_t,
                    combine_within = combine_type, minlen_bins = 5,
                            artif_handling=artif_handling, printLog=1, require_all_intervals_present = 0,
                                bindict_per_rawn = bindict_per_rawn_t)




    #display ( indsets)
    #display('means', means, 'std',stds )

    #########################

    assert indsets == [[0]]

    # total mean
    assert dat_t.mean(axis=-1)[0] > means[0]['notrem_R'][0]
    # naive interval mean
    assert dat_t[:,0:bins_t[-1]+1].mean(axis=-1)[0] >  means[0]['notrem_R'][0]

    assert utsne.robustMean(dat_t[:,nbins_artif:bins_t[-1]+1], axis=-1, per_dim = 1)[0]  == means[0]['notrem_R'][0]
