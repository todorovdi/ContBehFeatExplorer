have_feats = False
try:
    len(X)
except NameError as e:
    have_feats = False
else:
    have_feats = True

if not (have_feats and use_existing_feat):
    if os.path.exists(fname_feat_full):
        print('Loading feats from ',fname_feat_full)
        f = np.load(fname_feat_full)
        Xfull_almost =  f['X']  # more or less full
        Xtimes_almost = f['Xtimes']
        #rawtimes = f['rawtimes']
        skip_ =f['skip']
        windowsz_ =f['windowsz']
        nedgeBins = f['nedgeBins']
        feature_names_all = f['feature_names_all']

        assert windowsz_ == windowsz
        assert skip_ == skip_feat
    else:
        raise ValueError('{} not exists!'.format(fname_feat_full) )
else:
    print('Using existing features')

X = Xfull_almost[::subskip]

int_names = ['trem_{}'.format(mts_letter), 'notrem_{}'.format(mts_letter)]
int_names += ['{}_{}'.format(task,mts_letter) for task in tasks]


####################  Load emg
EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
if use_main_tremorside:
    chnames_emg = EMG_per_hand[maintremside]
else:
    chnames_emg = raw_emgrectmav.ch_names
print(chnames_emg)

rectconv_emg, ts_ = raw_emgrectmav[chnames_emg]
chnames_emg = [chn+'_rectconv' for chn in chnames_emg]

extdat = rectconv_emg  # for subsampled features it does not make sense to add ecg
extnames = chnames_emg



#############################

if do_plot_feat_stats:
    print('Starting plotting stats of features' )
    #  Plots stats for subsampled data
    utsne.plotBasicStatsMultiCh(X.T, feature_names_all, printMeans = 0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# Plot evolutions for subsampled data
if do_plot_feat_timecourse:
    print('Starting plotting timecourse of subsampled features' )

    for int_name in int_names:
        intervals = ivalis.get(int_name,[])
        for iv in intervals:
            start,end,_itp = iv

            tt = utsne.plotIntervalData(X.T,feature_names_all,iv,
                                        times = Xtimes,
                                        plot_types=['timecourse'],
                                        dat_ext = extdat[:,nedgeBins:-nedgeBins:totskip],
                                        extend=extend)

            pdf.savefig()
            plt.close()
