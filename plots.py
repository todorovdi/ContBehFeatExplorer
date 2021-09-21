import numpy as np
import matplotlib.pyplot as plt

def plotStepsWbd(dat_stepped, wbd, sfreq, ax=None):
    # to compare windowing with actual data
    n = wbd.shape[1]
    skip = wbd[0,1] - wbd[0,0]
    wsz = wbd[1,0] - wbd[0,0]
    ymax = np.max(dat_stepped)
    ymin = np.min(dat_stepped)
    if ax is None:
        ax =plt.gca()
    assert len(dat_stepped) == wbd.shape[1], ( len(dat_stepped) , wbd.shape[1]  )
    for i in range(n):
        #plt.fill_betweenx( [ymin,ymax], wbd[0,i], wbd[1,i], color='yellow', alpha=0.1)
        ax.axvline(wbd[0,i] , ls=':', c='y')
        ax.plot( [i * skip,i * skip + wsz], [dat_stepped[i], dat_stepped[i]] )

    plt.fill_betweenx( [ymin,ymax], wbd[0,n-1], wbd[1,n-1], color='yellow', alpha=0.1)

def shadeAnn(ax,anns,ymin,ymax,color='red',alpha=0.2, sfreq=256, skip=32, plot_bins = 0,
             shift_by_anntype = 1, seed=0, ann_color_dict = None,
             intervals_to_plot = None, printLog=False ):
    '''
    plot_bins means X axis is assumed to be bin number (and not times)
    intervals_to_plot -- have sides
    '''
    cmap = plt.cm.get_cmap('tab20', 20)
    #start,end pairs
    descrs = list(dict.fromkeys(anns.description))  # preserveas ordering
    if len(descrs) == 0:
        if printLog:
            print('shadeAnn: empty annotation, exiting')
        return {},[]
    if intervals_to_plot is not None:
        #descrs = [d  for d in descrs if d in intervals_to_plot ]
        descrs = [d for d in intervals_to_plot if d in descrs ]

    #print('shadeAnn lim ',ymin,ymax)
    #print( len(descrs), len(anns) )
    np.random.seed(seed)
    ri0 = np.random.randint(1e6)
    ri1 = np.random.randint(1e6)
    attrs_per_descr = {}
    if shift_by_anntype:
        height = (ymax - ymin ) / len(descrs)
        for i,descr in enumerate(descrs):
            attr = {}
            if ann_color_dict is None:
                attr['color'] = cmap( (ri0 + i*ri1) % 20)
            else:
                #print('fff ',descr)
                attr['color'] = ann_color_dict[descr]
                #attr['color'] = #cmap( (ri0 + i*ri1) % 20)
            #attr['ylim'] = ymin + i * height, ymin + (i+1) * height
            #attr['ylim'] = ymax - attr['ylim'][1], ymax - attr['ylim'][0]
            attr['ylim'] = ymax - (i+1) * height, ymax - i * height
            attrs_per_descr[descr] = attr
            #print(descr, attr['ylim'])
            #cmap(3)

    #intervals_se_pairs = list( zip( anns.onset, anns.onset + anns.duration ) )
    #for pa in intervals_se_pairs:
    descrs_were_shown = []
    for ann in anns:
        pa = ann['onset'],ann['onset'] + ann['duration']
        descr = ann['description']
        if (intervals_to_plot is not None) and (descr not in intervals_to_plot):
            continue
        if plot_bins:
            pa = np.array(list(pa)) * sfreq // skip
        attr = attrs_per_descr.get(descr,None)
        if attr is not None:
            color_cur = attr['color']
            ymin_cur, ymax_cur =  attr['ylim']
        #ax.fill_between( list(pa) , ymin_cur, ymax_cur, facecolor=color_cur, alpha=alpha)

        # I don't want duplicate entries in the legend if we have several intervals of same type
        if descr in descrs_were_shown:
            lab = None
        else:
            lab = descr
            #print('fff__ ',descr)
            descrs_were_shown += [descr]

        ax.fill_between( list(pa) , ymin_cur, ymax_cur,
                        facecolor=color_cur, alpha=alpha, label=lab)

    #assert tuple(descrs) == tuple(descrs_were_shown)
    assert set(descrs) == set(descrs_were_shown)

    #return attrs_per_descr, descrs_were_shown
    return attrs_per_descr, descrs


def plotMeansPerIt(ax,anns,means_per_it,chi, sfreq=256, plot_bins=0,
                   alpha=0.5,ls=':', attrs_per_descr=None, c=None,lw=None,
                   printLog = False):
    ctr = 0
    for ann in anns:
        pa = ann['onset'],ann['onset'] + ann['duration']
        descr = ann['description']

        if means_per_it is not None:
            if descr not in means_per_it:
                if printLog:
                    print(f'plotMeansPerIt: Warning: {descr} not in means_per_it')
                continue
            ctr += 1
            me = means_per_it[descr][chi]
            ys = [me,me]
            xs = np.array( list(pa) )
            if plot_bins:
                xs *= sfreq
            color = None
            if attrs_per_descr is not None:
                color = attrs_per_descr[descr]['color']
            elif c is not None:
                color = c
            ax.plot(xs,ys, ls=ls, alpha=alpha, c=color, lw=lw)
    if ctr == 0:
        print(f'plotMeansPerIt: nothing was plotted because {set(anns.description)} are not part of  {means_per_it.keys()}')


def plotDataAnnStat(rawnames,dat_pri,times_pri,chnames_pri,
                   dat_hires_pri=None,times_hires_pri=None,chnames_hires_pri=None,
                   anndict_per_intcat_per_rawn=None,
                   indsets=None,means_per_iset=None,suptitle='', sfreq=256,
                   dat_dict=None,band=None,pdf=None, legend_loc = 'lower right',
                    chis_to_show = None, q_thr = 1e-3, mult_std = 3.5,
                    artif_height_prop = 0.3, ww=6, hh=2):

    import utils_tSNE as utsne
    import matplotlib.pyplot as plt
    import utils_preproc as upre
    indset_mask,_ = upre.getIndsetMask(indsets)
    dat_to_plot = 0

    if chis_to_show is None:
        if (band is not None) and band.startswith('HFO'):
            chis_to_show = dat_hires_pri[0].shape[0]
        else:
            chis_to_show = dat_pri[0].shape[0]

    nr = len(chis_to_show)
        #nr = dat_pri[0].shape[0]
    nc = len(rawnames)
    fig,axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh))
    if not isinstance(axs,np.ndarray):
        axs = np.array( [[axs]] )
    axs = axs.reshape((nr,nc))
    for rawi,rawn in enumerate(rawnames):
        for i in range(nr):
            ax = axs[i,rawi]

            chi = chis_to_show[i]
            if (band is not None) and band.startswith('HFO') and chi >= len(chnames_hires_pri[rawi]):
                continue

            if dat_dict is not None:
                dat_to_plot = dat_dict[rawi][band][chi][0][0]
                times = times_pri[rawi]  # EVEN IF HFO
                ax.plot(times, dat_to_plot ,c='purple',alpha=0.5)


            if (band is not None) and band.startswith('HFO'):
                #dat_LFP_hires_pri=None,times_hires_pri=None,subfeature_order_hires_pri=None,
                times = times_hires_pri[rawi]
                print(chi)
                dat_to_plot0 = dat_hires_pri[rawi][chi]
                chn = chnames_hires_pri[rawi][chi]
                #print("BBBBBBBBAND ", band)
            else:
                times = times_pri[rawi]
                dat_to_plot0 = dat_pri[rawi][chi]
                chn = chnames_pri[rawi][chi]

            #print(rawn,i)
            #mx1,mx0 = np.max(dat_to_plot), np.max(dat_to_plot0)
            #mn1,mn0 = np.min(dat_to_plot), np.min(dat_to_plot0)
            rawc='g'
            q_list = [q_thr,0.5,1-q_thr]
            mn0,me0,mx0 = np.quantile(dat_to_plot0, q_list)
            #std0 = np.std(dat_to_plot0)

            me0,std0 = utsne.robustMean(dat_to_plot0,ret_std=1)
            if band is not None:
                mn1,me1,mx1 = np.quantile(dat_to_plot, q_list)
                std1 = np.std(dat_to_plot)
                mn1 = min(0,mn1) # if positive..

                ax.plot(times, (dat_to_plot0 - mn0) / (mx0-mn0) * (mx1-mn1) + me1  ,c=rawc,alpha=0.1)
                #ax.plot(times, dat_to_plot0  ,c=rawc,alpha=0.1)

                mn = mn1 - abs(mn1) * 0.2
                mx = mx1 + abs(mx1) * 0.2
                me = me1
                std = std1
            else:
                mx = mx0
                mn = mn0
                me = me0
                std = std0

                ax.plot(times, dat_to_plot0  ,c=rawc,alpha=0.1)


            ann = anndict_per_intcat_per_rawn[rawn]['beh_state']
            if chn.startswith('msrc'):
                ann_artif = anndict_per_intcat_per_rawn[rawn]['artif']['MEG']
            else:
                ann_artif = anndict_per_intcat_per_rawn[rawn]['artif']['LFP']
                #print(ann_artif)

            iseti0 = np.where( [ (rawi in iset) for iset in indsets ] )[0][0]
            iseti = indset_mask[rawi]
            assert iseti0 == iseti
            #means_per_it = stats_multiband_flt_per_ct[ct]['stats_per_indset'][band][iseti][0] #['notrem_L']

            means_per_it = means_per_iset[iseti]  #['notrem_L']

            #shadey_aftif = (mn, (mn+mx)/2); shadey_beh_state = ( (mn+mx)/2, mx)
            m = mult_std
            spl = (mn+mx)/2; shadey_aftif = [mn - abs(std)*m, spl]; shadey_beh_state = [ spl, mx + abs(std)*m]

            d = (shadey_aftif[1] - shadey_aftif[0])
            shadey_aftif[1] = shadey_aftif[0] + d * artif_height_prop / 0.5
            shadey_beh_state[0] = shadey_aftif[1]
            #shadey_beh_state[0] = d * (1-artif_height_prop)
            #spl = me; shadey_aftif = (me - std, spl); shadey_beh_state = ( spl, me+std)
            ax.set_title(f'{rawn} : {chn}')
            attrs = shadeAnn(ax,ann,*shadey_beh_state,color='red',alpha=0.4, sfreq=sfreq, skip=1, plot_bins = 0,
                     shift_by_anntype = 1, seed=1)
            shadeAnn(ax,ann_artif,*shadey_aftif,color='red',alpha=0.4, sfreq=sfreq, skip=1, plot_bins = 0,
                     shift_by_anntype = 1, seed=4)
            plotMeansPerIt(ax,ann,means_per_it,chi, c='red', alpha=1.,lw=3)#,attrs_per_descr = attrs)
            ax.set_ylim( shadey_aftif[0], shadey_beh_state[1] )
            ax.legend(loc=legend_loc)
    plt.suptitle(f'{suptitle}')
    if pdf is not None:
        pdf.savefig()
        plt.close()


def plotFeatsAndRelDat(rawnames,featnames_sel, dat_pri,chnames_all_pri,
                       X_pri,featnames_all_pri,
                       times_pri=None,Xtimes_pri=None, chnames_all_newsrcgrp_pri = None,
                       wbd_pri=None, extdat_dict = None,
                      dat_hires_pri=None,chnames_all_hires_pri=None,times_hires_pri=None,
                      anndict_per_intcat_per_rawn = None, artif_height_prop=0.3, sfreq=None,
                      legend_loc = 'lower right', ww=6, hh=2, mainLFP_per_rawn = None, roi_labels=None,
                       srcgrouping_names_sorted = None, alpha=0.5, main_side_let = None,
                       legend_alpha=0.5, legend_alpha_artif=0.8, beh_states_to_shade=None,
                      extdat_pri = None, xlim=None, main_color='purple', feat_comments=None ):
    from featlist import parseFeatNames
    r = parseFeatNames(featnames_sel)
    chnames_involved = [chn for chn in (r['ch1'] + r['ch2']) if chn is not None]
    chnames_involved = list(set(chnames_involved))
    assert len(chnames_involved) > 0
    print(chnames_involved)

    for X,featns in zip(X_pri,featnames_all_pri):
        assert X.shape[1] == len(featns), ( X.shape[1] ,len(featns) )

    if feat_comments is not None:
        assert len(feat_comments) == len(featnames_sel)

    import utils

    all_it = []
    all_it_a1 = []
    all_it_a2 = []
    for rawn in rawnames:
        ann = anndict_per_intcat_per_rawn[rawn]['beh_state']
        all_it += list(ann.description)

        ann = anndict_per_intcat_per_rawn[rawn]['artif'].get('LFP',None)
        if ann is not None:
            all_it_a1 += list(ann.description)

        ann = anndict_per_intcat_per_rawn[rawn]['artif'].get('MEG',None)
        if ann is not None:
            all_it_a2 += list(ann.description)
    all_it = list(dict.fromkeys(all_it))
    all_it_a1 = list(dict.fromkeys(all_it_a1))
    all_it_a2 = list(dict.fromkeys(all_it_a2))
    all_it = all_it + all_it_a1 + all_it_a2

    ann_color_dict = {}
    cmap = plt.cm.get_cmap('tab20', 20)
    for iit, it in enumerate(all_it):
        ann_color_dict[it] = cmap(iit % 20)
    #from globvars import gp
    # global color code, indep of function arg values
    #sided_int_types = [it + '_L' for it in gp.int_types_ext]
    ##[it + '_R' for it in gp.int_types_ext]
    #for iit, it in enumerate(sided_int_types):
    #    attrs_per_descr[it] = cmap()


    nr = len(chnames_involved) + len(featnames_sel)
    hratios = [0.75] * len(chnames_involved) + [1] * len(featnames_sel)
    if extdat_pri is not None:
        hratios = [0.4] + hratios
        nr += 1
        #nr = dat_pri[0].shape[0]
    nc = len(rawnames)
    fig,axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh), gridspec_kw={'height_ratios': hratios}  )
    axs = axs.reshape((nr,nc))


    rowi_offset = 0
    if extdat_pri is not None:
        rowi_offset = 1

        for rawi,rawn in enumerate(rawnames):
            ax = axs[0, rawi]
            ts = times_pri[rawi]
            extdat = extdat_pri[rawi]
            for chni in range(extdat.shape[0]):
                ax.plot(ts,extdat[chni,:] )

            ax.set_xlim(ts[0],ts[-1])
            ax.set_title(f'{rawn} EMG')

            if xlim is not None:
                ax.set_xlim(xlim)

        #for extdat_type,ed in extdat_dict:
        #    nr = 0
        #    if extdat_type in ['flt','bp']:
        #        for band in ed:
        #            #nr += len(ed[band].ch_names)
        #            nr += len(chnames_involved)
        #        #ts = times_pri[rawi]
        #        #d = raw_perband_bp_pri[rawi][band]._data[chni]
        #        #plt.plot(ts,d)

        #    nr = len(chnames_involved) + len(featnames_sel)
        #        #nr = dat_pri[0].shape[0]
        #    nc = len(rawnames)
        #    ww = 6; hh = 2
        #    fig,axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh))
        #    axs = axs.reshape((nr,nc))



    print('Plotting rawdata')
    # plot raw data
    for rawi,rawn in enumerate(rawnames):
        print(rawn)
        # LFP main chan would be 007
        for chni,chn in enumerate(chnames_involved):
            ax = axs[chni + rowi_offset, rawi]

            if chn.startswith('LFP') and mainLFP_per_rawn is not None:
                mainLFP_cur = mainLFP_per_rawn[rawn]
                chn = mainLFP_cur

            if chnames_all_newsrcgrp_pri is not None:
                orig_ind  = chnames_all_newsrcgrp_pri[rawi].index(chn)
            else:
                orig_ind  = chnames_all_pri[rawi].index(chn)


            if chn.startswith('LFP') and dat_hires_pri is not None:
                ax.plot(times_hires_pri[rawi], dat_hires_pri[rawi][chnames_all_hires_pri[rawi].index(chn)], alpha=alpha, c=main_color )
                ax.set_xlim( (times_hires_pri[rawi][0],times_hires_pri[rawi][-1] ) )
            else:
                ax.plot(times_pri[rawi], dat_pri[rawi][orig_ind],  alpha=alpha, c=main_color)
                ax.set_xlim( (times_pri[rawi][0],times_pri[rawi][-1] ) )

            if xlim is not None:
                ax.set_xlim(xlim)




            descr_order = None
            if anndict_per_intcat_per_rawn is not None:
                ann = anndict_per_intcat_per_rawn[rawn]['beh_state']
                if chn.startswith('msrc'):
                    ann_artif = anndict_per_intcat_per_rawn[rawn]['artif']['MEG']
                else:
                    ann_artif = anndict_per_intcat_per_rawn[rawn]['artif']['LFP']

                mn,mx = ax.get_ylim()
                shadey_aftif     = [mn, (mn+mx) * 0.5   ]
                shadey_beh_state = [shadey_aftif[0] , mx]

                d = (shadey_aftif[1] - shadey_aftif[0])
                shadey_aftif[1] = shadey_aftif[0] + d * artif_height_prop / 0.5
                shadey_beh_state[0] = shadey_aftif[1]
                ax.axhline(y=shadey_beh_state[0], ls=':' )

                attrs1, descr_order1 = shadeAnn(ax,ann,*shadey_beh_state,color='red',
                                                alpha=legend_alpha, sfreq=sfreq, skip=1, plot_bins = 0,
                        shift_by_anntype = 1, seed=1,
                        ann_color_dict=ann_color_dict,
                        intervals_to_plot = beh_states_to_shade)
                attrs2, descr_order2 = shadeAnn(ax,ann_artif,*shadey_aftif,color='red',
                                                alpha=legend_alpha_artif, sfreq=sfreq, skip=1, plot_bins = 0,
                        shift_by_anntype = 1, seed=4,ann_color_dict=ann_color_dict)
                descr_order = descr_order1 + descr_order2


            if roi_labels is not None:
                chn = utils.nicenMEGsrc_chnames([chn],roi_labels,srcgrouping_names_sorted)
                chn = chn[0]
            ax.set_title(f'{rawn} : {chn}')


            ax.legend(loc=legend_loc)
            if descr_order is None:
                ax.legend(loc=legend_loc)
            else:
                # default ordering of labels in legend is weird, I want it
                # consistent
                handles, labels = ax.get_legend_handles_labels()
                assert tuple(sorted(labels) ) ==  tuple(sorted(descr_order )), \
                    (sorted(labels), sorted(descr_order ))
                handles_reord = [ handles[labels.index(lbl)] \
                                 for lbl in descr_order ]

                # just in case, not really needed
                #p = [ (handles[labels.index(lbl)],lbl) \
                #                 for lbl in all_it if lbl in labels ]
                #handles2, labels2 = zip(*p)

                # sort both labels and handles by labels
                #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                ax.legend(handles_reord, descr_order, loc=legend_loc)

    # plot features
    print('Plotting features')
    rowi_offset += len(chnames_involved)
    for rawi,rawn in enumerate(rawnames):
        print(rawn)
        for feati,featn in enumerate(featnames_sel):
            ax = axs[feati + rowi_offset, rawi ]
            ts = Xtimes_pri[rawi]
            if wbd_pri is not None:
                ts = wbd_pri[rawi][1]
                ts = times_pri[rawi][ts]
            #print(ts,ts.shape,wbd_pri[rawi].shape)
            feati_true = list(featnames_all_pri[rawi]).index(featn)
            ax.plot(ts, X_pri[rawi] [:,feati_true],  alpha=alpha, c=main_color)

            descr_order = None
            if anndict_per_intcat_per_rawn is not None:
                ann = anndict_per_intcat_per_rawn[rawn]['beh_state']

                mn,mx = ax.get_ylim()
                shadey_aftif     = [mn, (mn+mx) * 0.5   ]
                shadey_beh_state = [shadey_aftif[0] , mx]

                d = (shadey_aftif[1] - shadey_aftif[0])
                shadey_aftif[1] = shadey_aftif[0] + d * artif_height_prop / 0.5
                shadey_beh_state[0] = shadey_aftif[1]
                ax.axhline(y=shadey_beh_state[0], ls=':' )

                attrs, descr_order  = shadeAnn(ax,ann,*shadey_beh_state,color='red',alpha=legend_alpha, sfreq=sfreq, skip=1, plot_bins = 0,
                        shift_by_anntype = 1, seed=1,
                        ann_color_dict=ann_color_dict,
                        intervals_to_plot = beh_states_to_shade)
                if featn.find('msrc') >= 0:
                    ann_artif = anndict_per_intcat_per_rawn[rawn]['artif']['MEG']
                    _,descr_order_cur = shadeAnn(ax,ann_artif,*shadey_aftif,
                            color='red',alpha=legend_alpha_artif, sfreq=sfreq, skip=1, plot_bins = 0,
                            shift_by_anntype = 1, seed=4, ann_color_dict=ann_color_dict)
                    descr_order += descr_order_cur
                if featn.find('LFP') >= 0:
                    ann_artif = anndict_per_intcat_per_rawn[rawn]['artif']['LFP']
                    _,descr_order_cur = shadeAnn(ax,ann_artif,*shadey_aftif,
                            color='red',alpha=legend_alpha_artif, sfreq=sfreq, skip=1, plot_bins = 0,
                            shift_by_anntype = 1, seed=4, ann_color_dict=ann_color_dict)
                    descr_order += descr_order_cur


            if roi_labels is not None:
                featn = utils.nicenFeatNames([featn],roi_labels,srcgrouping_names_sorted)
                featn = featn[0]
            ttl = f'{rawn} : {featn}'
            if feat_comments is not None and len(feat_comments[feati] ):
                ttl += '\n' + feat_comments[feati]
            ax.set_title(ttl)

            ax.set_xlim(0,ts[-1])
            if xlim is not None:
                ax.set_xlim(xlim)

            if descr_order is None:
                ax.legend(loc=legend_loc)
            else:
                handles, labels = ax.get_legend_handles_labels()
                assert tuple(sorted(labels) ) ==  tuple(sorted(descr_order )), \
                    (sorted(labels), sorted(descr_order ))
                handles_reord = [ handles[labels.index(lbl)] \
                                 for lbl in descr_order ]
                # sort both labels and handles by labels
                #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                ax.legend(handles, descr_order, loc=legend_loc)

    # all data

    # plot raw data
    # plot features
    #print('plotting')

    return axs



def plotTFRlike(dat_pri, dat_hires_pri,  tfrres_pri, tfrres_HFO_pri, csd_pri,
                bpow_abscsd_pri,
                wbd, feat_dict,
                times_pri, times_hires_pri,
                subfeature_order_pri, subfeature_order_newsrcgrp_pri,
                csdord_pri, csdord_strs_pri,
                bands,chns,
                freqs, sfreq,
                normalize=True):

    from featlist import selFeatsRegexInds
    import globvars as gv
    rawi = 0
    Xtimes = wbd[rawi][1] / sfreq

    #chn = subfeature_order[chi]
    freqis = {}
    for band in bands:
        freqis_cur = np.where( (freqs >= gv.fbands[band][0]) & (freqs <= gv.fbands[band][1]) )[0]
        freqis[band] = freqis_cur


    numLFPchs = len( selFeatsRegexInds(chns, '^LFP.*') )

    from scipy.special import  binom
    nplots_main = int( binom(len(chns), 2) ) + int( binom(len(chns), 1) )

    #nlowfreq = np.where( (n_cycles / freqs) < 2  )[0][0]
    #lowfreqbd = freqs[nlowfreq]
    N = dat_pri[0].shape[-1]
    print(tfrres_pri[0].shape)
    nc = len(chns) + nplots_main * len(bands) * 2 + numLFPchs

    fig,axs = plt.subplots(nc,1,figsize=(12,2*nc)) #, sharex='col')
    axind = 0


    ##################################################
    for band in bands:
        chis = []
        for chn in chns:
            chi = subfeature_order_pri[rawi].index(chn)
            chis += [chi]

            if band != 'HFO':
                ax = axs[axind];
                axind += 1
                for fi in freqis[band]:
                    d = np.abs( tfrres_pri[rawi][chi,fi] )
                    if normalize:
                        d /= np.std(d)
                    ax.plot(Xtimes,  d)
                ax.set_title(f'{rawi} TFR {chn} {band}')

                # recall that for LFP we have put hires TFR in normal TFR
                if chn.startswith('LFP'):
                    a,b = ax.get_ylim(); rng = b-a;
                    d = dat_hires_pri[rawi][chi]; mn,mx = np.min(d),np.max(d); ts = times_hires_pri[rawi]
                    ax.plot(ts, (d -mn) / (mx-mn) * rng + a, alpha = 0.5 )
                else:
                    a,b = ax.get_ylim(); rng = b-a; d = dat_pri[rawi][chi]; mn,mx = np.min(d),np.max(d); ts = times_pri[rawi]
                    assert mx - mn >= 1e-14
                    #if mx - mn <= 1e-10:
                    #    d = dat_hires_pri[rawi][chi]; mn,mx = np.min(d),np.max(d); ts = times_hires_pri[rawi]
                    ax.plot(ts, (d -mn) / (mx-mn) * rng + a, alpha = 0.5 )
                    print(a,b)

            if chn.startswith('LFP') and band == 'HFO':
                ax = axs[axind];
                axind += 1
                for fi in freqis[band]:
                    d = np.abs( tfrres_HFO_pri[rawi][chi,fi] )
                    if normalize:
                        d /= np.std(d)
                    ax.plot(Xtimes,  d)
                ax.set_title(f'{rawi} TFR hires {chn} {band}')
                a,b = ax.get_ylim(); rng = b-a;
                d = dat_hires_pri[rawi][chi]; mn,mx = np.min(d),np.max(d); ts = times_hires_pri[rawi]
                ax.plot(ts, (d -mn) / (mx-mn) * rng + a, alpha = 0.5 )

    ###################################
#     delim = 6
#     for chn in chns:
#         chi = subfeature_order_pri[rawi].index(chn)
#         chis += [chi]

#         ax = axs[axind];
#         axind += 1
#         for fi in freqis[:delim]:
#             ax.plot(Xtimes, np.abs( tfrres_pri[rawi][chi,fi] * np.conj(tfrres_pri[rawi][chi,fi] ) ) )
#         ax.set_title(f'{rawi} TFR {chn} {band}')
#         a,b = ax.get_ylim(); rng = b-a
#         #ax.plot(times_pri[rawi], dat_pri[rawi][chi] / rng, alpha = 0.5 )
#         print(a,b)

#     for chn in chns:
#         chi = subfeature_order_pri[rawi].index(chn)
#         chis += [chi]

#         ax = axs[axind];
#         axind += 1
#         for fi in freqis[delim:]:
#             ax.plot(Xtimes, np.abs( tfrres_pri[rawi][chi,fi] * np.conj(tfrres_pri[rawi][chi,fi] ) ) )
#         ax.set_title(f'{rawi} TFR {chn} {band}')
#         a,b = ax.get_ylim(); rng = b-a
#         #ax.plot(times_pri[rawi], dat_pri[rawi][chi] / rng, alpha = 0.5 )
#         print(a,b)

    #########################

    mask1 = np.in1d(csdord_pri[rawi][0], chis)
    mask2 = np.in1d(csdord_pri[rawi][1], chis)
    mask = mask1 & mask2 #& (csdord_pri[rawi][1] != csdord_pri[rawi][0] )


    #for chn in chns:
    #chi = subfeature_order_pri[rawi].index(chn)
    csdis_rel = np.unique ( np.where( mask )[0] )

    for band in bands:
        for csdi in csdis_rel:
            ax = axs[axind];
            axind += 1
            for fii,fi in enumerate(freqis[band]):
                chi1,chi2 = csdord_pri[rawi][:,csdi]
                if fii == 0:
                    # this is not guraranteed to be true {csdord_strs_pri[rawi][csdi]}
                    lab = f'{subfeature_order_newsrcgrp_pri[rawi][chi1],subfeature_order_newsrcgrp_pri[rawi][chi2]} '
                else:
                    lab = ''
                ax.plot(Xtimes, np.abs( csd_pri[rawi][csdi,fi] ), label=lab )
            ax.set_title(f'{rawi} CSD  {chns} {band}')
            ax.legend(loc='lower right')
        a,b = ax.get_ylim(); rng = b-a
        #ax.plot(times_pri[rawi], dat_pri[rawi][chi] / rng, alpha = 0.5 )
        print(a,b)



    chi = subfeature_order_pri[rawi].index(chn)
    newchn = subfeature_order_newsrcgrp_pri[rawi][chi]

    csdis_rel = np.unique ( np.where( (csdord_pri[rawi][0] == chi) | (csdord_pri[rawi][1] == chi) )[0] )

#     bpow_abscsd = bpow_abscsd_pri[rawi]
#     ncsds,nfreqs,ntimebins_ = bpow_abscsd.shape
#     bpow_abscds_reshaped = bpow_abscsd.reshape( ncsds*nfreqs, ntimebins_ )

    chn_str =  '|'.join( [subfeature_order_newsrcgrp_pri[rawi][chi] for chi in chis] )
    #chn_str2=  '|'.join(chns[::-1])
    for band in bands:
        ax = axs[axind];
        axind += 1

        regex = f'{band}_({chn_str}),({chn_str})'
        print(regex)
        inds = selFeatsRegexInds( feat_dict['con']['names'], regex )
        #ax = axs[axind]; axind += 1

        #for csdi in csdis_rel:
        #for feati in range(0,5):
        #for feati in range(bpow_abscds_reshaped.shape[0]):
        for feati in inds:
            s = csdord_strs_pri[rawi][feati]
    #             if s.find(newchn) < 0 or s.find('LFPR092') < 0:
    #                 continue
            ax.plot(Xtimes, np.abs( bpow_abscsd_pri[rawi][feati] ) , label=s)
        a,b = ax.get_ylim(); rng = b-a
        #ax.plot(times_pri[rawi], dat_pri[rawi][chi] / rng, alpha = 0.5 )
        ax.set_title(f'{chn}: bpow_abscds')
        ax.legend(loc='lower right')
        #ax.fill_betweenx([0,0.2],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)


    ############
    for chn in chns:
        chi = subfeature_order_pri[rawi].index(chn)
        if chn.startswith('LFP'):
            ax = axs[axind];
            axind += 1
            # tfrres_HFO_pri already contains only HFO
            for fi in range(tfrres_HFO_pri[rawi].shape[1] ):
                ax.plot(Xtimes, np.abs( tfrres_HFO_pri[rawi][chi,fi] ) )
            a,b = ax.get_ylim(); rng = b-a
            #ax.plot(times_pri[rawi], dat_pri[rawi][chi] / rng, alpha = 0.5 )
            ax.set_title(f'{chn}: tfrres LFP HFO only')
            #ax.fill_betweenx([0,0.1],  (offset_start-windowsz)//skip,  offset_start//skip, color='red', alpha=0.15)

            a,b = ax.get_ylim(); rng = b-a;
            d = dat_hires_pri[rawi][chi]; mn,mx = np.min(d),np.max(d); ts = times_hires_pri[rawi]
            ax.plot(ts, (d -mn) / (mx-mn) * rng + a, alpha = 0.5 )

    for ax in axs:
        ax.set_xlim(times_pri[rawi][0],times_pri[rawi][-1])

    print(axind)

    plt.tight_layout()
    return locals()

def plotErrorBarStrings(ax,xs,ys,xerr, add_args={},merge_bias=True, same_sets_only=1):
    from collections.abc import Iterable
    xs = xs[:]  # copy
    assert len(xs) == len(ys),  ( len(xs), len(ys)  )
    if xerr is not None:
        assert len(xerr) == len(xs)

    labs = [a.get_text() for a in ax.get_yticklabels() ]
    lens = [len(s) for s in labs]
    maxlen = 0
    if len(lens):
        maxlen = max(lens )
    #print(maxlen)

    slabs = set(labs)   # exiting labels
    sxs = set(xs)
    if slabs != sxs and maxlen > 0:
        diff1,diff2 = slabs-sxs,sxs-slabs
        if len(diff1) == len(diff2) and list(diff1)[0].startswith('bias') \
                and list(diff2)[0].startswith('bias') and merge_bias:
            #print(xs)
            #print(xs[0] )
            xs[ xs.index(list(diff2)[0] ) ] = list(diff1)[0]
            #print(xs)
            print('xs[0] after corr = ',xs[0] )

            if np.sum( np.array(list(xs[0])) == '/' ) == 2:
                import pdb; pdb.set_trace()

        elif same_sets_only:
            print('Warning, different label sets! ', (len(set(labs)), len(set(xs))), diff1,diff2  )
            raise ValueError('bad')

    if maxlen > 0 and same_sets_only:
        assert tuple(xs) == tuple(labs)


    if not isinstance(xs[0], str):
        xs = [ str(x) for x in xs]
    color = add_args.get('color',None)
    if color is not None and isinstance(color,Iterable):
        del add_args['color']
        xerr_orig = xerr
        if xerr is None:
            xerr = [0] * len(ys)
        #olids,xs_cur,ys_cur = zip(*inds)
        for y,x,xe,c in zip( ys,xs,xerr, color ):
        #for y,x,xe,c in zip( np.array(ys)[inds],np.array(xs)[inds],xerr, color ):
            assert len(c) < 5
            if xerr_orig is not None:
                ax.errorbar([y],[x],xerr=[xe],color=c,**add_args)
            else:
                fmt = add_args.get('fmt',None)
                if fmt is not None:
                    del add_args['fmt']
                    add_args['marker'] = fmt
                ax.plot([y],[x],color=c,**add_args)
    else:
        #olids,xs_cur,ys_cur = zip(*inds)
        if xerr is not None:
            ax.errorbar(ys,xs,xerr=xerr,**add_args)
        else:
            fmt = add_args.get('fmt',None)
            if fmt is not None:
                del add_args['fmt']
                add_args['marker'] = fmt
            ax.plot(ys,xs,**add_args)

def plotErrorBarStrings_old(ax,xs,ys,xerr, add_args={},merge_bias=True):
    # too complex and works better without my weird sorting
    from collections.abc import Iterable

    labs = [a.get_text() for a in ax.get_yticklabels() ]
    xs = [ str(x) for x in xs]
    lens = [len(s) for s in labs]
    maxlen = 0
    if len(lens):
        maxlen = max(lens )
    inds_list = []
    if maxlen > 0:
        slabs = set(labs)   # exiting labels
        sxs = set(xs)
        if slabs != sxs:
            #print(sxs,slabs)
            diff1,diff2 = slabs-sxs,sxs-slabs
            print('Warning, different label sets! ', (len(set(labs)), len(set(xs))), diff1,diff2  )
            raise ValueError('achtung!')
            union = slabs | sxs
            intersect = slabs & sxs
            # indices of intersection
            inds = [ (labs.index(x),x,  ys[xs.index(x)] ) for x in intersect]
            inds_list = [inds]

            inds = [ ( max(len(inds),maxlen) + xs.index(x),x,  ys[xs.index(x)] ) for x in sxs - slabs]
            inds_list += [inds]
        else:
            inds = [ (labs.index(x),x,ys[xs.index(x)]) for x in xs]
            inds_list = [inds]
    else:
        inds = list(zip(np.arange(len(xs)),xs,ys))
        inds_list = [inds]

        #print(list(inds))

    totinds = []
    for inds in inds_list:
        totinds += list(inds)
        #ax.set_yticklabels(xs)
        if xerr is not None:
            #xerr = np.array(xerr)[inds]
            xerr = [xerr[ xs.index(x) ] for (oldi,x,y) in inds ]
        color = add_args.get('color',None)
        if color is not None and isinstance(color,Iterable):
            del add_args['color']
            if xerr is None:
                xerr = [0] * len(ys)
            olids,xs_cur,ys_cur = zip(*inds)
            for y,x,xe,c in zip( ys_cur,xs_cur,xerr, color ):
            #for y,x,xe,c in zip( np.array(ys)[inds],np.array(xs)[inds],xerr, color ):
                assert len(c) < 5
                ax.errorbar([y],[x],xerr=[xe],color=c,**add_args)
        else:
            olids,xs_cur,ys_cur = zip(*inds)
            ax.errorbar(ys_cur,xs_cur,xerr=xerr,**add_args)
            #ax.errorbar(np.array(ys)[inds],np.array(xs)[inds],xerr=xerr,**add_args)
    if maxlen == 0:
        ax.set_yticks(olids)
        ax.set_yticklabels(xs)
    else:
        olids,xs_cur,ys_cur = zip(*totinds)
        ax.set_yticks(olids)
        ax.set_yticklabels(xs_cur)

def plotFeatsWithEverything(dat_pri, times_pri, X_pri, Xtimes_pri, dat_lfp_hires_pri, times_hires_pri,
                            rawnames,
                            subfeature_order_pri, subfeature_order_newsrcgrp_pri,
                            subfeature_order_lfp_hires_pri,
                            anndict_per_intcat_per_rawn,
                            feature_names_all, wbd_H_pri,
                            sfreq, raw_perband_flt_pri, raw_perband_bp_pri,
                            scale_data_combine_type,
                            stats_multiband_flt, stats_multiband_bp,
                            test_plots_descr, special_chns,
                           fband_names_inc_HFO ):
    '''
    plots features AND raw data AND intermediate data separately for reach raw
    '''
    from os.path import join as pjoin
    import globvars as gv

    from featlist import parseFeatNames
    from featlist import selectFeatNames
    from utils import freqs2relevantBands
    featnames_parse_res = parseFeatNames( feature_names_all)
    featnames_parse_res.keys()

    #from plots import plotFeatsAndRelDat
    #from plots import plotDataAnnStat
    #from plots import shadeAnn
    #from plots import plotMeansPerIt

    #%debug
    for tpd in test_plots_descr:
    #for tpd in test_plots_descr[:1]:
    #for tpd in test_plots_descr[0:1]:
    #for tpd in test_plots_descr[1:2]:
    #for tpd in test_plots_descr[2:3]:

        from matplotlib.backends.backend_pdf import PdfPages
        #fign = rawnstr + '__rawdata_vs_feat'
        fign = tpd['figname'] + '__rawdata_vs_feat'
        figfn_full = pjoin(gv.dir_fig, fign+'.pdf')
        #plt.savefig(figfn_full)
        pdf= PdfPages(figfn_full  )


        ds = tpd['chn_descrs']
        print('   ',tpd['informal_descr'])
        desired_chns = [special_chns[chnd] for chnd in ds]
        fts = tpd['feat_types_actual_coupling']
        relevant_freqs = tpd['relevant_freqs']
        print(ds,desired_chns)
        print(fts,relevant_freqs)

        featnames_sel = selectFeatNames(fts,relevant_freqs,desired_chns, feature_names_all,
                                        fband_names=fband_names_inc_HFO)
        print(featnames_sel)

        # select features related to these channels
        plotFeatsAndRelDat(rawnames, featnames_sel, dat_pri,subfeature_order_pri,
                        X_pri,[feature_names_all]*len(rawnames),times_pri,Xtimes_pri,
                        subfeature_order_newsrcgrp_pri, wbd_H_pri,
                        dat_hires_pri=dat_lfp_hires_pri,
                        chnames_all_hires_pri = subfeature_order_lfp_hires_pri,
                        times_hires_pri=times_hires_pri,
                        anndict_per_intcat_per_rawn=anndict_per_intcat_per_rawn, sfreq=sfreq )
        plt.tight_layout()
        pdf.savefig()
        plt.close()


    #ct = 'medcond'

        tpl1 = (raw_perband_flt_pri,stats_multiband_flt,'flt')
        tpl2 = (raw_perband_bp_pri, stats_multiband_bp,'bp')
        tpls = (tpl1,tpl2)
        for dat_dict,stat_dict,dat_type in tpls:
            bands_cur = freqs2relevantBands(relevant_freqs, fband_names_inc_HFO)
            for band in bands_cur:
            #for band in ['HFO']:
            #for band in ['beta']:
            #band = 'gamma'
            #band = 'beta'
            #band = 'tremor'
                means_per_iset = stat_dict['means'].get(band, None)
                indsets = stat_dict['indsets']
                suptitle = f'combin={scale_data_combine_type}  band={band} dat_type={dat_type}'

                if means_per_iset is None:
                    continue

                r = parseFeatNames(featnames_sel)
                chnames_involved = [chn for chn in (r['ch1'] + r['ch2']) if chn is not None]
                chnames_involved = list(set(chnames_involved))
                chis = [subfeature_order_newsrcgrp_pri[0].index( chn ) for chn in chnames_involved]

                plotDataAnnStat(rawnames, dat_pri, times_pri, subfeature_order_pri,
                                dat_lfp_hires_pri,times_hires_pri,subfeature_order_lfp_hires_pri,
                                anndict_per_intcat_per_rawn,
                                indsets,means_per_iset,suptitle=suptitle,
                                dat_dict=dat_dict,band=band,legend_loc='upper left',
                            chis_to_show = chis)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        pdf.close()

#funloc = plotTFRlike(dat_pri,bpow_abscds_all_reshaped,0, 'beta', chns=['msrcR_0_3_c5', 'LFPR092'])
#funloc = plotTFRlike(dat_pri,bpow_abscds_all_reshaped,0, 'beta', chns=['LFPR092'])

#funloc['bpow_abscds_reshaped'].shape
