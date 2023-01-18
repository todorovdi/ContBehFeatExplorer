import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
import os
import traceback

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

def plotStatsPerIt(ax,anns,means_per_it,stds_per_it,
                   chi, sfreq=256, plot_bins=0,
                   alpha=0.5,ls=':', attrs_per_descr=None, c=None,lw=None,
                   printLog = False):
    ctr = 0
    for ann in anns:
        pa = ann['onset'],ann['onset'] + ann['duration']
        descr = ann['description']

        if means_per_it is not None:
            if descr not in means_per_it:
                if printLog:
                    print(f'plotStatsPerIt: Warning: {descr} not in means_per_it')
                continue
            ctr += 1
            me = means_per_it[descr][chi]
            std = stds_per_it[descr][chi]
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

            middle = np.mean(list(pa) )
            xs = [middle, middle]
            ys = [me-std, me+std]
            #print('vline ',xs,ys)
            ax.plot(xs,ys, ls=ls, alpha=alpha, c=color, lw=lw)
    if ctr == 0:
        print(f'plotStatsPerIt: nothing was plotted because {set(anns.description)} are not part of  {means_per_it.keys()}')

def plotDataAnnStat(rawnames,dat_pri,times_pri,chnames_pri,
                   dat_hires_pri=None,times_hires_pri=None,chnames_hires_pri=None,
                   anndict_per_intcat_per_rawn=None,
                   indsets=None,
                    means_per_iset=None, stds_per_iset=None,
                    chnames_nicened_pri  = None,
                    suptitle='', sfreq=256,
                   dat_dict=None,band=None,pdf=None, legend_loc = 'lower right',
                    chis_to_show = None, q_thr = 1e-3, mult_std = 3.5,
                    artif_height_prop = 0.3, ww=6, hh=2):
    '''
    plot data with annotations
    '''

    import utils_tSNE as utsne
    import matplotlib.pyplot as plt
    import utils_preproc as upre
    indset_mask,_ = upre.getIndsetMask(indsets)
    dat_to_plot = 0

    if chis_to_show is None:
        if (band is not None) and band.startswith('HFO'):
            chis_to_show = range( dat_hires_pri[0].shape[0] )
        else:
            chis_to_show = range( dat_pri[0].shape[0] )

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
                # rescale all to same size?
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
            ax.set_xlim(times[0], times[-1] )


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


            #shadey_aftif = (mn, (mn+mx)/2); shadey_beh_state = ( (mn+mx)/2, mx)
            m = mult_std
            spl = (mn+mx)/2; shadey_aftif = [mn - abs(std)*m, spl]; shadey_beh_state = [ spl, mx + abs(std)*m]

            d = (shadey_aftif[1] - shadey_aftif[0])
            shadey_aftif[1] = shadey_aftif[0] + d * artif_height_prop / 0.5
            shadey_beh_state[0] = shadey_aftif[1]
            #shadey_beh_state[0] = d * (1-artif_height_prop)
            #spl = me; shadey_aftif = (me - std, spl); shadey_beh_state = ( spl, me+std)
            ttl = f'{rawn} : {chn}'
            if chnames_nicened_pri is not None:
                chn_nice = chnames_nicened_pri[rawi][chi]
                ttl += f' {chn_nice}'
            ax.set_title(ttl)
            attrs = shadeAnn(ax,ann,*shadey_beh_state,color='red',alpha=0.4, sfreq=sfreq, skip=1, plot_bins = 0,
                     shift_by_anntype = 1, seed=1)
            shadeAnn(ax,ann_artif,*shadey_aftif,color='red',alpha=0.4, sfreq=sfreq, skip=1, plot_bins = 0,
                     shift_by_anntype = 1, seed=4)

            if means_per_iset is not None:
                means_per_it = means_per_iset[iseti]  #['notrem_L']
                stds_per_it  = stds_per_iset[iseti]  #['notrem_L']
                #print('means_per_it',rawn,chn,means_per_it)
                plotStatsPerIt(ax,ann,means_per_it,stds_per_it,
                    chi, c='red', alpha=1.,lw=3)#,attrs_per_descr = attrs)
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


    nr = len(featnames_sel)
    hratios =  [1] * len(featnames_sel)
    if dat_pri is not None:
        nr += len(chnames_involved)
        hratios = [0.75] * len(chnames_involved) + hratios
    if extdat_pri is not None:
        hratios = [0.4] + hratios
        nr += 1
        #nr = dat_pri[0].shape[0]
    nc = len(rawnames)
    fig,axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh), gridspec_kw={'height_ratios': hratios}  )
    axs = axs.reshape((nr,nc))


    rowi_offset = 0
    # plot extdat
    if extdat_pri is not None:
        print('plotFeatsAndRelDat: Plotting extdat')
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



    print('plotFeatsAndRelDat: Plotting rawdata')
    # plot raw data
    if dat_pri is not None or dat_hires_pri is not None:
        for rawi,rawn in enumerate(rawnames):
            print('plotFeatsAndRelDat: ',rawn)
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
                    assert dat_hires_pri is not None
                    ax.plot(times_hires_pri[rawi], dat_hires_pri[rawi][chnames_all_hires_pri[rawi].index(chn)], alpha=alpha, c=main_color )
                    ax.set_xlim( (times_hires_pri[rawi][0],times_hires_pri[rawi][-1] ) )
                else:
                    assert dat_pri is not None
                    ax.plot(times_pri[rawi], dat_pri[rawi][orig_ind],  alpha=alpha, c=main_color)
                    ax.set_xlim( (times_pri[rawi][0],times_pri[rawi][-1] ) )

                if xlim is not None:
                    ax.set_xlim(xlim)
            #end for over chnames_involved


                # Shade artifacts
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
                #print('aaaaaaa ', f'{rawn} : {chn}')
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


        rowi_offset += len(chnames_involved)

    # plot features
    print('Plotting features')
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
    it plots not all chans, but only the special_chns
    test_plots_descr -- what actually we will plot (each in separate pdf):
        chn_descrs, feat_types_actual_coupling, relevant_freqs, informal_descr, figname

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
        if special_chns is not None:
            desired_chns = [special_chns[chnd] for chnd in ds]
            fts = tpd['feat_types_actual_coupling']
            relevant_freqs = tpd['relevant_freqs']
            print(ds,desired_chns)
            print(fts,relevant_freqs)

            featnames_sel = selectFeatNames(fts,relevant_freqs,desired_chns,
                                            feature_names_all,
                                            fband_names=fband_names_inc_HFO)
            print(featnames_sel)
        else:
            from featlist import getFreqsFromParseRes
            featnames_sel = feature_names_all
            r = parseFeatNames(featnames_sel)
            relevant_freqs = getFreqsFromParseRes(r)

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
        # plot second plot (for intermediate data -- filtered and bandpower) elsewhere

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
                stds_per_iset = stat_dict['stds'].get(band, None)
                indsets = stat_dict['indsets']
                suptitle = f'combin={scale_data_combine_type}  band={band} dat_type={dat_type}'

                if means_per_iset is None:
                    continue

                r = parseFeatNames(featnames_sel)
                chnames_involved = [chn for chn in (r['ch1'] + r['ch2']) if chn is not None]
                chnames_involved = list(set(chnames_involved))
                chis = [subfeature_order_newsrcgrp_pri[0].index( chn ) for chn in chnames_involved]

                plotDataAnnStat(rawnames, dat_pri, times_pri, subfeature_order_pri,
                                dat_lfp_hires_pri,times_hires_pri,
                                subfeature_order_lfp_hires_pri,
                                anndict_per_intcat_per_rawn,
                                indsets,means_per_iset,stds_per_iset,
                                suptitle=suptitle,
                                dat_dict=dat_dict,band=band,
                                legend_loc='upper left',
                            chis_to_show = chis)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        pdf.close()

def plotComponents(components, feature_names_all, comp_inds, nfeats_show, q,
                  toshow_decide_0th_component, explained_variance_ratio  = None,
                  inds_toshow = None,  nfeats_show_pc= None, nfeats_highlight_pc= None,
                   hh=4):
    # Ncomponens x Ncomponent_corrds
    if components is None:
        return None
    dd = np.abs(components[comp_inds[0] ] )

    ncomp_to_plot = min( len(comp_inds), components.shape[0] )
    nr = min( ncomp_to_plot, components.shape[0] )
    if inds_toshow is None:
        if toshow_decide_0th_component:
            print('0th component')
            inds_sort = np.argsort(dd)  # smallest go first
            inds_toshow = inds_sort[-nfeats_show:]

            dd_toshow = dd[inds_toshow]
            #strong_inds = np.where(dd_toshow   > np.quantile(dd_toshow,q) ) [0]
            strong_inds = inds_toshow
            strongest_ind = np.argmax(dd_toshow)
            strong_inds_pc = [strong_inds]
            strongest_inds_pc = [strongest_ind]
        else:
            strong_inds_pc = []
            strongest_inds_pc = []
            if nfeats_show_pc is None:
                nfeats_show_pc = max(1, nfeats_show // ncomp_to_plot)
            if nfeats_highlight_pc is None:
                nfeats_highlight_pc = nfeats_show_pc // 2
            print('Per component we will plot {} feats'.format(nfeats_show_pc) )
            inds_toshow = []
            #for i in range(nr):
            for i in comp_inds:
                dd = np.abs(components[i  ] )

                inds_sort = np.argsort(dd)  # smallest go first
                inds_toshow_cur = inds_sort[-nfeats_show_pc:]
                inds_toshow += [inds_toshow_cur]

                #dd_toshow = dd[inds_toshow_cur]
                strong_inds = np.where(dd   > np.quantile(dd,q) ) [0]
                #print(i, strong_inds )
                strongest_ind = np.argmax(dd)
                assert  strongest_ind == inds_toshow_cur[-1]
                strongest_inds_pc += [strongest_ind]

                #strong_inds_pc += [strong_inds.copy() ]
                strong_inds_pc += [inds_sort[-nfeats_highlight_pc:]  ]

            inds_toshow = np.sort( np.unique( inds_toshow) )

    #print(inds_toshow, strong_inds_pc, strongest_inds_pc)


    nc = 1
    ww = max(14 , min(40, components.shape[1]/3 ) )
    fig,axs = plt.subplots(nrows=nr, ncols=nc, figsize=(ww*nc, hh*nr), sharex='col')
    if nr == 1:
        axs = [axs]
    for i,compi in enumerate(comp_inds):
        ax = axs[i]
        #dd = np.abs(pca.components_[i] )
        dd = np.abs(components[compi,inds_toshow  ] )
        ax.plot( dd, lw=0, marker='o' )
        ax.axhline( np.quantile(dd, q), ls=':', c='r' )
        ttl = '(abs of) component {}' .format(compi)
        if explained_variance_ratio is not None:
            ttl += ', expl {:.4f} of variance (ratio)'.format(explained_variance_ratio[compi])
        ax.set_title(ttl)

        ax.grid()
        ax.set_xlim(0, len(inds_toshow) )


    ax.set_xticks(np.arange(len(inds_toshow) ))
    if feature_names_all is not None:
        ax.set_xticklabels( np.array(feature_names_all)[inds_toshow], rotation=90)

    tls = ax.get_xticklabels()
    ratio = 0.5 * len(inds_toshow) / len(strong_inds_pc )
    for compi in range(len(strong_inds_pc ) ):
        sipc = 0
        #print(compi, strong_inds_pc[compi] )
        si_cur = strong_inds_pc[compi]
        for i in si_cur[::-1]:
            ii = np.where(inds_toshow == i)[0]
            #print(ratio, sipc  )
            if len(ii) > 0 and (sipc < ratio ):
                sipc += 1
                ii = ii[0]
                tls[ii].set_color("purple")
    for compi in range(len(strong_inds_pc ) ):
        ii = np.where(inds_toshow == strongest_inds_pc[compi])[0][0]
        tls[ii ].set_color("red")

    #plt.suptitle('PCA first components info')
    #plt.savefig('PCA_info.png')
    return strong_inds_pc

#funloc = plotTFRlike(dat_pri,bpow_abscds_all_reshaped,0, 'beta', chns=['msrcR_0_3_c5', 'LFPR092'])
#funloc = plotTFRlike(dat_pri,bpow_abscds_all_reshaped,0, 'beta', chns=['LFPR092'])

#funloc['bpow_abscds_reshaped'].shape

#def plotICA(ica):
#    from mne.preprocessing import ICA
#    ica = ICA(n_components = n_components_ICA, random_state=0).fit(filt_raw2)
#
#    compinds =  range( ica.get_components().shape[1] )  #all components
#    icacomp = ica.get_sources(filt_raw)

def array2png(im, figname_full):
    from imageio import imwrite
    imwrite(figname_full, im )

def saveRenderVisBrainScene(sc, figname_full, resolution = 300,
                            render_only = False, crop_out = None):
    # This hack is needed when running visbrain in jupyter
    import matplotlib.pyplot as plt
    from mpl_render import RenderingImShow

    ww,hh = sc.canvas.size
    x,y=0,0
    if crop_out is not None:
        x += crop_out[0]
        y += crop_out[1]
        ww -= crop_out[2]
        hh -= crop_out[3]
    render_result = sc.render()
    region = x,y,ww,hh

    render_result =  render_result[y:y+hh,:][:,x:x+ww]
    #render_result = np.flip(render_result, axis = 0)

    #from imageio import imwrite
    #imwrite(figname_full, render_result )
    if not render_only:
        array2png(render_result, figname_full)
    return render_result

    # produces wierd stuff that needs to be cropped
    #fig, ax = plt.subplots(1, 1)
    #ax = plt.gca()
    #fig = plt.gcf()
    ## |  The `user_render()` method is expected to return an image with
    ## |  size `self.size`, representing area `self.extent`, where `extent`
    ## |  describes a rectangle `(x0, x1, y0, y1)`
    ## size is just size of the image (in pixels)
    #extent = (0, 7, 0, 5)
    #extent = (-5, 0, 0, 5)
    #p = RenderingImShow( ax, extent = extent, render_callback = (lambda size, extent: render_result))

    ## here I delete axis and colorbar, which somehow get added while applying mpl_render
    #plt.axis('off')
    #plt.delaxes(fig.axes[1])

    ## save the figure in high resolution now possible
    #fig.savefig(figname_full, dpi=resolution)


def plot3DValPerParcel(vals_per_source, val_LFP, vis_info, title,
                       show_supp=True, sources_visible=False,
                       radius_project=1.2, bgcolor='lightgrey',
                       brain_translucent=False,
                      cblabel='performance', fit_to_brain = True, fit_supp_to_brain = False,
                       views=['left','right'], clim = None, ww=None, hh=None,
                      use_mod_sourcegrid = True, colorbar = True, cmap='plasma' ):
    '''
    vals_per_source -- values per source according to what was in srcgroups
    (so with potential duplication if for every source within same brain area
    has same number)

    show_supp -- show supplementary info -- creating
    additional brain objects (one for each view) with sources drawn

    if val_LFP (float) is not None, it tries to plot BG (so one more brain object to the right)

    title will be the actual title, without any modification

    vi['headsurf_tris']
    vi['headsurf_verts']
    vi['headsurfgrid_mod_verts']
    '''
    import numpy as np
    from visbrain.objects import BrainObj, ColorbarObj, SceneObj, SourceObj, RoiObj
    # Scene creation
    if ww is None:
        ww = 300 * len(views) + 200

    if hh is None:
        hh = 500
        if show_supp:
            hh = 1000
    sc = SceneObj(bgcolor=bgcolor, size=(ww, hh))
    # Colorbar default arguments. See `visbrain.objects.ColorbarObj`
    cbrect = (-0.15, -2., 1., 4.)
    CBAR_STATE = dict(cbtxtsz=12, txtsz=10., width=.1, cbtxtsh=2.,
                      rect=cbrect)

    zoom_STN_area = 2.
    zoom_brain = 2.5
    zoom_brain = 10.
    KW = dict(title_size=16., zoom=zoom_brain)
    if bgcolor in ['lightgrey', 'white']:
        KW['title_color'] = 'black'
        CBAR_STATE['txtcolor'] = 'black'
        CBAR_STATE['bgcolor'] = 'white'


    import globvars as gv

    sind_str = ''
    vi = vis_info
    tris =   vi['headsurf_tris']
    verts =  vi['headsurf_verts']  #- 1.
    # Translucent inflated BrainObj with both hemispheres displayed
    #tc = True;
    tc = brain_translucent
    #hemisphere = 'left'
    hemisphere = 'both'

    colids = {}
    brain_obj_per_rot = {}
    for i,v in enumerate(views):
        b_obj = BrainObj(f'{sind_str}_brainsurf', translucent=tc, vertices=verts,
                            faces=tris,  hemisphere=hemisphere)
        brain_obj_per_rot[v] = b_obj
        colids[v] = i

    #b_obj_left = BrainObj(f'{sind_str}_brainsurf', translucent=tc, vertices=verts,
    #                     faces=tris,  hemisphere=hemisphere)

    #colids = {'back':0, 'left':1}
    # Add the brain to the scene. Note that `row_span` means that the plot will
    # occupy two rows (row 0 and 1);  row_span=1
    was = False
    for rot,colid in colids.items():
        title_cur = title
        if was:
            title_cur = ''
        sc.add_to_subplot(brain_obj_per_rot[rot], row=0, col=colid,
                          title=title_cur, **KW, rotate=rot)
        was = True

    if show_supp:
        tc_supp = True
        brain_obj_per_rot_supp = {}
        for v,colid in colids.items():
            b_obj = BrainObj(f'{sind_str}_brainsurf', translucent=tc_supp, vertices=verts,
                                faces=tris,  hemisphere=hemisphere)
            brain_obj_per_rot_supp[v] = b_obj

            sc.add_to_subplot(brain_obj_per_rot_supp[v], row=1, col=colid,
                          title=f'{title} supp', **KW, rotate=v)


    # verticesarray_like | None
    # Mesh vertices to use for the brain. Must be an array of shape (n_vertices, 3).

    # facesarray_like | None
    # Mesh faces of shape (n_faces, 3).
    ################################################################
    #if vals_per_source is None:
    #    import utils
    #    #vals = np.arange(len(roi_labels))
    #    vals_per_source = utils.dupValsWithinParcel(roi_labels,srcgroups, vals)

    if use_mod_sourcegrid:
        xyz = vi['headsurfgrid_mod_verts']
    else:
        xyz = vi['headsurfgrid_verts']
            #headsurfgrid_mod_verts
    #xyz = xyz * 10 / 3
#     data = np.zeros( xyz.shape[0]  )
#     #data = np.random.uniform(size=data.shape[0])
#     data[10:100] = 1
    data = vals_per_source
    data_bad_mask = np.isnan(data)

    data_good_mask = ~data_bad_mask
    print( xyz.shape, data.shape )

    mask_nan = True  # DEBUG ONLY!
    if mask_nan:
        xyz = xyz[data_good_mask]
        data_good = data[data_good_mask]
    else:
        data[data_bad_mask] = -100
        data_good = data

    #%debug
    radius_supp = 6
    RADINFO = dict(radius_min=radius_supp, radius_max = radius_supp)

    verts = brain_obj_per_rot[views[0] ].vertices

    source_obj_per_rot = {}
    for v,colid in colids.items():
        s_obj  = SourceObj ('mysrc', xyz, data=data_good, cmap=cmap, **RADINFO)
        # mask=~data_bad_mask, mask_color='gray'
        if fit_to_brain:
            s_obj.fit_to_vertices(verts)
        source_obj_per_rot[v] = s_obj

        #mask=mask, mask_color='gray'


    if show_supp:
        source_obj_per_rot_supp = {}
        data_supp = data_good
        data_supp = None
        for v,colid in colids.items():
            s_obj_unfit  = SourceObj('mysrc',  xyz, data=data_supp, cmap=cmap, **RADINFO )

            if fit_supp_to_brain:
                s_obj_unfit.fit_to_vertices(verts)

            source_obj_per_rot_supp[v] = s_obj_unfit

    # Just for fun, color sources according to the data :)
    #s_obj.color_sources(data=data)


    #s_obj_unfit.project_sources(b_obj_back, cmap='plasma', radius=20)
    # s_obj2_unfit.project_sources(b_obj_left, cmap='plasma', radius=4)

    # Finally, add the source and brain objects to the subplot
    for rot,colid in colids.items():
        # Project source's activity
        source_obj_per_rot[rot].project_sources(brain_obj_per_rot[rot], cmap=cmap,
                                                radius=radius_project)

        if not sources_visible:
            source_obj_per_rot[rot].set_visible_sources('none')
        sc.add_to_subplot(source_obj_per_rot[rot],  row=0, col=colid, title='', **KW, rotate=rot)
        #sc.add_to_subplot(s_obj2, row=0, col=2, title='', **KW, rotate='right')

    if show_supp:
        for rot,colid in colids.items():
            sc.add_to_subplot(source_obj_per_rot_supp[rot],  row=1,
                              col=colid, title='', **KW, rotate=rot, )


    #sc.add_to_subplot(b_obj_proj, row=0, col=2, rotate='left', use_this_cam=True)
    # Finally, add the colorbar :

    colorbar_colind = len(colids)
    if val_LFP is not None:
        colorbar_colind = len(colids) + 1
    if clim is None:
        dd = data[ ~data_bad_mask ]
        clim = np.min(dd),np.max(dd)

    if colorbar:
        cb_proj = ColorbarObj(s_obj, cblabel=cblabel, **CBAR_STATE, clim=clim )
        sc.add_to_subplot(cb_proj, row=0, col=colorbar_colind)#, width_max=100)

    #########################
    if val_LFP is not None:
        b_obj = BrainObj('brtr', translucent=True, vertices=verts*10,
                        faces=tris,  hemisphere=hemisphere)
        roi_aal = RoiObj('aal')
        idx_th = roi_aal.where_is('Thalamus (L)')
        r = roi_aal.select_roi(select=idx_th)
        mean_v = np.mean(roi_aal.vertices, axis=0)
        xyz = mean_v[None,:]
        radius_supp = 2
        RADINFO = dict(radius_min=radius_supp, radius_max = radius_supp)
        s_obj  = SourceObj ('mysrc', xyz, data=[val_LFP], cmap=cmap, **RADINFO)

        s_obj.project_sources(roi_aal, cmap=cmap, radius = 20, clim=clim)
        #, clim=(-1., 1.), vmin=-.5, vmax=.7, under='gray', over='red', radius = 5


        stn_colid = len(colids)
        #sc = SceneObj(bgcolor='white', size=(1500, 500))
        sc.add_to_subplot(b_obj,   row=0, col = stn_colid, rotate='left', zoom=zoom_STN_area,
                          title='STN LFP')
        sc.add_to_subplot(roi_aal, row=0, col = stn_colid, rotate='left')
        sc.add_to_subplot(s_obj,   row=0, col = stn_colid, rotate='left')


    return sc


def plotBrainPerSubj(sind_strs, vis_info_per_subj, source_coords, subdir, clim,
                    countinfo=None,
                    fix_vis_info = True,
                    plot_intremed = True,
                    figtitle_inc_durations = False,
                    figtitle_inc_LFP_plus_best = False,
                    use_mod_sourcegrid = 1,
                    show_supp = 0,
                    fit_to_brain_def = 0,
                    use_common_colorbar = 1,
                    colorbar_individ_show = 0,
                    df_per_mode = None,
                    base_key_name = 'base_low',
                    hhdef = 900,
                    wwdef = 300*4 + 200,
                    crop_out_def = (0,230,0,480),  #x,-y_top,-x_right,-y_bottom  # y counted top to bottom
                    radius_project = 1.3,
                    fit_to_brain = 0,
                    plotinfos_pre = None,
                    verbose=0,
                    save = True,
                    crop_out=None,
                    pctize_LFP = False,
                    modes = ['LFPand_only']  ):
    '''
    format of plotinfos_pre:
        plotinfo_cur['info_pgn_rel_LFP']
        plotinfo_cur['info_pgn_abs']

            format of info_pgn_rel_LFP field
        info_rel_LFP['brain_area_labels'], info_rel_LFP['intensities']
        info_rel_LFP['srcgrp_new']
        info_rel_LFP['coords']
        info_abs['impr_per_medcond_per_pgn']
    '''

    import globvars as gv
    import utils

    ####################################################################

    if not use_common_colorbar:
        clim = None

    print('clim = ',clim)

    renders = []

    subdir_fig = pjoin(gv.dir_fig ,subdir) #'output'
    if not os.path.exists(subdir_fig):
        os.makedirs(subdir_fig)
    #plot_intremed = False


    ERRORS = []
    empty = np.zeros( (hhdef, wwdef, 4) )
    all_renders = np.zeros( (len(sind_strs) * hhdef, 2* wwdef, 4) )
    fig_fnames_full = []
    figtitles = []

    if plotinfos_pre is None:
        plotinfos = {}
    else:
        plotinfos = plotinfos_pre


    for rowi,sind_str in enumerate(sind_strs):
    #for sind_str in ['S03']:
        if not fix_vis_info:
            if sind_str == 'S07':
                fit_to_brain = 0
                radius_project = 2.1
            else:
                fit_to_brain = fit_to_brain_def
                radius_project = 1.3
        #infos[sind_str] = {}

        for mode in  modes:  # exclude
            for mmci,medcond in enumerate( ['off', 'on'] ):
                if plotinfos_pre is None:
                    plotinfo_cur = {}
                    plotinfos[(sind_str,medcond,mode)] = plotinfo_cur
                plotinfo_cur = plotinfos[(sind_str,medcond,mode)]

                if medcond == 'on' and sind_str == 'S03':
                    ww2,hh2 = wwdef,hhdef
                    x2,y2=0,0
                    if crop_out is not None:
                        x2 += crop_out_def[0]
                        y2 += crop_out_def[1]
                        ww2 -= crop_out_def[2]
                        hh2 -= crop_out_def[3]
                    render_result = empty[y2:y2+hh2,:][:,x2:x2+ww2]
                    renders += [render_result]
                    continue
                ttl = f'brain_map_area_strength_{sind_str}_{medcond}'
                print('Starting producing figure ',ttl)
                #fname_full = pjoin(gv.data_dir,subdir,f'EXPORT_{ttl}_medcond={medcond}_mode={mode}.npz')
                #f = np.load(fname_full,allow_pickle=1)
                #info_rel_LFP = f['info'][()]

                #fname_full2 = pjoin(gv.data_dir,subdir,\
                #    f'EXPORT_{ttl}_medcond={medcond}_mode=only.npz')
                #f2 = np.load(fname_full2,allow_pickle=1)
                #info_abs = f2['info'][()]


                #plotinfo_cur['fname_full'] = fname_full
                #plotinfo_cur['fname_full2'] = fname_full2
                #plotinfo_cur['info_pgn_rel_LFP'] = info_rel_LFP
                #plotinfo_cur['info_pgn_abs'] = info_abs

                #infos[sind_str][medcond]['info2'] = info2

                info_rel_LFP  = plotinfo_cur['info_pgn_rel_LFP']
                info_abs      = plotinfo_cur['info_pgn_abs']

                if sind_str == 'S01':
                    info0 = info_rel_LFP

                d = dict( zip(info_rel_LFP['brain_area_labels'], info_rel_LFP['intensities']) )
                #intensitites = np.array(info['intensities'][1:] ) #/100
                intensitites = np.array(info_rel_LFP['intensities'] ) #/100

                try:
                    vals_per_source = utils.dupValsWithinParcel(info_rel_LFP['brain_area_labels'],
                                                                info_rel_LFP['srcgrp_new'], intensitites)
                    #vals_per_source_list += [vals_per_source.copy()]
            #         bads = np.isnan(vals_per_source)
            #         #vals_per_source[bads ] = np.min( vals_per_source[ ~bads ]  )
            #         vals_per_source[ bads ] = 0
                    print('vals_per_source  min=', np.min( vals_per_source[~np.isnan(vals_per_source) ]),
                          np.max( vals_per_source[~np.isnan(vals_per_source) ]) )

                    if fix_vis_info:
                        vi = vis_info_per_subj[sind_strs[0]  ]
                        vi = dict(vi.items() ) # copy
                        vi['headsurfgrid_verts'] = info0['coords']
                        vi['headsurfgrid_mod_verts'] = source_coords
                    else:
                        vi = vis_info_per_subj[sind_str]
                        vi = dict(vi.items() ) # copy
                        vi['headsurfgrid_verts'] = info_rel_LFP['coords']

                    title = f'{sind_str}_medcond={medcond.upper()}_mode={mode}_fit{fit_to_brain}'
                    LFP_val = info_rel_LFP.get(base_key_name, None)

                    # percent
                    if pctize_LFP:
                        LFP_val = LFP_val * 100
                    plotinfo_cur['LFP_val_mode=only'] = LFP_val
                    if countinfo is not None:
                        plotinfo_cur['countinfo'] = countinfo[f'{sind_str}_{medcond}']

                    #LFP_val = info_rel_LFP['impr_per_medcond_per_pgn'][medcond].get(base_key_name, None)
                    #LFP_val = info_rel_LFP['impr_per_medcond_per_pgn'].get(base_key_name, None)
                    #continue
                    #if LFP_val is None:
                    #    #LFP_val = info_abs['impr_per_medcond_per_pgn'][medcond][base_key_name]
                    #    LFP_val = info_abs['impr_per_medcond_per_pgn'][base_key_name]


                    # just to see numbers in text
                    CB_vals = d.get("Cerebellum",None)
                    if CB_vals is None:
                        CB_vals = d["Cerebellum_L"],d["Cerebellum_R"]
                    print(f'title={title}, CB intensity={CB_vals}, LFP={LFP_val}%, clim={clim}' )


                    #assert np.abs(best - best_) < 1e-2
    #                 figtitle = f'{sind_str} {medcond.upper()}' +\
    #                 f',   LFP perf={LFP_val:.1f}%,  LFP+{best_area}={best:.1f}%'
                    figtitle = f'{sind_str} {medcond.upper()}' +\
                    f',   LFP perf={LFP_val:.0f}%\n'
                    if figtitle_inc_LFP_plus_best:
                        impr_absolute = info_abs['impr_per_medcond_per_pgn']
                        best_area = max(impr_absolute, key=impr_absolute.get)
                        #best = info2[best_area] #NO, it is single area W/O LFP!
                        best = np.max(intensitites[~np.isnan(intensitites)]) + LFP_val
                        figtitle += f'LFP+{best_area}={best:.0f}%'
                    #figtitle += '\n' + countinfo2[f'{sind_str}_{medcond}']
                    if figtitle_inc_durations:
                        assert countinfo is not None
                        figtitle += '\n' + plotinfo_cur['countinfo']


                    plotinfo_cur['figtitle'] = figtitle


                    # max(my_dict, key=my_dict.get)
                    #for fit_to_brain in [ 0,1]:
                    #info['srcgrp_new'], info['color_group_labels'],
                    figtitle_on_individ_plot = figtitle #+ f',   LFP perf={LFP_val:.2f}'
                    figtitle_on_individ_plot = ''

                    if show_supp:
                        hh = None
                    else:
                        hh = hhdef
                    #vis_info_per_subj[sind_str]
                    val_LFP_for_3D = None #val_LFP
                    if save:
                        sc = plot3DValPerParcel(vals_per_source, val_LFP_for_3D, vi,
                            figtitle_on_individ_plot, show_supp=show_supp, fit_supp_to_brain = fit_to_brain,
                            radius_project=radius_project,
                            brain_translucent=False , bgcolor='black',
                            sources_visible = False,
                            fit_to_brain= fit_to_brain, use_mod_sourcegrid=use_mod_sourcegrid,
                            clim = clim, views=['top', 'left', 'right', 'bottom'],
                            ww = wwdef, hh = hh,
                            colorbar = colorbar_individ_show)

                except AssertionError as e:
                    #sys.exc_info()
                    ERRORS += [(ttl + '_' + mode, e, traceback.format_exc())]
                    print(str(e))
                    continue


                figfname_full= pjoin(subdir_fig,
                        f'{title}_visbrain_rad{radius_project}.png')
                figfname_crp_full= pjoin(subdir_fig,
                        f'{title}_visbrain_rad{radius_project}_crp.png')
                fig_fnames_full += [figfname_full]
                figtitles += [figtitle]
                if show_supp:
                    crop_out = None
                else:
                    crop_out = crop_out_def

                if save:
                    render_result = saveRenderVisBrainScene(sc,figfname_full, render_only = not plot_intremed,
                                            crop_out = None)
                    render_result = saveRenderVisBrainScene(sc,figfname_crp_full, render_only = not plot_intremed,
                                            crop_out = crop_out)
                else:
                    render_result = None



                if save and (not show_supp):
                    all_renders[rowi * hhdef:(rowi+1) * hhdef,:,:][:, mmci * wwdef: (mmci+1) * wwdef, : ] = sc.render()
                #sc,render_result = r
                renders += [render_result]
                plotinfo_cur['render_result'] = render_result
                plotinfo_cur['figfname_full'] = figfname_full
                plotinfo_cur['figfname_crp_full'] = figfname_crp_full

    return all_renders, renders, plotinfos
