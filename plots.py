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
             shift_by_anntype = 1, seed=0 ):
    '''
    plot_bins means X axis is assumed to be bin number (and not times)
    '''
    cmap = plt.cm.get_cmap('tab20', 20)
    #start,end pairs
    descrs = list(sorted( set(anns.description)) )
    if len(descrs) == 0:
        print('shadeAnn: empty annotation')
        return {}
    #print( len(descrs), len(anns) )
    np.random.seed(seed)
    ri0 = np.random.randint(1e6)
    ri1 = np.random.randint(1e6)
    attrs_per_descr = {}
    if shift_by_anntype:
        height = (ymax - ymin ) / len(descrs)
        for i,descr in enumerate(descrs):
            attr = {}
            attr['color'] = cmap( (ri0 + i*ri1) % 20)
            attr['ylim'] = ymin + i * height, ymin + (i+1) * height
            attrs_per_descr[descr] = attr
            #print(descr, attrs_per_descr)
            #cmap(3)

    #intervals_se_pairs = list( zip( anns.onset, anns.onset + anns.duration ) )
    #for pa in intervals_se_pairs:
    descrs_were_shown = []
    for ann in anns:
        pa = ann['onset'],ann['onset'] + ann['duration']
        descr = ann['description']
        if plot_bins:
            pa = np.array(list(pa)) * sfreq // skip
        attr = attrs_per_descr.get(descr,None)
        if attr is not None:
            color_cur = attr['color']
            ymin_cur, ymax_cur =  attr['ylim']
        #ax.fill_between( list(pa) , ymin_cur, ymax_cur, facecolor=color_cur, alpha=alpha)
        if descr in descrs_were_shown:
            lab = None
        else:
            lab = descr
        ax.fill_between( list(pa) , ymin_cur, ymax_cur, facecolor=color_cur, alpha=alpha, label=lab)
        descrs_were_shown += [descr]

    return attrs_per_descr


def plotMeansPerIt(ax,anns,means_per_it,chi, sfreq=256, plot_bins=0,
                   alpha=0.5,ls=':', attrs_per_descr=None, c=None,lw=None,
                   printLog = False):
    for ann in anns:
        pa = ann['onset'],ann['onset'] + ann['duration']
        descr = ann['description']

        if means_per_it is not None:
            if descr not in means_per_it:
                if printLog:
                    print(f'plotMeansPerIt: Warning: {descr} not in means_per_it')
                continue
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
                print("BBBBBBBBAND ", band)
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
                      legend_loc = 'lower right', ww=6, hh=2  ):
    from featlist import parseFeatNames
    r = parseFeatNames(featnames_sel)
    chnames_involved = [chn for chn in (r['ch1'] + r['ch2']) if chn is not None]
    chnames_involved = list(set(chnames_involved))
    assert len(chnames_involved) > 0
    print(chnames_involved)

#     if extdat_dict is not None:
#         for extdat_type,ed in extdat_dict:
#             nr = 0
#             if extdat_type in ['flt','bp']:
#                 for band in ed:
#                     #nr += len(ed[band].ch_names)
#                     nr += len(chnames_involved)
#                 #ts = times_pri[rawi]
#                 #d = raw_perband_bp_pri[rawi][band]._data[chni]
#                 #plt.plot(ts,d)

#             nr = len(chnames_involved) + len(featnames_sel)
#                 #nr = dat_pri[0].shape[0]
#             nc = len(rawnames)
#             ww = 6; hh = 2
#             fig,axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh))
#             axs = axs.reshape((nr,nc))



    nr = len(chnames_involved) + len(featnames_sel)
        #nr = dat_pri[0].shape[0]
    nc = len(rawnames)
    fig,axs = plt.subplots(nr,nc,figsize=(nc*ww,nr*hh))
    axs = axs.reshape((nr,nc))
    for rawi,rawn in enumerate(rawnames):
        for chni,chn in enumerate(chnames_involved):
            ax = axs[chni, rawi]
            if chnames_all_newsrcgrp_pri is not None:
                orig_ind  = chnames_all_newsrcgrp_pri[rawi].index(chn)
            else:
                orig_ind  = chnames_all_pri[rawi].index(chn)

            if chn.startswith('LFP') and dat_hires_pri is not None:
                ax.plot(times_hires_pri[rawi], dat_hires_pri[rawi][chnames_all_hires_pri[rawi].index(chn)] )
            else:
                ax.plot(times_pri[rawi], dat_pri[rawi][orig_ind])


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

                attrs = shadeAnn(ax,ann,*shadey_beh_state,color='red',alpha=0.4, sfreq=sfreq, skip=1, plot_bins = 0,
                        shift_by_anntype = 1, seed=1)
                shadeAnn(ax,ann_artif,*shadey_aftif,color='red',alpha=0.4, sfreq=sfreq, skip=1, plot_bins = 0,
                        shift_by_anntype = 1, seed=4)


            ax.set_title(f'{rawn} : {chn}')
            ax.legend(loc=legend_loc)

    for rawi,rawn in enumerate(rawnames):
        for feati,featn in enumerate(featnames_sel):
            ax = axs[feati + len(chnames_involved), rawi ]
            ts = Xtimes_pri[rawi]
            if wbd_pri is not None:
                ts = wbd_pri[rawi][1]
                ts = times_pri[rawi][ts]
            #print(ts,ts.shape,wbd_pri[rawi].shape)
            ax.plot(ts, X_pri[rawi] [:,featnames_all_pri[rawi].index(featn)])

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

                attrs = shadeAnn(ax,ann,*shadey_beh_state,color='red',alpha=0.4, sfreq=sfreq, skip=1, plot_bins = 0,
                        shift_by_anntype = 1, seed=1)
                shadeAnn(ax,ann_artif,*shadey_aftif,color='red',alpha=0.4, sfreq=sfreq, skip=1, plot_bins = 0,
                        shift_by_anntype = 1, seed=4)




            ax.set_title(f'{rawn} : {featn}')
            ax.set_xlim(0,ts[-1])
            ax.legend(loc=legend_loc)

    # all data

    # plot raw data
    # plot features
    print('plotting')

    return axs
