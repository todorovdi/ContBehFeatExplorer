from matplotlib.backends.backend_pdf import PdfPages
import utils_postprocess as pp
from utils_postprocess_HPC import loadFullScores
from matplotlib.lines import Line2D

use_light_file = 1
#subdir_short = 'joint_noskip'

#markers_mean = ['o','*']
#markers_max = ['x','+']

rname_crop = slice(-3,None)
rname_crop2 = slice(None,-3)
legend_alpha = 0.25
legend_marker_size = 15

clear_yticks_for_middle_axes = True
separate_by_band = 1
use_same_ax = True

suptitle_to_put_ = ''
legend_loc = 'lower right'

collect_SHAP_outs = []
for tpl_ in pref_hh_tuples:
    if isinstance(tpl_,str):
        prefix_cur = tpl_
    else:
        prefix_cur,hh_ = tpl_

    prefixes_to_use = [prefix_cur]

#     outputs_grouped = pp.groupOutputs(output_per_raw, prefixes_to_use,
#                                       ['merge_movements'],['trem_vs_hold&move'])
#     outputs_grouped = pp.groupOutputs(output_per_raw, prefixes_to_use,
#                                       ['merge_nothing'],['trem_vs_quiet'])
    outputs_grouped = pp.groupOutputs(output_per_raw, prefixes_to_use,
                                      [grpit_tpl[0]],[grpit_tpl[1]])

    if not len(outputs_grouped):
        print(f'{prefix_cur}: Found not outputs.. skipping')
        continue

    #(prefix,grp,int_type), output = u
    print(f'-- {prefix_cur}: Starting plotting for {len(outputs_grouped)} grouped outputs')

    postp.loadFullScores(outputs_grouped)
    #sens,spec = res[rn].get(pref, (np.nan, np.nan))
    (prefix,grp,int_type), mult_clf_output = list(outputs_grouped.values())[0]

    output_subdir = ''
    rnstr = ';'.join( list( outputs_grouped.values() )[0][0] )
    out_name_plot = f'{",".join((prefix,grp,int_type) ) }_feat_signif'
    #chnames_LFP = ['LFPR01', 'LFPR12', 'LFPR23']

    #%debug
    merge_Hjorth = prefix_cur != 'onlyH'
    merge_Hjorth = False

    #Hjorth_diff_color = (prefix_cur == 'onlyH') and not merge_Hjorth
    Hjorth_diff_color = not merge_Hjorth

    #clear_yticks_for_middle_axes = False

    if hh_ is None:
        hh = 12
        if prefix_cur == 'onlyH':
            if merge_Hjorth:
                hh = 5
            else:
                hh = 9
        if prefix_cur == 'LFPrel_noself_onlyBpcorr':
            hh = 9
        if prefix_cur == 'LFPrel_noself_onlyRbcorr':
            hh = 7
            if modSrc_self:
                hh = 12
        if prefix_cur == 'all':
            hh = 18
            if modSrc_self:
                hh = 25
    else:
        hh = hh_
    #outputs_grouped   #
    #(prefix,grp,int_type), mult_clf_output = output_list[0]
    #%debug
    figfname_full = pjoin(gv.dir_fig, subdir_short, out_name_plot + '.pdf')
    figfname_full = figfname_full.replace('&','_AND_')
    pdf= PdfPages(figfname_full )
    axs = None
    legend_elements = []
    for i,og in enumerate(outputs_grouped.items() ):
        rn = og[0][0]
        (prefix,grp,int_type), mult_clf_output = og[1]
        assert prefix == prefix_cur
        ogg = dict( [og] )
        if not use_same_ax:
            axs = None
        axs, collect_SHAP_outs_cur = postp.plotFeatSignifSHAP_list(pdf=None,
                                 outputs_grouped=ogg, fshs='XGB_Shapley',
                                 figname_prefix=prefix, roi_labels=labels_dict['all_raw'],
                                 body_side='left', chnames_LFP=None,
                                 hh = hh, ww = 7,
                                 tickfontsize = 10, markersize=8,
                                 suptitle = suptitle_to_put_, use_best_LFP=False,
                                 suptitle_fontsize=20, show_bias=1, show_max = True,
                                 show_std = False, average_over_subjects = True,
                                 merge_Hjorth = merge_Hjorth,alpha=0.8,
                                      use_full_scores=True, show_abs_plots=1, reconstruct_from_VIF=1,
                                    marker_mean=markers_mean[i] ,marker_max = markers_max[i],
                                    Hjorth_diff_color = Hjorth_diff_color,
                                    axs=axs, grand_average_per_feat_type=0,
                                        separate_by_band=separate_by_band);
        legend_elements += [Line2D([0], [0], marker=markers_mean[i], color='orange', lw=0,
                               label=f'mean {rn[rname_crop]}',
                                   markerfacecolor='orange',
                                   markersize=legend_marker_size,alpha=legend_alpha) ]
        legend_elements += [Line2D([0], [0], marker=markers_max[i],
                                   color='orange', lw=0, label=f'max {rn[rname_crop]}',
                                   markerfacecolor='orange',
                                   markersize=legend_marker_size,alpha=legend_alpha) ]

        if not use_same_ax:
            pdf.savefig();    plt.close()
            axs[0,0].set_title(axs[0,0].get_title() + f' {rn[rname_crop2]}')

            if clear_yticks_for_middle_axes:
                for i in range(len(axs) ):
                    for j in range( 1,4 ):
                        axs[i,j].set_yticklabels([])
                        #axs[i,j].set_yticks([])
                    for j in range(4):
                        axs[i,j].grid()
        collect_SHAP_outs += collect_SHAP_outs_cur

    if use_same_ax:
        axs[0,0].set_title( str(axs[0,0].get_title() ) + f' {rn[rname_crop2]}*')
        axs[0,1].legend(handles= legend_elements, loc=legend_loc)

        if clear_yticks_for_middle_axes:
            for i in range(len(axs) ):
                for j in range (1,4):
                    axs[i,j].set_yticklabels([])
                    #axs[i,j].set_yticks([])
                for j in range (4):
                    axs[i,j].grid()

        plt.tight_layout()
        pdf.savefig();    plt.close()

    cax,clrb = postp.plotConfmats(outputs_grouped, best_LFP=0, ww=9,hh=9)
    kind = ';'.join( list( outputs_grouped.values() )[0][0] )
    figname = ','.join( [k[0] for k in outputs_grouped.keys()] ) + f'_confmats_{kind}.pdf'
    kind = ';  '.join( list( outputs_grouped.values() )[0][0] )
    plt.suptitle(kind)
    pdf.savefig();    plt.close();    pdf.close()
    print('saved to ',figname)
