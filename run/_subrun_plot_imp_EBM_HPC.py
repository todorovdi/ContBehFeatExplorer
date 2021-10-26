from matplotlib.backends.backend_pdf import PdfPages
import utils_postprocess as pp
from utils_postprocess_HPC import loadFullScores, loadEBMExplainer_
from matplotlib.lines import Line2D
#EBM_feat_subset = 'all'
#%debug

use_same_ax = True
#clear_yticks_for_middle_axes = False
clear_yticks_for_middle_axes = True
show_max = False
use_light_file = 1
rname_crop = slice(-3,None)
rname_crop2 = slice(None,-3)

legend_marker_size = 10
legend_alpha = .3

suptitle_to_put_ = ''
#legend_loc = 'lower right'
legend_loc = -1,0.5

load_explainers = True

max_nfeats_to_sum = 20
max_nfeats_to_sum = None

markers_mean = ['o','*']
markers_max = ['x','+']
#markers_mean = [r'$\mathbf{' +f'{int(subj[0][1:])}' + r'}$' for subj in outputs_grouped.keys()]
#markers_max = [r'$\tilde{' + f'{int(subj[0][1:])  }' + r'}$' for subj in outputs_grouped.keys()]

#for prefix_cur in list(sorted( set(prefixes) - set(['all']) )):
#for prefix_cur in ['cross_freqmod_beta,gamma:HFO']:
#for prefix_cur in [ 'LFPrel_noself']:
#for prefix_cur in [ 'onlyH']:
#for prefix_cur in [ 'allb_beta', 'onlyH']:
#for prefix_cur in [  'allb_beta', 'allb_tremor', 'allb_gamma', 'onlyH']:
#for prefix_cur in [ 'LFPrel_noself_onlyRbcorr']:
#for prefix_cur in [ 'LFPrel_noself_onlyBpcorr', 'LFPrel_noself']:
#for prefix_cur in [ 'all']:
#for prefix_cur in ['LFPrel_noself']
#for prefix_cur in [ 'modLFP']:
#for prefix_cur in prefixes:
collect_SHAP_outs = []
for tpl_ in pref_hh_tuples:
    if isinstance(tpl_,str):
        prefix_cur = tpl_
        hh_ = None
    else:
        prefix_cur,hh_ = tpl_
    prefixes_to_use = [prefix_cur]

    outputs_grouped = pp.groupOutputs(output_per_raw, prefixes_to_use,
                                      [grpit_tpl[0]],[grpit_tpl[1]])

    if not len(outputs_grouped):
        print(f'{prefix_cur}: Found not outputs.. skipping')
        continue

    #pp.printDict(outputs_grouped,1,print_leaves=1)


    #(prefix,grp,int_type), output = u
    print(f'-- {prefix_cur}: Starting plotting for {len(outputs_grouped)} grouped outputs')
    #sens,spec = res[rn].get(pref, (np.nan, np.nan))
    (prefix,grp,int_type), mult_clf_output = list(outputs_grouped.values())[0]

    rnstr = ';'.join( list( outputs_grouped.values() )[0][0] )
    sfl = max_nfeats_to_sum is not None
    out_name_plot = f'{",".join((prefix,grp,int_type) ) }_EBM_feat_signif_sfl{sfl}'
    #chnames_LFP = ['LFPR01', 'LFPR12', 'LFPR23']

    #%debug
    merge_Hjorth = prefix_cur != 'onlyH'
    merge_Hjorth = False

    Hjorth_diff_color = (prefix_cur == 'onlyH') and not merge_Hjorth


    if hh_ is not None:
        hh = hh_
    else:
        hh = 12
        if prefix_cur == 'onlyH':
            hh = 5
        if prefix_cur == 'LFPrel_noself_onlyBpcorr':
            hh = 9
        if prefix_cur == 'LFPrel_noself_onlyRbcorr':
            hh = 7
    #outputs_grouped   #
    #(prefix,grp,int_type), mult_clf_output = output_list[0]
    #%debug
    figfname_full = pjoin(gv.dir_fig, subdir_short, out_name_plot + '.pdf')
    #figfname_full = figfname_full.replace('&','_AND_')
    pdf= PdfPages(figfname_full )

    for featsel_feat_subset_name_cur in EBM_feat_subsets:
    #for featsel_feat_subset_name_cur in ['all']:
        featsel_on_VIF_cur = featsel_feat_subset_name_cur == 'VIFsel'

        axs = None
        legend_elements = []
        for i,og in enumerate(outputs_grouped.items() ):
            rn = og[0][0]
            (prefix,grp,int_type), mult_clf_output = og[1]
            if load_explainers:
                loadEBMExplainer_(mult_clf_output,
                                featsel_feat_subset_name_cur, force=1)
            ogg = dict( [og] )
            if not use_same_ax:
                axs = None
            axs, collect_SHAP_outs_cur = postp.plotFeatSignifEBM_list(pdf=None,
                             outputs_grouped=ogg, fshs='interpret_EBM',
                             figname_prefix=prefix, roi_labels=labels_dict['all_raw'],
                             body_side='left', chnames_LFP=None,
                             hh = hh, ww = 7,
                             tickfontsize = 10, markersize=8,
                             suptitle = suptitle_to_put_, use_best_LFP=False,
                             suptitle_fontsize=20, show_bias=1, show_max = show_max,
                             show_std = False, average_over_subjects = True,
                             merge_Hjorth = merge_Hjorth,alpha=0.8,
                                    featsel_on_VIF=featsel_on_VIF_cur,
                                  use_full_scores=0, reconstruct_from_VIF=0,
                                show_abs_plots=1, marker_mean=markers_mean[i],
                                marker_max = markers_max[i], Hjorth_diff_color = Hjorth_diff_color,
                                axs=axs, grand_average_per_feat_type=0,
                                featsel_feat_subset_name = featsel_feat_subset_name_cur,
                                perf_marker_size=25, allow_dif_feat_group_sets=featsel_on_VIF_cur,
                                separate_by_band = True,
                                separate_by_band2 = True,
                                max_nfeats_to_sum  = max_nfeats_to_sum );
            legend_elements += [Line2D([0], [0], marker=markers_mean[i], color='orange', lw=0,
                                   label=f'mean {rn[rname_crop]}', markerfacecolor='orange',
                                       markersize=legend_marker_size,alpha=legend_alpha) ]
            if show_max:
                legend_elements += [Line2D([0], [0], marker=markers_max[i], color='orange', lw=0,
                                       label=f'max {rn[rname_crop]}', markerfacecolor='orange',
                                           markersize=legend_marker_size,alpha=legend_alpha) ]

            if not use_same_ax:
                pdf.savefig();    plt.close()
                axs[0,0].set_title(axs[0,0].get_title() + f' {rn[rname_crop2]}')

            collect_SHAP_outs += collect_SHAP_outs_cur

        #outs +=   [ (rn_,prefix,grp,int_type,fsh,featnames_nice_sub,\
        #    label_str,scores_per_class[lblind],feat_imp_stats )  ]


        if use_same_ax:
            axs[0,0].set_title( str(axs[0,0].get_title() ) + f' {rn[rname_crop2]}*')
            axs[0,1].legend(handles= legend_elements, loc=legend_loc)

            for i in range(len(axs) ):
                for j in range( 0,2 ):
                    axs[i,j].grid()

            plt.tight_layout()
            pdf.savefig();    plt.close()

    cax,clrb = postp.plotConfmats(outputs_grouped, best_LFP=0, ww=9,hh=9)
    kind = ';'.join( list( outputs_grouped.values() )[0][0] )
    figname = ','.join( [k[0] for k in outputs_grouped.keys()] ) + f'_confmats_{kind}.pdf'
    kind = ';  '.join( list( outputs_grouped.values() )[0][0] )
    plt.suptitle(kind)
    pdf.savefig();    plt.close();    pdf.close()
    print('saved to ',subdir,out_name_plot)

#             break+
#         break
#     break
