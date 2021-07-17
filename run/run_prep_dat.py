import sys,os,getopt
import utils_tSNE as utsne
import utils_genfeats as ugf
import utils_preproc as upre
import utils
import globvars as gv
import numpy as np
import gc
import matplotlib.pyplot as plt
from globvars import gp
from matplotlib.backends.backend_pdf import PdfPages

from os.path import join as pjoin
import mne
import multiprocessing as mpr

pdf = None

plot_stat_scatter = 0
use_main_moveside = 1
use_lfp_HFO = 1
src_type_to_use  = 'parcel_ICA'     # or mean_td
sources_type     = 'HirschPt2011'
save_dat   = 1
save_stats = 1
use_main_LFP_chan=0
crop_start = None
crop_end   = None
bands_precision = 'crude'
calc_stats_multi_band = 0

n_free_cores = gp.n_free_cores
n_jobs = max(1, mpr.cpu_count() - n_free_cores)

allow_CUDA_MNE = mne.utils.get_config('MNE_USE_CUDA')
allow_CUDA = True

input_subdir = ""
msrc_inds = np.arange(8,dtype=int)  #indices appearing in channel (sources) names, not channnel indices

data_modalities = ['LFP', 'msrc']
params_read = {}
params_cmd = {}

exit_after = 'end'
use_preloaded_raws = False

print('sys.argv is ',sys.argv)
effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

helpstr = 'Usage example\nrun_prep_dat.py --rawnames <rawname_naked,rawname_naked> '
opts, args = getopt.getopt(effargv,"hr:",
        ["mods=","msrc_inds=","rawnames=",
            "show_plots=", "side=", "LFPchan=", "useHFO=",
          "sources_type=", "crop=" ,
         "src_grouping=", "src_grouping_fn=",
         "input_subdir=", "save_dat=", "save_stats=", "param_file=",
         "bands_precision=", "calc_stats_multi_band=", "exit_after=",
         "use_preloaded_raws=" ])
print('opts is ',opts)
print('args is ',args)

for opt, arg in opts:
    print(opt,arg)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == "--param_file":
        param_fname_full = pjoin(gv.param_dir,arg)
        params_read = gv.paramFileRead(param_fname_full)
    else:
        if opt.startswith('--'):
            optkey = opt[2:]
        elif opt == '-r':
            optkey = 'rawnames'
        else:
            raise ValueError('wrong opt {}'.format(opt) )
        params_cmd[optkey] = arg


#overwriting values from param file with those from command line
#print(params_cmd,params_read)
params_read.update(params_cmd)
pars = params_read

for opt,arg in pars.items():
    if opt == "msrc_inds":
        msrc_inds = [ int(indstr) for indstr in arg.split(',') ]
        msrc_inds = np.array(msrc_inds, dtype=int)
    elif opt == "crop":
        if len(arg) > 2:
            cr =  arg.split(',')
            crop_start = float(cr[0])
            crop_end = float(cr[1] )
    elif opt == "save_stats":
        save_stats = int(arg)
    elif opt == "save_dat":
        save_dat = int(arg)
    elif opt == "exit_after":
        exit_after = arg
    elif opt == "src_grouping":
        src_grouping = int(arg)
    elif opt == "input_subdir":
        input_subdir = arg
        if len(input_subdir) > 0:
            subdir = pjoin(gv.data_dir,input_subdir)
            assert os.path.exists(subdir )
    elif opt == "src_grouping_fn":
        src_file_grouping_ind = int(arg)
    elif opt == "mods":
        data_modalities = arg.split(',')   #lfp of msrc
    elif opt == "calc_stats_multi_band":
        calc_stats_multi_band = int(arg)
    elif opt == "bands_precision":
        assert arg in ['fine', 'crude']
        bands_precision = arg
    elif opt == "sources_type":
        if len(arg):
            sources_type = arg
    elif opt == "use_preloaded_raws":
        use_preloaded_raws = int(arg)
    elif opt in 'rawnames':
        if len(arg) < 5:
            print('Empty raw name provided, exiting')
            sys.exit(1)
        rawnames = arg.split(',')  #lfp of msrc
        for rn in rawnames:
            assert len(rn) > 3
        if len(rawnames) > 1:
            print('Using {} datasets at once'.format(len(rawnames) ) )
        #rawname_ = arg
    elif opt == 'show_plots':
        show_plots = int(arg)
    elif opt == 'side':
        if arg == 'both':
            use_main_moveside == 0
        elif arg == 'main':
            use_main_moveside == 1
        elif arg == 'other':
            use_main_moveside == -1
        elif arg in ['left','right']:
            raise ValueError('to be implemented')
    elif opt == 'LFPchan':
        if arg == 'main':
            use_main_LFP_chan = 1
        elif arg == 'other':
            use_main_LFP_chan = 0
        elif arg.find('LFP') >= 0:   # maybe I want to specify expliclity channel name
            raise ValueError('to be implemented')
    elif opt == 'useHFO':
        use_lfp_HFO = int(arg)
    elif opt == 'plot_only':
        load_feat = 1
        save_feat = 0
        show_plots = 1
    elif opt == 'plot_types':
        plot_types = arg.split(',')  #lfp of msrc
        if 'raw_stats' in plot_types:
            do_plot_raw_stats = 1
        else:
            do_plot_raw_stats = 0
        if 'raw_stats_scatter' in plot_types:
            plot_stat_scatter = 1
        else:
            plot_stat_scatter = 0
        if 'raw_timecourse' in plot_types:
            do_plot_raw_timecourse = 1
        else:
            do_plot_raw_timecourse = 0
        if 'raw_psd' in plot_types:
            do_plot_raw_psd = 1
        else:
            do_plot_raw_psd = 0
        if 'csd' in plot_types:
            do_plot_CSD = 1
        else:
            do_plot_CSD = 0
    else:
        raise ValueError('Unknown option:arg {}:{}'.format(opt,arg) )

if allow_CUDA and allow_CUDA_MNE:
    n_jobs = 'cuda'
    print('Using CUDA')
#sys.exit(0)

#curpar = "mods"
#arg = params.get(curpar,None)
#if curpar not in opts and arg is not None:  # cmd input overides
#    data_modalities = arg.split(',')
#
#curpar = "rawname"
#arg = params.get(curpar,None)
#if curpar not in opts and '-r' not in opts and arg is not None:  # cmd input overides
#    if len(arg) < 5:
#        print('Empty raw name provided, exiting')
#        sys.exit(1)
#    rawnames = arg.split(',')  #lfp of msrc
#    for rn in rawnames:
#        assert len(rn) > 3
#    if len(rawnames) > 1:
#        print('Using {} datasets at once'.format(len(rawnames) ) )
#
#curpar = "mods"
#arg = params.get(curpar,None)
#if curpar not in opts and arg is not None:  # cmd input overides
#    data_modalities = arg.split(',')
#
# "msrc_inds=","rawname=",
#    "show_plots=", "side=", "LFPchan=", "useHFO=",
#    "sources_type=", "crop=" ,
#    "src_grouping=", "src_grouping_fn=",
#    "input_subdir=", "save_dat=", "save_stats=", "param_file=" ])


mods_to_load = ['LFP', 'src', 'EMG']
#mods_to_load = ['LFP', 'src', 'EMG', 'SSS','resample', 'FTraw']
#mods_to_load = ['LFP', 'src', 'EMG', 'resample', 'afterICA']
#mods_to_load = ['src', 'FTraw']
if use_lfp_HFO:
    mods_to_load += ['LFP_hires']

if not use_preloaded_raws:
    raws_permod_both_sides = upre.loadRaws(rawnames,mods_to_load, sources_type, src_type_to_use,
                src_file_grouping_ind,input_subdir=input_subdir,n_jobs=n_jobs)
else:
    print('----------  USING PRELOADED RAWS!!!!!')

sfreqs = [ int(raws_permod_both_sides[rn]['LFP'].info['sfreq']) for rn in rawnames]
assert len(set(sfreqs)) == 1
sfreq = sfreqs[0]

if use_lfp_HFO:
    sfreqs_hires = [ int(raws_permod_both_sides[rn]['LFP_hires'].info['sfreq']) for rn in rawnames]
    assert len(set(sfreqs_hires)) == 1
    sfreq_hires = sfreqs_hires[0]

####################  data processing params

def_main_side = 'left'
side_to_use = 'move_side'
force_consistent_main_sides = 1 # consistent across datasets, even not loaded ones
if force_consistent_main_sides:
    new_main_side = def_main_side

rec_info_pri = []
for rawname_ in rawnames:
    src_rec_info_fn = utils.genRecInfoFn(rawname_,sources_type,src_file_grouping_ind)
    src_rec_info_fn_full = pjoin(gv.data_dir, input_subdir, src_rec_info_fn)
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
    rec_info_pri += [rec_info]

# the output is dat only from the selected hemisphere
r = ugf.collectDataFromMultiRaws(rawnames, raws_permod_both_sides, sources_type,
                             src_file_grouping_ind, src_grouping, use_main_LFP_chan,
                             side_to_use, new_main_side, data_modalities,
                             crop_start,crop_end,msrc_inds, rec_info_pri)

dat_pri, dat_lfp_hires_pri, extdat_pri, anns_pri, anndict_per_intcat_per_rawn_, times_pri,\
times_hires_pri, subfeature_order_pri, subfeature_order_lfp_hires_pri, aux_info_perraw = r

if not use_preloaded_raws:
    anndict_per_intcat_per_rawn = anndict_per_intcat_per_rawn_

bindict_per_rawn = {}
for rawn,anndict_per_intcat in anndict_per_intcat_per_rawn.items():
    # here side is not important
    times_cur = raws_permod_both_sides[rawn]['LFP'].times
    bindict_per_rawn[rawn] = upre.markedIntervals2Bins(anndict_per_intcat,times_cur,sfreq)

if exit_after == 'collectDataFromMultiRaws':
    sys.exit(0)

#fn_suffix_dat = 'dat_{}_newms{}_mainLFP{}_grp{}-{}.npz'.format(new_main_side,
#                                                              ','.join(data_modalities),
#                                                               use_main_LFP_chan,
#                                                            src_file_grouping_ind, src_grouping)
if save_dat:
    for rawi in range(len(rawnames) ):
        rawn = rawnames[rawi]
        fname = utils.genPrepDatFn(rawn, new_main_side, data_modalities,
                                   use_main_LFP_chan, src_file_grouping_ind,
                                   src_grouping)
        #fname = '{}_'.format(rawn) + fn_suffix_dat
        fname_dat_full = pjoin(gv.data_dir, input_subdir, fname)
        print('Saving ',fname_dat_full)
        np.savez(fname_dat_full,
                 dat=dat_pri[rawi],
                 dat_lfp_hires=dat_lfp_hires_pri[rawi],
                 extdat = extdat_pri[rawi],
                 ivalis = utils.ann2ivalDict(anns_pri[rawi] ),
                 ivalis_artif = None,
                 anndict_per_intcat = anndict_per_intcat_per_rawn[rawn],
                 times = times_pri[rawi],
                 times_hires = times_hires_pri[rawi],
                 subfeature_order_pri = subfeature_order_pri[rawi],
                 subfeature_order_lfp_hires_pri = subfeature_order_lfp_hires_pri[rawi],
                 aux_info = aux_info_perraw[rawn],
                 data_modalities=data_modalities,
                 sfreq=sfreq, sfreq_hires = sfreq_hires, cmd=(opts,args), pars=pars)

n_channels_pri = [ datcur.shape[0] for datcur in dat_pri ];
main_sides_pri = [ aux_info_perraw[rn]['main_body_side'] for rn in rawnames]
side_switched_pri = [ aux_info_perraw[rn]['side_switched'] for rn in rawnames]

subfeature_order = subfeature_order_pri[0]
subfeature_order_lfp_hires = subfeature_order_lfp_hires_pri[0]


main_side_let = new_main_side[0].upper()
artif_handling = 'reject'
intervals_for_stats = [it + '_{}'.format(main_side_let) for it in gp.int_types_basic]
# here we plot even if we don't actually rescale
if plot_stat_scatter and len(set( n_channels_pri ) ) == 1 :
    dat_T_pri = [0]*len(dat_pri)
    for dati in range(len(dat_pri) ):
        dat_T_pri[dati] = dat_pri[dati].T

    upre.plotFeatStatsScatter(rawnames,dat_T_pri, intervals_for_stats,
                        subfeature_order_pri,sfreq,
                        times_pri,side_switched_pri, wbd_pri=None,
                                save_fig=False, separate_by='mod', artif_handling=artif_handling )
    plt.suptitle('Stats of nonHFO data before rescaling')
    pdf.savefig()
    plt.close()
    gc.collect()


baseline_int = 'notrem_{}'.format(main_side_let)
stats_per_ct = {}
stats_HFO_per_ct = {}
for combine_type in gv.rawnames_combine_types_rawdata:
    dat_T_pri = [0]*len(dat_pri)
    for dati in range(len(dat_pri) ):
        dat_T_pri[dati] = dat_pri[dati].T

    indsets, means, stds, stats_per_indset = \
        upre.gatherFeatStats(rawnames, dat_T_pri, subfeature_order_pri, None, sfreq, times_pri,
                intervals_for_stats, side_rev_pri = side_switched_pri,
                combine_within = combine_type, minlen_bins = 5*sfreq, require_intervals_present = [baseline_int],
                        artif_handling=artif_handling, bindict_per_rawn=bindict_per_rawn )
    curstatinfo = {'indsets':indsets, 'means':means, 'stds':stds, 'rawnames':rawnames, 'stats_per_indset':stats_per_indset }
    stats_per_ct[combine_type] = curstatinfo

    #X_pri_rescaled, indsets, means, stds
    #dat_T_scaled, indsets, means, stds = upre.rescaleFeats(rawnames, dat_T_pri, subfeature_order_pri, None,
    #                sfreq, times_pri, int_type = intervals_for_stats,
    #                main_side = None, side_rev_pri = side_switched_pri,
    #                minlen_bins = 5 * sfreq, combine_within=combine_type,
    #                artif_handling=artif_handling )
    #for dati in range(len(dat_pri) ):
    #    dat_pri[dati] = dat_T_scaled[dati].T

    if use_lfp_HFO:
        dat_T_pri = [0]*len(dat_lfp_hires_pri)
        for dati in range(len(dat_lfp_hires_pri) ):
            dat_T_pri[dati] = dat_lfp_hires_pri[dati].T

        indsets, means, stds, stats_per_indset = upre.gatherFeatStats(rawnames, dat_T_pri, subfeature_order_lfp_hires_pri,
                             None, sfreq_hires, times_hires_pri,
                intervals_for_stats, side_rev_pri = side_switched_pri,
                combine_within = combine_type, minlen_bins = 5*sfreq, require_intervals_present = [baseline_int],
                        artif_handling=artif_handling, bindict_per_rawn=bindict_per_rawn)

        curstatinfo = {'indsets':indsets, 'means':means, 'stds':stds, 'rawnames':rawnames, 'stats_per_indset':stats_per_indset }
        stats_HFO_per_ct[combine_type] = curstatinfo


if exit_after == 'gatherFeatStats':
    sys.exit(0)

if save_stats:
    subjs_analyzed, subjs_analyzed_glob = upre.getRawnameListStructure(rawnames, ret_glob=True)
    #ind_strs = map(lambda x: int(x[1:3]), subjs_analyzed.keys() )
    #inds_str = ','.join( map(str, list(sorted(ind_strs)) ) )
    #inds_str = ','.join( sorted(subjs_analyzed.keys() ) )
    #nr = len(rawnames)
    #fname_stats = 'stats_{}_{}_'.format(inds_str,nr)  + fn_suffix_dat
    fname_stats = utils.genStatsFn(rawnames, new_main_side, data_modalities,
                                   use_main_LFP_chan, src_file_grouping_ind,
                                   src_grouping )
    fname_stats_full = pjoin( gv.data_dir, input_subdir, fname_stats)
    print('Saving ',fname_stats_full)
    np.savez(fname_stats_full, stats_per_ct=stats_per_ct, stats_HFO_per_ct=stats_HFO_per_ct,
             rawnames=rawnames, cmd=(opts,args), pars=pars)


if calc_stats_multi_band:
    print('Filtering and Hilbert')

    if bands_precision == 'fine':
        fband_names = gv.fband_names_fine
    else:
        fband_names = gv.fband_names_crude

    if bands_precision == 'fine':
        fband_names_inc_HFO = gv.fband_names_fine_inc_HFO
    else:
        fband_names_inc_HFO = gv.fband_names_crude_inc_HFO
        fband_names_HFO = fband_names_inc_HFO[len(fband_names):]  # that HFO names go after

    fbands = gv.fbands

    sfreqs = [sfreq, sfreq_hires]
    skips = None
    dat_pri_persfreq = [dat_pri, dat_lfp_hires_pri]

    ann_MEGartif_prefix_to_use = '_ann_MEGartif_flt'
    n_jobs_flt = max(1, mpr.cpu_count()-gp.n_free_cores )

    # note that we can have different channel names for different raws
    #raw_perband_flt_pri_persfreq = []
    #raw_perband_bp_pri_persfreq = []

    smoothen_bandpow = 0

    raw_perband_flt_pri, raw_perband_bp_pri, chnames_perband_flt_pri, chnames_perband_bp_pri  = \
        ugf.bandFilter(rawnames, times_pri, main_sides_pri, side_switched_pri,
                sfreqs, skips, dat_pri_persfreq, fband_names_inc_HFO, gv.fband_names_HFO_all,
                fbands, n_jobs_flt, allow_CUDA and n_jobs == 'cuda',
                subfeature_order, subfeature_order_lfp_hires,
                smoothen_bandpow, ann_MEGartif_prefix_to_use,
                anndict_per_intcat_per_rawn= anndict_per_intcat_per_rawn)

    #raws_flt_pri_perband_ = {}

    stats_multiband_flt_per_ct = {}
    stats_multiband_bp_per_ct  = {}

    for combine_type in gv.rawnames_combine_types_rawdata:


        #means_perband_flt_pri, stds_perband_flt_pri = \
        indsets, means_per_indset_per_band_flt, stds_per_indset_per_band_flt, stats_per_indset_per_band_flt = \
        ugf.gatherMultiBandStats(rawnames,raw_perband_flt_pri, times_pri,
                                    chnames_perband_flt_pri, side_switched_pri, sfreq,
                                    intervals_for_stats, combine_type,
                                    artif_handling, require_intervals_present = [baseline_int],
                                    bindict_per_rawn=bindict_per_rawn)
        #d = {}
        #d['means'] = means_perband_flt_pri
        #d['stds'] = stds_perband_flt_pri
        #d['rawnames'] = rawnames
        #stats_multiband_flt_per_ct[combine_type] = d
        curstatinfo = {'indsets':indsets, 'rawnames':rawnames, 'stats_per_indset':stats_per_indset_per_band_flt }
        curstatinfo['means' ]  = means_per_indset_per_band_flt
        curstatinfo['stds' ]  = stds_per_indset_per_band_flt
        stats_multiband_flt_per_ct[combine_type] = curstatinfo

        #means_perband_bp_pri,stds_perband_bp_pri = \
        indsets, means_per_indset_per_band_bp, stds_per_indset_per_band_bp, stats_per_indset_per_band_bp =  \
        ugf.gatherMultiBandStats(rawnames,raw_perband_bp_pri, times_pri,
                                    chnames_perband_bp_pri, side_switched_pri, sfreq,
                                    intervals_for_stats, combine_type,
                                    artif_handling, require_intervals_present = [baseline_int],
                                    bindict_per_rawn=bindict_per_rawn)
        #d = {}
        #d['means'] = means_perband_bp_pri
        #d['stds'] = stds_perband_bp_pri
        #d['rawnames'] = rawnames
        #stats_multiband_bp_per_ct[combine_type] = d

        curstatinfo = {'indsets':indsets, 'rawnames':rawnames, 'stats_per_indset':stats_per_indset_per_band_bp }
        curstatinfo['means' ]  = means_per_indset_per_band_bp
        curstatinfo['stds' ]  = stds_per_indset_per_band_bp
        stats_multiband_bp_per_ct[combine_type] = curstatinfo


    if save_stats:
        subjs_analyzed, subjs_analyzed_glob = upre.getRawnameListStructure(rawnames, ret_glob=True)
        #ind_strs = map(lambda x: int(x[1:3]), subjs_analyzed.keys() )
        #inds_str = ','.join( map(str, list(sorted(ind_strs)) ) )
        #inds_str = ','.join( sorted(subjs_analyzed.keys() ) )
        #nr = len(rawnames)
        #fname_stats = 'stats_{}_{}_'.format(inds_str,nr)  + fn_suffix_dat
        fname_stats = utils.genStatsMultiBandFn(rawnames, new_main_side, data_modalities,
                                    use_main_LFP_chan, src_file_grouping_ind,
                                    src_grouping, bands_precision )
        fname_stats_full = pjoin( gv.data_dir, input_subdir, fname_stats)
        print('Saving ',fname_stats_full)
        np.savez(fname_stats_full, stats_multiband_bp_per_ct=stats_multiband_bp_per_ct,
                stats_multiband_flt_per_ct=stats_multiband_flt_per_ct,
                rawnames=rawnames, cmd=(opts,args))
