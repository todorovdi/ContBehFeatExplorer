import os,sys
import mne
import utils  #my code
import json
import matplotlib.pyplot as plt
import numpy as np
import h5py
import multiprocessing as mpr
import matplotlib as mpl
import time
import gc;
import scipy.signal as sig

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import utils_tSNE as utsne
import globvars as gv

from matplotlib.backends.backend_pdf import PdfPages
import utils_preproc as upre

from globvars import gp

mpl.use('Agg')

############################

if os.environ.get('DATA_DUSS') is not None:
    data_dir = os.path.expandvars('$DATA_DUSS')
else:
    data_dir = '/home/demitau/data'

############################

rawname_ = 'S01_on_hold'  # no trem
#rawname_ = 'S01_on_move'
rawname_ = 'S01_off_hold'
#rawname_ = 'S01_off_move'
#
#rawname_ = 'S02_off_hold'
#rawname_ = 'S02_on_hold'

rawnames = [rawname_]



#############################################################
#######  Main params

use_main_tremorside = 1  # 0 both , -1 opposite
bands_only = 'fine'  #'fine'  or 'no'
#subsample_type = 'half'
#subsample_type = 'prescribedSkip'  # or 'desiredNpts'
#desiredNpts = 7000
#desiredNpts = 4000
desiredNpts = 5000
#prescribedSkip = 2  # #  32 * 4 = 128 = 256/2
data_modalities = ['lfp', 'msrc']
#data_modalities = ['lfp']
#data_modalities = ['msrc']

prefix = ''

n_channels = 7  # should it be hardcoded?

extend = 3

skip_PCA = 32
skip_feat = 32
subskip = 2

n_feats = 609
n_feats_PCA = 609
dim_PCA = 329  # number
dim_inp_tSNE = 120

nfeats_per_comp_LDA = -1
# this is used to determine if we take data
# from PCA fitted to single dataset or to merged dataset
nraws_used_PCA = 2

crop_time = -1

#######   tSNE params

#perplex_values = [5, 10, 30, 40, 50]
#seeds = range(5)
#lrate = 200.

lrate = 200.  # for tSNE only
#perplex_values = [10, 30, 50, 65]
perplex_values = [10, 30, 50]
n_neighbors_values = [5, 10, 20]
seeds = range(0,2)

#######################

##########################  ploting params

load_tSNE                    = 0
save_tSNE                    = 1
use_existing_tSNE            = 1
use_existing_feat            = 1

show_plots                   = 1
do_tSNE                      = 1
#
do_plot_feat_timecourse_full = 0 * show_plots
do_plot_feat_stats_full      = 0 * show_plots
do_plot_feat_timecourse      = 0 * show_plots
do_plot_feat_stats           = 1 * show_plots
do_plot_CSD                  = 0 * show_plots
do_plot_PCA                  = 0 * show_plots

nPCAcomponents_to_plot = 5

set_explicit_dim_PCA = 0
set_explicit_n_feats_PCA = 0
set_explicit_nraws_used_PCA = 0
####################  data processing params
windowsz  =  1 * 256

src_file_grouping_ind = 9  # motor-related_vs_CB_vs_rest
src_grouping = 0  # src_grouping is used to get info from the file
##############################
import sys, getopt

effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

print(effargv)
sources_type = ''

use_avCV_LDA = 1

helpstr = 'Usage example\nrun_tSNE.py --rawnames <rawname_naked1,rawnames_naked2> '
opts, args = getopt.getopt(effargv,"hr:n:s:w:p:",
        ["rawnames=", "n_channels=", "skip_feat=", "windowsz=",
         "show_plots=", 'subskip=', 'n_feats=', 'n_feats_PCA=', 'dim_PCA=',
         'dim_inp_tSNE=', 'perplex_values=', 'seeds_tSNE=', 'lrate_tSNE=',
         'use_existing_tSNE=', 'load_tSNE=', 'prefix=', 'crop=', 'nrPCA=',
         'nfeats_per_comp_LDA=', 'sources_type=', 'skip_calc=',
         "src_grouping=", "src_grouping_fn=" ])
print(sys.argv, opts, args)

for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == '--skip_calc':
        do_tSNE = not bool(int(arg))
    elif opt in ('-r','--rawnames'):
        rawnames = arg.split(',')
    elif opt == "--n_channels":
        n_channels = int(arg)
    elif opt == '--prefix':
        prefix = arg
    elif opt == "--src_grouping":
        src_grouping = int(arg)
    elif opt == "--src_grouping_fn":
        src_file_grouping_ind = int(arg)
    elif opt == "--skip_feat":
        skip_feat = int(arg)
    elif opt == "--subskip":
        subskip = int(arg)
    elif opt == '--show_plots':
        show_plots = int(arg)
    elif opt == '--sources_type':
        sources_type = arg
    elif opt == "--windowsz":
        windowsz = int(arg)
    elif opt == "--n_feats":
        n_feats = int(arg)
    elif opt == "--nfeats_per_comp_LDA":
        nfeats_per_comp_LDA = int(arg)
    elif opt == "--n_feats_PCA":
        n_feats_PCA = int(arg)
        set_explicit_n_feats_PCA = 1
    elif opt == "--dim_PCA":
        dim_PCA = int(arg)
        set_explicit_dim_PCA = 1
    elif opt == "--dim_inp_tSNE":
        dim_inp_tSNE = int(arg)
    elif opt == "--preplex_values":
        preplex_values = map(float, arg.split(',') )
    elif opt == "--seeds_tSNE":
        seeds = map(int, arg.split(',') )
    elif opt == "--lrate_tSNE":
        lrate = float( arg )
    elif opt == '--use_existing_tSNE':
        use_existing_tSNE = int(arg)
    elif opt == '--load_tSNE':
        load_tSNE = int(arg)
    elif opt == '--crop':
        crop_time = float(arg)
    elif opt == '--nrPCA':
        nraws_used_PCA = int(arg)
        set_explicit_nraws_used_PCA = 1
    else:
        raise ValueError('Unk option {}'.format(str(arg) ) )

print('!!!!!! current rawnames --- ',rawnames)

if dim_inp_tSNE < 0:
    dim_inp_tSNE = dim_PCA
############################

a = '{}_feats_{}chs_skip{}_wsz{}.npz'.\
    format(rawname_,n_channels, skip_feat, windowsz)
fname_feat_full = os.path.join( data_dir,a)


#############################################################


#############################################################


# get info about bad MEG channels (from separate file)
with open('subj_info.json') as info_json:
    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)


tasks = []
for rawname_ in rawnames:
    subj,medcond,task  = utils.getParamsFromRawname(rawname_)
    tasks += [task]

# for current raw
maintremside = gen_subj_info[subj]['tremor_side']
mainLFPchan = gen_subj_info[subj]['lfpchan_used_in_paper']
tremfreq_Jan = gen_subj_info[subj]['tremfreq']
print('main trem side and LFP ',maintremside, mainLFPchan)


mts_letter = maintremside[0].upper()


################### Load PCA


pcapts_pri = []
ldapts_pri = []
pca_pri = []
lda_pri = []
lda_avCV_pri = []
Xtimes_almost_pri = []
PCA_info_pri = []
feat_info_pri = []
lda_pg_pri = []
rawtimes_pri = []

for rawname_ in rawnames:
    subj,medcond,task  = utils.getParamsFromRawname(rawname_)

    if len(prefix):
        if set_explicit_nraws_used_PCA:
            regex_nrPCA = str(nraws_used_PCA)
        else:
            regex_nrPCA = '[0-9]+'
        if set_explicit_n_feats_PCA:
            regex_nfeats = str(n_feats_PCA)
        else:
            regex_nfeats = '[0-9]+'
        if set_explicit_dim_PCA:
            regex_pcadim = str(dim_PCA)
        else:
            regex_pcadim = '[0-9]+'
        regex = '{}_{}_grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_pcadim({}).*'.\
            format(rawname_, sources_type, src_file_grouping_ind, src_grouping,
                   prefix, regex_nrPCA, regex_nfeats, regex_pcadim)

        # here prefix should be without '_' in front or after
        fnfound = utsne.findByPrefix(data_dir, rawname_, prefix, regex=regex)
        if len(fnfound) > 1:
            fnt = [0] * len(fnfound)
            for fni in range(len(fnt) ):
                fnfull = os.path.join(data_dir, fnfound[fni])
                fnt[fni] = os.path.getmtime(fnfull)
            fni_max = np.argmax(fnt)
            fnfound = [ fnfound[fni_max] ]
        assert len(fnfound) == 1, 'For {} with regex {} found not single fnames {}'.format(rawname_,regex,fnfound)
        fname_PCA_full = os.path.join( data_dir, fnfound[0] )
    else:
        prefix += '_'
        out_name_templ = '{}_{}PCA_nr{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}'
        out_name = (out_name_templ ).\
            format(sources_type,prefix, nraws_used_PCA, n_channels, n_feats_PCA, dim_PCA, skip_PCA, windowsz)
        fname_PCA_full = os.path.join( data_dir, '{}{}.npz'.format(rawname_,out_name))

    print('run_tSNE: Loading PCA from {}'.format(fname_PCA_full) )
    f = np.load(fname_PCA_full, allow_pickle=1)
    pcapts_cur = f['pcapts']
    pca_cur = f['pcaobj'][()]

    lda_pg_cur = f['lda_output_pg'][()]
    lda_pg_pri += [lda_pg_cur]

    #lda_output = lda_pg_cur['merge_all_not_trem']['basic']
    lda_output = lda_pg_cur['merge_nothing']['basic']

    #ldapts_cur = lda_output['transformed_imputed']
    ldapts_cur = lda_output['transformed_imputed_CV']
    lda_cur    = lda_output['ldaobj']

    pca_pri += [pca_cur]
    lda_pri += [lda_cur]
    lda_avCV_pri += [lda_output['ldaobj_avCV'] ]

    pcapts_pri += [pcapts_cur]
    ldapts_pri += [ldapts_cur]

    feature_names_all = f['feature_names_all']
    feat_info = f['feat_info'][()]
    nedgeBins = feat_info['nedgeBins']

    rawtimes_pri += [ f['rawtimes'] ]

    #TODO: set dim_PCA and   n_feats_PCA from data here
    dim_PCA = pcapts_cur.shape[1]
    n_feats_PCA = len(feature_names_all)


    sfreq = int(feat_info.get('sfreq', 256) )

    PCA_info = f['info'][()]

    PCA_info_pri += [PCA_info]
    feat_info_pri += [feat_info]

    Xtimes_almost_pri += [ f['Xtimes'] ]



    #if do_plot_PCA:
    #    utsne.plotPCA(pcapts_cur,pca_cur, nPCAcomponents_to_plot,feature_names_all, colors, markers,
    #                mrk, mrknames, color_per_int_type, task,
    #                pdf=pdf,neutcolor=neutcolor)

if dim_inp_tSNE < dim_PCA:
    print('Of PCA with dim {} using only {} first dimensions'.format(dim_PCA,dim_inp_tSNE) )
else:
    dim_inp_tSNE = dim_PCA
pcapts = np.vstack(pcapts_pri)[:, :dim_inp_tSNE]
ldapts = np.vstack(ldapts_pri)


#assert len(Xtimes_almost_pri) <= 2
#if len(Xtimes_almost_pri) == 2:
#    dt = Xtimes_almost_pri[0][1] - Xtimes_almost_pri[0][0]
#    timeshift = Xtimes_almost_pri[0][-1] + dt
#    Xtimes_almost = np.hstack( [Xtimes_almost_pri[0], Xtimes_almost_pri[1] + timeshift ] )
#
#    anns = anns_pri[0]
#    anns.append(anns_pri[1].onset + timeshift,anns_pri[1].duration,anns_pri[1].description)
#elif len(Xtimes_almost_pri)==1:
#    anns = anns_pri[0]
#    Xtimes_almost = Xtimes_almost_pri[0]

side_switch_happened_pri = [ fi['side_switched'] for fi in feat_info_pri ]


anns, anns_pri, Xtimes_almost, dataset_bounds = utsne.concatAnns(rawnames, Xtimes_almost_pri,
                                                                 side_rev_pri=side_switch_happened_pri)

# TODO: uncomment when I fix prepColorMarkers
#anns, anns_pri, times_pri, dataset_bounds = utsne.concatAnns(rawnames, rawtimes_pri,
#                                                                 side_rev_pri=side_switch_happened_pri)


if crop_time > 0:
    end_ind = np.where(Xtimes_almost < crop_time)[0][-1]
    Xtimes_almost = Xtimes_almost[:end_ind+1]
    pcapts = pcapts[:end_ind+1]
    ldapts = ldapts[:end_ind+1]

Xtimes = Xtimes_almost[::subskip]
X = utsne.downsample(pcapts, subskip)

#lda_to_use = lda_pri[0]
lda_to_use = lda_avCV_pri[0]
#if not:

if nfeats_per_comp_LDA > 0:
    r = utsne.getImporantCoordInds(lda_to_use.coef_, nfeats_show = nfeats_per_comp_LDA, q=0.8, printLog = 1)
    inds_toshow, strong_inds_pc, strongest_inds_pc  = r
    print('From LDA using {} features'.format( len(inds_toshow ) ) )
    ldapts_ = np.matmul(ldapts , lda_to_use.coef_[:,inds_toshow ].T  ) + lda_to_use.intercept_
else:
    print('From LDA using ALL features' )
    ldapts_ = ldapts

X_LDA = utsne.downsample(ldapts_, subskip)
print('Loaded PCAs together have shape {}'.format(pcapts.shape ) )
assert len(Xtimes) == len(X), (len(Xtimes),len(X))
assert len(Xtimes) == len(X_LDA)


#if subsample_type == 'half':
#    subskip = 4 #  32 * 4 = 128 = 256/2
#elif subsample_type == 'desiredNpts':
#    subskip = pcapts.shape[0] // desiredNpts
#elif subsample_type == 'prescribedSkip':
#    subskip = prescribedSkip

totskip = skip_feat * subskip
print('totskip = {},  Xtimes number = {}'.format(totskip, Xtimes.shape[0  ] ) )

################  Prep colors
mrk = ['<','>','o','^','v']
mrknames = ['_pres','_posts','','_pree','_poste']

tremcolor = 'r'
notremcolor = 'g'
movecolor = 'b'  #c,y
holdcolor = 'purple'  #c,y
neutcolor = 'grey'

color_per_int_type = { 'trem':tremcolor, 'notrem':notremcolor, 'neut':neutcolor,
                      'move':movecolor, 'hold':holdcolor }


colors, markers =utsne.prepColorsMarkers(mts_letter, anns, Xtimes,
           nedgeBins, windowsz, sfreq, totskip, mrk,mrknames, color_per_int_type,
                                        dataset_bounds = dataset_bounds)



###############################  tSNE
from umap import UMAP

if do_tSNE:

    legend_elements = utsne.prepareLegendElements(mrk,mrknames,color_per_int_type, tasks )

    print('Starting tSNE')
    # function to be run in parallel

    def run_tsne(p):
        t0 = time.time()
        pi,si, pts, seed, param, lrate, dtype, dim_red_alg = p
        print('Starting {} for {} points of shape {}'.format(dim_red_alg,dtype,pts.shape) )
        #n_component is desired dim of output

        if dim_red_alg == 'UMAP':
            reducer = UMAP(n_neighbors=param)
        elif dim_red_alg == 'tSNE':
            reducer = TSNE(n_components=2, random_state=seed, perplexity=param, learning_rate=lrate)
        else:
            raise ValueError('wrong dim_red_alg')
        #X_embedded = tsne.fit_transform(pts)
        #reducer.fit(pcapts[good_inds])
        #X_embedded = reducer.transform(pcapts)

        X_embedded = reducer.fit_transform(pts)


        dur = time.time() - t0
        print('tSNE {} computed in {:.3f}s: param = {};  lrate = {}; seed = {}'.
            format(pts.shape, dur,param, lrate, seed))

        return pi,si,X_embedded, seed, param, lrate, dtype, dim_red_alg


    #perplex_values = [30]
    #seeds = [0]
    use_tSNE = 0
    use_UMAP = 1

    params_per_alg_type = {}
    if use_tSNE:
        params_per_alg_type['tSNE'] = perplex_values
    if use_UMAP:
        params_per_alg_type['UMAP'] = n_neighbors_values
    dim_red_algs = list(params_per_alg_type.keys() )

    res = []
    args = []
    for dim_red_alg in dim_red_algs:
        for pi in range(len(perplex_values) ):
            param = params_per_alg_type[dim_red_alg][pi]
            subres = []
            for si,seed in enumerate(seeds):

                if si == 0:
                    dat_to_use = X.copy()
                    dtype = 'PCA'
                elif si == 1:
                    dat_to_use = X_LDA.copy()
                    dtype = 'LDA'
                args += [ (pi,si, dat_to_use, seed, param, lrate, dtype, dim_red_alg)]


    rn_str = ','.join(rawnames)
    #TODO put before tSNE
    if prefix[-1] != '_':
        prefix += '_'
    out_name_templ = '{}_{}tSNE_nrPCA{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}_effdim{}_subskip{}'
    if crop_time > 0:
        out_name_templ += '_crop{.1f}'.format(crop_time)

    out_name = (out_name_templ ).\
        format(rn_str,prefix,nraws_used_PCA,
               n_channels, n_feats_PCA, dim_PCA, skip_PCA, windowsz, dim_inp_tSNE, subskip)





    #a = '{}_tsne_{}chs_skip{}_wsz{}.npz'.format(rawname_,n_channels, totskip, windowsz)
    fname_tsne_full = os.path.join( data_dir, out_name + '.npz')

    have_tSNE = False
    try:
        len(tSNE_result)
    except NameError as e:
        have_tSNE = False
    else:
        have_tSNE = True

    if not (have_tSNE and use_existing_tSNE):
        if load_tSNE and os.path.exists(fname_tsne_full):
            print('Loading tSNE from ',fname_tsne_full)
            ff = np.load(fname_tsne_full, allow_pickle=True)
            tSNE_result =  ff['tSNE_result']
            feat_info_pri = ff['feat_info_pri'][()]
            PCA_info_pri = ff['PCA_info_pri'][()]
        else:
            ncores = min(len(args) , mpr.cpu_count()- gp.n_free_cores)
            if ncores > 1:
                pool = mpr.Pool(ncores)
                print('tSNE:  Starting {} workers on {} cores'.format(len(args), ncores))
                tSNE_result = pool.map(run_tsne, args)

                pool.close()
                pool.join()
            else:
                tSNE_result = [run_tsne(args[0])]

            if save_tSNE:
                print('Saving tSNE to {}'.format(fname_tsne_full) )
                np.savez(fname_tsne_full, tSNE_result=tSNE_result, colors=colors,
                         markers=markers, legend_elements=legend_elements,
                         PCA_info_pri = PCA_info_pri, feat_info_pri = feat_info_pri,
                         anns=anns, Xtimes_almost_pri = Xtimes_almost_pri )
    else:
        assert len(tSNE_result) == len(args)
        print('Using exisitng tSNE')

if show_plots and do_tSNE:
    print('Starting tSNE plotting ')
    pdf= PdfPages( os.path.join(gv.dir_fig, out_name+'.pdf' ) )

    #cols = [colors, colors2, colors3]
    cols = [colors]

    colind = 0
    nr = len(seeds) * len(dim_red_algs )
    nc = len(perplex_values)
    ww = 8; hh=8
    fig,axs = plt.subplots(ncols =nc, nrows=nr, figsize = (nc*ww, nr*hh))
    if not isinstance(axs,np.ndarray):
        axs = np.array([[axs]] )
    # for pi,pv in enumerate(perplex_values):
    #     for si,sv in enumerate(seeds):
    for tpl in tSNE_result:
        #for algi in range(len(dim_red_algs)):
        pi,si,X_embedded, seed, param, lrate, dtype, alg = tpl
        algi = dim_red_algs.index(alg)
        ax = axs[si + len(seeds)*algi,pi ]
        #X_embedded = res[si][pi]
        #ax.scatter(X_embedded[:,0], X_embedded[:,1], c = cols[colind], s=5)

        utsne.plotMultiMarker(ax,X_embedded[:,0], X_embedded[:,1], c = cols[colind], s=5,
                                m=markers)
        ax.set_title('{} from {} param = {};\nlrate = {}; seed = {}'.
                    format(alg, dtype, param, lrate, seed))

    axs[0,0].legend(handles=legend_elements)
    #figname = 'tSNE_inpdim{}_PCAdim{}_Npts{}_hside{}_skip{}_wsz{}_lrate{}.pdf'.\
    #    format(n_feats, pcapts.shape[-1], X.shape, mts_letter, totskip, windowsz, lrate)


    str_feats = ','.join(PCA_info['features_to_use'])
    str_mods = ','.join(PCA_info['data_modalities'])
    out_name += '\nmainLFP{}_HFO{}_{}_{}\nXshape={}'.\
        format(int(PCA_info['use_main_LFP_chan']), int(PCA_info['use_lfp_HFO']),
               str_mods, str_feats, X.shape)

    figname = out_name + '.pdf'
    plt.suptitle(figname)
    #plt.savefig(rawname_ +'_'+figname)

    pdf.savefig()
    pdf.close()

#plt.savefig(figname)
gc.collect()
