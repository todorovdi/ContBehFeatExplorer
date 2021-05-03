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
from umap import UMAP

from globvars import gp
import datetime


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

#subsample_type = 'half'
#subsample_type = 'prescribedSkip'  # or 'desiredNpts'
#desiredNpts = 7000
#desiredNpts = 4000
desiredNpts = 5000
#prescribedSkip = 2  # #  32 * 4 = 128 = 256/2

prefix = ''

n_channels = 7  # should it be hardcoded?

extend = 3

skip_ML = 32
skip_feat = 32
subskip = 2

n_feats = 609
n_feats_PCA = 609
dim_PCA = 329  # number
dim_inp_nlproj = 120

nfeats_per_comp_LDA = -1
# this is used to determine if we take data
# from PCA fitted to single dataset or to merged dataset
nraws_used_PCA = 2

crop_time = -1

#######   nlproj params

#perplex_values = [5, 10, 30, 40, 50]
#seeds = range(5)
#lrate = 200.

lrate = 200.  # for nlproj only
#perplex_values = [10, 30, 50, 65]
perplex_values = [10, 30, 50]
n_neighbors_values = [5, 10, 20]
seeds = range(0,2)

seeds = [0]

#######################

##########################  ploting params

load_nlproj                    = 0
save_nlproj                    = 1
use_existing_nlproj            = 1

show_plots                   = 1
do_nlproj                      = 1
#
do_plot_PCA                  = 0 * show_plots

nPCAcomponents_to_plot = 5

set_explicit_dim_PCA = 0
set_explicit_n_feats_PCA = 0
set_explicit_nraws_used_PCA = 0
####################  data processing params
windowsz  =  1 * 256

src_file_grouping_ind = 10  # motor-related_vs_CB_vs_rest
src_grouping = 0  # src_grouping is used to get info from the file
##############################
import sys, getopt

effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three
else:
    mpl.use('Agg')

print(effargv)
sources_type = 'parcel_aal'

use_avCV_LDA = 1
input_subdir = ""
output_subdir = ""

feat_subset_types = ['all_present_features', 'strongest_features_LDA_selMinFeatSet', 'best_PCA-derived_features_0.75',
             'strongest_features_XGB_opinion']
#points_type = ["PCA", "LDA_CV_aver", "X_XGB_bestfeats"]
points_type = ["PCA", "X_XGB_bestfeats" ]
points_type += feat_subset_types

LDA_scalings_type_to_use = 'CV_aver'
it_grouping = 'merge_nothing'
it_set = 'basic'

helpstr = 'Usage example\nrun_nlproj.py --rawnames <rawname_naked1,rawnames_naked2> '
opts, args = getopt.getopt(effargv,"hr:n:s:w:p:",
        ["rawnames=", "n_channels=", "skip_feat=", "windowsz=",
         "show_plots=", 'subskip=', 'n_feats=', 'n_feats_PCA=', 'dim_PCA=',
         'dim_inp_nlproj=', 'perplex_values=',
         'n_neighbors_values=', 'seeds_nlproj=', 'lrate_tSNE=',
         'use_existing_nlproj=', 'load_nlproj=', 'prefix=', 'crop=', 'nrML=',
         'nfeats_per_comp_LDA=', 'sources_type=', 'skip_calc=',
         "src_grouping=", "src_grouping_fn=", "input_subdir=", "output_subdir=",
         "load_only=", "points_type=", "it_grouping=", "it_set=" ])
print(sys.argv, opts, args)

for opt, arg in opts:
    print(opt,arg)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == '--load_only':
        do_nlproj = not bool(int(arg))
    elif opt in ('-r','--rawnames'):
        rawnames = arg.split(',')
    elif opt == "--n_channels":
        n_channels = int(arg)
    elif opt == "--it_grouping":
        it_grouping = arg
    elif opt == "--it_set":
        it_set = arg
    elif opt == '--prefix':
        prefix = arg
    elif opt == "--src_grouping":
        src_grouping = int(arg)
    elif opt == "--src_grouping_fn":
        src_file_grouping_ind = int(arg)
    elif opt == "--input_subdir":
        input_subdir = arg
        if len(input_subdir) > 0:
            subdir = os.path.join(gv.data_dir,input_subdir)
            assert os.path.exists(subdir )
    elif opt == "--output_subdir":
        output_subdir = arg
        if len(output_subdir) > 0:
            subdir = os.path.join(gv.data_dir,output_subdir)
            if not os.path.exists(subdir ):
                print('Creating output subdir {}'.format(subdir) )
                os.makedirs(subdir)
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
    elif opt == "--dim_inp_nlproj":
        dim_inp_nlproj = int(arg)
    elif opt == "--preplex_values":
        perplex_values = map(float, arg.split(',') )
    elif opt == "--seeds_nlproj":
        seeds = map(int, arg.split(',') )
    elif opt == "--points_type":
        points_type = arg.split(',')
    elif opt == "--lrate_tSNE":
        lrate = float( arg )
    elif opt == '--use_existing_nlproj':
        use_existing_nlproj = int(arg)
    elif opt == '--load_nlproj':
        load_nlproj = int(arg)
    elif opt == '--crop':
        crop_time = float(arg)
    elif opt == '--nrML':
        nraws_used_PCA = int(arg)
        set_explicit_nraws_used_PCA = 1
    else:
        raise ValueError('Unk option {}'.format(str(arg) ) )

print('!!!!!! current rawnames --- ',rawnames)

if dim_inp_nlproj < 0:
    dim_inp_nlproj = dim_PCA
############################

# get info about bad MEG channels (from separate file)
with open('subj_info.json') as info_json:
    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)


tasks = []
for rawname_ in rawnames:
    subj,medcond,task  = utils.getParamsFromRawname(rawname_)
    tasks += [task]



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
wbd_pri = []
strong_inds_XGB_pri = []
lda_anver_pri = []

X_pri = []
Ximp_pri = []


X_imp_valid = True

for rawname_ in rawnames:
    subj,medcond,task  = utils.getParamsFromRawname(rawname_)

    if len(prefix):
        if set_explicit_nraws_used_PCA:
            regex_nrML = str(nraws_used_PCA)
        else:
            regex_nrML = '[0-9]+'
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
                   prefix, regex_nrML, regex_nfeats, regex_pcadim)

        # here prefix should be without '_' in front or after
        inp_subdir = os.path.join(gv.data_dir, input_subdir)
        fnfound = utsne.findByPrefix(inp_subdir, rawname_, prefix, regex=regex)
        if len(fnfound) > 1:
            fnt = [0] * len(fnfound)
            for fni in range(len(fnt) ):
                fnfull = os.path.join(inp_subdir, fnfound[fni])
                fnt[fni] = os.path.getmtime(fnfull)
            fni_max = np.argmax(fnt)
            fnfound = [ fnfound[fni_max] ]
        assert len(fnfound) == 1, 'For {} with regex {} in {} found not single fnames {}'.\
            format(rawname_,regex,inp_subdir, fnfound)
        fname_PCA_full = os.path.join( inp_subdir, fnfound[0] )
    else:
        prefix += '_'
        out_name_templ = '{}_{}PCA_nr{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}'
        out_name = (out_name_templ ).\
            format(sources_type,prefix, nraws_used_PCA, n_channels, n_feats_PCA, dim_PCA, skip_ML, windowsz)
        fname_PCA_full = os.path.join( inp_subdir, '{}{}.npz'.format(rawname_,out_name))


    modtime = datetime.datetime.fromtimestamp(os.path.getmtime(fname_PCA_full)  )
    #print('Loading feats from {}, modtime {}'.format(fname_feat_full, modtime ))
    print('run_nlproj: Loading PCA from {} modtime {}'.format(fname_PCA_full, modtime) )
    f = np.load(fname_PCA_full, allow_pickle=1)

    lda_pg_cur = f['lda_output_pg'][()]
    lda_pg_pri += [lda_pg_cur]
    lda_output = lda_pg_cur[it_grouping][it_set]
    lda_anver = lda_output['lda_analysis_versions']

    pcapts_cur = f['pcapts']
    pca_cur = f['pcaobj'][()]
    X_imp = f.get('X_imputed', None)
    Ximp_pri += [X_imp  ]
    if X_imp is None:
        X_imp_valid = False


    #lda_output = lda_pg_cur['merge_all_not_trem']['basic']

    lda_anver_pri += [lda_anver]
    #['strongest_features_LDA_opinion'][

    #ldapts_cur = lda_output['transformed_imputed']
    ldapts_cur = lda_output['transformed_imputed_CV']
    lda_cur    = lda_output['ldaobj']

    pca_pri += [pca_cur]
    lda_pri += [lda_cur]
    lda_avCV_pri += [lda_output['ldaobj_avCV'] ]

    strong_inds_XGB_pri += [ lda_output['strong_inds_XGB'] ]

    pcapts_pri += [pcapts_cur]
    ldapts_pri += [ldapts_cur]


    if 'feature_names_filtered' in f:
        featnames_filtered = f['feature_names_filtered']
    else:
        featnames_filtered = f['feature_names_all']
    feat_info = f['feat_info'][()]
    nedgeBins = feat_info['nedgeBins']

    rawtimes_pri += [ f['rawtimes'] ]

    wbd_pri += [f['wbd'] ]
    X_pri += [f['X'] ]

    #TODO: set dim_PCA and   n_feats_PCA from data here
    dim_PCA = pcapts_cur.shape[1]
    n_feats_PCA = len(featnames_filtered)


    sfreq = int(feat_info.get('sfreq', 256) )

    PCA_info = f['info'][()]

    PCA_info_pri += [PCA_info]
    feat_info_pri += [feat_info]

    Xtimes_almost_pri += [ f['Xtimes'] ]



    #if do_plot_PCA:
    #    utsne.plotPCA(pcapts_cur,pca_cur, nPCAcomponents_to_plot,featnames_filtered, colors, markers,
    #                mrk, mrknames, color_per_int_type, task,
    #                pdf=pdf,neutcolor=neutcolor)

new_main_side_pri= [fi['new_main_side'] for fi in feat_info_pri]
main_side_pri    = [fi['main_side_before_switch'] for fi in feat_info_pri]


# for current raw
#maintremside = gen_subj_info[subj]['tremor_side']
#mainLFPchan = gen_subj_info[subj]['lfpchan_used_in_paper']
#tremfreq_Jan = gen_subj_info[subj]['tremfreq']
#mts_letter = maintremside[0].upper()

assert len(set(new_main_side_pri) ) == 1
#print('main side and LFP ',main_side_pri, mainLFPchan)
print('main sides ',main_side_pri)
ms_letter = new_main_side_pri[0][0].upper()

###########################################

if dim_inp_nlproj < dim_PCA:
    print('Of PCA with dim {} using only {} first dimensions'.format(dim_PCA,dim_inp_nlproj) )
else:
    dim_inp_nlproj = dim_PCA

pcapts = np.vstack(pcapts_pri)[:, :dim_inp_nlproj]
print('Loaded PCAs together have shape {}'.format(pcapts.shape ) )

ldapts = np.vstack(ldapts_pri)

siXGB = strong_inds_XGB_pri[0]
sisetXGB = set(siXGB)
for fdi in range(len(rawnames) ):
    assert set(strong_inds_XGB_pri[fdi] ) == sisetXGB

Xconcat = np.vstack(X_pri)
X_XGBsel = Xconcat[:,siXGB]

if X_imp_valid:
    Xconcat_imputed = np.vstack(Ximp_pri)
    Ximp_XGBsel = Xconcat_imputed[:,siXGB]
else:
    Ximp_XGBsel = None


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


anns, anns_pri, Xtimes_almost, dataset_bounds, wbd_merged = utsne.concatAnns(rawnames, Xtimes_almost_pri,
                                                                 side_rev_pri=side_switch_happened_pri,
                                                                 sfreq=sfreq,
                                                                 wbd_pri = wbd_pri, ret_wbd_merged=1)

# TODO: uncomment when I fix prepColorMarkers
#anns, anns_pri, times_pri, dataset_bounds = utsne.concatAnns(rawnames, rawtimes_pri,
#                                                                 side_rev_pri=side_switch_happened_pri)


if crop_time > 0:
    end_ind = np.where(Xtimes_almost < crop_time)[0][-1]
    Xtimes_almost = Xtimes_almost[:end_ind+1]
    pcapts = pcapts[:end_ind+1]
    ldapts = ldapts[:end_ind+1]

Xtimes = Xtimes_almost[::subskip]
X_PCA = utsne.downsample(pcapts, subskip)

#lda_to_use = lda_pri[0]
lda_to_use = lda_avCV_pri[0]
#if not:

if nfeats_per_comp_LDA > 0:
    r = utsne.getImporantCoordInds(lda_to_use.coef_, nfeats_show = nfeats_per_comp_LDA, q=0.8, printLog = 1)
    inds_toshow, strong_inds_pc, strongest_inds_pc  = r
    print('From LDA using {} features'.format( len(inds_toshow ) ) )
    ldapts_strongLDA = np.matmul(ldapts , lda_to_use.coef_[:,inds_toshow ].T  ) + lda_to_use.intercept_
else:
    ldapts_strongLDA = None

#X_LDA = utsne.downsample(ldapts, subskip)
X_XGBsel = utsne.downsample(X_XGBsel, subskip)
#assert len(Xtimes) == len(X_PCA), (len(Xtimes),len(X_PCA))
#assert len(Xtimes) == len(X_LDA)
#X_LDA = None
#X_XGBsel = None


dat_per_ptype={}
for feat_subset_type_cur in feat_subset_types:
    if feat_subset_type_cur  not in lda_anver:
        print("{} not in lda_anver, skipping",format(feat_subset_type_cur) )
    X_cur_lda_type_ = [lda_anver[feat_subset_type_cur][LDA_scalings_type_to_use]['X_transformed'] for lda_anver in lda_anver_pri ]
    X_cur_lda_type = np.vstack(X_cur_lda_type_)
    X_cur_lda_type = utsne.downsample(X_cur_lda_type, subskip)

    assert len(Xtimes) == len(X_cur_lda_type), (feat_subset_type_cur, len(Xtimes), len(X_cur_lda_type))
    dat_per_ptype[feat_subset_type_cur] = X_cur_lda_type

dat_per_ptype.update(  {'PCA': X_PCA, 'X_XGB_bestfeats':X_XGBsel} )
if Ximp_XGBsel is not None:
    dat_per_ptype['Ximp_XGB_bestfeats'] = X_XGBsel

for k,dpp in dat_per_ptype.items():
    if dpp is None:
        continue
    print(' type vs shape  ',k,dpp.shape)

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


# anns have been prepared taking reversal into consideration
colors, markers =utsne.prepColorsMarkers(anns, Xtimes, nedgeBins, windowsz,
                                         sfreq, totskip, mrk,mrknames,
                                         color_per_int_type, dataset_bounds = dataset_bounds,
                                         side_letter = ms_letter )



###############################  nlproj

if do_nlproj:

    legend_elements = utsne.prepareLegendElements(mrk,mrknames,color_per_int_type, tasks )

    print('Starting calc of nonlinear projections')
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
        print('{} for {} computed in {:.3f}s: param={}; lrate={}; seed={}'.
            format(dim_red_alg,pts.shape, dur,param, lrate, seed))

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
        for point_type in points_type:
            if point_type not in dat_per_ptype:
                print('{} not in dat_to_use, skipping'.format(point_type) )
                continue
            for pi in range(len(perplex_values) ):
                param = params_per_alg_type[dim_red_alg][pi]
                subres = []
                for si,seed in enumerate(seeds):

                    dat_to_use = dat_per_ptype[point_type]
                    dtype = point_type
                    #if si == 0:
                    #    dat_to_use = X_PCA.copy()
                    #    dtype = 'PCA'
                    #elif si == 1:
                    #    dat_to_use = X_LDA.copy()
                    #    dtype = 'LDA'
                    args += [ (pi,si, dat_to_use, seed, param, lrate, dtype, dim_red_alg)]


    rn_str = ','.join(rawnames)
    #TODO put before tSNE
    if prefix[-1] != '_':
        prefix += '_'
    out_name_templ = '{}_{}nlproj_nrML{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}_effdim{}_subskip{}'
    if crop_time > 0:
        out_name_templ += '_crop{.1f}'.format(crop_time)

    out_name = (out_name_templ ).\
        format(rn_str,prefix,nraws_used_PCA,
               n_channels, n_feats_PCA, dim_PCA, skip_ML, windowsz, dim_inp_nlproj, subskip)





    #a = '{}_tsne_{}chs_skip{}_wsz{}.npz'.format(rawname_,n_channels, totskip, windowsz)
    fname_tsne_full = os.path.join( gv.data_dir, output_subdir, out_name + '.npz')

    have_nlproj = False
    try:
        len(nlproj_result)
    except NameError as e:
        have_nlproj = False
    else:
        have_nlproj = True

    if not (have_nlproj and use_existing_nlproj):
        if load_nlproj and os.path.exists(fname_tsne_full):
            print('Loading nlproj from ',fname_tsne_full)
            ff = np.load(fname_tsne_full, allow_pickle=True)
            nlproj_result =  ff['nlproj_result']
            feat_info_pri = ff['feat_info_pri'][()]
            PCA_info_pri = ff['PCA_info_pri'][()]
        else:
            ncores = min(len(args) , mpr.cpu_count()- gp.n_free_cores)
            if ncores > 1:
                print('nlproj:  Starting {} workers on {} cores'.format(len(args), ncores))
                #pool = mpr.Pool(ncores)
                #nlproj_result = pool.map(run_tsne, args)

                #pool.close()
                #pool.join()

                from joblib import Parallel, delayed
                res = Parallel(n_jobs=ncores)(delayed(run_tsne)(arg) for arg in args)
            else:
                nlproj_result = [run_tsne(args[0])]

            if save_nlproj:
                print('Saving nlproj to {}'.format(fname_tsne_full) )
                np.savez(fname_tsne_full, nlproj_result=nlproj_result, colors=colors,
                         markers=markers, legend_elements=legend_elements,
                         PCA_info_pri = PCA_info_pri, feat_info_pri = feat_info_pri,
                         anns=anns, Xtimes_almost_pri = Xtimes_almost_pri )
    else:
        assert len(nlproj_result) == len(args)
        print('Using exisitng nlproj')

if show_plots and do_nlproj:
    print('Starting nlproj plotting ')
    print('!!! only zero seed !! ')
    out_subdir_fig = os.path.join(gv.dir_fig, output_subdir)
    if not os.path.exists(out_subdir_fig):
        os.makedirs( out_subdir_fig)
        print('Creating dir for fig {}'.format(out_subdir_fig) )
    fig_fname = os.path.join(out_subdir_fig, out_name+'.pdf' )
    pdf= PdfPages(fig_fname  )

    #cols = [colors, colors2, colors3]
    cols = [colors]

    colind = 0
    nr = len(points_type) * len(dim_red_algs )
    nc = len(perplex_values)
    ww = 8; hh=8
    fig,axs = plt.subplots(ncols =nc, nrows=nr, figsize = (nc*ww, nr*hh))
    if not isinstance(axs,np.ndarray):
        axs = np.array([[axs]] )
    # for pi,pv in enumerate(perplex_values):
    #     for si,sv in enumerate(seeds):
    for tpl in nlproj_result:
        #print('fdsfsd')
        #for algi in range(len(dim_red_algs)):
        pi,si,X_embedded, seed, param, lrate, dtype, alg = tpl
        algi = dim_red_algs.index(alg)
        rowind = points_type.index(dtype )
        print(dtype,lrate,alg)
        #ax = axs[si + len(seeds)*algi,pi ]
        ax = axs[rowind,pi ]
        #X_embedded = res[si][pi]
        #ax.scatter(X_embedded[:,0], X_embedded[:,1], c = cols[colind], s=5)

        utsne.plotMultiMarker(ax,X_embedded[:,0], X_embedded[:,1], c = cols[colind], s=5,
                                m=markers)
        ax.set_title('{} from {} param = {};\nlrate = {}; seed = {}, in_shape={}'.
                    format(alg, dtype, param, lrate, seed, dat_per_ptype[dtype].shape ))

    axs[0,0].legend(handles=legend_elements)
    #figname = 'nlproj_inpdim{}_PCAdim{}_Npts{}_hside{}_skip{}_wsz{}_lrate{}.pdf'.\
    #    format(n_feats, pcapts.shape[-1], X_PCA.shape, mts_letter, totskip, windowsz, lrate)


    str_feats = ','.join(PCA_info['features_to_use'])
    str_mods = ','.join(PCA_info['data_modalities'])
    fig_title = out_name + '\nmainLFP{}_HFO{}_{}_{}\nXshape={}'.\
        format(int(PCA_info['use_main_LFP_chan']), int(PCA_info['use_lfp_HFO']),
               str_mods, str_feats, X_PCA.shape)

    figname = out_name # + '.pdf'
    plt.suptitle(fig_title)
    #print('Savfing fig to ',figname)
    #plt.savefig(rawname_ +'_'+figname)

    pdf.savefig()
    pdf.close()

    print('Save figure to ',fig_fname)

#plt.savefig(figname)
gc.collect()
