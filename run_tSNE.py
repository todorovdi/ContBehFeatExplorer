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

from matplotlib.backends.backend_pdf import PdfPages
import utils_preproc as upre

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

print('!!!!!! current rawnames --- ',rawnames)


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

n_feats = 600
n_feats_PCA = 600
dim_PCA = 329  # number
dim_inp_tSNE = 40

crop_time = -1

#######   tSNE params

#perplex_values = [5, 10, 30, 40, 50]
#seeds = range(5)
#lrate = 200.

lrate = 200.
#perplex_values = [10, 30, 50, 65]
perplex_values = [10, 30, 50]
seeds = range(0,2)

#######################

##########################  ploting params

load_tSNE                    = 0
save_tSNE                    = 1
use_existing_tSNE            = 1
load_feat                    = 0
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

####################  data processing params
windowsz  =  1 * 256

##############################
import sys, getopt

effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

print(effargv)

helpstr = 'Usage example\nrun_tSNE.py --rawnames <rawname_naked1,rawnames_naked2> '
opts, args = getopt.getopt(effargv,"hr:n:s:w:p:",
        ["rawnames=", "n_channels=", "skip_feat=", "windowsz=",
         "show_plots=", 'subskip=', 'n_feats=', 'n_feats_PCA=', 'dim_PCA=',
         'dim_inp_tSNE=', 'perplex_values=', 'seeds_tSNE=', 'lrate_tSNE=',
         'use_existing_tSNE=', 'load_tSNE=', 'prefix=', 'crop='])
print(sys.argv, opts, args)

for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt in ('-r','--rawnames'):
        rawnames = arg.split(',')
    elif opt == "--n_channels":
        n_channels = int(arg)
    elif opt == '--prefix':
        prefix = arg + '_'
    elif opt == "--skip_feat":
        skip_feat = int(arg)
    elif opt == "--subskip":
        subskip = int(arg)
    elif opt == '--show_plots':
        show_plots = int(arg)
    elif opt == "--windowsz":
        windowsz = int(arg)
    elif opt == "--n_feats":
        n_feats = int(arg)
    elif opt == "--n_feats_PCA":
        n_feats_PCA = int(arg)
    elif opt == "--dim_PCA":
        dim_PCA = int(arg)
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
    else:
        raise ValueError('Unk option {}'.format(str(arg) ) )


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
        #raise TypeError

    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)


tasks = []
ivalis_pri = []
anns_pri = []
for rawname_ in rawnames:
    subj,medcond,task  = utils.getParamsFromRawname(rawname_)
    tasks += [task]

    anns_fn = rawname_ + '_anns.txt'
    anns_fn_full = os.path.join(data_dir, anns_fn)
    anns = mne.read_annotations(anns_fn_full)
    #raw.set_annotations(anns)

    anns_pri += [anns]
    ivalis_pri += [ utils.ann2ivalDict(anns) ]

 # for current raw
maintremside = gen_subj_info[subj]['tremor_side']
mainLFPchan = gen_subj_info[subj]['lfpchan_used_in_paper']
tremfreq_Jan = gen_subj_info[subj]['tremfreq']
print('main trem side and LFP ',maintremside, mainLFPchan)


mts_letter = maintremside[0].upper()


################### Load PCA


pcapts_pri = []
pca_pri = []
Xtimes_almost_pri = []
for rawname_ in rawnames:
    subj,medcond,task  = utils.getParamsFromRawname(rawname_)

    out_name_templ = '_{}PCA_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}'
    out_name = (out_name_templ ).\
        format(prefix,n_channels, n_feats_PCA, dim_PCA, skip_PCA, windowsz)
    fname_PCA_full = os.path.join( data_dir, '{}{}.npz'.format(rawname_,out_name))

    f = np.load(fname_PCA_full, allow_pickle=1)
    pcapts_cur = f['pcapts']
    pca_cur = f['pcaobj'][()]


    pcapts_pri += [pcapts_cur]

    feature_names_all = f['feature_names_all']
    feat_info = f['feat_info'][()]
    nedgeBins = feat_info['nedgeBins']

    sfreq = int(feat_info.get('sfreq', 256) )

    PCA_info = f['info'][()]

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


assert len(Xtimes_almost_pri) <= 2
if len(Xtimes_almost_pri) == 2:
    dt = Xtimes_almost_pri[0][1] - Xtimes_almost_pri[0][0]
    timeshift = Xtimes_almost_pri[0][-1] + dt
    Xtimes_almost = np.hstack( [Xtimes_almost_pri[0], Xtimes_almost_pri[1] + timeshift ] )

    anns = anns_pri[0]
    anns.append(anns_pri[1].onset + timeshift,anns_pri[1].duration,anns_pri[1].description)
elif len(Xtimes_almost_pri)==1:
    anns = anns_pri[0]
    Xtimes_almost = Xtimes_almost_pri[0]

if crop_time > 0:
    np.where(Xtimes_almost)


if crop_time > 0:
    end_ind = np.where(Xtimes_almost < crop_time)[0][-1]
    Xtimes_almost = Xtimes_almost[:end_ind+1]
    pcapts = pcapts[:end_ind+1]

Xtimes = Xtimes_almost[::subskip]
X = utsne.downsample(pcapts, subskip)
print('Loaded PCAs together have shape {}'.format(pcapts.shape ) )


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


colors,markers =utsne.prepColorsMarkers(mts_letter, anns, Xtimes,
           nedgeBins, windowsz, sfreq, totskip, mrk,mrknames, color_per_int_type )



###############################  tSNE


if do_tSNE:

    legend_elements = utsne.prepareLegendElements(mrk,mrknames,color_per_int_type, tasks )

    print('Starting tSNE')
    # function to be run in parallel
    def run_tsne(p):
        t0 = time.time()
        pi,si, pcapts, seed, perplex_cur, lrate = p
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplex_cur, learning_rate=lrate)
        X_embedded = tsne.fit_transform(X)

        dur = time.time() - t0
        print('computed in {:.3f}s: perplexity = {};  lrate = {}; seed = {}'.
            format(dur,perplex_cur, lrate, seed))

        return pi,si,X_embedded, seed, perplex_cur, lrate


    #perplex_values = [30]
    #seeds = [0]


    res = []
    args = []
    for pi,perplex_cur in enumerate(perplex_values):
        subres = []
        for si,seed in enumerate(seeds):

            args += [ (pi,si, X.copy(), seed, perplex_cur, lrate)]


    rn_str = ','.join(rawnames)
    out_name_templ = '{}_{}tSNE_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}_effdim{}_subskip{}'
    out_name = (out_name_templ ).\
        format(rn_str,prefix,n_channels, n_feats_PCA, dim_PCA, skip_PCA, windowsz, dim_inp_tSNE, subskip)





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
            feat_info = ff['feat_info'][()]
            PCA_info = ff['PCA_info'][()]
        else:
            ncores = min(len(args) , mpr.cpu_count()-2)
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
                         PCA_info = PCA_info, feat_info = feat_info)
    else:
        assert len(tSNE_result) == len(args)
        print('Using exisitng tSNE')

if show_plots and do_tSNE:
    print('Starting tSNE plotting ')
    pdf= PdfPages( out_name+'.pdf'  )

    #cols = [colors, colors2, colors3]
    cols = [colors]

    colind = 0
    nr = len(seeds)
    nc = len(perplex_values)
    ww = 8; hh=8
    fig,axs = plt.subplots(ncols =nc, nrows=nr, figsize = (nc*ww, nr*hh))
    if not isinstance(axs,np.ndarray):
        axs = np.array([[axs]] )
    # for pi,pv in enumerate(perplex_values):
    #     for si,sv in enumerate(seeds):
    for tpl in tSNE_result:
        pi,si,X_embedded, seed, perplex_cur, lrate = tpl
        ax = axs[si,pi]
        #X_embedded = res[si][pi]
        #ax.scatter(X_embedded[:,0], X_embedded[:,1], c = cols[colind], s=5)

        utsne.plotMultiMarker(ax,X_embedded[:,0], X_embedded[:,1], c = cols[colind], s=5,
                                m=markers)
        ax.set_title('perplexity = {};  lrate = {}; seed = {}'.format(perplex_cur, lrate, seed))

    axs[0,0].legend(handles=legend_elements)
    #figname = 'tSNE_inpdim{}_PCAdim{}_Npts{}_hside{}_skip{}_wsz{}_lrate{}.pdf'.\
    #    format(n_feats, pcapts.shape[-1], X.shape, mts_letter, totskip, windowsz, lrate)


    str_feats = ','.join(PCA_info['features_to_use'])
    str_mods = ','.join(PCA_info['data_modalities'])
    out_name += '\nmainLFP{}_HFO{}_{}_{}'.\
        format(int(PCA_info['use_main_LFP_chan']), int(PCA_info['use_lfp_HFO']),
               str_mods, str_feats)

    figname = out_name + '.pdf'
    plt.suptitle(figname)
    #plt.savefig(rawname_ +'_'+figname)

    pdf.savefig()
    pdf.close()

#plt.savefig(figname)
gc.collect()
