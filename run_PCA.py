import os
import sys
import mne
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

import utils
import utils_tSNE as utsne
import utils_preproc as upre


#rawnames_for_PCA = ['S01_on_hold', 'S01_on_move']
rawnames_for_PCA = ['S01_off_hold', 'S01_off_move']

if os.environ.get('DATA_DUSS') is not None:
    data_dir = os.path.expandvars('$DATA_DUSS')
else:
    data_dir = '/home/demitau/data'

#nPCA_comp = 0.95
nPCA_comp = 0.95
n_channels = 7
skip = 32
windowsz = 256

nPCAcomponents_to_plot = 5
show_plots = 0

discard = 1e-2
qshift = 1e-2
##########################

feat_types_all = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr']
data_modalities_all = ['LFP', 'msrc']

features_to_use = [ 'con',  'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr']
data_modalities = ['LFP', 'msrc']
msrc_inds = np.arange(8,dtype=int)  #indices appearing in channel (sources) names, not channnel indices
use_main_LFP_chan = False
use_lfp_HFO = 1

fband_names_crude = ['tremor', 'beta', 'gamma']
fband_names_fine = ['tremor', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma' ]
fband_names_crude_inc_HFO = fband_names_crude + ['HFO']
fband_names_fine_inc_HFO = fband_names_fine + ['HFO1', 'HFO2', 'HFO3']
fbands_to_use = fband_names_fine_inc_HFO

prefix = ''

load_only = 0
do_LDA = 1
n_feats = 609
##############################
import sys, getopt

effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

print(effargv)

helpstr = 'Usage example\nrun_PCA.py --rawnames <rawname_naked1,rawnames_naked2> '
opts, args = getopt.getopt(effargv,"hr:n:s:w:p:",
        ["rawnames=", "n_channels=", "skip_feat=", "windowsz=", "pcexpl=",
         "show_plots=","discard=", 'feat_types=', 'use_HFO=', 'mods=',
         'prefix=', 'load_only=', 'fbands='])
print(sys.argv, opts, args)
for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == "--n_channels":
        n_channels = int(arg)
    elif opt == "--n_feats":
        n_feats = int(arg)
    elif opt == "--mods":
        data_modalities = arg.split(',')
    elif opt == "--skip_feat":
        skip = int(arg)
    elif opt == '--load_only':
        load_only = int(arg)
    elif opt == '--show_plots':
        show_plots = int(arg)
    elif opt == "--windowsz":
        windowsz = int(arg)
    elif opt == "--discard":
        discard = float(arg)
    elif opt == "--prefix":
        prefix = arg + '_'
    elif opt == "--msrc_inds":
        msrc_inds = [ int(indstr) for indstr in arg.split(',') ]
        msrc_inds = np.array(msrc_inds, dtype=int)
    elif opt == "--feat_types":
        features_to_use = arg.split(',')
        for ftu in features_to_use:
            assert ftu in feat_types_all, ftu
    elif opt == "--mods":
        data_modalities = arg.split(',')   #lfp of msrc
    elif opt == '--fbands':
        fbands_to_use = arg.split(',')
    elif opt == '--LFPchan':
        if arg == 'main':
            use_main_LFP_chan = 1
        elif arg == 'other':
            use_main_LFP_chan = 0
        elif arg.find('LFP') >= 0:   # maybe I want to specify expliclity channel name
            raise ValueError('to be implemented')
    elif opt == '--use_HFO':
        use_lfp_HFO = int(arg)
    elif opt == "--pcexpl":
        nPCA_comp = float(arg)  #crude of fine
        if nPCA_comp - int(nPCA_comp) < 1e-6:
            nPCA_comp = int(nPCA_comp)
    elif opt in ('-r','--rawnames'):
        rawnames_for_PCA = arg.split(',')

print('nPCA_comp = ',nPCA_comp)

############################

with open('subj_info.json') as info_json:
        #raise TypeError

    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)

##############################

tasks = []
Xs = []
lens = []
feature_names_pri = []
good_inds_pri = []

feat_fnames = []
feat_file_pri = []
Xtimes_pri = []
for rawname_ in rawnames_for_PCA:

    a = '{}_feats_{}chs_nfeats{}_skip{}_wsz{}.npz'.\
        format(rawname_,n_channels, n_feats, skip, windowsz)
    feat_fnames += [a]
    fname_feat_full = os.path.join( data_dir,a)

    print('Loading feats from ',fname_feat_full)
    f = np.load(fname_feat_full, allow_pickle=True)
    feat_file_pri += [f]
    sfreq = f['sfreq']

    X =  f['X']
    Xtimes = f['Xtimes']
    skip_ =f['skip']
    feature_names_all = f['feature_names_all']
    chnames_src = f['chnames_src']
    chnames_LFP = f['chnames_LFP']
    assert skip_ == skip


    Xtimes_pri += [Xtimes]

    subj,medcond,task  = utils.getParamsFromRawname(rawname_)
    tasks += [task]

    mainLFPchan = gen_subj_info[subj]['lfpchan_used_in_paper']

    bad_inds = set([] )

    remove_crossLFP = 1
    if remove_crossLFP:
        regex_sameLFP = r'.*(LFP.[0-9]+),.*\1.*'
        inds_sameLFP = utsne.selFeatsRegexInds(feature_names_all,[regex_sameLFP])

        regex_biLFP = r'.*(LFP.[0-9]+),.*(LFP.[0-9]+).*'
        inds_biLFP = utsne.selFeatsRegexInds(feature_names_all,[regex_biLFP])

        inds_notsame_LFP = set(inds_biLFP) - set(inds_sameLFP)
        print('Removing cross LFPs {}'.format( inds_notsame_LFP) )
        #print( np.array(feature_names_all)[list(inds_notsame_LFP)] )
        bad_inds.update(inds_notsame_LFP  ) #same LFP are fine, it is just power

    if len(fbands_to_use) < len(fband_names_fine_inc_HFO):
        fbnames_bad = set(fband_names_fine_inc_HFO) - set(fbands_to_use)
        print('Removing bands ',fbnames_bad)
        regexs = []
        for fbname in fbnames_bad:
            regexs += [ '.*{}.*'.format(fbname)  ]
        inds_bad_fbands = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_bad_fbands)
    # here 'bad' means nothing essentially, just something I want to remove
    if set(feat_types_all) != set(features_to_use):
        badfeats = set(feat_types_all) - set(features_to_use)
        print('Removing features ',badfeats)
        regexs = [ '{}.*'.format(feat_name) for feat_name in  badfeats]
        inds_badfeats = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_badfeats)

    if set(data_modalities_all) != set(data_modalities):
        badmod = list( set(data_modalities_all) - set(data_modalities) )
        print('Removing modalities ',badmod)
        assert len(badmod) == 1
        badmod = badmod[0]
        regexs = [ '.*{}.*'.format(badmod) ]
        inds_badmod = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_badmod)

    if use_main_LFP_chan:
        # take mainLFPchan, extract <side><number>,  select names where there
        # is LFP with other <side><number>
        # similar with msrc_inds

        chnames_bad_LFP = set(chnames_LFP) - set([mainLFPchan] )

        regexs = [ '.*{}.*'.format(chname) for chname in  chnames_bad_LFP]
        inds_bad_LFP = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)

        print('Removing non-main LFPs ',chnames_bad_LFP)
        bad_inds.update(inds_bad_LFP)

    # collecting indices of all msrc that we have used
    import re
    regex = 'msrc._([0-9]+)'
    res = []
    for fn in feature_names_all:
        r = re.findall(regex,fn)
        res += r
        #print(r)
    #     if r is not None:
    #         print(fn, r.groups() )
    res = list( map(int,res) )
    res = np.unique(res)

    msrc_inds_undesired = np.setdiff1d( msrc_inds, res)
    if len(msrc_inds_undesired):
        #chnames_bad_src =

        regexs = [ '.*msrc.{}.*'.format(ind) for ind in msrc_inds_undesired]
        inds_bad_src = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_bad_src)

    if not use_lfp_HFO:
        regexs = [ '.*HFO.*' ]
        inds_HFO = utsne.selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update( inds_HFO )

    print('Removing {} features out of {}'.format( len(bad_inds) , len(feature_names_all) ) )
    good_inds = set( range(len(feature_names_all) ) ) - bad_inds
    good_inds = list( sorted(good_inds) )


    X = X[:, good_inds]

    Xs += [ X ]
    lens += [ X.shape[0] ]
    feature_names_pri += [ feature_names_all[ good_inds] ]
    good_inds_pri += [good_inds]

    feat_info = f.get('feat_info',None)[()]
    mts_letter = gen_subj_info[subj]['tremor_side'][0].upper()
    nedgeBins = feat_info['nedgeBins']

Xconcat = np.concatenate(Xs,axis=0)

out_bininds, qvmult, discard_ratio = \
    utsne.findOutlierLimitDiscard(Xconcat,discard=discard,qshift=1e-2)
good_inds = np.setdiff1d( np.arange(Xconcat.shape[0] ), out_bininds)

anns, anns_pri, Xtimes, dataset_bounds = utsne.concatAnns(rawnames_for_PCA, Xtimes_pri)
ivalis = utils.ann2ivalDict(anns)
ivalis_tb, ivalis_tb_indarrays = utsne.getAnnBins(ivalis, Xtimes, nedgeBins, sfreq, skip, windowsz, dataset_bounds)
ivalis_tb_indarrays_merged = utsne.mergeAnnBinArras(ivalis_tb_indarrays)
all_interval_inds = np.hstack( [inds for inds in ivalis_tb_indarrays_merged.values()] )
unset_inds = np.setdiff1d(np.arange(len(Xtimes)), all_interval_inds)

remove_unlabeled = 1
if remove_unlabeled:
    #do somthing
    good_inds_ = np.setdiff1d( good_inds, unset_inds)
    print('Removing {} unlabeled pts before PCA'.format(len(good_inds) - len(good_inds_) ) )
    good_inds = good_inds_
else:
    print('Warning not removing unlabeled before PCA')

print('Outliers selection result: qvmult={:.3f}, len(out_bininds)={} of {} = {:.3f}s, discard_ratio={:.3f} %'.
    format(qvmult, len(out_bininds), Xconcat.shape[0],
           len(out_bininds)/sfreq,  100 * discard_ratio ) )

if load_only:
    print('Got load_only, exiting!')
    sys.exit(0)


print('Input PCA dimension ', (len(good_inds),Xconcat.shape[1]) )
pca = PCA(n_components=nPCA_comp)
Xconcat_good = Xconcat[good_inds]
pca.fit(Xconcat_good )   # fit to not-outlier data
pcapts = pca.transform(Xconcat)  # transform outliers as well

print('Output PCA dimension {}, total explained variance proportion {}'.
      format( pcapts.shape[1] , np.sum(pca.explained_variance_ratio_) ) )
print('PCA First several var ratios ',pca.explained_variance_ratio_[:5])


if do_LDA:
    #TODO: make possible non-main side
    sides_hand = [mts_letter]
    int_types_basic = ['trem', 'notrem', 'hold', 'move']

    int_types = set()
    for itb in int_types_basic:
        for side in sides_hand:
            assert len(side) == 1
            int_types.update(['{}_{}'.format(itb,side)])
    #int_types = ['trem_L', 'notrem_L', 'hold_L', 'move_L']
    int_types = list(int_types)
    #print(int_types)

    classes = [k for k in ivalis_tb_indarrays.keys() if k in int_types]  #need to be ordered
    #classes

    defclass = 0
    class_labels = np.repeat(defclass,len(Xconcat))
    assert defclass == 0
    for i,k in enumerate(classes):
        #print(i,k)
        for bininds in ivalis_tb_indarrays[k]:
            #print(i,len(bininds), bininds[0], bininds[-1])
            class_labels[ bininds ] = i + 1


    class_labels_good = class_labels[good_inds]

    rem_neut = 1
    if rem_neut:
        neq = class_labels_good != defclass
        inds = np.where( neq)[0]
        Xconcat_good = Xconcat_good[inds]
        class_labels_good = class_labels_good[inds]
    else:
        classes = ['neut'] + classes  # will fail if run more than once

    n_components_LDA = len(set(class_labels_good)) - 1
    print('n_components_LDA =', n_components_LDA)

    # first axis gives best separation, second does the second best job, etc
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #y = np.array([1, 1, 1, 2, 2, 2])
    lda = LinearDiscriminantAnalysis(n_components=n_components_LDA)
    lda.fit(Xconcat_good, class_labels_good)

    print('LDA var explained = ', lda.explained_variance_ratio_)
    print('LDA priors ', list(zip(classes,lda.priors_) ) )

    X_LDA = lda.transform(Xconcat)  # we transform all points, even bad and ulabeled ones. Transform is done using scalings

    class_to_check = 'trem_{}'.format(mts_letter)
    class_ind = classes.index(class_to_check) + 1
    sens,spec = utsne.getLDApredPower(lda,Xconcat, class_labels, class_ind)
    print('--!! LDA on training data, identifying {}: sens = {:.3f};  spec = {:.3f}'.format(class_to_check,sens,spec))
else:
    lda = None
    class_labels_good = None
    X_LDA = None
    sens = np.nan
    spec = np.nan


###### Save result
indst = 0
indend = lens[0]
# save PCA output separately
for i,rawname_ in enumerate(rawnames_for_PCA):
    # note that we use number of features that we actually used, not that we
    # read

    str_feats = ','.join(features_to_use)
    str_mods = ','.join(data_modalities)
    #use_lfp_HFO
    #use_main_LFP_chan

    # I don't include rawname in template because I want to use it for PDF name
    # as well
    out_name_templ = '_{}PCA_nr{}_{}chs_nfeats{}_pcadim{}_skip{}_wsz{}'
    out_name = (out_name_templ ).\
        format(prefix, len(rawnames_for_PCA), n_channels, Xconcat.shape[1],
               pcapts.shape[1], skip, windowsz)
    fname_PCA_full = os.path.join( data_dir, '{}{}.npz'.format(rawname_,out_name))
    sl = slice(indst,indend)
    print(sl, sl.stop - sl.start)
    assert (sl.stop-sl.start) == len(Xtimes_pri[i])

    info = {}
    info['n_channels_featfile'] = 7
    info['features_to_use'] = features_to_use
    info['data_modalities'] = data_modalities
    info['msrc_inds' ]  = msrc_inds
    info['use_main_LFP_chan'] = use_main_LFP_chan
    info['use_lfp_HFO'] = use_lfp_HFO
    info['nPCA_comp'] = nPCA_comp
    info['feat_fnames'] = feat_fnames
    info['good_feat_inds_pri'] = good_inds_pri
    info['out_bininds'] = out_bininds
    info['qvmult'] = qvmult
    info['discard_ratio'] = discard_ratio
    info['prefix'] = prefix
    info['LDAsens'] = sens
    info['LDAspec'] = spec

    np.savez(fname_PCA_full, pcapts = pcapts[sl], pcaobj=pca, ldaobj=lda,
             feature_names_all = feature_names_pri[i] , good_inds = good_inds_pri[i],
             info = info, feat_info = feat_file_pri[i].get('feat_info',None),
             Xtimes=Xtimes_pri[i] , X_LDA=X_LDA[sl], class_labels_good=class_labels_good )
    print('Saving PCA to ',fname_PCA_full)

    if i+1 < len(lens):
        indst += lens[i]
        indend += lens[i+1]



######################## Plotting
if show_plots:
    from matplotlib.backends.backend_pdf import PdfPages
    mpl.use('Agg')

    print('Starting to plot')

    use_main_tremorside = 1 # for now, temporarily
    bands_only = 'fine' # until I code merging

    rn_str = ','.join(rawnames_for_PCA)


    #str_feats = ','.join(features_to_use)
    #str_mods = ','.join(data_modalities)

    out_name_plot = rn_str + out_name + \
        'mainLFP{}_HFO{}_{}_{}'.\
        format(int(use_main_LFP_chan), int(use_lfp_HFO), str_mods, str_feats)
    #a = out_name_templ.\
    #    format(rn_str,n_channels, Xconcat.shape[1], pcapts.shape[1], skip, windowsz)
    pdf= PdfPages(out_name_plot + '.pdf')
    #pdf= PdfPages(   )


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
            nedgeBins, windowsz, sfreq, skip, mrk,mrknames, color_per_int_type )

    utsne.plotPCA(pcapts,pca, nPCAcomponents_to_plot,feature_names_all, colors, markers,
                mrk, mrknames, color_per_int_type, tasks,
                pdf=pdf,neutcolor=neutcolor)

    if do_LDA:
        n_LDA_feats_toshow = 140
        utsne.plotPCA(X_LDA,lda, n_components_LDA,feature_names_all, colors, markers,
                    mrk, mrknames, color_per_int_type, tasks,
                    pdf=pdf,neutcolor=neutcolor, nfeats_show=n_LDA_feats_toshow)

    pdf.close()

gc.collect()