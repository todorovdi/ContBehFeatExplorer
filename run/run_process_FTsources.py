#!/usr/bin/python3
print(f'Starting {__file__}')

import sys, os
sys.path.append( os.path.expandvars('$OSCBAGDIS_DATAPROC_CODE') )

import mne
import utils  #my code
import json
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
import scipy.io as sio
import globvars as gv
from globvars import gp
import sys, getopt
from os.path import join as pjoin

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

#if os.environ.get('DATA_DUSS') is not None:
#    data_dir = os.path.expandvars('$DATA_DUSS')
#else:
#    data_dir = '/home/demitau/data'
data_dir = gv.data_dir




effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

print(effargv)

#rawname_ = 'S01_off_hold'
rawname_ = 'S01_on_hold'
#rawname_ = 'S01_off_move'
#rawname_ = 'S01_on_move'

#rawname_ = 'S02_off_hold'
#rawname_ = 'S02_on_hold'
#rawname_ = 'S02_off_move'
#rawname_ = 'S02_on_move'

#rawname_ = 'S03_off_move'
#rawname_ = 'S03_off_hold'

#rawname_ = 'S04_off_hold'
#rawname_ = 'S04_off_move'
#rawname_ = 'S04_on_hold'
#rawname_ = 'S04_on_move'

#rawname_ = 'S05_off_hold'
#rawname_ = 'S05_off_move'
#rawname_ = 'S05_on_hold'
#rawname_ = 'S05_on_move'

#rawname_ = 'S07_off_hold'
#rawname_ = 'S07_off_move'
#rawname_ = 'S07_on_hold'
#rawname_ = 'S07_on_move'


#desired_selected_by_me = False

#sources_type = 'HirschPt2011'
sources_type = 'parcel_aal'

groupings_to_use = ['all']
save_info = 1
save_data = 1

nPCA_comp = 0.95
#algType = 'PCA+ICA' #'PCA' # 'mean'
algType = 'all_sources'

input_subdir = ""
output_subdir = ""

helpstr = 'Usage example\nrun_process_FTsources.py --rawname <rawname_naked> --sources_type <srct> '
opts, args = getopt.getopt(effargv,"hr:s:",
        ["rawnames=", 'sources_type=', 'groupings=', 'alg_type=',
         'save_info=',
         'save_data=',
         "input_subdir=", "output_subdir="])
print(sys.argv, opts, args)
for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt == '--groupings':
        groupings_to_use = arg.split(',')
    elif opt == '--save_info':
        save_info = int(arg)
    elif opt == '--save_data':
        save_data = int(arg)
    elif opt == "--input_subdir":
        input_subdir = arg
        if len(input_subdir) > 0:
            subdir = pjoin(gv.data_dir,input_subdir)
            assert os.path.exists(subdir ), subdir
    elif opt == "--output_subdir":
        output_subdir = arg
        if len(output_subdir) > 0:
            subdir = pjoin(gv.data_dir,output_subdir)
            if not os.path.exists(subdir ):
                print('Creating output subdir {}'.format(subdir) )
                os.makedirs(subdir)
    elif opt == '--alg_type':
        algType = arg
        assert algType in ['PCA', 'PCA+ICA', 'mean', 'all_sources']
    #elif opt == '--desired_custom':
    #    desired_selected_by_me = int(arg)
    elif opt in ('-r','--rawnames'):
        rawnames = arg.split(',')
        rawnames_nonempty = []
        for rn in rawnames:
            if len(rn):
                rawnames_nonempty += [rn]
        assert len(rawnames_nonempty) > 0
        rawnames = rawnames_nonempty
    elif opt in ('-s','--sources_type'):
        sources_type = str(arg)
    else:
        raise ValueError(f'Unrecognized option {opt} with value {arg}')

if groupings_to_use[0] == 'all_raw':
    assert algType == 'all_sources'

import pymatreader as pymr
p = 'cortical_grid.mat'
if not os.path.exists(p):
    r = pymr.read_mat(pjoin(data_dir,p) )
else:
    r = pymr.read_mat(p )
template_grid_cm = r['cortical_grid'] * 100


######




times_pri = len(rawnames) * [0]
custom_raws_pri = len(rawnames) * [0]
coords_pri = len(rawnames) * [0]
for rawni,rawname_ in enumerate(rawnames):
    sind_str,medcond,task = utils.getParamsFromRawname(rawname_)

    # this one does not have to be from the subdir
    #rawname = rawname_ + '_resample_notch_highpass_raw.fif'
    rawname = rawname_ + '_LFPonly.fif'
    fname_full = pjoin(data_dir,rawname)

    #rawname = rawname_ + '_resample_raw.fif'
    #fname_full = pjoin(data_dir,rawname)

    ## read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
    raw = mne.io.read_raw_fif(fname_full, None)

    #reconst_name = rawname_ + '_resample_afterICA_raw.fif'
    #reconst_fname_full = pjoin(data_dir,reconst_name)
    #reconst_raw = mne.io.read_raw_fif(reconst_fname_full, None)

    times_pri[rawni] = raw.times

    src_fname_noext = 'srcd_{}_{}'.format(rawname_, sources_type)

    src_fname = src_fname_noext + '.mat'
    src_fname_full = pjoin(data_dir,input_subdir,src_fname)
    print(src_fname_full)
    # if I try to use just pymatreader, I get an error:
    # "Please use HDF reader for matlab v7.3 files"
    src_ft = h5py.File(src_fname_full, 'r')
    ff = src_ft


    #timeinfo = os.stat(src_fname_full)
    #print('Last mod of src was ', time.ctime(timeinfo.st_mtime) )
    #timeinfo = os.stat(reconst_fname_full)
    #print('Last raw was        ', time.ctime(timeinfo.st_mtime) )

    f = ff[ ff['source_data'][0,0] ]

    #sys.exit(0)

    #########


    coords_MNI = None
    if sources_type in [ 'parcel_aal', 'parcel_aal_surf' ]:
        srcCoords_fn = sind_str + '_modcoord_parcel_aal.mat'

        p = srcCoords_fn
        if not os.path.exists(p):
            crdf = pymr.read_mat(pjoin(data_dir,'modcoord',p) )
        else:
            crdf = pymr.read_mat(p)

        # here we do not have label 'unlabeled'
        labels = crdf['labels']  #indices of sources - labels_corresp_coordance
        #crdf = sio.loadmat(srcCoords_fn)
        #lbls = crdf['labels'][0]
        #labels = [  lbls[i][0] for i in range(len(lbls)) ]

        coords = crdf['coords_Jan_actual']
        srcgroups_ = crdf['point_ind_corresp']

        coords_pri += [coords]

        # here we shift by 1 to add label 'unlabeled' in the beginning
        if np.min(srcgroups_) == 0:
            srcgroups_ #+= 1  # we don't need to add one because we have Matlab notation starting from 1
            labels = ['unlabeled'] + labels

        print(labels)
        #crdf['pointlabel']

    elif sources_type == 'HirschPt2011':
        #<0 left
        with open(pjoin('.','coord_labels.json') ) as jf:
            coord_labels = json.load(jf)
        #    gv.gparams['coord_labels'] = coord_labels

        srcCoords_fn = 'coordsJan.mat'
        coords_MNI_f = sio.loadmat(srcCoords_fn)
        coords_MNI = coords_MNI_f['coords']

        srcCoords_fn = sind_str + '_modcoord.mat'
        labels = ['']* len(coords_MNI)
        assert len(labels ) == len(coord_labels) * 2
        for coordi in range(len(coords_MNI)):
            labeli = coordi // 2
            if coords_MNI[coordi,0] < 0:
                side = 'L'
            else:
                side = 'R'
            labels[coordi]= coord_labels[labeli] + '_{}'.format(side)

        srcgroups_ = crdf['point_ind_corresp'][0]
        print(coords_MNI,labels)


    print(srcgroups_)

    numcenters = np.max(srcgroups_)
    print(numcenters)
    print(coords.shape, coords)

    #dispGroupInfo(srcgroups_)

    ##########

    #if desired_selected_by_me:
    #    desired = gp.areas_list_aal_my_guess
    #else:
    #    desired = labels

    ######3

    #scrgroups_dict = {}
    stcs = []
    fbands = {}
    custom_raws = {}
    vertices_inds_dict = {} # values are lists of lists of indices in the original data

    if sources_type == 'HirschPt2011':
        indsets = {'centers': slice(0,numcenters), 'surround': slice(numcenters,None)}
    elif sources_type in [ 'parcel_aal', 'parcel_aal_surf' ]:
        indsets = {'surround':slice(None,None)}

    pos_ = f['source_data']['pos'][:,:].T

    # check that correspondence of sides is correct
    if sources_type == 'HirschPt2011':
        for posi in range(len(pos_) ):
            corresp_cnt_coord = coords[ srcgroups_[posi] - 1 ]
            cur_pt_coord = pos_[posi]
            assert np.sign(cur_pt_coord[0] ) == np.sign( corresp_cnt_coord[0] )


    ##########

    defSideFromLabel = True
    if ff['source_data'].shape[0] == 1:
        bandnames = ['allf']
    else:
        bandnames = ['tremor', 'beta', 'gamma', 'allf']
    for srcdi in range(ff['source_data'].shape[0] ):
        bandname = bandnames[srcdi]
        f = ff[ ff['source_data'][srcdi,0] ]

        freqBand = f['bpfreq'][:].flatten()

        t0 = f['source_data']['time'][0,0]
        tstep = np.diff( f['source_data']['time'][:10,0] ) [0]

        mom = f['source_data']['avg']['mom']
        # extractring unsorted data
        if sources_type == 'HirschPt2011':
            srcRefs = mom[0,:]
            srcData_ = [0]* len(srcRefs)
            for srci in range(len(srcRefs)):
                print(srci)
                srcData_[srci] = f[srcRefs[srci] ][:,0]
                srcData_ = np.vstack(srcData_)
        else:
            srcData_= mom[:,:].T

        assert pos_.shape[0] == srcData_.shape[0]

        numsrc_total = len(srcData_)
        for indset_name in indsets:
            allinds = np.arange(numsrc_total)[indsets[indset_name] ]
            pos = pos_[allinds]
            # first coord is the left-right coord
            #posinds = np.argsort( pos[:,0] )  # I'll need to sort correspondance as well
            #print(posinds)
            #sortedPos = pos[posinds]
            #leftInds = np.where(sortedPos[:,0]<= 0)[0]
            #rightInds = np.where(sortedPos[:,0] > 0)[0]

            if sources_type == 'HirschPt2011':
                labels_deford = np.array(labels)[srcgroups_[allinds ]-1 ]
            else:
                labels_deford = np.array(labels)[srcgroups_[allinds ] ]  # because 0 is unlabeled
            Lchnis = [labi for labi,lab in enumerate(labels_deford) if lab.endswith('_L') ]
            #Rchnis = [labi for labi,lab in enumerate(labels_deford) if lab.endswith('_R') ]
            Rchnis = [labi for labi,lab in enumerate(labels_deford) if not lab.endswith('_L') ]

            # create MNE structure that I'm actually not using later
            #(data, vertices=None, tmin=None, tstep=None, subject=None, verbose=None


            ####  Create my
            if sources_type == 'parcel_aal' and defSideFromLabel:
                lhi = map(str, Lchnis )
                rhi = map(str, Rchnis )

                concat = Lchnis + Rchnis
                vertices = [Lchnis, Rchnis]
            else:  #define side from coordinate
                leftInds_coord = np.where(pos[:,0]<= 0)[0]
                rightInds_coord = np.where(pos[:,0] > 0)[0]
                vertices = [leftInds_coord, rightInds_coord]

                lhi = map(str, list( vertices[0] ) )
                rhi = map(str, list( vertices[1] ) )

                concat = np.concatenate((leftInds_coord, rightInds_coord))

                #srcData = srcData_[posinds]

                if sources_type == 'HirschPt2011':
                    labels_coord_ord = np.array(labels)[srcgroups_[allinds[concat] ]-1 ]
                else:
                    labels_coord_ord = np.array(labels)[srcgroups_[allinds[concat] ] ]
                #print(labels)
            vertices_inds_dict[indset_name] = vertices

            srcData = srcData_[ allinds [concat]  ]   # with special ordering

            stc = mne.SourceEstimate(data = srcData, tmin = t0, tstep= tstep  ,
                                    subject = sind_str , vertices=vertices)
            stcs += [stc]

            fbands[bandname] = freqBand

            if indset_name == 'centers':
                # for compatibility
                srcnames =  [ 'msrcL_{}_'.format(bandname) + s for s in lhi ]
                srcnames += [ 'msrcR_{}_'.format(bandname) + s for s in rhi ]
            else:
                srcnames =  [ 'srcL_{}_'.format(bandname) + s for s in lhi ]
                srcnames += [ 'srcR_{}_'.format(bandname) + s for s in rhi ]

            # Initialize an info structure
            info = mne.create_info(
                ch_names=srcnames,
                ch_types=['csd'] * len(srcnames),
                sfreq=int ( 1/tstep ))

            if sources_type == 'HirschPt2011':
                srcgroups = srcgroups_[indsets[indset_name] ]-1
            else:
                srcgroups = srcgroups_[indsets[indset_name] ]
            srcgroups = srcgroups [ allinds [concat] ]
            #srcgroups = srcgroups_[posinds]-1
            assert min(srcgroups) == 0
            if sources_type == 'HirschPt2011':
                assert max(srcgroups) == len(coords)-1, ( max(srcgroups)+1, len(coords) )
            info['temp'] = { 'srcgroups': srcgroups }
            #scrgroups_dict[indset_name] = srcgroups

            custom_raw_cur = mne.io.RawArray(srcData, info)
            for chi in range(len(custom_raw_cur.info['chs']) ):
                custom_raw_cur.info['chs'][chi]['loc'][:3] = pos[concat][chi,:3]
            coords_resorted = pos[concat]
            new_src_order = concat # to save later

            rawtmp = custom_raws.get(indset_name, None)
            if rawtmp is None:
                custom_raws[indset_name] = custom_raw_cur
            else:
                custom_raws[indset_name].add_channels([custom_raw_cur])


            #print(custom_raw)
            #reconst_raw.add_channels([custom_raw])


    #################  Find inconsistent sides

    chns = custom_raw_cur.ch_names
    Lchns = [chn for chn in chns if chn.find('srcL') >= 0]
    for chni,chn in enumerate(chns):
        lab = labels[srcgroups[chni] ]
        loc = custom_raw_cur.info['chs'][chni]['loc']
        leftlab = lab.find('_L') >= 0
        leftside = loc[0] <= 0
        if leftlab ^ leftside:
            print(chni, chn, loc[0], lab)
    print(len(Lchns), len(chns))


    ##################

    if sources_type == 'HirschPt2011':
        newsrc_fname_full = pjoin( data_dir, output_subdir, 'cnt_' + src_fname_noext + '.fif' )
        print( newsrc_fname_full )

        custom_raws['centers'].save(newsrc_fname_full, overwrite=1)
    else:
        print("We use center-lees sources, so don't save centers")

    custom_raws_pri[rawni] = custom_raws



# yes I want them to be different!
assert abs( np.linalg.norm(coords_resorted - pos, 2)  ) > 1e-20

########################### preparing groups ###################3

srcgroups_all = custom_raws['surround'].info['temp']['srcgroups']
label_groups_dict, srcgroups_dict  = utils.prepareSourceGroups(labels,srcgroups_all)

####################### Test
#sys.exit(1)
########################

#if 'Cerebellum_B' not in desired:
#    desired += ['Cerebellum_B']

###############################################

kks = list(srcgroups_dict.keys())
for k in kks:
    if k not in groupings_to_use:
        del srcgroups_dict[k]
        del label_groups_dict[k]

assert len(srcgroups_dict) > 0

################################### Copmute
import utils_tSNE as utsne
windowsz_for_artifacts = 1
skip = 1
sfreq = raw.info['sfreq']
nedgeBins = 0



suffixes = ['_ann_MEGartif']
anns_artif, anns_artif_pri, Xtimes, dataset_bounds = utsne.concatAnns(rawnames,times_pri, suffixes)
ivalis_artif = utils.ann2ivalDict(anns_artif)
ivalis_artif_tb, ivalis_artif_tb_indarrays = \
    utsne.getAnnBins(ivalis_artif, Xtimes, nedgeBins, sfreq, skip, windowsz_for_artifacts, dataset_bounds)
ivalis_artif_tb_indarrays_merged = utsne.mergeAnnBinArrays(ivalis_artif_tb_indarrays)

nbinstot = len(Xtimes) #sum( [len(times) for times in times_pri] )
num_nans = sum( [ len(ar) for ar in ivalis_artif_tb_indarrays_merged.values() ] )
print('Artifact NaN percentage is {:.4f}%'.format(100 * num_nans/ nbinstot  ) )

#Xconcat_artif_nan  = utils.setArtifNaN(Xconcat, ivalis_artif_tb_indarrays_merged, feature_names_pri[0])


#srcgroups_list += [custom_raws['surround'].info['srcgroups']]
skip_unlabeled = True
duplicate_merged_across_sides = True # applies to 'merged' and 'CBmerged_vs_rest'
newchnames = []
newdatas = []
avpos = []
pcas = []
icas = []
srcgroups_keys_ordered = list( sorted(srcgroups_dict.keys()) )
#srcgroups_keys_ordered = ['CBmerged_vs_rest']
# cycle over parcellations
for srcgi,srcgroups_key in enumerate(srcgroups_keys_ordered):
    print('   Starting working wtih grouping ',srcgroups_key)
    srcgroups = srcgroups_dict[srcgroups_key]
    if srcgroups_key == 'all':
        assert min(srcgroups) == 0
    if sources_type == 'HirschPt2011':
        assert max(srcgroups) == len(coords)-1, ( max(srcgroups)+1, len(coords) )

    labels_list = label_groups_dict[srcgroups_key]

    # cycle over parcel indices
    for i in range( np.max(srcgroups)+1 ):
        merge_needed = False
        # get the (orderd) list of parcels in the current parcellation
        cur_parcel = labels_list[i]
        desired_ind = False  # whether we desire this index or not
        #for des in desired:
        #    desired_ind = desired_ind or (cur_parcel.find(des) >= 0)
        #if not desired_ind or (skip_unlabeled and cur_parcel.find('unlabeled') >= 0):
        if (skip_unlabeled and cur_parcel.find('unlabeled') >= 0):
            continue

        inds = np.where(srcgroups == i)[0]
        #srcData[inds]
        if cur_parcel.endswith('_B'):  # NOT 'Cerebellum_L' or 'Cerebellum_R':
            if duplicate_merged_across_sides:
                merge_needed = True
            brainside = 'B'
        else:
            if defSideFromLabel:
                if cur_parcel.endswith('_L'):
                    brainside = 'L'
                else:
                    brainside = 'R'
            else:
                #L or R?
                if coords_resorted[i][0] <= 0:
                    brainside = 'L'
                else:
                    brainside = 'R'

        for bandname in bandnames:
            if sources_type == 'HirschPt2011':
                chnames = [ 'src{}_{}_{}'.format(brainside,bandname,s) for s in inds ]
            else:
                chnames = np.array(custom_raws['surround'].ch_names)[inds]

            if len(chnames) == 0:
                print('! {}: no channels found for {}, index {}'.
                      format(cur_parcel, srcgroups_key, i))
                continue

            chdatas = []
            for cri,custom_raws_cur in enumerate(custom_raws_pri):
                chdata_cur, times_cur = custom_raws_cur['surround'][chnames]
                chdatas += [chdata_cur]
            chdata = np.hstack(chdatas)
            assert not ( np.any(np.isnan(chdata) ) and np.any(np.isinf(chdata) ) )

            chdata_wnans = utils.setArtifNaN(chdata.T,ivalis_artif_tb_indarrays_merged, None)
            isnan = np.isnan( chdata_wnans)
            if np.sum(isnan):
                artif_bininds = np.where( isnan )[0] # note that it is transposed
            else:
                artif_bininds = []
            bininds_noartif = np.setdiff1d( np.arange(len(isnan) ) , artif_bininds)


            if algType.startswith('PCA'):
                pca = PCA(n_components=nPCA_comp)
                pca_succeed = False
                try:
                    pca.fit(chdata.T)
                    pca_succeed = True
                    newdata = pca.transform(chdata.T[bininds_noartif] ).T
                except np.linalg.LinAlgError as e:
                    print('PCA calc exception ',e)
                    newdata = chdata
                    pca = None

                #newchnames += ['msrc{}_{}_{}_c{}'.format(brainside,bandname,i,ci) \
                #               for ci in range(newdata.shape[0])]

                pcas += [pca]
                if algType.find('ICA') >= 0 and pca_succeed:
                    max_iter = 500
                    ica = FastICA(n_components=len(newdata),
                                  random_state=0, max_iter=max_iter)
                    ica.fit(newdata.T[bininds_noartif])
                    if ica.n_iter_ < max_iter:
                        newdata = ica.transform(newdata.T).T
                    else:
                        # newdata does not change in this case
                        print('Did not converge')

                    icas += [ica]
                else:
                    icas += [ica]

                if merge_needed:
                    pcas += [pca] # same pca and ica again
                    icas += [ica]

                    cur_newchnames = ['msrc{}_{}_{}_{}_c{}'.
                                   format('L',bandname,srcgi,i,ci) \
                                   for ci in range(newdata.shape[0])]
                    cur_newchnames += ['msrc{}_{}_{}_{}_c{}'.
                                   format('R',bandname,srcgi,i,ci) \
                                   for ci in range(newdata.shape[0])]
                else:
                    cur_newchnames = ['msrc{}_{}_{}_{}_c{}'.
                                   format(brainside,bandname,srcgi,i,ci) \
                                   for ci in range(newdata.shape[0])]

                newchnames += cur_newchnames

            elif algType == 'mean':
                newdata = np.mean(chdata,axis=0)[None,:]
                if merge_needed:
                    cur_newchnames = ['msrc{}_{}_{}_{}_c{}'.
                                    format('L',bandname,srcgi,i,0) ]
                    cur_newchnames += ['msrc{}_{}_{}_{}_c{}'.
                                    format('R',bandname,srcgi,i,0) ]
                else:
                    cur_newchnames = ['msrc{}_{}_{}_{}_c{}'.
                                    format(brainside,bandname,srcgi,i,0) ]
                    #newchnames += ['msrc{}_{}_{}'.format(brainside,bandname,i)]
                newchnames += cur_newchnames
            elif algType == 'all_sources':
                newdata = chdata
                #TODO in this case I may want to put brainside differently
                if merge_needed:
                    cur_newchnames = ['msrc{}_{}_{}_{}_c{}'.
                                    format('L',bandname,srcgi,i,ci) \
                                    for ci in range(newdata.shape[0])]
                    cur_newchnames += ['msrc{}_{}_{}_{}_c{}'.
                                    format('R',bandname,srcgi,i,ci) \
                                    for ci in range(newdata.shape[0])]
                else:
                    cur_newchnames = ['msrc{}_{}_{}_{}_c{}'.
                                    format(brainside,bandname,srcgi,i,ci) \
                                    for ci in range(newdata.shape[0])]
                newchnames += cur_newchnames
            else:
                raise ValueError('Wrong algType')


            #print(chnames)
            print('{}={}: {} over {}, newdata shape {}'.
                  format(i,labels_list[i], algType,
                         chdata.shape[0],newdata.shape))

            newdatas    += [newdata]
            if merge_needed: # adding again
                newdatas    += [newdata]

            avpos_cur = np.zeros(3)
            for chi in mne.pick_channels(custom_raws['surround'].ch_names,chnames):
                avpos_cur += custom_raws['surround'].info['chs'][chi]['loc'][:3]
            avpos_cur /= len(chnames)
            avpos += [avpos_cur] * newdata.shape[0]
            if merge_needed: # adding again
                avpos += [avpos_cur] * newdata.shape[0]

assert len(newchnames) == len(avpos)
assert len(icas) == len(pcas)
assert len(newchnames) == np.sum([nd.shape[0] for nd in newdatas])
print('We got {} new channels '.format( len(newchnames)) )






#-------------------


scrgroups_per_indset = {}
for crt in custom_raws:
    scrgroups_per_indset[crt] = custom_raws[crt].info['temp']['srcgroups']

sl = sorted([ gp.src_grouping_names_order.index(k) for k in srcgroups_dict.keys() ])
assert len(sl) == 1  # just for now, I don't want to code extra in gen_features
grp_id_str = ','.join(map(str,sl) )


if save_info:
    for rawname_ in rawnames:
        src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.\
            format(rawname_,sources_type,  grp_id_str  )
        src_rec_info_fn_full = pjoin(gv.data_dir, output_subdir, src_rec_info_fn + '.npz')
        print(src_rec_info_fn_full)
        np.savez(src_rec_info_fn_full,
                 srcgroups_dict=srcgroups_dict,
                scrgroups_per_indset = scrgroups_per_indset,
                coords_Jan_actual=coords_resorted,
                coords_Jan_actual_unsorted=pos,
                label_groups_dict = label_groups_dict,
                srcgroups_key_order = srcgroups_keys_ordered,
                coords_MNI=coords_MNI,
                pcas=pcas, icas=icas,
                avpos=avpos, algType=algType, vertices_inds_dict=vertices_inds_dict,
                new_src_order = new_src_order )
        # coords_Jan_actual -- indeed actual coords (not MNI)


#------------------

ml = 0
for chn in newchnames:
    ml = max (ml, len(chn))
print('Max chname len = ',ml)


if ml > 15:
    for chni in range(len(newchnames) ):
        newchnames[chni] = newchnames[chni].replace('allf_','')
#     for chni in range(len(newchnames) ):
#         newchnames[chni] = newchnames[chni].replace('Cerebelum','CB')
    shortened_chnames = True
    print('Shortened namees')
else:
    shortened_chnames = False

print('newchnames =', newchnames )




## Choose data from which groupings do we want to save (not all perhpas)
#------------------------
dd = np.vstack(newdatas)

#onlyCB_and_notCB = True
#group_names_to_keep = ['CB_vs_rest', 'CBmerged_vs_rest']
onlyCB_and_notCB = False
group_names_to_keep = ['all']
if onlyCB_and_notCB:
    inds = []
    newchnames_filtered = []
    newdatas_filtered = []
    for chni,chn in enumerate(newchnames):
        nn = utils.getMEGsrc_chname_nice(chn,label_groups_dict, srcgroups_keys_ordered)
        srcgroup_ind, ind, subind = utils.parseMEGsrcChnameShort(chn)
        #print(nn)
        # computed CB twice, so we don't want to save it twice
        if nn.find('Cerebellum') >= 0 and srcgroups_keys_ordered[srcgroup_ind] in srcgroups_keys_ordered:

            print(chni,nn, srcgroup_ind, ind, subind)
            inds += [chni]
            newchnames_filtered += [chn]
            newdatas_filtered += [dd[chni] ]
            #newdatas_filtered += [ newdatas[ind][subind] ]

else:
    newchnames_filtered = newchnames
    newdatas_filtered = newdatas
#---------------
dd = np.vstack(newdatas_filtered)
rawdata_flt = dd

ml = 0
for chn in newchnames_filtered:
    ml = max (ml, len(chn))
assert ml <= 15


info = mne.create_info(
    ch_names=newchnames_filtered,
    ch_types=['csd'] * len(newchnames_filtered),
    sfreq=int ( 1/tstep ))

for chi in range(len(info['chs']) ):
    info['chs'][chi]['loc'][:3] = avpos[chi]

srcgroups_backup = custom_raws['surround'].info['temp']['srcgroups']

newraw = mne.io.RawArray(dd, info)

if save_data:
    curstart = 0
    for rawni,rawname_ in enumerate(rawnames):
        if sources_type == 'HirschPt2011':
            newraw_cur = custom_raws['surround'].copy()
        else:
            newraw_cur = newraw.copy()
        tt = custom_raws_pri[rawni]['surround'].times
        newraw_cur.crop( curstart + tt[0], curstart + tt[-1], include_tmax = True )
        curstart += tt[-1] + tstep

        src_fname_noext = 'srcd_{}_{}_grp{}'.format(rawname_, sources_type, grp_id_str)

        #  Save
        #newsrc_fname_full = pjoin( data_dir, 'av_' + src_fname_noext + '.fif' )
        newsrc_fname_full = pjoin( data_dir, output_subdir, 'pcica_' + src_fname_noext + '.fif' )
        print( newsrc_fname_full )

        newraw_cur.save(newsrc_fname_full, overwrite=1)
