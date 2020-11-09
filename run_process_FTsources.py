#!/usr/bin/python3

import os,sys
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


#sources_type = 'HirschPt2011'
sources_type = 'parcel_aal'

helpstr = 'Usage example\nrun_process_FTsources.py --rawname <rawname_naked> --sources_type <srct> '
opts, args = getopt.getopt(effargv,"hr:s:",
        ["rawnames=", 'sources_type='])
print(sys.argv, opts, args)
for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt in ('-r','--rawnames'):
        rawnames = arg.split(',')
    elif opt in ('-s','--sources_type'):
        sources_type = str(arg)

import pymatreader as pymr
r = pymr.read_mat('cortical_grid.mat')
template_grid_cm = r['cortical_grid'] * 100

srcCoords_fn = 'coordsJan.mat'
coords_MNI_f = sio.loadmat(srcCoords_fn)
coords_MNI = coords_MNI_f['coords']

######

with open(os.path.join('.','coord_labels.json') ) as jf:
    coord_labels = json.load(jf)
#    gv.gparams['coord_labels'] = coord_labels

def dispGroupInfo(grps):
    ugrps = np.unique(grps)
    print('min {}, max {}; ulen {}, umin {}, umax {}'.
          format(min(grps), max(grps), len(ugrps), min(ugrps), max(ugrps)) )


custom_raws_pri = len(rawnames) * [0]
for rawni,rawname_ in enumerate(rawnames):
    sind_str,medcond,task = utils.getParamsFromRawname(rawname_)

    rawname = rawname_ + '_resample_raw.fif'
    fname_full = os.path.join(data_dir,rawname)

    # read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
    raw = mne.io.read_raw_fif(fname_full, None)

    reconst_name = rawname_ + '_resample_afterICA_raw.fif'
    reconst_fname_full = os.path.join(data_dir,reconst_name)
    reconst_raw = mne.io.read_raw_fif(reconst_fname_full, None)

    src_fname_noext = 'srcd_{}_{}'.format(rawname_, sources_type)

    src_fname = src_fname_noext + '.mat'
    src_fname_full = os.path.join(data_dir,src_fname)
    print(src_fname_full)
    src_ft = h5py.File(src_fname_full, 'r')
    ff = src_ft


    #os.path.getmtime(src_fname_full)
    timeinfo = os.stat(src_fname_full)
    print('Last mod of src was ', time.ctime(timeinfo.st_mtime) )
    timeinfo = os.stat(reconst_fname_full)
    print('Last raw was        ', time.ctime(timeinfo.st_mtime) )

    f = ff[ ff['source_data'][0,0] ]

    #########


    if sources_type == 'parcel_aal':
        srcCoords_fn = sind_str + '_modcoord_parcel_aal.mat'

        crdf = pymr.read_mat(srcCoords_fn)

        coord_labels_corresp_coord = crdf['labels']
        #crdf = sio.loadmat(srcCoords_fn)
        #lbls = crdf['labels'][0]
        #coord_labels_corresp_coord = [  lbls[i][0] for i in range(len(lbls)) ]

        coords = crdf['coords_Jan_actual']
        srcgroups_ = crdf['point_ind_corresp']

        if min(srcgroups_) == 0:
            srcgroups_ += 1
            coord_labels_corresp_coord = ['unlabeled'] + coord_labels_corresp_coord

        print(coord_labels_corresp_coord)
        #crdf['pointlabel']

    elif sources_type == 'HirschPt2011':
        #<0 left

        srcCoords_fn = sind_str + '_modcoord.mat'
        coord_labels_corresp_coord = ['']* len(coords_MNI)
        assert len(coord_labels_corresp_coord ) == len(coord_labels) * 2
        for coordi in range(len(coords_MNI)):
            labeli = coordi // 2
            if coords_MNI[coordi,0] < 0:
                side = 'L'
            else:
                side = 'R'
            coord_labels_corresp_coord[coordi]= coord_labels[labeli] + '_{}'.format(side)

        srcgroups_ = crdf['point_ind_corresp'][0]
        print(coords_MNI,coord_labels_corresp_coord)


    print(srcgroups_)

    numcenters = np.max(srcgroups_)
    print(numcenters)
    print(coords.shape, coords)

    #dispGroupInfo(srcgroups_)

    ##########

    desired_selected_by_me = False
    if desired_selected_by_me:
        desired = gp.areas_list_aal_my_guess
    else:
        desired = coord_labels_corresp_coord

    ######3

    scrgroups_dict = {}
    stcs = []
    fbands = {}
    custom_raws = {}
    vertices_inds_dict = {} # values are lists of lists of indices in the original data

    if sources_type == 'HirschPt2011':
        indsets = {'centers': slice(0,numcenters), 'surround': slice(numcenters,None)}
    elif sources_type == 'parcel_aal':
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

            labels_deford = np.array(coord_labels_corresp_coord)[srcgroups_[allinds ]-1 ]
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

                #if sources_type == 'HirschPt2011':
                labels_coord_ord = np.array(coord_labels_corresp_coord)[srcgroups_[allinds[concat] ]-1 ]
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

            srcgroups = srcgroups_[indsets[indset_name] ]-1
            srcgroups = srcgroups [ allinds [concat] ]
            #srcgroups = srcgroups_[posinds]-1
            assert min(srcgroups) == 0
            if sources_type == 'HirschPt2011':
                assert max(srcgroups) == len(coords)-1, ( max(srcgroups)+1, len(coords) )
            info['srcgroups'] = srcgroups
            scrgroups_dict[indset_name] = srcgroups

            custom_raw_cur = mne.io.RawArray(srcData, info)
            for chi in range(len(custom_raw_cur.info['chs']) ):
                custom_raw_cur.info['chs'][chi]['loc'][:3] = pos[concat][chi,:3]


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
        lab = coord_labels_corresp_coord[srcgroups[chni] ]
        loc = custom_raw_cur.info['chs'][chni]['loc']
        leftlab = lab.find('_L') >= 0
        leftside = loc[0] <= 0
        if leftlab ^ leftside:
            print(chni, chn, loc[0], lab)
    print(len(Lchns), len(chns))


    ##################

    if sources_type == 'HirschPt2011':
        newsrc_fname_full = os.path.join( data_dir, 'cnt_' + src_fname_noext + '.fif' )
        print( newsrc_fname_full )

        custom_raws['centers'].save(newsrc_fname_full, overwrite=1)
    else:
        print('other sources, do nothing')

    custom_raws_pri[rawni] = custom_raws



nPCA_comp = 0.95
algType = 'PCA+ICA' #'PCA' # 'mean'

########################### preparing groups ###################3

coord_labels_corresp_coord_cb_vs_rest = ['Cerebellum_L', 'Cerebellum_R',
                                         'notCerebellum_L', 'notCerebellum_R',
                                         'unlabeled']

srcgroups_all = custom_raws['surround'].info['srcgroups']

cbinds = [i for i in range(len(coord_labels_corresp_coord) ) if
 coord_labels_corresp_coord[i].find('Cerebellum') >= 0 ]  # indices ofcb in original

assert len(cbinds )  == 2

srcgroups_cb_vs_rest = srcgroups_all.copy() # copy needed
b = [True] * len(srcgroups_cb_vs_rest)  # first include all
# mark those that are not cerebellum
for i in cbinds:
    b = np.logical_and( srcgroups_cb_vs_rest != i, b)

#srcgroups_cb_vs_rest

#addCBinds = True

for j in range(len(srcgroups_cb_vs_rest)):
    curi = srcgroups_cb_vs_rest[j]
    if b[j]:  # rename not-cerebellum
        parcel_name = coord_labels_corresp_coord[curi]
        if parcel_name == 'unlabeled':
            newlabi = coord_labels_corresp_coord_cb_vs_rest.index(parcel_name)
        else:
            sidelet = parcel_name[-1]
            newlabi = coord_labels_corresp_coord_cb_vs_rest.index('notCerebellum_' + sidelet)
        srcgroups_cb_vs_rest[srcgroups_all == curi] = newlabi
#     else if srcgroups[j] in cbinds  and addCBinds:
#         srcgroups_cb_vs_rest

set_CB_unlab = False #make sense if we compute Cerbellum for other srcgroups
for newlabi,newlab in enumerate(coord_labels_corresp_coord_cb_vs_rest[:2] ):
    oldind = np.array(coord_labels_corresp_coord)[cbinds].tolist().index(newlab)
    print(oldind,newlab,newlabi)
    if set_CB_unlab:
        newlabi = coord_labels_corresp_coord_cb_vs_rest.index('unlabeled')
    srcgroups_cb_vs_rest[srcgroups_all == cbinds[oldind] ] = newlabi  # its on purpose that I use 'old' srcgroups

dispGroupInfo(srcgroups_cb_vs_rest)
print(srcgroups_cb_vs_rest)

#-------

srcgroups_cbm_vs_rest = srcgroups_cb_vs_rest.copy()

lind = coord_labels_corresp_coord_cb_vs_rest.index('Cerebellum_L')
rind = coord_labels_corresp_coord_cb_vs_rest.index('Cerebellum_R')
srcgroups_cbm_vs_rest[srcgroups_cbm_vs_rest == rind] = lind
srcgroups_cbm_vs_rest[srcgroups_cbm_vs_rest > 1] -= 1

#-------------

coord_labels_corresp_coord_cbm_vs_rest = coord_labels_corresp_coord_cb_vs_rest[:]
del coord_labels_corresp_coord_cbm_vs_rest[0]
coord_labels_corresp_coord_cbm_vs_rest[0] = 'Cerebellum'

#srcgroups_cbm_vs_rest

unlabi = coord_labels_corresp_coord_cbm_vs_rest.index('unlabeled'); unlabi
srcgroups_merged = srcgroups_cbm_vs_rest.copy()
srcgroups_merged[srcgroups_merged != unlabi] = 0
srcgroups_merged[srcgroups_merged == unlabi] = 1
coord_labels_corresp_coord_merged = ['cortex', 'unlabeled']

coord_labels_corresp_coord_merged_by_side = ['left_hemisphere',
                                             'right_hemisphere', 'unlabeled']
srcgroups_merged_by_side = srcgroups_all.copy()
for labi, label in enumerate(coord_labels_corresp_coord):
    if label.endswith('_L'):
        srcgroups_merged_by_side[srcgroups_all == labi] = 0
    elif label.endswith('_R'):
        srcgroups_merged_by_side[srcgroups_all == labi] = 1
    else:
        srcgroups_merged_by_side[srcgroups_all == labi] = 2


#------------------

srcgroups_dict = {}
srcgroups_dict['all'] = srcgroups_all
srcgroups_dict['CB_vs_rest'] = srcgroups_cb_vs_rest
srcgroups_dict['CBmerged_vs_rest'] = srcgroups_cbm_vs_rest
srcgroups_dict['merged'] = srcgroups_merged
srcgroups_dict['merged_by_side'] = srcgroups_merged_by_side



#coord_labels_corresp_coord

coord_labels_corresp_dict = {}
coord_labels_corresp_dict['all'] = coord_labels_corresp_coord
coord_labels_corresp_dict['CB_vs_rest'] = coord_labels_corresp_coord_cb_vs_rest
coord_labels_corresp_dict['CBmerged_vs_rest'] = coord_labels_corresp_coord_cbm_vs_rest
coord_labels_corresp_dict['merged'] = coord_labels_corresp_coord_merged
coord_labels_corresp_dict['merged_by_side'] = coord_labels_corresp_coord_merged_by_side


if 'Cerebellum' not in desired:
    desired += ['Cerebellum']

###############################################

#del srcgroups_dict['all']
del srcgroups_dict['CB_vs_rest']
del srcgroups_dict['CBmerged_vs_rest']
del srcgroups_dict['merged']
del srcgroups_dict['merged_by_side']
#----------------


################################### Copmute


#srcgroups_list += [custom_raws['surround'].info['srcgroups']]
skip_ulabeled = True
duplicate_merged_across_sides = True # applies to 'merged' and 'CBmerged_vs_rest'
newchnames = []
newdatas = []
avpos = []
pcas = []
icas = []
sort_keys = list( sorted(srcgroups_dict.keys()) )
#sort_keys = ['CBmerged_vs_rest']
# cycle over parcellations
for srcgi,srcgroups_key in enumerate(sort_keys):
    print('   Starting working wtih grouping ',srcgroups_key)
    srcgroups = srcgroups_dict[srcgroups_key]
    if srcgroups_key == 'all':
        assert min(srcgroups) == 0
    if sources_type == 'HirschPt2011':
        assert max(srcgroups) == len(coords)-1, ( max(srcgroups)+1, len(coords) )

    # cycle over parcel indices
    for i in range( max(srcgroups)+1 ):
        merge_needed = False
        # get the (orderd) list of parcels in the current parcellation
        coord_labels_corresp = coord_labels_corresp_dict[srcgroups_key]
        cur_parcel = coord_labels_corresp[i]
        desired_ind = False  # whether we desire this index or not
        for des in desired:
            desired_ind = desired_ind or (cur_parcel.find(des) >= 0)
        if not desired_ind or (skip_ulabeled and cur_parcel.find('unlabeled') >= 0):
            continue

        inds = np.where(srcgroups == i)[0]
        #srcData[inds]
        if cur_parcel == 'Cerebellum':  # NOT 'Cerebellum_L' or 'Cerebellum_R':
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
                if coords[i][0] <= 0:
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

            if algType.startswith('PCA'):
                pca = PCA(n_components=nPCA_comp)
                pca.fit(chdata.T)
                newdata = pca.transform(chdata.T).T
                #newchnames += ['msrc{}_{}_{}_c{}'.format(brainside,bandname,i,ci) \
                #               for ci in range(newdata.shape[0])]

                pcas += [pca]
                if algType.find('ICA') >= 0:
                    max_iter = 500
                    ica = FastICA(n_components=len(newdata),
                                  random_state=0, max_iter=max_iter)
                    ica.fit(newdata.T)
                    if ica.n_iter_ < max_iter:
                        newdata = ica.transform(newdata.T).T
                    else:
                        # newdata does not change in this case
                        print('Did not converge')

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
                newchnames += ['msrc{}_{}_{}'.format(brainside,bandname,i)]
            #print(chnames)
            print('{}={}: {} over {}, newdata shape {}'.
                  format(i,coord_labels_corresp[i], algType,
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
    scrgroups_per_indset[crt] = custom_raws[crt].info['srcgroups']


for rawname_ in rawnames:
    src_rec_info_fn = '{}_{}_src_rec_info'.format(rawname_,sources_type)
    src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    print(src_rec_info_fn_full)
    np.savez(src_rec_info_fn_full, scrgroups_dict=scrgroups_dict,
            scrgroups_per_indset = scrgroups_per_indset,
            coords_Jan_actual=coords,
            coord_labels_corresp_dict = coord_labels_corresp_dict,
            srcgroups_key_order = sort_keys,
            coords_MNI=coords_MNI,
            pcas=pcas, icas=icas,
            avpos=avpos, algType=algType, vertices_inds_dict=vertices_inds_dict)


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
        nn = utils.getMEGsrc_chname_nice(chn,coord_labels_corresp_dict, sort_keys)
        srcgroup_ind, ind, subind = utils.parseMEGsrcChnameShort(chn)
        #print(nn)
        # computed CB twice, so we don't want to save it twice
        if nn.find('Cerebellum') >= 0 and sort_keys[srcgroup_ind] in sort_keys:

            print(chni,nn, srcgroup_ind, ind, subind)
            inds += [chni]
            newchnames_filtered += [chn]
            newdatas_filtered += [dd[chni] ]
            #newdatas_filtered += [ newdatas[ind][subind] ]

else:
    newchnames_filtered = newchnames
    newdatas_filtered = newdatas
#------------
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

srcgroups_backup = custom_raws['surround'].info['srcgroups']

newraw = mne.io.RawArray(dd, info)

curstart = 0
for rawni,rawname_ in enumerate(rawnames):
    if sources_type == 'HirschPt2011':
        newraw_cur = custom_raws['surround'].copy()
    else:
        newraw_cur = newraw.copy()
    tt = custom_raws_pri[rawni]['surround'].times
    newraw_cur.crop( curstart + tt[0], curstart + tt[-1], include_tmax = True )
    curstart += tt[-1] + tstep

    src_fname_noext = 'srcd_{}_{}'.format(rawname_, sources_type)

    #  Save
    newsrc_fname_full = os.path.join( data_dir, 'av_' + src_fname_noext + '.fif' )
    print( newsrc_fname_full )

    newraw_cur.save(newsrc_fname_full, overwrite=1)
