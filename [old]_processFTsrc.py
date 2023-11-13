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
import sys, getopt

data_dir = gv.data_dir



print('sys.argv is ',sys.argv)
effargv = sys.argv[1:]  # to skip first
if sys.argv[0].find('ipykernel_launcher') >= 0:
    effargv = sys.argv[3:]  # to skip first three

helpstr = 'Usage example\n<filename>.py --rawname <rawname_naked> '
opts, args = getopt.getopt(effargv,"hr:",
        ["rawname="])
print(sys.argv, opts, args)

for opt, arg in opts:
    print(opt)
    if opt == '-h':
        print (helpstr)
        sys.exit(0)
    elif opt in ('-r','--rawname'):
        rawnames = arg.split(',')  #lfp of msrc
        if len(rawnames) > 1:
            print('Using {} datasets at once'.format(len(rawnames) ) )
            sys.exit(0)
        rawname_ = rawnames[0]
        #rawname_ = arg


#rawname_ = 'S01_off_hold'
#rawname_ = 'S01_on_hold'
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

sind_str,medcond,task = utils.getParamsFromRawname(rawname_)

rawname = rawname_ + '_resample_raw.fif'
fname_full = os.path.join(data_dir,rawname)

# read file -- resampled to 256 Hz,  Electa MEG, EMG, LFP, EOG channels
raw = mne.io.read_raw_fif(fname_full, None)

reconst_name = rawname_ + '_resample_afterICA_raw.fif'
reconst_fname_full = os.path.join(data_dir,reconst_name)
reconst_raw = mne.io.read_raw_fif(reconst_fname_full, None)

src_fname_noext = 'srcd_{}_HirschPt2011'.format(rawname_)

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

print( ff.keys() )
print( ff['source_data'] )
#print( ff['source_data'].keys() )
print( ff['source_data'][0,0] )
f = ff[ ff['source_data'][0,0] ]
print( f.keys() )
print( ''.join( map(chr, f['source_data']['method'][:,0] ) ) )
print( f['source_data']['avg'].keys() )


# Gather info
with open(os.path.join('.','coord_labels.json') ) as jf:
    coord_labels = json.load(jf)
#    gv.gparams['coord_labels'] = coord_labels


srcCoords_fn = 'coordsJan.mat'
coords_MNI_f = sio.loadmat(srcCoords_fn)
coords_MNI = coords_MNI_f['coords']

coord_labels_corresp_coord = ['']* len(coords_MNI)
assert len(coord_labels_corresp_coord ) == len(coord_labels) * 2
for coordi in range(len(coords_MNI)):
    labeli = coordi // 2
    if coords_MNI[coordi,0] < 0:
        side = 'L'
    else:
        side = 'R'
    coord_labels_corresp_coord[coordi]= coord_labels[labeli] + '_{}'.format(side)
#<0 left
print(coords_MNI,coord_labels_corresp_coord)

srcCoords_fn = sind_str + '_modcoord.mat'
crdf = sio.loadmat(srcCoords_fn)
coords = crdf['coords_Jan_actual']
srcgroups_ = crdf['point_ind_corresp'][0]
print(srcgroups_)

numcenters = np.max(srcgroups_)
print(numcenters)
print(coords)



# Load LCMV
sind_str,medcond,task = utils.getParamsFromRawname(rawname)

scrgroups_dict = {}
stcs = []
fbands = {}
custom_raws = {}
indsets = {'centers': slice(0,numcenters), 'surround': slice(numcenters,None)}

pos_ = f['source_data']['pos'][:,:].T

# check that correspondence of sides is correct
for posi in range(len(pos_) ):
    corresp_cnt_coord = coords[ srcgroups_[posi] - 1 ]
    cur_pt_coord = pos_[posi]
    assert np.sign(cur_pt_coord[0] ) == np.sign( corresp_cnt_coord[0] )


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

    srcRefs = f['source_data']['avg']['mom'][0,:]
    srcData_ = [0]* len(srcRefs)
    for srci in range(len(srcRefs)):
        srcData_[srci] = f[srcRefs[srci] ][:,0]
    srcData_ = np.vstack(srcData_)

    numsrc_total = len(srcRefs)
    for indset_name in indsets:
        allinds = np.arange(numsrc_total)[indsets[indset_name] ]
        pos = pos_[allinds]
        # first coord is the left-right coord
        #posinds = np.argsort( pos[:,0] )  # I'll need to sort correspondance as well
        #print(posinds)
        #sortedPos = pos[posinds]
        #leftInds = np.where(sortedPos[:,0]<= 0)[0]
        #rightInds = np.where(sortedPos[:,0] > 0)[0]
        leftInds = np.where(pos[:,0]<= 0)[0]
        rightInds = np.where(pos[:,0] > 0)[0]
        vertices = [leftInds, rightInds]

        concat = np.concatenate((leftInds, rightInds))
        srcData = srcData_[ allinds [concat]  ]
        #srcData = srcData_[posinds]

        print(np.array(coord_labels_corresp_coord)[srcgroups_[allinds[concat] ]-1 ])
        # create MNE structure that I'm actually not using later
        #(data, vertices=None, tmin=None, tstep=None, subject=None, verbose=None
        stc = mne.SourceEstimate(data = srcData, tmin = t0, tstep= tstep  ,
                                 subject = sind_str , vertices=vertices)
        stcs += [stc]

        fbands[bandname] = freqBand

        ####  Create my

        lhi = map(str, list( vertices[0] ) )
        rhi = map(str, list( vertices[1] ) )

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
        #srcgroups = srcgroups_[posinds]-1
        assert min(srcgroups) == 0
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


#Average cources
newsrc_fname_full = os.path.join( data_dir, 'cnt_' + src_fname_noext + '.fif' )
print( newsrc_fname_full )

custom_raws['centers'].save(newsrc_fname_full, overwrite=1)


srcgroups = custom_raws['surround'].info['srcgroups']
assert min(srcgroups) == 0
assert max(srcgroups) == len(coords)-1, ( max(srcgroups)+1, len(coords) )
newchanames = []
newdatas = []
avpos = []
for i in range( max(srcgroups)+1 ):
    inds = np.where(srcgroups == i)[0]
    #srcData[inds]
    #L or R?
    if coords[i][0] <= 0:
        brainside = 'L'
    else:
        brainside = 'R'

    for bandname in bandnames:
        chnames = [ 'src{}_{}_{}'.format(brainside,bandname,s) for s in inds ]
        chdata, times = custom_raws['surround'][chnames]
        newdata = np.mean(chdata,axis=0)[None,:]
        #print(chnames)
        print('{}: Mean over {}'.format(i,chdata.shape[0]))
        newchname = 'msrc{}_{}_{}'.format(brainside,bandname,i)
        newchanames += [newchname]
        newdatas    += [newdata]

        avpos_cur = np.zeros(3)
        for chi in mne.pick_channels(custom_raws['surround'].ch_names,chnames):
            avpos_cur += custom_raws['surround'].info['chs'][chi]['loc'][:3]
        avpos_cur /= len(chnames)
        avpos += [avpos_cur]


# save supplementary info
src_rec_info_fn = '{}_src_rec_info'.format(rawname_)
src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
print(src_rec_info_fn_full)
np.savez(src_rec_info_fn_full, scrgroups_dict=scrgroups_dict,coords_Jan_actual=coords,
         coords_MNI=coords_MNI,coord_labels_corresp_coord=coord_labels_corresp_coord,
        avpos=avpos)


# Save surround

info = mne.create_info(
    ch_names=newchanames,
    ch_types=['csd'] * len(newchanames),
    sfreq=int ( 1/tstep ))

for chi in range(len(info['chs']) ):
    info['chs'][chi]['loc'][:3] = avpos[chi]

srcgroups_backup = custom_raws['surround'].info['srcgroups']


custom_raws['surround'].add_channels([ mne.io.RawArray(np.vstack(newdatas), info)] )
custom_raws['surround'].info['srcgroups'] = srcgroups_backup

newsrc_fname_full = os.path.join( data_dir, 'av_' + src_fname_noext + '.fif' )
print( newsrc_fname_full )

custom_raws['surround'].save(newsrc_fname_full, overwrite=1)
