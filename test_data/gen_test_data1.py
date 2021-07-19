main_side = 'left'
rawnames = ['S95_off_move','S95_off_hold','S95_on_move']
test_data_descr = '''Here I want to test just scaling of entire, very basic
'''

######################
anndict_per_intcat_per_rawn = {}
for rawn in rawnames:
    anndict_per_intcat_per_rawn[rawn] = {'artif':{
        'MEG':mne.Annotations([],[],[]), 'LFP': mne.Annotations([],[],[])  },
        'beh_state':[]}
ann1 = mne.Annotations([0.24,0.8],[0.5,0.6],['notrem_L','trem_L'])
ann15 = mne.Annotations([0.24,2.5,0.8],[0.5,0.9,0.6],['notrem_L','notrem_L','trem_L'])
ann2 = mne.Annotations([1.24,0.0,3.0],[0.1,0.3,1.2],['notrem_L','trem_L','trem_L'])

i__ = 0; rawn = rawnames[i__]; ann_dict = {'artif':{}, 'beh_state':[]}
ann_dict['artif']['MEG'] = mne.Annotations([0.2],[0.23],['BAD_MEGR'])
ann_dict['artif']['LFP'] = mne.Annotations([1.2],[1.23],['BAD_LFPR'])
ann_dict['beh_state']= ann1
anndict_per_intcat_per_rawn[rawn] = ann_dict
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_anns.txt'), overwrite=1 )
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_ann_MEGartif_flt.txt'), overwrite=1 )
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_ann_LFPartif.txt'), overwrite=1 )


i__ += 1; rawn = rawnames[i__]; ann_dict = {'artif':{}, 'beh_state':[]}
ann_dict['artif']['MEG'] = mne.Annotations([0.3,2],[0.33,0.32],['BAD_MEGR','BAD_MEGL'])
ann_dict['artif']['LFP'] = mne.Annotations([2,3.1],[1.33,0.4],['BAD_LFPR','BAD_LFPR02'])
# anndict_per_intcat_per_rawn[rawnames[i__]]['artif']['LFP'] = mne.Annotations([1.3],[1.33],
#                                                                              ['BAD_LFPR'])
ann_dict['beh_state']= ann15
anndict_per_intcat_per_rawn[rawn] = ann_dict
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_anns.txt') , overwrite=1)
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_ann_MEGartif_flt.txt'), overwrite=1 )
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_ann_LFPartif.txt'), overwrite=1 )

i__ += 1; rawn = rawnames[i__]; ann_dict = {'artif':{}, 'beh_state':[]}
ann_dict['artif']['MEG'] = mne.Annotations([0.4],[0.43],['BAD_MEGR'])
ann_dict['artif']['LFP'] = mne.Annotations([1.4,3.1],[1.43,0.4],
                                                                             ['BAD_LFPR','BAD_LFPR092'])
ann_dict['beh_state']= ann2
anndict_per_intcat_per_rawn[rawn] = ann_dict
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_anns.txt') , overwrite=1)
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_ann_MEGartif_flt.txt'), overwrite=1 )
ann_dict['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_ann_LFPartif.txt'), overwrite=1 )

#parcels 2,18 are coupled, 3,5 are not;  60 -- Cerebellum_L (coupled to 2,18 too)
########################
#sfo_LFP = ['LFPL001','LFPL002','LFPR12']
sfo_LFP = ['LFPL001','LFPL002', 'LFPR092', 'LFPR015']
sfo = sfo_LFP +     ['msrcR_0_2_c1', 'msrcL_0_60_c34', 'msrcL_0_19_c1']
sfo += ['msrcR_0_18_c0', 'msrcR_0_3_c5']
##sfo += ['msrcL_0_19_c1','msrcL_0_47_c0','msrcR_0_59_c33','msrcR_0_46_c4']
special_chns = {}

sfo_pri = [sfo]*len(rawnames)
sfo_lfp_hires_pri = [sfo_LFP]*len(rawnames)

datlen_s = 5
nbins = sfreq * datlen_s
nbins_hires = sfreq_hires * datlen_s
noise_size = 3e-3
#dat =  defdgen( (len(sfo), nbins )) * noise_size
#dat_hires = defdgen ( (len(sfo_LFP), nbins_hires ) ) * noise_size

dat_pri = [0] * len(rawnames)
dat_LFP_hires_pri = [0] * len(rawnames)
# make independent noise
for rawi in range(len(rawnames)):
    dat_pri[rawi]           = defdgen( (len(sfo), nbins ))            * noise_size
    dat_LFP_hires_pri[rawi] = defdgen( (len(sfo_LFP), nbins_hires ) ) * noise_size

times = np.arange( dat_pri[0].shape[1] )/sfreq
times_hires = np.arange( dat_LFP_hires_pri[0].shape[1] )/sfreq_hires

def stepf(ts,s,e):
    return np.heaviside(ts - s,0) * (1- np.heaviside(ts - e,0) )

###########################

#ss,se=2,3
bdshiftL = 0.1
bdshiftR = 0.3
ss = ann1[0]['onset'] + bdshiftL
se = ss + ann1[0]['duration'] + bdshiftR
dati = 1

#src_chi,src_chi2 = 0,1
ssbad,sebad = 2,2.5

#int(ss*sfreq):int(se*sfreq)
step =       noise_size + stepf(times,ss,se)       + stepf(times,ssbad,sebad)
step_hires = noise_size + stepf(times_hires,ss,se) + stepf(times_hires,ssbad,sebad)
freq = 0.3
d       = np.sin(times * 2 *np.pi * freq)
d_hires = np.sin(times_hires * 2 *np.pi * freq)

# set_data
LFPchi = special_chnis['LFP_coupled_to_src']
dat_pri[dati][:,  :]         += d[None,:]
dat_LFP_hires_pri[dati][:,:] += d_hires[None,:]

test_plots_descr = []


##############  put same data in all raws
data_mult_per_rawi = [1] * len(rawnames)
for rawi in range(len(rawnames)):
    if rawi == dati:
        continue
    # mulitply entire raws to check how cross-raw rescaling workgs
    mult = (rawi + 1) * 10
    data_mult_per_rawi[rawi] = mult
    dat_LFP_hires_pri[rawi] += dat_LFP_hires_pri[dati] * mult
    dat_pri[rawi]           += dat_pri[dati]           * mult
