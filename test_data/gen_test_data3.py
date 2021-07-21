#import mne
#import numpy as np
main_side = 'left'
rawnames = ['S95_off_move','S95_off_hold','S95_on_move']

test_data_descr = '''Nontriv data, no artifacs'''

######################
anndict_per_intcat_per_rawn = {}
for rawn in rawnames:
    anndict_per_intcat_per_rawn[rawn] = {'artif':{
        'MEG':mne.Annotations([],[],[]), 'LFP': mne.Annotations([],[],[])  },
        'beh_state':[]}
ann1 = mne.Annotations([2.64,3.6],[0.5,0.6],['notrem_L','trem_L'])
ann2 = mne.Annotations([0.24,2.5,0.8],[0.5,0.9,0.6],['notrem_L','notrem_L','trem_L'])
ann3 = mne.Annotations([1.24,0.0,3.0],[0.1,0.3,1.2],['notrem_L','trem_L','trem_L'])

i__ = 0; rawn = rawnames[i__];
ann_dict=anndict_per_intcat_per_rawn[rawn]
ann_dict['beh_state']= ann1
anndict_per_intcat_per_rawn[rawn] = ann_dict

i__ += 1; rawn = rawnames[i__];
ann_dict=anndict_per_intcat_per_rawn[rawn]
ann_dict['beh_state']= ann2
anndict_per_intcat_per_rawn[rawn] = ann_dict

i__ += 1; rawn = rawnames[i__];
ann_dict=anndict_per_intcat_per_rawn[rawn]
ann_dict['beh_state']= ann3
anndict_per_intcat_per_rawn[rawn] = ann_dict

for rawn,ad in anndict_per_intcat_per_rawn.items():
    ad['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_anns.txt') , overwrite=1)
    ad['artif']['MEG'].save( pjoin(gv.data_dir, f'{rawn}_ann_MEGartif_flt.txt'), overwrite=1 )
    ad['artif']['LFP'].save( pjoin(gv.data_dir, f'{rawn}_ann_LFPartif.txt'), overwrite=1 )

#parcels 2,18 are coupled, 3,5 are not;  60 -- Cerebellum_L (coupled to 2,18 too)
########################
#sfo_LFP = ['LFPL001','LFPL002','LFPR12']
sfo_LFP = ['LFPL001','LFPL002', 'LFPR092', 'LFPR015']
sfo = sfo_LFP +     ['msrcR_0_2_c1', 'msrcL_0_60_c34', 'msrcL_0_19_c1']
sfo += ['msrcR_0_18_c0', 'msrcR_0_3_c5']
##sfo += ['msrcL_0_19_c1','msrcL_0_47_c0','msrcR_0_59_c33','msrcR_0_46_c4']
special_chns = {}
special_chns['LFP_uncoupled_from_everything'] = 'LFPR015'
special_chns['LFP_coupled_to_src'] = 'LFPR092'
special_chns['src_coupled_to_LFP'] = 'msrcR_0_3_c5'
special_chns['src_coupled_to_LFP_HFO_cross_freq'] = 'msrcL_0_60_c34'
special_chns['LFP_coupled_to_src_HFO_cross_freq'] = 'LFPR015'
special_chns['src_coupled_to_src1'] = 'msrcR_0_2_c1'
special_chns['src_coupled_to_src2'] = 'msrcR_0_18_c0'
special_chns['src_coupled_to_src3'] = 'msrcL_0_60_c34'
#special_chns['chn_src_uncoupled_from_everything'] = 'msrcR_0_3_c5'

special_chnis = {}
for chn_descr,chn in special_chns.items():
    special_chnis[chn_descr] = sfo.index(chn)

sfo_pri = [sfo]*len(rawnames)
sfo_lfp_hires_pri = [sfo_LFP]*len(rawnames)

datlen_s = 6
nbins = sfreq * datlen_s
nbins_hires = sfreq_hires * datlen_s
noise_size = 0

dat_pri = [0] * len(rawnames)
dat_LFP_hires_pri = [0] * len(rawnames)
# make independent noise
for rawi in range(len(rawnames)):
    dat_LFP_hires_pri[rawi] = defdgen( (len(sfo_LFP), nbins_hires ) ) * noise_size
    dat_pri[rawi]           = defdgen( (len(sfo), nbins )) * noise_size

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
freq, freq2 = 0.3, 0.13
sine_slow       = np.sin(times * 2 *np.pi * freq)
sine_slow_hires = np.sin(times_hires * 2 *np.pi * freq)

sine_slow2      = np.sin(times * 2 *np.pi * freq2)

ss2,se2 = se-0.5, se+0.4
step2       = noise_size + stepf(times,ss2,se2)
step2_hires = noise_size + stepf(times_hires,ss2,se2)

freq_beta = 20
sine_beta       = np.sin(times       * 2 *np.pi * freq_beta)
sine_beta_hires = np.sin(times_hires * 2 *np.pi * freq_beta)

freq_gamma = 46
sine_gamma       = np.sin(times       * 2 *np.pi * freq_gamma)
sine_gamma_hires = np.sin(times_hires * 2 *np.pi * freq_gamma)

ss3,se3 = 2.6,3.8
freq_HFO = 200
sine_HFO_hires = np.sin(times_hires * 2 *np.pi * freq_HFO)
step3_hires = noise_size + stepf(times_hires,ss3,se3)

ss4,se4 = 2.6,4.6
freq_tremor = 8
sine_tremor = np.sin(times * 2 *np.pi * freq_tremor)
step4       = noise_size + stepf(times,      ss4,se4)
step4_hires = noise_size + stepf(times_hires,ss4,se4)

# set_data
LFPchi = special_chnis['LFP_coupled_to_src']
dat_pri[dati][LFPchi,  :]         += sine_slow       + step * sine_beta + step2 * sine_gamma

dat_LFP_hires_pri[dati][LFPchi,:] += sine_slow_hires + step_hires * sine_beta_hires +\
    step2_hires * sine_gamma_hires + step3_hires * sine_HFO_hires * 5e-2
dat_pri[dati][special_chnis['src_coupled_to_LFP'], :]  += (sine_slow  + step * sine_beta + step2 * sine_gamma ) * 0.5

# 2 and 3 are coupled between each other, but only after filtering (bandpower as well)
dat_pri[dati][special_chnis['src_coupled_to_src1'], :] += (sine_slow       + step4 * sine_tremor ) * 1e-2 #* 1e-5
dat_pri[dati][special_chnis['src_coupled_to_src2'], :] += (sine_slow   + sine_slow2  + step4 * sine_tremor ) * 5e-2
dat_pri[dati][special_chnis['src_coupled_to_src3'], :] += (sine_slow2      + step4 * sine_tremor ) * 1e2
# yes, I want to use 4 and 3 here (not equal numbers) because I want to have HFO sine
dat_LFP_hires_pri[dati][special_chnis['LFP_coupled_to_src_HFO_cross_freq'], :] += \
    step4_hires * sine_HFO_hires


test_plots_descr = []
test_plots_descr += [{'chn_descrs':['LFP_coupled_to_src','src_coupled_to_LFP'],
                      'feat_types_actual_coupling':['rbcorr','bpcorr','con'],
                      'relevant_freqs':[freq_beta,freq_gamma],
                      'informal_descr':'should show within freq, not cross freq',
                       'figname':f'LFP_to_src_within_{freq_beta,freq_gamma}' }]

test_plots_descr += [{'chn_descrs':['src_coupled_to_src1','src_coupled_to_src2','src_coupled_to_src3'],
                      'feat_types_actual_coupling':['rbcorr','con'],
                      'relevant_freqs':[freq_tremor],
                      'informal_descr':'should show within freq, not cross freq',
                       'figname':f'src_to_src_within_{freq_tremor}' }]

test_plots_descr += [{'chn_descrs':['src_coupled_to_LFP_HFO_cross_freq','LFP_coupled_to_src_HFO_cross_freq'],
                      'feat_types_actual_coupling':['bpcorr'],
                      'relevant_freqs':[freq_tremor,freq_HFO],
                      'informal_descr':'should show cross freq, not within-freq',
                       'figname':f'src_to_LFP_cross_{freq_tremor,freq_HFO}' }]

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
