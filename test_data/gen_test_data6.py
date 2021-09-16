#import mne
#import numpy as np
main_side = 'left'
rawnames = ['S95_off_hold', 'S95_off_move']

test_data_descr = '''Two raws (same) nontriv data, no artifacs, to test coupling, only 2 channels,
all beh intervals have different LFP activity, to test basic ML.
Intervals are the same but in second raw description is move, not hold'''

######################
anndict_per_intcat_per_rawn = {}
for rawn in rawnames:
    anndict_per_intcat_per_rawn[rawn] = {'artif':{
        'MEG':mne.Annotations([],[],[]), 'LFP': mne.Annotations([],[],[])  },
        'beh_state':[]}
ann1 = mne.Annotations([2.64,3.6,5],[0.5,0.6,1],['notrem_L','trem_L','hold_L'])
ann2 = mne.Annotations([2.64,3.6,5],[0.5,0.6,1],['notrem_L','trem_L','move_L'])

ann1 = mne.Annotations( ann1.onset * 2, ann1.duration * 2, ann1.description)
ann2 = mne.Annotations( ann2.onset * 2, ann2.duration * 2, ann2.description)

i__ = 0; rawn = rawnames[i__];
ann_dict=anndict_per_intcat_per_rawn[rawn]
ann_dict['beh_state']= ann1
anndict_per_intcat_per_rawn[rawn] = ann_dict

i__ = 1; rawn = rawnames[i__];
ann_dict=anndict_per_intcat_per_rawn[rawn]
ann_dict['beh_state']= ann2
anndict_per_intcat_per_rawn[rawn] = ann_dict

for rawn,ad in anndict_per_intcat_per_rawn.items():
    ad['beh_state'].save( pjoin(gv.data_dir, f'{rawn}_anns.txt') , overwrite=1)
    ad['artif']['MEG'].save( pjoin(gv.data_dir, f'{rawn}_ann_MEGartif_flt.txt'), overwrite=1 )
    ad['artif']['LFP'].save( pjoin(gv.data_dir, f'{rawn}_ann_LFPartif.txt'), overwrite=1 )

#parcels 2,18 are coupled, 3,5 are not;  60 -- Cerebellum_L (coupled to 2,18 too)
########################
sfo_LFP = ['LFPR092']
sfo = sfo_LFP +     ['msrcR_0_3_c5']
special_chns = {}
special_chns['LFP_coupled_to_src'] = 'LFPR092'
special_chns['src_coupled_to_LFP'] = 'msrcR_0_3_c5'
#special_chns['chn_src_uncoupled_from_everything'] = 'msrcR_0_3_c5'

special_chnis = {}
for chn_descr,chn in special_chns.items():
    special_chnis[chn_descr] = sfo.index(chn)

sfo_pri = [sfo]*len(rawnames)
sfo_lfp_hires_pri = [sfo_LFP]*len(rawnames)

# we need longer otherwise ML would not work since size of intervals
# is comparable with windows size and wavelet size effects play too much role
datlen_s = 20
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
ss = ann1.onset[1]  # trem_L
se = ss + ann1.duration[1]
dati = 0

#src_chi,src_chi2 = 0,1
# SLOW SINE DOES NOT BELONG HERE! It is assumed that data was 1.5 hz
# high-passed, it just creates DC offset

#int(ss*sfreq):int(se*sfreq)
# trem_L
step =       noise_size + stepf(times,ss,se)
step_hires = noise_size + stepf(times_hires,ss,se)

ss2 = ann1.onset[2]  # hold_L
se2 = ss2 + ann1.duration[2]
step2       = noise_size + stepf(times,ss2,se2)
step2_hires = noise_size + stepf(times_hires,ss2,se2)

freq_beta = 20
sine_beta       = np.sin(times       * 2 *np.pi * freq_beta)
sine_beta_hires = np.sin(times_hires * 2 *np.pi * freq_beta)

freq_gamma = 46
sine_gamma       = np.sin(times       * 2 *np.pi * freq_gamma)
sine_gamma_hires = np.sin(times_hires * 2 *np.pi * freq_gamma)
freq_gamma2 = 63
sine_gamma2       = np.sin(times       * 2 *np.pi * freq_gamma2)

ss3,se3 = max(0,ann1.onset[0] - 2), datlen_s - 2.  # unrelated to beh state
freq_HFO = 200
sine_HFO_hires = np.sin(times_hires * 2 *np.pi * freq_HFO)
step3_hires = noise_size + stepf(times_hires,ss3,se3)

# set_data
LFPchi = special_chnis['LFP_coupled_to_src']
#step2       = 0.
#step2_hires = 0.
dat_pri[dati][LFPchi,  :]         +=     step * sine_beta + step2 * sine_gamma

# beta is everywhere, gamma is only in LFP
dat_LFP_hires_pri[dati][LFPchi,:] += step_hires * sine_beta_hires +\
    step2_hires * sine_gamma_hires + step3_hires * sine_HFO_hires * 1e-1
dat_pri[dati][special_chnis['src_coupled_to_LFP'], :]  += \
    ( step * sine_beta  ) * 0.5


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


# in msrc there is also gamma but of different freq,
# so they won't be con and rbcorr
dat_pri[1] +=  step2 * sine_gamma2


# LFP has beta associated with tremor and gamma assoc with hold. And also HFO
# not related to anything, covering both tremor and hold intervals
# msrc has only beta assoc with tremor

# so we should see good coupling con and rbcorr between the two modalities
# in beta -- for tremor and not much in gamma for hold

test_plots_descr = []
test_plots_descr += [{'chn_descrs':['LFP_coupled_to_src','src_coupled_to_LFP'],
                      'feat_types_actual_coupling':['rbcorr','bpcorr','con'],
                      'relevant_freqs':[freq_beta,freq_gamma, freq_HFO],
                      'informal_descr':'tremor assoc with beta in LFP and src, hold assoc with gamma in LFP , HFO is indep of beh state ',
                       'figname':f'LFP_to_src_within_{freq_beta,freq_gamma,freq_HFO}' }]
