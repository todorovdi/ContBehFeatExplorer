import os,sys
import json
import socket
import numpy as np

from os.path import join as pjoin

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
try:
    import StringIO
except ImportError:
    import io as StringIO

global specgrams
global freqBands
global raws
global srcs
global glob_stats_perint
global glob_stats
global chanTypes_toshow
global timeIntervalPerRaw_processed
global plot_timeIntervalPerRaw
global gen_subj_info
global subjs_analyzed

global gparams
global artifact_intervals
global code_dir
global data_dir
global dir_fig
global dir_fig_preproc
global fbands
global EMG_per_hand
global feat_types_all
global DEBUG_MODE
global DEBUG_PLOT_TFR2CSD

specgrams                           = None
freqBands                           = None
raws                                = None
glob_stats_perint                   = None
glob_stats_perint_nms               = None
glob_stats                          = None
chanTypes_toshow                    = None
timeIntervalPerRaw_processed        = None
plot_timeIntervalPerRaw             = None
gen_subj_info                       = None
subjs_analyzed                      = None
srcs                                = None
artifact_intervals                  = None

import mne
#print(f'mne version is {mne.__version__}')
#assert mne.__version__ == '0.23.0', f'mne version is {mne.__version__}'

gparams                              = {}
code_dir = os.path.expandvars('$OSCBAGDIS_DATAPROC_CODE')
gparams['intTypes'] = ['pre', 'post', 'initseg', 'endseg', 'middle_full', 'no_tremor', 'unk_activity_full' ]
gparams['intType2col'] =  {'pre':'blue', 'post':'gold', 'middle_full':'red', 'no_tremor':'green',
            'unk_activity_full': 'cyan', 'initseg': 'teal', 'endseg':'blueviolet' }


if os.environ.get('DATA_DUSS') is not None:
    data_dir = os.path.expandvars('$DATA_DUSS')
else:
    data_dir = '/home/demitau/data'

if os.environ.get('OUTPUT_OSCBAGDIS') is not None:
    dir_fig = os.path.expandvars('$OUTPUT_OSCBAGDIS')
else:
    dir_fig = '.'

param_dir = pjoin(code_dir,'params')

with open(pjoin(code_dir,'subj_info.json') ) as info_json:
    #raise TypeError
    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)

dir_fig_preproc = os.path.join(dir_fig,'preproc')


hostname = socket.gethostname()

DEBUG_MODE = False
DEBUG_PLOT_TFR2CSD = False

CUDA_state = 'no'
try:
    import GPUtil
    import pycuda
    GPUs_list = GPUtil.getAvailable()

    try:
        import pycuda.driver as cuda_driver
        try:
            cuda_driver.init()
        except pycuda.driver.Error as e:
            print(f'CUDA problem: {e}')
    except AttributeError as e:
        print(f'CUDA presence: {e}')
    CUDA_state = 'ok'
    print('GPU found, total GPU available = ',GPUs_list)
except (ImportError,ValueError) as e:
    if not hostname.startswith('jsfc'):
        print(e)
    GPUs_list = []
    CUDA_state = 'bad'



fbands = {'tremor': [3,10], 'low_beta':[11,22], 'high_beta':[22,30],
           'low_gamma':[30,60], 'high_gamma':[60,90],
          'HFO1':[91,200], 'HFO2':[200,300], 'HFO3':[300,400],
          'beta':[11,30],   'gamma':[30,90], 'HFO':[91,400]}

fband_names_crude = ['tremor', 'beta', 'gamma']
fband_names_fine = ['tremor', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma' ]
fband_names_HFO_crude = ['HFO']
fband_names_HFO_fine =  ['HFO1', 'HFO2', 'HFO3']
fband_names_HFO_all = fband_names_HFO_crude + fband_names_HFO_fine
fband_names_crude_inc_HFO = fband_names_crude + fband_names_HFO_crude
fband_names_fine_inc_HFO = fband_names_fine + fband_names_HFO_fine

wband_feat_types = ['rbcorr', 'bpcorr', 'con' ]
bichan_feat_types = ['rbcorr', 'bpcorr', 'con' ]
bichan_bifreq_feat_types = ['rbcorr', 'bpcorr' ]
bichan_bifreq_cross_feat_types = [ 'bpcorr' ]
noband_feat_types = [ 'H_act', 'H_mob', 'H_compl']
feat_types_all = [ 'con',  'Hjorth', 'H_act', 'H_mob', 'H_compl', 'bpcorr', 'rbcorr']
# used for ML and VIF selection
desired_feature_order = ['H_act', 'H_mob',  'H_compl', 'con', 'rbcorr', 'bpcorr' ]

EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
EMG_per_hand_base = {'right':['EMG061', 'EMG062'], 'left':['EMG063', 'EMG064' ] }

rawnames_combine_types = ['no', 'subj', 'medcond', 'task', 'across_everything',
                          'medcond_across_subj', 'task_across_subj']
# we cannot combine across subjects beacause we may have different channel
# numbers in different subjects
rawnames_combine_types_rawdata = ['no', 'subj', 'medcond', 'task']

data_coupling_types_all = ['self', 'LFP_vs_all', 'CB_vs_all', 'motorlike_vs_motorlike']

import re
common_regexs = {}
# r is needed because by def python3 treats \w as escape for unicode
common_regexs[ 'match_feat_beg_H'] = re.compile(r'^(H_[a-z]{1,5})_(\w+)')
common_regexs[ 'match_feat_beg_notH'] = re.compile(r'^([a-zA-Z0-9]+)_')
common_regexs[ 'match_band_ch_band_ch_beg'] = re.compile(r'^([a-zA-Z0-9]+)_(.+),([a-zA-Z0-9]+)_(.+)$')


def paramFileRead(fname,recursive=True):
    print('--Log: reading paramFile {0}'.format(fname) )

    file = open(pjoin(code_dir, fname), 'r')
    ini_str = '[root]\n' + file.read()
    file.close()
    ini_fp = StringIO.StringIO(ini_str)
    preparams = ConfigParser.RawConfigParser(allow_no_value=True)
    preparams.optionxform = str
    preparams.read_file(ini_fp)
    #preparams.readfp(ini_fp)
    #sect = paramsEnv_pre.sections()
    items= preparams.items('root')
    params = dict(items)

    if(recursive):
        addParamKeys = sorted( [ k for k in params.keys() if 'iniAdd' in k ] )
        lenAddParamKeys = len(addParamKeys)
        if(lenAddParamKeys ):
            print('---Log: found {0} iniAdd\'s, reading them'.format(lenAddParamKeys) )
        for pkey in addParamKeys:
            paramFileName = paramFileRead(params[pkey])
            params.update(paramFileName)

        # we actually want to overwrite some of the params from the added inis
        if(lenAddParamKeys):
            paramsAgain = paramFileRead(fname,recursive=False)
            params.update(paramsAgain)

    return params

class globparams:
    def __init__(self):
        self.time_format_str = "%d.%m.%Y   %H:%M:%S"
        import datetime
        dt = datetime.datetime.now()
        # it will be convenient
        print("NOW is ", dt.strftime(self.time_format_str) )

        self.hostname = socket.gethostname()
        if self.hostname.startswith('jsfc'):
            print('Hostname = ',self.hostname)
        else:
            try:
                from jupyter_helpers.notifications import Notifications
                p = '/usr/share/sounds/gnome/default/alerts/'
                sound_file  = '../beep-06.mp3'
                sound_file2 = '../glitch-in-the-matrix-600.mp3'
                #p1 = p + 'glass.ogg'; p2 = p + 'sonar.ogg';
                p1 = sound_file; p2 = sound_file2
                Notifications(success_audio=p1, time_threshold=2,
                    failure_audio=p2)  #    ,integration='GNOME')
                print('Jupyter sounds setting succeded')
            except:
                print('Jupyter sounds setting failed')


        self.hostname_home = 'demitau-ZBook'
        if self.hostname == self.hostname_home:
            self.n_free_cores = 2
        else:
            self.n_free_cores = 0

        fn = os.path.join(code_dir, 'subj_corresp.json')
        if os.path.exists(fn):
            with open(fn, 'r') as f:
                svd = json.load(f)
            self.subj_corresp = svd
        else:
            self.subj_corresp = None


        self.tremcolor = 'r'
        self.notremcolor = 'g'
        self.movecolor = 'b'  #c,y
        self.holdcolor = 'purple'  #c,y
        self.neutcolor = 'grey'

        self.color_per_int_type = { 'trem':self.tremcolor, 'notrem':self.notremcolor,
                                   'neut':self.neutcolor,
                            'move':self.movecolor, 'hold':self.holdcolor }
        self.mrk = ['<','>','o','^','v']
        self.mrknames = ['_pres','_posts','','_pree','_poste']


        self.int_types_basic = ['trem', 'notrem', 'hold', 'move']
        self.int_types_basic_sided = ['trem_L', 'notrem_L', 'hold_L', 'move_L'] + \
                ['trem_R', 'notrem_R', 'hold_R', 'move_R']
        #self.int_types_trem_and_mov = ['trem', 'hold', 'move']
        #self.int_types_trem_and_rest = ['trem', 'notrem']
        self.int_types_aux = ['undef', 'holdtrem', 'movetrem']
        self.subj_strs_all = [ 'S{:02d}'.format(i) for i in range(1,11) ] + ['S95', 'S97', 'S98', 'S99']

        medconds = ['on', 'off']
        self.subj_medcond_strs_all = []
        for subj_str in self.subj_strs_all:
            self.subj_medcond_strs_all += [ '{}_{}'.format(subj_str,mc) for mc in medconds  ]

        tasks = ['hold', 'move', 'rest' ]
        self.subj_medcond_task_strs_all = []
        for smc in self.subj_medcond_strs_all:
            self.subj_medcond_task_strs_all += [ '{}_{}'.format(smc,task) for task in tasks  ]

        int_types_all = self.int_types_basic + self.int_types_aux

        # we don't want to use undef becuse it is not clear how it should be
        # classified even on ground truth level
        self.groupings = {'merge_movements':['hold','move'],
                    'merge_all_not_trem':['notrem','hold','move'],
                    'merge_nothing':[],
                          'merge_within_task':int_types_all,
                          'merge_within_medcond':int_types_all,
                          'merge_within_subj':int_types_all     ,
                          'merge_within_medcond_across':int_types_all     }

        # I prefer to have globally non-intersecting class ids
        basic_shift = len(int_types_all) * 2 + 10  # just in case
        self.int_types_aux_cid_shift = { 'subj':basic_shift}
        self.int_types_aux_cid_shift['subj_medcond'] = \
            self.int_types_aux_cid_shift['subj'] + len(self.subj_strs_all) * 2
        self.int_types_aux_cid_shift['subj_medcond_task'] = \
            self.int_types_aux_cid_shift['subj_medcond'] + len(self.subj_medcond_strs_all) * 2
        self.int_types_aux_cid_shift['medcond'] = \
            self.int_types_aux_cid_shift['subj_medcond_task'] + len(self.subj_medcond_task_strs_all) * 2
        #self.int_types_aux_cid_shift['subj_medcond_task'] = \
        #    self.int_types_aux_cid_shift['subj_medcond'] + len(self.subj_medcond_strs_all) * 2

        self.int_type_datset_rel = list(sorted(self.int_types_aux_cid_shift.keys() ))

        # extended list of interval types
        self.int_types_ext = self.int_types_basic + self.int_types_aux
        # what is on '0' index is important
        self.int_types_to_include = {'basic': self.int_types_basic,
                                'basic+ext': self.int_types_ext,
                                'trem_vs_quiet':['trem','notrem'],
                                'trem_vs_quiet&undef':['trem','notrem','undef'],
                                'trem_vs_hold&move':['trem','hold','move'],
                                'hold_vs_quiet':['hold','notrem'],
                                'move_vs_quiet':['move','notrem'],
                                 'subj_medcond_task':self.subj_medcond_task_strs_all,
                                 'subj_medcond':self.subj_medcond_strs_all,
                                 'subj': self.subj_strs_all  ,
                                 'medcond': medconds  }

        # in trem_vs_quiet we dont' want merge_movements because it will do
        # nothing and just eat computation time
        self.group_vs_int_type_allowed = {'basic':['merge_movements', 'merge_all_not_trem', 'merge_nothing'],
                                'trem_vs_quiet':['merge_all_not_trem', 'merge_nothing'],
                                'trem_vs_quiet&undef':['merge_movements', 'merge_all_not_trem', 'merge_nothing'],
                                'trem_vs_hold&move':['merge_movements', 'merge_nothing'],
                                'hold_vs_quiet': ['merge_nothing'],
                                'move_vs_quiet': ['merge_nothing'],
                                          'subj_medcond_task': ['merge_within_task'],
                                          'subj_medcond': ['merge_within_medcond'],
                                          'subj': ['merge_within_subj'],
                                          'medcond': ['merge_within_medcond_across']}

        #motor-related
        self.areas_list_aal_my_guess = ["Precentral", "Rolandic_Oper",
                                        "Supp_Motor_Area", "Postcentral",
                                        "Parietal_Sup", "Parietal_Inf",
                                        "Precuneus", "Paracentral_Lobule",
                                        "Cerebellum" ];

        self.parcel_groupings_post0 = {}
        self.parcel_groupings_post0['M1-ish'] = ['Precentral','Supp_Motor_Area', 'Rolandic_Oper',
                                        'Postcentral', 'Paracentral_Lobule']
        self.parcel_groupings_post0['Parietal'] = ['Parietal_Sup', 'Parietal_Inf']
        self.parcel_groupings_post0['Cerebellum'] = ['Cerebellum']

        # need to cover whole brain
        # sent by Jan
        #old_labels_MATLAB = [['Precentral','Postcentral','Rolandic_Oper','Supp_Motor_Area','Paracentral'],
        # ['Frontal_Sup'],['Frontal_Mid','Frontal_Med'],['Frontal_Inf'],['Parietal_Sup','Precuneus'],
        # ['Parietal_Inf'],['Temporal_Sup','Temporal_Pole_Sup'],['Temporal_Mid','Temporal_Pole_Mid'],
        # ['Temporal_Inf','Fusiform'],['Occipital_Sup','Cuneus'],['Occipital_Mid'],
        # ['Occipital_Inf','Calcarine','Lingual'],['Angular'],['Supra_Marginal'],['Cerebellum']];
        #new_labels_MATLAB = [['Senorimotor'], ['FrontalSup'],['FrontalMed'], ['FrontalInf'],['ParietalSup'], ['ParietalInf'],['TemporalSup'], ['TemporalMid'] ,['TemporalInf'], ['OccipitalSup'], ['OccipitalMid'],['OccipitalInf'], ['Angular'],['SupraMarginal'],['Cerebellum']];
        new_labels = ['Sensorimotor', 'FrontalSup', 'FrontalMed', 'FrontalInf', 'ParietalSup',
                      'ParietalInf', 'TemporalSup', 'TemporalMid', 'TemporalInf', 'OccipitalSup', 'OccipitalMid',
                      'OccipitalInf', 'Angular', 'SupraMarginal', 'Cerebellum']

        # sent by Jan was similar to AAL but not exactly
        old_labels_MATLAB_mod = [['Precentral','Postcentral','Rolandic_Oper','Supp_Motor_Area','Paracentral_Lobule'],
         ['Frontal_Sup','Frontal_Sup_Medial', 'Frontal_Sup_Orb'],['Frontal_Mid','Frontal_Mid_Orb','Frontal_Med_Orb'],
                                 ['Frontal_Inf_Oper','Frontal_Inf_Orb','Frontal_Inf_Tri'],['Parietal_Sup','Precuneus'],
         ['Parietal_Inf'],['Temporal_Sup'],['Temporal_Mid','Temporal_Pole_Mid'],
         ['Temporal_Inf'],['Occipital_Sup','Cuneus'],['Occipital_Mid'],
         ['Occipital_Inf','Calcarine','Lingual'],['Angular'],['SupraMarginal'],['Cerebellum']];

        self.parcel_groupings_post = dict(zip(new_labels,old_labels_MATLAB_mod) )

        self.parcel_groupings_post_sided = {}
        for side in ['left','rignt']:
            sidelet = side[0].upper()
            for pgn,oldlabs in self.parcel_groupings_post.items():
                self.parcel_groupings_post_sided[pgn + '_' + sidelet] = [lab + '_' + sidelet  for lab in  oldlabs]




        self.src_grouping_names_order = ['all', 'CB_vs_rest',
                                         'CBmerged_vs_rest', 'merged',
                                         'merged_by_side',
                                         'motor-related_only',
                                         'motor-related_vs_CB_vs_rest_merge_across_sides',
                                         'motor-related_incCB_vs_rest',
                                         'motor-related_vs_CBmerged_vs_rest',
                                         'motor-related_vs_CB_vs_rest', 'all_raw',
                                         'test']


        # convert labels to what I had
        #s = ''
        #for l in labels:
        #    for ll in gp.areas_list_aal_my_guess:
        #        if l.find(ll) >= 0:
        #            print(l)
        #            s+='"{}", '.format(l[:-2])
        #s = s[:-2]


        with open(pjoin(code_dir,'subj_info.json') ) as info_json:
            self.gen_subj_info = json.load(info_json)

        self.int_types_ps = {'L':[],'R':[] }
        for side_letter in self.int_types_ps:
            self.int_types_ps[side_letter] = ['{}_{}'.format( itcur , side_letter) for \
                                        itcur in self.int_types_ext ]

        self.class_ids_def = {}
        for ind, it in enumerate(self.int_types_ps['L'] ):
            self.class_ids_def[it] = ind+1
        for ind, it in enumerate(self.int_types_ps['R'] ):
            self.class_ids_def[it] = -ind-1

        self.class_id_neut = 0

        templs = ['{}_off_move','{}_on_move','{}_off_hold','{}_on_hold']
        self.rawnames_debug =  []
        for ds in ['S95', 'S94', 'S97', 'S99', 'S98']:
            for t in templs:
                self.rawnames_debug += [t.format(ds) ]
        #self.rawnames_debug += ['S94_off_move','S94_on_move','S94_off_hold','S94_on_hold']


global gp
gp = globparams()


#normally
#roi_labels['all_raw'] =
roi_labels_def = \
 ['unlabeled', 'Precentral_L', 'Precentral_R', 'Frontal_Sup_L',
  'Frontal_Sup_R', 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L',
  'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
  'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
  'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
  'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
  'Supp_Motor_Area_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
  'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Calcarine_L', 'Calcarine_R',
  'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L',
  'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
  'Occipital_Inf_R', 'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L',
  'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
  'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
  'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Temporal_Mid_L',
  'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R',
  'Temporal_Inf_L', 'Temporal_Inf_R', 'Temporal_Sup_L', 'Temporal_Sup_R',
  'Cerebellum_R', 'Cerebellum_L']
