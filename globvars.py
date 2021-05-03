import os
import json
import socket
import numpy as np

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
global data_dir
global dir_fig
global dir_fig_preproc
global fbands
global EMG_per_hand

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

gparams                              = {}

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

with open('subj_info.json') as info_json:
    #raise TypeError
    #json.dumps({'value': numpy.int64(42)}, default=convert)
    gen_subj_info = json.load(info_json)

dir_fig_preproc = os.path.join(dir_fig,'preproc')


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

EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }
EMG_per_hand_base = {'right':['EMG061', 'EMG062'], 'left':['EMG063', 'EMG064' ] }

rawnames_combine_types = ['no', 'subj', 'medcond', 'task', 'across_everything',
                          'medcond_across_subj', 'task_across_subj']
# we cannot combine across subjects beacause we may have different channel
# numbers in different subjects
rawnames_combine_types_rawdata = ['no', 'subj', 'medcond', 'task']

data_coupling_types_all = ['self', 'LFP_vs_all', 'CB_vs_all', 'motorlike_vs_motorlike']

def paramFileRead(fname,recursive=True):
    print('--Log: reading paramFile {0}'.format(fname) )

    file = open(fname, 'r')
    ini_str = '[root]\n' + file.read()
    file.close()
    ini_fp = StringIO.StringIO(ini_str)
    preparams = ConfigParser.RawConfigParser(allow_no_value=True)
    preparams.optionxform = str
    preparams.readfp(ini_fp)
    #sect = paramsEnv_pre.sections()
    items= preparams.items('root')
    params = dict(items)

    if(recursive):
        addParamKeys = sorted( [ k for k in params.keys() if 'iniAdd' in k ] )
        l = len(addParamKeys)
        if(l ):
            print('---Log: found {0} iniAdd\'s, reading them'.format(l) )
        for pkey in addParamKeys:
            paramFileName = paramFileRead(params[pkey])
            params.update(paramFileName)

        # we actually want to overwrite some of the params from the added inis
        if(l):
            paramsAgain = paramFileRead(fname,recursive=False)
            params.update(paramsAgain)

    return params

class globparams:
    def __init__(self):

        self.hostname = socket.gethostname()
        if not self.hostname.startswith('jsfc'):
            print('Hostname = ',self.hostname)

        self.hostname_home = 'demitau-ZBook'
        if self.hostname == self.hostname_home:
            self.n_free_cores = 2
        else:
            self.n_free_cores = 0

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
        self.int_types_aux = ['undef', 'holdtrem', 'movetrem']
        self.subj_strs_all = [ 'S{:02d}'.format(i) for i in range(1,11) ] + ['S98', 'S99']

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
                          'merge_within_subj':int_types_all     }

        # I prefer to have globally non-intersecting class ids
        basic_shift = len(int_types_all) * 2 + 10  # just in case
        self.int_types_aux_cid_shift = { 'subj':basic_shift}
        self.int_types_aux_cid_shift['subj_medcond'] = \
            self.int_types_aux_cid_shift['subj'] + len(self.subj_strs_all) * 2
        self.int_types_aux_cid_shift['subj_medcond_task'] = \
            self.int_types_aux_cid_shift['subj_medcond'] + len(self.subj_medcond_strs_all) * 2

        self.int_type_datset_rel = list(sorted(self.int_types_aux_cid_shift.keys() ))

        # extended list of interval types
        self.int_types_ext = self.int_types_basic + self.int_types_aux
        # what is on '0' index is important
        self.int_types_to_include = {'basic': self.int_types_basic,
                                'basic+ext': self.int_types_ext,
                                'trem_vs_quiet':['trem','notrem'],
                                'trem_vs_quiet&undef':['trem','notrem','undef'],
                                'hold_vs_quiet':['hold','notrem'],
                                'move_vs_quiet':['move','notrem'],
                                 'subj_medcond_task':self.subj_medcond_task_strs_all,
                                 'subj_medcond':self.subj_medcond_strs_all,
                                 'subj': self.subj_strs_all  }

        # in trem_vs_quiet we dont' want merge_movements because it will do
        # nothing and just eat computation time
        self.group_vs_int_type_allowed = {'basic':['merge_movements', 'merge_all_not_trem', 'merge_nothing'],
                                'trem_vs_quiet':['merge_all_not_trem', 'merge_nothing'],
                                'trem_vs_quiet&undef':['merge_movements', 'merge_all_not_trem', 'merge_nothing'],
                                'hold_vs_quiet': ['merge_nothing'],
                                'move_vs_quiet': ['merge_nothing'],
                                          'subj_medcond_task': ['merge_within_task'],
                                          'subj_medcond': ['merge_within_medcond'],
                                          'subj': ['merge_within_subj'],
                                          }

        self.areas_list_aal_my_guess = ["Precentral", "Rolandic_Oper",
                                        "Supp_Motor_Area", "Postcentral",
                                        "Parietal_Sup", "Parietal_Inf",
                                        "Precuneus", "Paracentral_Lobule",
                                        "Cerebellum" ];

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


        with open('subj_info.json') as info_json:
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


global gp
gp = globparams()
