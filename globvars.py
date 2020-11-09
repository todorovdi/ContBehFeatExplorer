import os
import json

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

dir_fig_preproc = os.path.join(dir_fig,'preproc')


fbands = {'tremor': [3,10], 'low_beta':[11,22], 'high_beta':[22,30],
           'low_gamma':[30,60], 'high_gamma':[60,90],
          'HFO1':[91,200], 'HFO2':[200,300], 'HFO3':[300,400],
          'beta':[15,30],   'gamma':[30,100], 'HFO':[91,400]}

EMG_per_hand = {'right':['EMG061_old', 'EMG062_old'], 'left':['EMG063_old', 'EMG064_old' ] }

class globparams:
    def __init__(self):
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

        # we don't want to use undef becuse it is not clear how it should be
        # classified even on ground truth level
        self.groupings = {'merge_movements':['hold','move'],
                    'merge_all_not_trem':['notrem','hold','move'],
                    'merge_nothing':[]}
        self.int_types_ext = self.int_types_basic + self.int_types_aux
        # what is on '0' index is important
        self.int_types_to_include = {'basic': self.int_types_basic,
                                'basic+aux': self.int_types_ext,
                                'trem_vs_quiet':['trem','notrem'],
                                'trem_vs_quiet&undef':['trem','notrem','undef'],
                                'hold_vs_quiet':['hold','notrem'],
                                'move_vs_quiet':['move','notrem'] }

        # in trem_vs_quiet we dont' want merge_movements because it will do
        # nothing and just eat computation time
        self.group_vs_int_type_allowed = {'basic':['merge_movements', 'merge_all_not_trem', 'merge_nothing'],
                                'trem_vs_quiet':['merge_all_not_trem', 'merge_nothing'],
                                'trem_vs_quiet&undef':['merge_movements', 'merge_all_not_trem', 'merge_nothing'],
                                'hold_vs_quiet': ['merge_nothing'],
                                'move_vs_quiet': ['merge_nothing'] }

        self.areas_list_aal_my_guess = [ "Precentral", "Supp", "Cerebellum", "Parietal", "Supramarginal",
           "Paracentral", "Temporal_lob", "Temporal_med",
           "Rolandic", "Postcentral", "Precuneus"];


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
