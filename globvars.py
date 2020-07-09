import os

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
