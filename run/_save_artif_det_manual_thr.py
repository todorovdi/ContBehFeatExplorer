import json
from os.path import join as pjoin
import sys, os
sys.path.append( os.path.expandvars('$OSCBAGDIS_DATAPROC_CODE') )
from globvars import dir_fig, data_dir, code_dir, code_ver

ct = {}
d = {}
d['threshold_muscle'] = 10
d['flt_thr_mult'] = 2
d['thr_mult'] = 2.5
d['ICA_thr_mult'] = 2.2
ct['S01_off_hold'] = d

d = {}
d['ICA_thr_mult'] = 2.3
d['thr_mult'] = 2.5
d['threshold_muscle'] = 9
d['flt_thr_mult'] = 2
ct['S02_off_hold'] = d

d = {}
d['ICA_thr_mult'] = 2.5
d['thr_mult'] = 3
d['threshold_muscle'] = 7
d['flt_thr_mult'] = 2
ct['S03_off_hold'] = d

d = {}
d['ICA_thr_mult'] = 2.3
d['thr_mult'] = 3
d['threshold_muscle'] = 8
d['flt_thr_mult'] = 2.25
ct['S04_off_hold'] = d

d = {}
d['ICA_thr_mult'] = 2.3
d['thr_mult'] = 3
d['threshold_muscle'] = 8
d['flt_thr_mult'] = 2.25
ct['S05_off_hold'] = d

### ON

d = {}
d['ICA_thr_mult'] = 2.5
d['thr_mult'] = 2.5
d['threshold_muscle'] = 5
d['flt_thr_mult'] = 2
ct['S01_on_hold'] = d

d = {}
d['ICA_thr_mult'] = 2.5
d['thr_mult'] = 2.5
d['threshold_muscle'] = 8
d['flt_thr_mult'] = 1.5
ct['S02_on_hold'] = d

d = {}
d['ICA_thr_mult'] = 2.15
d['thr_mult'] = 2.4
d['threshold_muscle'] = 8.5
d['flt_thr_mult'] = 1.85
ct['S04_on_hold'] = d

d = {}
d['ICA_thr_mult'] = 2.3
d['thr_mult'] = 2.9
d['threshold_muscle'] = 19
d['flt_thr_mult'] = 2.
#Idk has a lot of muscle apparently
ct['S05_on_hold'] = d

d = {}
d['ICA_thr_mult'] = 2.3
d['thr_mult'] = 2.6
d['threshold_muscle'] = 10
d['flt_thr_mult'] = 1.9
ct['S07_on_hold'] = d

#ON MOVE

d=   { 'thr_mult':3,
       'ICA_thr_mult':2.3 ,
       'threshold_muscle':8,
       'flt_thr_mult':2 }
ct['S01_on_move'] = d

d=   {'ICA_thr_mult':2.3,
      'thr_mult':2.5,
      'threshold_muscle':9,
      'flt_thr_mult':2.25}
ct['S02_on_move'] = d

d=   {'ICA_thr_mult':2.3,
      'thr_mult':2.5,
      'threshold_muscle':7,
      'flt_thr_mult':1.6}
ct['S04_on_move'] = d

d=   {'ICA_thr_mult':2.3,
      'thr_mult':2.8,
      'threshold_muscle':8,
      'flt_thr_mult':3}
# a lot of muscle activ
ct['S05_on_move'] = d

d=   {'ICA_thr_mult':2.3,
      'thr_mult':2.5,
      'threshold_muscle':10,
      'flt_thr_mult':1.7}
ct['S07_on_move'] = d

#OFF move

d=   {'ICA_thr_mult':2.2,
      'thr_mult':2.5,
      'threshold_muscle':9,
      'flt_thr_mult':1.85}
ct['S01_off_move'] = d

d=   {'ICA_thr_mult':2.3,
      'thr_mult':2.75,
      'threshold_muscle':12,
      'flt_thr_mult':1.9}
ct['S02_off_move'] = d

d=   {'ICA_thr_mult':2.,
      'thr_mult':3.2,
      'threshold_muscle':10,
      'flt_thr_mult':2.}
ct['S03_off_move'] = d
#
#to set
#
d=   {'ICA_thr_mult':1.9,
      'thr_mult':2.5,
      'threshold_muscle':8.5,
      'flt_thr_mult':1.85}
ct['S04_off_move'] = d
#
d=   {'ICA_thr_mult':2.15,
      'thr_mult':2.6,
      'threshold_muscle':8.6,
      'flt_thr_mult':2.25}
ct['S05_off_move'] = d

d=   {'ICA_thr_mult':2.3,
      'thr_mult':2.5,
      'threshold_muscle':9,
      'flt_thr_mult':2.25}
ct['S07_off_move'] = d

# TODO maybe save date and git hash

#fnf_codever = pjoin(code_dir,'last_code_ver_synced_with_HPC.txt')
#with open(fnf_codever, 'r') as f:
#    codever = f.read()
#codever = codever[:-1]

import datetime
envelope = { 'artif_detection_params' : ct }
envelope['date'] = datetime.datetime.today().strftime("%d.%m.%Y %H:%M:%S")
envelope['codever'] = code_ver

import json
fnf = pjoin(data_dir,'artif_detection_params.json')
with open( fnf,'w') as f:
    f.write( json.dumps(envelope) )
print(f'Saved to {fnf}')
