import globvars as gv
from os.path import join as pjoin
import csv

##############################################

# from Jan
Initials =  ['ps',  'tm','hhw','an',  'hg','mg', 'lk2', 'mf','ft', 'hb'];
subj_codes = ['PGY6BWAXHZ','DRFRMVMB13','GY24G81RK7','XT73PAA4R0','7GDX14CLEN','9QOETOVN7U','NDW513CVOD','XEAR6DG6KY','K6WNF99ZBZ','LGLN5B06N0'];

# my subject numbers
chosen_subjects_paper = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10']
subj_codes2 = ['PGY6BWAXHZ','DRFRMVMB13','GY24G81RK7','XT73PAA4R0','7GDX14CLEN','9QOETOVN7U','NDW513CVOD','XEAR6DG6KY','K6WNF99ZBZ','LGLN5B06N0']

# internal UKD subject numbers
i2ic = {'hb':'sub037',
'ft' : 'sub022',
'ps' : 'sub011',
'tm' : 'sub012',
'hhw' : 'sub13',
'an' : 'sub015',
'hg' : 'sub018',
'mg' : 'sub020',
'lk2' : None,
'mf' : 'sub021'}
ksvs = list(i2ic.items())
ks,vs = zip(*ksvs)
ic2i = dict( zip(vs,ks) )


i2c = dict(zip(Initials,subj_codes) )
c2i = dict(zip(subj_codes,Initials) )

s2c = dict(zip(chosen_subjects_paper,subj_codes2) )
c2s = dict(zip(subj_codes2,chosen_subjects_paper) )

#set(subj_codes2) == set(subj_codes)

tpls = []
for s in chosen_subjects_paper:
    code = s2c[s]
    initial = c2i [ code ]
    internal = i2ic[initial]
    if internal is None:
        internal = ''
    print(f'{initial:4} = {internal:7} = {s :3} = {code:10} ')
    tpls += [(initial,internal,s,code)]


##############################################

fld = 'patients_info_xls'
fname_trem_checked = 'pd_patients_tremor_checked.csv'
fname_subj_corresp = 'fuer_alex.csv'

fname_full = pjoin(gv.code_dir, fld,fname_subj_corresp)
entries_full = {}
with open(fname_full, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    # pre, peri, post
    for row in csvreader:
        if len(row) == 5:
            oldname,newname = row[0], row[-1]
            # dates
            pre, peri, post = row[1:4]
            if not( len(oldname) and len(newname)):
                continue
            print(oldname,newname, '   ', ', '.join(row))
            if oldname != 'Subject ID':
                entries_full[oldname] = {'new_ID':newname, 'pre':pre, 'peri':peri, 'post':post}


fname_full = pjoin(gv.code_dir, fld,fname_trem_checked)
entries_trem_info = {}
ind = 0
legend_row = None
with open(fname_full, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    # pre, peri, post
    for row in csvreader:
        print(ind,row)
        if ind == 0:
            legend_row = row
        else:
            entries_trem_info[row[0]] = {}
            for ci,le in enumerate(legend_row[1:] ):
                entries_trem_info[row[0]][le] = row[ci+1]

        ind += 1
