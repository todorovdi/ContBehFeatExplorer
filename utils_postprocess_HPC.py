import utils
import utils_tSNE as utsne
import os
from datetime import datetime
import globvars as gv
import numpy as np
from os.path import join as pjoin
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from utils_postprocess import multiLevelDict2TupleList
from pathlib import Path
from collections import OrderedDict as odict

STS2ststr = {'%':'contra', '^':'ipsi', 'B':'bilat'}

# compared to 2 it adds warid to tuple keys
def collectPerformanceInfo3(rawnames, prefixes, interval_groupings= None, interval_sets = None, ndays_before = None,
                           n_feats_PCA=None,dim_PCA=None, nraws_used=None,
                           sources_type = None, printFilenames = False,
                           group_fn = 10, group_ind=0, subdir = '', old_file_format=False,
                           use_main_LFP_chan=False,
                           remove_large_items=1, list_only = False,
                           allow_multi_fn_same_prefix = False, use_light_files=True,
                            rawname_regex_full =False,
                           start_time=None,end_time=None,
                           lighter_light = False, load=True, verbose=1, ret_df = False, use_tmpdir_to_load = False):
    '''
    rawnames can actually be just subject ids (S01  etc)
    red means smallest possible feat set as found by XGB

    label tuples is a list of tuples ( <newname>,<grouping name>,<int group name> )
    rawname_regex_full means very general regex for rawnames (meaningful when one combines many rawnames (or its parts) into one
    '''

    set_explicit_nraws_used_PCA = nraws_used is not None and isinstance(nraws_used,(int,str))
    set_explicit_n_feats_PCA = n_feats_PCA is not None
    set_explicit_dim_PCA = dim_PCA is not None

    #assert perf_to_use in [ 'perf', 'perfs_XGB']


    if set_explicit_nraws_used_PCA:
        regex_nrPCA = str(nraws_used)
    else:
        regex_nrPCA = '[0-9]+'
    if set_explicit_n_feats_PCA:
        regex_nfeats = str(n_feats_PCA)
    else:
        regex_nfeats = '[0-9]+'
    if set_explicit_dim_PCA:
        regex_pcadim = str(dim_PCA)
    else:
        regex_pcadim = '[0-9]+'

    from globvars import gp
    Ximputed_per_raw = {}
    good_bininds_per_raw = {}

    n_old = 0

    res = {}
    res_strongfeat = {}
    res_red = {}
    time_earliest, time_latest =  np.inf, 0
    output_per_raw = {}
    #TODO remove iteration alltogether
    #if rawname_regex_full:
    #    rawnames__ = ['xx']
    #else:
    #    rawnames__ = rawnames
    #subres = {}
    #subres_red = {}
    #subres_strongfeat = {}
    if rawname_regex_full:
        rawname_regexs = [ '([a-zS,_0-9]+)' ]
    else:
        #raise ValueError('in the version of the function it is not allowed')
        #rnstr= ','.join(rawnames)
        #letters = ''.join(list(set(list(rnstr))))
        prelet = 'S0123456789_,onoffholdmoverest'
        letters = ''.join(sorted(list(set(list(prelet))) ))
        rawname_regexs = [ r'([' + letters + r']+)']

    #S99_on_move_parcel_aal_grp10-0_test_PCA_nr4_7chs_nfeats1128_pcadim29_skip32_wsz256.npz

    Ximputed_per_prefix = {}
    good_bininds_per_prefix = {}

    ################
    #for prefix_expr in prefixes:
    ################
    # TODO: collect all prefixes somehow. Maybe just init with 100 and
    # remember which ones I have found..
    # or do completely differetn cycle
    # or better first get all by prefixed remove old and then of those
    # extract all found prefixes, select oldest within prefix and
    # if I want specific prefixes, only keep them

    prefix_implicit = True
    #prefix_expr = '(\w+)_'
    #prefix_expr = '([a-zA-Z0-9_,:]+)_'
    prefix_expr = r'([a-zA-Z0-9_,:\-@]+)_'
    #regex = ('_{}_{}grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_' +\
    #    'pcadim({}).*wsz[0-9]+__\(.+,.+\)\.npz').\
    #    format(rawgroup_idstr[:3], sources_type_, group_fn, group_ind, prefix,
    #    regex_nrPCA, regex_nfeats, regex_pcadim)


    #def genMLresFn(rawnames, sources_type, src_file_grouping_ind, src_grouping,
    #            prefix, n_channels, nfeats_used,
    #                pcadim, skip, windowsz,use_main_LFP_chan,
    #               grouping_key,int_types_key, nr=None, regex_mode=False,
    #               smart_rawn_grouping = False, rawname_format= 'subj',
    #               custom_rawname_str=None):

    # we look for files for given rawname
    regex =  utils.genMLresFn(rawnames=rawname_regexs,
            sources_type=sources_type, src_file_grouping_ind=group_fn,
            src_grouping=group_ind, prefix=prefix_expr, n_channels='[0-9]+',
            nfeats_used = regex_nfeats, pcadim='[0-9]+',
            skip='[0-9]+', windowsz='[0-9]+', use_main_LFP_chan=use_main_LFP_chan,
                                grouping_key=r'([a-zA-Z0-9_&]+)',
                              int_types_key=r'([a-zA-Z0-9_&]+)',
                                nr=nraws_used, regex_mode=1)

    if use_light_files:
        regex = '_!' + regex

    print('REGEX = ',regex)

    # just take latest
    dir_to_use = os.path.join(gv.data_dir, subdir)
    # here only regex param is improtant
    #fnfound, match_infos = utsne.findByPrefix(dir_to_use, rawgroup_idstr, prefix_expr, regex=regex, ret_aux=1)
    fnfound, match_infos = utsne.findByPrefix(dir_to_use,
        None, None, regex=regex, ret_aux=1)
    if len(fnfound) == 0:
        print('Nothing found :(((( for REGEX')
        return None
    else:
        print(f'In {dir_to_use} found {len(fnfound)} files matching regex')

    strs = []
    inds = []
    rows = []
    # dict of (prefix,int_grouping,intset) -> filename,mod_time
    fn_per_fntype = {}
    n_old = 0
    for fni,fnf in enumerate(fnfound) :
        fname_full = os.path.join(dir_to_use,fnf)

        mod_time = os.stat( fname_full ).st_mtime
        dt = datetime.fromtimestamp(mod_time)

        date_ok = True
        if start_time is not None:
            date_ok &= dt >= start_time
        if end_time is not None:
            date_ok &= dt <= end_time

        if not date_ok:
            n_old += 1
            if printFilenames:
                print('skipping due to being old ',dt)
            continue
        else:
            if ndays_before is not None:
                nh = getFileAge(fname_full)
                nd = nh / 24
                if nd > ndays_before:
                    #print('  !!! too old ({} days before), skipping {}'.format(nd,fnf) )
                    n_old += 1
                    continue

        # if we had'nt skiiped due to age
        match_info = match_infos[fni]
        mg = match_info.groups()
        #import pdb;pdb.set_trace()
        rawstrid = mg[0]
        rawels = rawstrid.split(',')
        subjs = list( set( [rawel[:3] for rawel in rawels] ) )
        #subjs_analyzed, subjs_analyzed_glob = \
        #    getRawnameListStructure(rawels, ret_glob=True)

        intset = mg[-1]
        int_grouping = mg[-2]
        prefix = mg[-3]

        mod_time = os.stat( fname_full ).st_mtime
        if printFilenames:
            print( dt.strftime(" Time: %d %b %Y %H:%M" ), ': ', fnf )
        time_earliest = min(time_earliest, mod_time)
        time_latest   = max(time_latest, mod_time)

        #s = '{}:{}:{}'.format(prefix,int_grouping,intset)
        s = (rawstrid,prefix,int_grouping,intset)
        # filter not-selected prefixed
        if prefixes is not None and prefix not in prefixes:
            if printFilenames:
                print('skipping {} due to bad prefix'.format(fnf) )
            continue
        if rawnames is not None and rawstrid not in rawnames:
            if printFilenames:
                print('skipping {} due to bad rawname'.format(fnf) )
            continue
        if interval_groupings is not None and int_grouping not in interval_groupings:
            if printFilenames:
                print(f'Skipping {s} due to bad grouping')
            continue
        if interval_sets is not None and intset not in interval_sets:
            if printFilenames:
                print(f'Skipping {s} due to bad intset')
            continue

        # here we only take the latest
        if s in fn_per_fntype:
            cur_tpl = (fnf,mod_time)
            if allow_multi_fn_same_prefix:
                fn_per_fntype[s] += [ cur_tpl ]
            else:
                fnf_other,mod_time_other = fn_per_fntype[s][0]
                if mod_time > mod_time_other:  # if younger than keep it
                    fn_per_fntype[s] = [ cur_tpl ]
        else:
            fn_per_fntype[s] = [ ( fnf,mod_time ) ]


        row = {'filename_full':fnf, 'mod_time':mod_time }

    if printFilenames and verbose >= 3:
        print( 'fn_per_fntype keys are', fn_per_fntype.keys() )

    #################3
    for s in fn_per_fntype:
        tuples = fn_per_fntype[s]
        if verbose > 1:
            print(f'   {s}: {len( tuples ) } tuples')
    ####################

    if not list_only:
        rows = []
        for s in fn_per_fntype:
            if verbose > 0:
                print(f'!!!!!!!!!!   Start loading (={load}) files for {s}')
            tuples = fn_per_fntype[s]
            #print(f'   {s}: {len( tuples ) } tuples')
            for tpli,tpl in enumerate(tuples ):
                rawstrid,prefix,int_grouping,intset = s

                prefix_eff = prefix[:]

                if allow_multi_fn_same_prefix:
                    prefix_eff += f'#{tpli}'
                #if prefix_eff not in output_per_prefix:
                #    output_per_prefix[prefix_eff] = {}
                #
                #if int_grouping not in output_per_prefix[prefix_eff]:
                #    output_per_prefix[prefix_eff][int_grouping] = {}
                #
                #if intset in output_per_prefix[prefix_eff][int_grouping]:
                #    raise ValueError('Already there!!!')

                #fnf,mod_time = fn_per_fntype[s]
                fnf,mod_time = tpl
                fname_full = os.path.join(dir_to_use,fnf )


                res_cur = {}
                # if it we were asking for light files it will be light filename
                res_cur['_fname_full'] = fname_full
                res_cur['filename_full'] = fname_full
                res_cur['mod_time'] = mod_time
                res_cur['mod_time_datetime'] = datetime.fromtimestamp( mod_time ) 
                res_cur['loaded'] = False


                if load:
                    
                    loadSingleRes(res_cur,
                        use_light_files=use_light_files,
                        lighter_light=lighter_light,
                        remove_large_items=remove_large_items,
                        use_tmpdir = use_tmpdir_to_load)
                    ######################

                rows += [ res_cur.copy() ]

                if rawstrid not in output_per_raw:
                    output_per_raw[rawstrid] = {}
                if prefix_eff not in output_per_raw[rawstrid] :
                    output_per_raw[rawstrid][prefix_eff] = {}
                if int_grouping not in output_per_raw[rawstrid][prefix_eff] :
                    output_per_raw[rawstrid][prefix_eff][int_grouping] = {}
                if intset in output_per_raw[rawstrid][prefix_eff][int_grouping]:
                    raise ValueError('Already there!!!')
                #    output_per_raw[rawstrid][prefix_eff][int_grouping][intset] = {}

                output_per_raw[rawstrid][prefix_eff][int_grouping][intset] \
                    = res_cur
                output_per_raw[rawstrid][prefix_eff]['feature_names_filtered'] = \
                    res_cur.get('feature_names_filtered', None)

                #if fnf.find('S07') >= 0:
                #    import pdb;pdb.set_trace()

    else:
        output_per_prefix = None
        Ximputed_per_prefix = None
        good_bininds_per_prefix = None

    if not np.isinf(time_earliest):
        print('Earliest file {}, latest file {}'.format(
            datetime.fromtimestamp(time_earliest).strftime("%d %b %Y %H:%M:%S" ),
            datetime.fromtimestamp( time_latest).strftime("%d %b %Y %H:%M:%S" ) ) )
    else:
        print('Found nothing 2 :( ')

    print(f'In total found {n_old} old files')

    #feat_counts = {'full':res, 'red':res_red}
    #if output_per_raw_ is not None:
    #    output_per_raw = output_per_raw_
    #return feat_counts, output_per_raw

    if ret_df:
        df = pd.DataFrame(rows)
        ret = df,output_per_raw,Ximputed_per_raw, good_bininds_per_raw 
    else:
        ret = output_per_raw,Ximputed_per_raw, good_bininds_per_raw 
    return ret

def loadSingleRes(res_cur, use_light_files=True,
                  lighter_light=False, remove_large_items=1,
                  in_place = True, use_tmpdir = False):
    # lighter light will involve re-generation of light file (maybe based on something that was already light)
    fname_full = res_cur['filename_full']
    if use_tmpdir:
        fname_full = fname_full.replace(gv.data_dir, gv.data_dir_tmp)
    t0 = time()
    from zipfile import BadZipFile
    from utils_postprocess import removeLargeItems
    try:
        f = np.load(fname_full, allow_pickle=1)

        from pathlib import Path
        if use_light_files:
            assert Path(fname_full).name.startswith('_!'), fname_full
            res = f['results_light'][()]
        else:
            res = f['results_cur'][()]

        if in_place:
            res_cur.update(res  )
        else:
            res_cur = res


        if 'featsel_shap_res' in f and 'featsel_shap_res' not in res_cur:
            print('Moving Shapley values')
            res_cur['featsel_shap_res'] = f['featsel_shap_res'][()]

        if (not use_light_files) or lighter_light:
            if 'class_labels_good' in f:
                res_cur['class_labels_good'] = f['class_labels_good']
            else:
                print('class_labels_good is not in the archive!')
            res_cur = removeLargeItems(res_cur)

            if remove_large_items:
                pfsz = Path(fname_full)
                dir_to_use = pfsz.parent
                fnf = pfsz.name
                newfn = fnf
                if not use_light_files:
                    newfn = '_!' + fnf
                fname_light = pjoin( dir_to_use, newfn)
                print('resaving LIGHT file ',fname_light)
                np.savez(fname_light, results_light=res_cur)
        del f
        t1 = time()
        import gc; gc.collect()
        tnow=time()
        print(f'------- Loading and processing {fname_full} took {tnow-t0:.2f}s, of it gc={tnow-t1:.2f}')

        res_cur['loaded'] = True

    except BadZipFile as e:
        print(f'!!!! BadZipFile Error reading file {fname_full}')
        print(str(e) )
        res_cur['ERROR'] = ('BadZipFile', str(e) )
        #continue

    return res_cur

def checkTupleListTableCompleteness(arg, grouping_to_check=None, it_to_check=None , prefixes_ignore=[]):
    if isinstance(arg,(list,np.ndarray) ):
        tpll_reshaped = arg
    elif isinstance(arg,dict):
        tpll = multiLevelDict2TupleList(arg,4,3)
        tpll_reshaped = list( zip(*tpll) ) # it is a tuple of lists
    rawnames_found = set(tpll_reshaped[0])
    prefixes_found = set(tpll_reshaped[1])
    grps_found = set(tpll_reshaped[2])
    its_found = set(tpll_reshaped[3])
    complen = len(rawnames_found) * len(prefixes_found) * len(grps_found) * len(its_found)
    ll = len(tpll_reshaped[0])
    print(f'Total len {ll}, complete table would be of len {complen}')

    prefixes_a  = np.array( tpll_reshaped[1] )
    rns_a       = np.array( tpll_reshaped[0] )
    groupings_a = np.array( tpll_reshaped[2] )
    its_a       = np.array( tpll_reshaped[3] )

    has_errors = [ ('ERROR' in res) for res in tpll_reshaped[-1] ]
    has_errors = np.array(has_errors)
    has_no_errors = ~has_errors

    if grouping_to_check is None:
        grouping_to_check = groupings_a [0 ]
    if it_to_check is None:
        it_to_check = its_a [0 ]
    prefixes_missing = {}
    for rn in sorted(rawnames_found):
        cond = (rns_a == rn) & (groupings_a == grouping_to_check) & (its_a == it_to_check)
        prefixes_cur = prefixes_a[cond & has_no_errors]
        prefixes_missing_cur = set(prefixes_found) - set(prefixes_cur) - set(prefixes_ignore)
        #print(len(prefixes_cur))
        prefixes_missing[rn] = prefixes_missing_cur
        nmiss = len(prefixes_missing_cur)
        s = f'{rn:8} npresent={len(prefixes_cur)}, nmissing={nmiss}'
        if nmiss:
            s = '* ' + s
        print(s)

    return complen == ll, prefixes_missing

def listComputedData(subdir,prefixes,start_time, end_time,
                                      use_main_LFP_chan=1,
                light_only=1 ):

    sources_type ='parcel_aal'
    ndaysBefore = None
    r = collectPerformanceInfo3(None,prefixes, nraws_used='[0-9]+',
            sources_type = sources_type,
            printFilenames=0,
            ndays_before=ndaysBefore,
            use_main_LFP_chan=use_main_LFP_chan,
            subdir=subdir, remove_large_items = 1,
            list_only=0, allow_multi_fn_same_prefix=0,
            use_light_files = light_only, rawname_regex_full=0,
            start_time=start_time, end_time=end_time,
            load=False, verbose=0)
    if r is None:
        return None
    output_per_raw_notload = r[0]

    rawnames_found, groupings_found, its_found, prefixes_found =\
        getOutputSetInfo( output_per_raw_notload )
    return rawnames_found, groupings_found, its_found, prefixes_found


def getOutputSetInfo( output_per_raw ):
    if len(output_per_raw) == 0:
        return [],[],[],[]
    if isinstance(output_per_raw ,dict):
        tpll_notload = multiLevelDict2TupleList(output_per_raw,4,3)
    elif isinstance(output_per_raw ,list) and len(output_per_raw[0]) == 5:
        tpll_notload = output_per_raw
    else:
        raise ValueError(f'Wrong type {type(output_per_raw)}, {len(output_per_raw[0])}')

    tpll_notload_reshaped = list(zip(*tpll_notload))

    rawnames_found = list(sorted(set(tpll_notload_reshaped[0] ) ));
    groupings_found = list(sorted(set(tpll_notload_reshaped[2] ) ));
    its_found = list(sorted(set(tpll_notload_reshaped[3] ) ));
    prefixes_found = list(sorted(set(tpll_notload_reshaped[1] ) ));

    return rawnames_found, groupings_found, its_found, prefixes_found

    # can be not loaded

def fillStatinfo(df_orig, grp, operation, operation_col):
    if operation == 'max':
        statinfo = grp.max()
        res = grp.idxmax(numeric_only = 1)
    elif operation == 'min':
        statinfo = grp.min()
        res = grp.idxmin(numeric_only = 1)

    #statinfo.reset_index(inplace=True)
    #statinfo.insert(0,'operation',operation)
    #idmax
    statinfo['operation'] = operation
    statinfo['operation_col'] = operation_col
    # statinfo[f'{bestname}_prefix'] = idmax.apply(lambda x: subdf3.iloc[x['bacc']]['prefix'] , 1)
    # statinfo[f'{bestname}_prefix_nice'] = idmax.apply(lambda x: subdf3.iloc[x['bacc']]['parcel_group_name_nice'] , 1)
    # statinfo[f'{bestname}_parcel_group_name'] = idmax.apply(lambda x: subdf3.iloc[x['bacc']]['parcel_group_name'] , 1)
    # statinfo[f'{bestname}_bacc'] = statinfo['bacc']
    col2col = {'name':'prefix','name_nice':'parcel_group_name_nice_RC',
               'parcel_group_names':'parcel_group_names'}

    def tmpf(x):
        if not np.isnan(x[operation_col]):
         r = df_orig.loc[ int(x[operation_col] ),col2]
        else:
            #display(statinfo)
            display(operation_col, operation,  x)
            raise ValueError('bad')
            r = None

        return r
    #display(res)
    #return None
    for col,col2 in col2col.items():
        statinfo[col] = res.apply(tmpf  , 1)
    return statinfo

# returns row with modLFP corresponding to current row (if possible)
def getCorrespModLFP(row, df, bad_pref_parts = ['_modLFP_'],
                     base_pref_start = 'onlyH_act_modLFP_subskip8',
                     base_pref_start_dict = None, use_opside = False):
    # we will search for keys of base_pref_start_dict in pref templ
    #prefix = row['prefix'][0]
    # other one or another is not None, not both
    assert (base_pref_start_dict is None) or (base_pref_start is None)
    prefix_templ = row['prefix_templ']
    prefix = row['prefix']
    if row['prefix'].find('_only') >= 0:
        mode_only = 1
    else:
        mode_only = 0

    if prefix_templ is None and not mode_only:
        return None

    prefix_LFP,prefix_templ_LFP = None,None
    if prefix_templ is not None:
        if use_opside:
            #raise ValueError('Not impl')
            return None
        # we return None if we ask about corresp LFP for modLFP itself
        # (we could return the original row but I decided not to in order
        # to have clear separation between modLFP and others)
        for bps in bad_pref_parts:
            if prefix_templ.find(bps) >= 0:
                return None

        # detect side code
        #allowed = ['B-B','%-%','^-^']
        allowed = ['B-B','%_exCB-%','^_exCB-^']
        side_code = None
        for ae in allowed:
            if prefix_templ.endswith(ae):
                sidelet = ae[0]  
                assert sidelet in ['B','%','^']
                side_code = ae 
        if side_code is None:
            print(f'Side code is none for prefix_templ={prefix_templ}')
            return None

        prefix_templ_LFP = None
        if base_pref_start is not None:
            prefix_templ_LFP = base_pref_start + side_code
        else:
            found = False
            # check if compatible with our base pref start dict
            for bps,bpsv in base_pref_start_dict.items():
                if prefix_templ.find(bps) >= 0:
                    prefix_templ_upd = prefix_templ.replace(bps,bpsv)
                    prefix_templ_LFP = prefix_templ_upd# + side_code
                    #print(side_code, prefix_templ_upd,prefix_templ_LFP)
                    found = True
                    #import pdb; pdb.set_trace()
                    break
            if not found:  # if the prefix is really different, difficult to decide wrt what compute improvemenet
                print( f'getCorrespModLFP: not found for prefix_templ = {prefix_templ}')
                return None
        cond = df[ 'prefix_templ' ] == prefix_templ_LFP
        print('sum cond = ',sum(cond) )
        #print( f'getCorrespModLFP: cond = df[ prefix_templ ] == {prefix_templ_LFP}')
    else:
        # this is the case for onlyH_act_LFPand_only*

        LFP_side_to_use = row['LFP_side_to_use']
        assert LFP_side_to_use is not None
        #sidelet = LFP_side_to_use#[0].upper()
        # TODO: maybe I can try to allow both-both
        copyLFPstr = 'copy_from_search_LFP'
        if LFP_side_to_use == 'both':
            addstr = ''
        else:
            addstr = '_exCB'
        # side code we'll use to search for modLFP calc res in our dataframe later
        if use_opside:
            if LFP_side_to_use == 'both':
                return None
            LFP_side_to_use = utils.getOppositeSideStr(LFP_side_to_use)
        side_code = '@' + f'{LFP_side_to_use}{addstr}-{copyLFPstr}'

        if base_pref_start is not None:
            prefix_LFP = base_pref_start + side_code
            print(f'getCorrespModLFP: setting due to base_pref_start = {base_pref_start}')
        found = False
        # check if we  base prefs in prefix
        for bps,bpsv in base_pref_start_dict.items():
            #print(prefix,bps)
            ind = prefix.find(bps)
            if ind >= 0:
                prefix_upd = prefix[:ind] + bpsv
                prefix_LFP = prefix_upd + side_code
                found = True
                print('getCorrespModLFP: found ',prefix,bps)
                break
        if not found:  # if the prefix is really different, difficult to decide wrt what compute improvemenet
            print( f'getCorrespModLFP: not found for prefix = {prefix}')
            return None
        cond = df[ 'prefix' ] == prefix_LFP
        print('sum cond (prefix = {}) = {}'.format(prefix_LFP,sum(cond) ) )
        #print( f'getCorrespModLFP: cond = df[ prefix ] == {prefix_LFP}')
        #todo get side

    cols = ['rawname', 'grouping', 'interval_set']

    for col in cols:
        cond &= df[ col ] == row[col]
    subdf = df[cond]

    #print(prefix, prefix_templ)
    assert len(subdf) == 1, (len(subdf), prefix,prefix_templ,prefix_LFP,prefix_templ_LFP)

    #print(prefix_templ,side_code,prefix_templ_LFP, subdf.iloc[0]['prefix_templ'])

    return subdf.iloc[0]
    #assert len(subdf) == 1
    #row = subdf.head(1)
    #df[ df[]]
    #subdf = df[df['barname_pre'] == barname_pre]

# for joint I make some columns have list values, it is not very convenient
def makeSubjParamsStr(df, inplace=True):
    if not inplace:
        df = df.copy()
    for coln in ['rawname','subject','medcond',
                 'move_hand_side_letter','move_hand_opside_letter']:
        lbd = lambda x: ','.join(x)
        df[coln] = df[coln].apply(lbd,1)
    return df

## add some more columns
def addParsColumns(df_all, output_per_raw):
    extract_pars = ['SLURM_job_id','runstring_ind','parcel_group_names', 'brain_side_to_use','LFP_side_to_use_final', 'subskip_fit']
    #%debug

    for p in extract_pars:
        def tmpf(row):
            moc = getMocFromRow(row,output_per_raw)
            if moc is None:
                return None
            if 'pars' not in moc:
                return None
            return moc['pars'].get(p,None)
        #lbd = lambda row:
        if p == 'LFP_side_to_use_final':
            df_all['LFP_side_to_use'] = df_all.apply(tmpf,1)
        else:
            df_all[p] = df_all.apply(tmpf,1)

    return df_all

def addRunCorrespCols(df, output_per_raw, inplace = True):
    if not inplace:
        df = df.copy()

    def lbd(row):
        moc = getMocFromRow(row,output_per_raw)
        if moc is None:
            print('did not find moc for row',row)
            return None
        corresp,all_info = loadRunCorresp(moc)
        prefix = row['prefix']
        ind,pgn,nice_name = corresp.get(prefix, (None,None,None) )
        return pgn, nice_name
    #df['parcel_group_name'] = df.apply(lbd,1)

#     def lbd(row):
#         moc = getMocFromRow(row,output_per_raw)
#         if moc is None:
#             return None
#         corresp,all_info = loadRunCorresp(moc)
#         prefix = row['prefix']
#         ind,pgn,nice_name = corresp.get(prefix, (None,None,None) )
#         return nice_name
    df[['parcel_group_name_RC','parcel_group_name_nice_RC']] = df.apply(lbd,1,result_type='expand')
    return df

# adds improvement and improv shuffled cols to the df produced by plotTableInfo
def addImprovColDfAll(df, inplace=True, score='bacc', score_shuffled = 'bacc_shuffled', base_pref_start_dict = None):
    if not inplace:
        df = df.copy()
    df['improv']                   = len(df) * [np.nan]
    df['improv_shuffled']          = len(df) * [np.nan]
    df['corresp_LFP_prefix_templ'] = len(df) * ['']
    df['corresp_LFP_prefix']       = len(df) * ['']
    df[f'corresp_LFP_{score}']         = len(df) * ['']
    for index,row in df.iterrows():
        for use_opside in [0, 1]:
            row_modLFP = getCorrespModLFP(row,df,
                base_pref_start_dict=base_pref_start_dict, 
                base_pref_start=None, use_opside = use_opside)
            if row_modLFP is None:
                print(f'None for {index}:  {row["prefix"]} ( {row["prefix_templ"]} ) use_opside = {use_opside}')
                continue

            if use_opside:
                cp = 'opside_'
            else:
                cp = ''
            df.at[index,cp + 'corresp_LFP_prefix_templ'] = row_modLFP['prefix_templ']
            df.at[index,cp + 'corresp_LFP_prefix']       = row_modLFP['prefix']

            area_and_LFP = float( row[score] )
            LFP = float( row_modLFP[score] )
            df.at[index,cp + 'improv'] = area_and_LFP - LFP

            df.at[index,cp + f'corresp_LFP_{score}'] = LFP
            #row['improv'] = area_and_LFP - LFP

            area_and_LFP = float( row[score_shuffled] )
            LFP = float( row_modLFP[score_shuffled] )
            df.at[index,cp + 'improv_shuffled'] = area_and_LFP - LFP
        #row['improv_shuffled'] = area_and_LFP - LFP
    return df

# old (wrt Jan 31, 2023), not used anymore
def addImprovCol(df, barnames_pre, inplace=True, bad_pref_starts = ['onlyH_act_modLFP_subskip8']):
    # this adds improv col to onlyBar output
    cnvd = {'clmove':'%%', 'ilmove':'^^'}
    if not inplace:
        df = df.copy()
    df['improv'] = len(df) * [np.nan]
    df['improv_shuffled'] = len(df) * [np.nan]
    for barname_pre in barnames_pre:
        exit  = 0
        for bps in bad_pref_starts:
            if barname_pre.startswith(bps):
                exit = 1
        if exit:
            continue

        subdf = df[df['barname_pre'] == barname_pre]
        side = barname_pre.split('_')[-1]
        #print(barname_pre, side)
        sidetempl = cnvd.get(side,None)

        for index, row in subdf.iterrows():
            #print(row['barname'])
            if sidetempl is None:
                if barname_pre.endswith('among_single-sided'):
                    area_name_sided = row['barname'].split(' ')[-2]
                    sidelet = area_name_sided[-1]
                    #print(area_name_sided,sidelet)
                    sidetempl_cur = sidelet2sideTempl(sidelet, row['subject']) # 1 symbol
                    sidetempl_cur += sidetempl_cur
                elif barname_pre.startswith('onlyH_act_subskip8'):
                    sidetempl_cur = barname_pre[-2:]
            else:
                sidetempl_cur = sidetempl

            cur = df [(df['rawname'] == row['rawname'] ) & (df['barname_pre'] == f'onlyH_act_modLFP_subskip8{sidetempl_cur}' )]
            area_and_LFP = float( row['p'] )
            LFP = float( cur['p'] )
            df['improv'][index] = area_and_LFP - LFP

            area_and_LFP = float( row['p_red'] )
            LFP = float( cur['p_red'] )
            df['improv_shuffled'][index] = area_and_LFP - LFP
    return df

def tupleList2multiLevelDict(tpll): #,min_depth=0,max_depth=99999, cur_depth = 0, prefix_sort = None):
    rawnames_found, groupings_found, its_found, prefixes_found = getOutputSetInfo( tpll )
    tpll_reshaped = list( zip(*tpll) ) # it is a tuple of lists

    rawnames =  np.array(tpll_reshaped[0] )
    prefixes =  np.array(tpll_reshaped[1] )
    groupings = np.array(tpll_reshaped[2] )
    its =       np.array(tpll_reshaped[3] )

    d = {}
    for rn in rawnames_found:
        d[rn] = {}
        for prefix in prefixes_found:
            mask1 = (rawnames == rn) & (prefixes == prefix)
            n = sum( mask1 )
            if n == 0:
                continue
            d[rn][prefix] = {}
            for grp in groupings_found:
                mask2 = mask1 & (groupings == grp)
                n = sum( mask2 )
                if n == 0:
                    continue
                d[rn][prefix][grp] = {}
                for it in its_found:
                    mask3 = mask2 & (its == it)
                    n = sum( mask3 )
                    if n == 0:
                        continue
                    else:
                        assert n == 1
                    ind = np.where(mask3)[0][0]
                    d[rn][prefix][grp][it] = tpll[ind][-1]

    return d

def checkPrefixCollectionConsistencty(subdir,prefixes,start_time, end_time,
                                      grouping_to_check, it_to_check,
                                      use_main_LFP_chan=1, light_only=1,
                                     prefixes_ignore  = [], preloaded = None, use_tmpdir_to_load = False):
    sources_type ='parcel_aal'
    ndaysBefore = None
    if preloaded is not None:
        output_per_raw_notload = preloaded
    else:
        r = collectPerformanceInfo3(None,prefixes, nraws_used='[0-9]+',
                sources_type = sources_type,
                printFilenames=0,
                ndays_before=ndaysBefore,
                use_main_LFP_chan=use_main_LFP_chan,
                subdir=subdir, remove_large_items = 1,
                list_only=0, allow_multi_fn_same_prefix=0,
                use_light_files = light_only, rawname_regex_full=0,
                start_time=start_time, end_time=end_time,
                load=False, verbose=0, use_tmpdir_to_load=use_tmpdir_to_load)

        if r is None:
            return None,None
        output_per_raw_notload = r[0]
    if isinstance(output_per_raw_notload ,dict):
        tpll_notload = multiLevelDict2TupleList(output_per_raw_notload,4,3)
    elif isinstance(output_per_raw_notload ,list) and len(output_per_raw_notload) == 4:
        tpll_notload = output_per_raw_notload
    else:
        raise ValueError('Wrong type')
    tpll_notload_reshaped = list(zip(*tpll_notload))

    rawnames_found, groupings_found, its_found, prefixes_found = getOutputSetInfo( output_per_raw_notload )
    for tmp in [rawnames_found, groupings_found, its_found, prefixes_found]:
        assert tmp is not None
    #rawnames_found = list(sorted(set(tpll_notload_reshaped[0] ) ));
    #groupings_found = list(sorted(set(tpll_notload_reshaped[2] ) ));
    #its_found = list(sorted(set(tpll_notload_reshaped[3] ) ));
    #prefixes_found = list(sorted(set(tpll_notload_reshaped[1] ) ));

    print(rawnames_found)
    print(groupings_found)
    print(its_found)
    nprefshow = 20
    if len(prefixes) < nprefshow:
        print(prefixes_found)
    else:
        print(f'found {len(prefixes_found)} prefixes, first are {prefixes_found[nprefshow]}')

    #if prefixes_ignore is None:
    #    prefixes_ignore = ['LFPrel_noself',
    #      'allb_beta_noH',
    #      'allb_gamma_noH',
    #      'allb_tremor_noH',
    #      'modSrc',
    #      'modSrc_self',
    #      'onlyH']

    rns_a       = np.array( tpll_notload_reshaped[0] )
    prefixes_a  = np.array( tpll_notload_reshaped[1] )
    groupings_a = np.array( tpll_notload_reshaped[2] )
    its_a       = np.array( tpll_notload_reshaped[3] )
    prefixes_missing = {}
    for rn in rawnames_found:
        cond = (rns_a == rn) & (groupings_a == grouping_to_check) & (its_a == it_to_check)
        prefixes_cur = prefixes_a[cond]
        assert prefixes_cur is not None
        prefixes_missing_cur = set(prefixes_found) - set(prefixes_cur) - set(prefixes_ignore)
        #print(len(prefixes_cur))
        prefixes_missing[rn] = prefixes_missing_cur
        print(f'{rn:8} npresent={len(prefixes_cur)}, nmissing={len(prefixes_missing_cur)}')

        #if rn == 'S02_on':
        #    #print(tpll_notload_reshaped,
        #    assert 'onlyH_act_only15' in  prefixes_cur


    return prefixes_missing, output_per_raw_notload

def loadCalcOutput(subdir, output_per_raw=None, save_collected=True, ignore_missing=False,
                  verbose=0 ):
    import globvars as gv
    import utils
    import utils_tSNE as utsne
    import utils_preproc as upre

    import os
    import sys
    import mne
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import multiprocessing as mpr
    import matplotlib as mpl
    import time
    import gc;
    import scipy.signal as sig
    import pandas as pd

    import numpy as np
    import utils_postprocess_HPC as postp
    import pymatreader
    import re

    from IPython.display import Audio
    sound_file = '../beep-06.mp3'

    data_dir = gv.data_dir
    from os.path import join as pjoin

    light_only = 1
    #light_only = 0
    ndaysBefore = None

    from dateutil import parser
    # start_time = parser.parse("6 Sept 2021 19:05:15")
    # end_time = parser.parse("8 Sept 2021 21:21:45")
    start_time = parser.parse("26 Sept 2015 01:00:15")
    #end_time = parser.parse("30 Oct 2021 21:21:45")
    end_time = parser.parse("30 Oct 2029 21:21:45")

    ndaysBefore = None
    #subdir = 'nointerp'
    #subdir = 'nofeatsel'
    subdir = 'per_subj_per_medcond_best_LFP_wholectx'
    lookup_dir = pjoin(gv.data_dir,subdir)
    recent = postp.listRecent(days=ndaysBefore, lookup_dir= lookup_dir,
                            start_time=start_time,
                                    end_time=end_time)
    print(len(recent))
    rawnames = []
    for lf in recent:
        st = 0
        if light_only:
            if not lf.startswith('_!'):
                continue
        rawname_regex = '([S0-9]+_[a-z]+)'
        if light_only:
            r = re.match('_\!_'+rawname_regex+'_.*',lf)
        else:
            r = re.match('_'+rawname_regex+'_.*',lf)
        if r is None:
            print('None ',lf)
            continue
        cr = r.groups()[0]
        rawnames += [ cr ]
    rawnames = list(sorted(set(rawnames)))


    rawname_regex = '([S0-9]+_[a-z]+)'
    #a0 = re.findall('_\!_'+rawname_regex+'_.*',lf)
    #a1 = re.match('_\!_'+rawname_regex+'_.*',lf)
    #print(a0,a1.groups())#

    import utils_postprocess_HPC as postp
    prefixes = postp.listRecentPrefixes(days = ndaysBefore, light_only=light_only,
                                        lookup_dir= lookup_dir,
                                        custom_rawname_regex = rawname_regex,
                                        start_time=start_time,
                                    end_time=end_time)

    print(rawnames)
    #display(prefixes)
    preloaded = None
    ###############################
    from utils_postprocess_HPC import checkPrefixCollectionConsistencty
    grouping_to_check= 'merge_nothing'
    it_to_check = 'basic'

    r = checkPrefixCollectionConsistencty(subdir,prefixes,start_time, end_time,
                                        grouping_to_check, it_to_check,
                                        use_main_LFP_chan=1, light_only=1,
                                        prefixes_ignore  = None, preloaded=preloaded)
    missing, preloaded = r
    print(missing)
    if max( [len(m) for m in missing.values()] ) != 0 and not ignore_missing:
        raise ValueError('something is missing')
    import gc; gc.collect()
    ##################################
    printFilenames = verbose > 2
    if output_per_raw is None:
        sources_type = 'parcel_aal'  # or ''
        r = postp.collectPerformanceInfo3(None,prefixes, nraws_used='[0-9]+',
                                                sources_type = sources_type,
                                                printFilenames=printFilenames,
                                                    ndays_before=ndaysBefore,
                                                    use_main_LFP_chan=1,
                                                    subdir=subdir, remove_large_items = 1,
                                        list_only=0, allow_multi_fn_same_prefix=0,
                                        use_light_files = light_only, rawname_regex_full=0,
                                        start_time=start_time,
                                        end_time=end_time)
        #output_per_raw,Ximp_per_raw,gis_per_raw = r
        output_per_raw,_,_ = r
        print('len(output_per_raw) =', len(output_per_raw))
        import gc; gc.collect()

        if save_collected:
            np.savez(pjoin(gv.data_dir,subdir,'gathered.npz'), output_per_raw=output_per_raw )
            import gc; gc.collect()

    Audio(filename=sound_file, autoplay=True)
    return output_per_raw



def plotCalcOutput(subdir, output_per_raw, to_show = [('trem_vs_all','merge_nothing','basic') ] ,
                   ignore_missing = False, rawnames=None, prefixes=None, pref_quick = 'onlyH_actBB' ):
    import globvars as gv
    sd = pjoin(gv.dir_fig, subdir)
    if not os.path.exists(sd):
        os.makedirs(sd)


    import utils_postprocess_HPC as postp
    tpll = multiLevelDict2TupleList(output_per_raw,4,3)

    z0 = [tpl[:-1] for tpl in tpll]
    #rns_ord, prefs_ord, grp_ord, it_ord = list (zip(*z0  ) )
    tpll_reshaped = np.array( list (zip(*z0  ) ) )
    #################################################
    from utils_postprocess_HPC import checkTupleListTableCompleteness
    b,missing = checkTupleListTableCompleteness(tpll_reshaped)

    if rawnames is None:
        rawnames = list(sorted(output_per_raw.keys()))
    if prefixes is None:
        prefixes = list(sorted(output_per_raw[rawnames[0]].keys() ) )
    outputs_filtered = postp.filterOutputs(output_per_raw, rns=rawnames ,
                        prefs=prefixes, grps = ['merge_nothing'] )
    b,missing =  checkTupleListTableCompleteness(outputs_filtered)
    print(missing)
    if max( [len(m) for m in missing.values()] ) != 0 and not ignore_missing:
        raise ValueError('something is missing')
    ################################################
    #tpll [0][:4]
    mult_clf_output = tpll[0][-1]
    all_thrs = mult_clf_output['feat_variance_q_thr'][-1:]
    #thr0,thr1,thr2='0.87','0.92','0.99'
    all_LDA =  []
    all_XGB = ['all_present_features'] #,'after_VF_threshold']
    for thr_cur in all_thrs:
        all_XGB += [ f'best_PCA-derived_features_{thr_cur}']

    perf_to_use_list = []
    for v in all_XGB[1:]:
        perf_to_use_list += [('XGB',all_XGB[0],v,'across_medcond')]
        #perf_to_use_list += [('XGB',all_XGB[0],v,'across_subj')]

    perf_to_use_list = [('XGB',all_XGB[0], f'best_PCA-derived_features_{all_thrs[-1]}','across_subj')]

    ###########################################  Table info
    print(f'perf_to_use_list={perf_to_use_list}')

    # to_show = [('allsep','merge_nothing','basic'), ('trem_vs_all','merge_all_not_trem','basic'),
    #         ('trem_vs_2class','merge_movements','basic')]
    #to_show = [('trem_vs_mvt','merge_movements','trem_vs_hold&move'),
    #           ('trem_vs_all','merge_all_not_trem','basic') ]


    #             ('trem_vs_2class','merge_movements','basic'),
    #           ('trem_vs_quiet','merge_nothing','trem_vs_quiet') ]
    #          ('allsep','merge_nothing','basic')]

    # warnings.simplefilter('error')
    # table_info_per_perf_type, table_per_perf_type = \
    #     postp.prepTableInfo2(output_per_raw, prefixes=prefixes,
    #     perf_to_use_list=perf_to_use_list)

    #%debug
    # import warnings
    # with warnings.catch_warnings():
    #warnings.simplefilter('error')
    table_info_per_perf_type, table_per_perf_type = \
        prepTableInfo3(output_per_raw, prefixes=prefixes,
        perf_to_use_list=perf_to_use_list, to_show=to_show)

    ##################################


    plotOnePrefQuick(rawnames,table_info_per_perf_type, perf_to_use_list[0],
                     pref = pref_quick)

    plt.tight_layout()
    plt.savefig( pjoin(gv.dir_fig, subdir,f'{pref_quick}_summary.pdf') )
    plt.close()

    ##########################################  Plot table
    for perf_tuple in perf_to_use_list:
        print(f'Satrting plotTableInfos2 for {perf_tuple}')
        postp.plotTableInfos2(table_info_per_perf_type, perf_tuple=perf_tuple,
                            output_subdir=subdir,use_recalc_perf=False,
                            prefixes_sorted=prefixes, crop_rawname=slice(None,None))
    plt.close()
    import gc;gc.collect()
    ####################################

    score = 'bacc'
    ##############################################

    plotname_pref = 'noLFP'
    good_prefs_permod = {'msrc': ['onlyH_act_exclude15'], 'LFP': ['onlyH_act_only15']}
    #good_prefs_permod =
    prefix2final_name = {good_prefs_permod['LFP'][0]:'LFP',
                         'onlyH_act_only_best': 'best area',
                         'onlyH_act_LFPand_quasibest': '*LFP + best area',
                         'onlyH_act_LFPand_best': 'LFP + best area',
                         good_prefs_permod['msrc'][0]:'cortex',
                         'onlyH_act':'LFP + cortex'  }
    #%debug

    #score = 'bacc'
    prefixes_final = list( prefix2final_name.keys() )
    #prefixes_final = ['modLFP','modSrc_self','onlyH_act']
    #for perf_tuple in [('XGB', 'all_present_features', 'interpret_EBM', 'across_subj')]:
    #for perf_tuple in [('XGB', 'all_present_features', 'interpret_EBM', 'across_subj')]:
    for perf_tuple in table_info_per_perf_type:
        print(f'Starting plotTableInfos_onlyBar for {perf_tuple}')
        addBestParcelGroups(output_per_raw, table_info_per_perf_type, perf_tuple, score )

        k = list( table_info_per_perf_type.keys() )[0]
        axs = plotTableInfos_onlyBar(table_info_per_perf_type,
                                           perf_tuple=perf_tuple,
                              output_subdir=subdir,use_recalc_perf=False,
                              prefixes_sorted=prefixes_final, prefix2final_name=prefix2final_name,
                                     crop_rawname='no',
                                           score= score,
                                           rawnames=rawnames, per_medcond =1,
                                          expand_best = 1,
                                           allow_missing_prefixes = 1)
        #axs[0,0].set_xlabel('')
        axs[2,1].set_visible(False)
        #frame1.axes.get_yaxis().set_visible(False)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=16)
        plt.rc('ytick', labelsize=16)

    defsp = 'special:min(sens,spec)'
    if score != defsp:
        scstr = score
    fn_full = pjoin(gv.dir_fig,subdir, f'bars_perf_dif_subsets_{plotname_pref}_{scstr}.pdf')
    plt.savefig(fn_full)
    plt.close()


    ##########################################################################
    ############################ Confmats ####################################
    ##########################################################################

    ############################revdict_user = {'trem_L':0, 'notrem_L':1, 'hold_L':2, 'move_L':3}
    #%debug
    from utils_postprocess_HPC import filterOutputs
    # 'onlyH%%'  #brain is contralat move
    # 'onlyH^^'  #brain is ipsilat move
    pref_confmat_plots = ['onlyHBB', 'onlyH_act%%' ,'onlyH_act^^' ,'onlyH%%', 'onlyH^^']
    #pref_confmat_plot = 'onlyHBB'
    #pref_confmat_plot = 'modLFP'
    #pref_confmat_plot = 'onlyH_act_only15' # CB
    #pref_confmat_plot = 'onlyH_act_only14' # CB
    #pref_confmat_plot = 'onlyH_act_only0'  #Senosorimotor
    #pref_confmat_plot = 'onlyH_act_LFPand_only14' # CB
    #pref_confmat_plot = 'onlyH_act_LFPand_only0'  #Senosorimotor

    for pref_confmat_plot in pref_confmat_plots:
        grps = ['merge_nothing'] #, 'merge_movements']
        #grps = [ 'merge_movements']
        outputs_filtered = filterOutputs(output_per_raw,prefs=[pref_confmat_plot],
                                            grps=grps)
        plt.rcParams.update({'font.size': 15})
        plt.rc('ytick',labelsize=22)
        plt.rc('xtick',labelsize=22)
        plt.rc('axes',labelsize=24)

        colorbar_axes_bbox = [0.80, 0.2, 0.025, 0.7]
        plotConfmats(outputs_filtered, ww = 5, hh =5, keep_beh_state_sides=0,
                        keep_subj_list_title=1,
                        labelpad_cbar=140, colorbar_axes_bbox= colorbar_axes_bbox,
                        rename_class_names = {'notrem':'quiet'})
        #plt.gcf().axes[-1].set_visible(False)
        s = pref_confmat_plot.replace('%%','_contramove')
        s = s.replace('^^','_ipsimove')
        figname = f'confmats_{s}_{grps}.pdf'
        figname_full = pjoin(gv.dir_fig,subdir,figname)
        plt.savefig(figname_full)####################
        plt.close()

    print('Plotting finised successfully!')


def genBestPGTempls():

    # templ_regex ,bestname, side_to_collect, side_used_in_fit, templ_grp) in enumerate(templs) :
    templs = []
    tmplg = 0
    templs += [ ('onlyH_act_only[0-9]+.*',       'onlyH_act_only_best_among_single-sided','both','single',tmplg)  ]
    templs += [ ('onlyH_act_LFPand_only[0-9]+.*','onlyH_act_LFPand_best_among_single-sided','both','single',tmplg) ]
    tmplg += 1
    templs += [ ('onlyH_act_only[0-9]+.*',       'onlyH_act_only_best_among_two-sided','both','both',tmplg)  ]
    templs += [ ('onlyH_act_LFPand_only[0-9]+.*','onlyH_act_LFPand_best_among_two-sided','both','both',tmplg) ]
    tmplg += 1
    templs += [ ('onlyH_act_only[0-9]+.*',       'onlyH_act_only_best_among_clmove','contralat_to_move','single',tmplg)  ]
    templs += [ ('onlyH_act_LFPand_only[0-9]+.*','onlyH_act_LFPand_best_among_clmove','contralat_to_move','single',tmplg) ]
    tmplg += 1
    templs += [ ('onlyH_act_only[0-9]+.*',       'onlyH_act_only_best_among_ilmove','ipsilat_to_move','single',tmplg)  ]
    templs += [ ('onlyH_act_LFPand_only[0-9]+.*','onlyH_act_LFPand_best_among_ilmove','ipsilat_to_move','single',tmplg) ]
    ######################
    tmplg += 1
    templs += [ ('onlyH_act_only[0-9]+.*',       'onlyH_act_only_worst_among_clmove','contralat_to_move','single',tmplg)  ]
    templs += [ ('onlyH_act_LFPand_only[0-9]+.*','onlyH_act_LFPand_worst_among_clmove','contralat_to_move','single',tmplg) ]
    tmplg += 1
    templs += [ ('onlyH_act_only[0-9]+.*',       'onlyH_act_only_worst_among_two-sided','both','both',tmplg)  ]
    templs += [ ('onlyH_act_LFPand_only[0-9]+.*','onlyH_act_LFPand_worst_among_two-sided','both','both',tmplg) ]

    return templs

def findBestParcelGroups(df, exCB = False, remove_bad = 1, verbose=0):
    from collections import Counter
    t2p = {}
    templs = genBestPGTempls()
    stats = []
    grps = []
    for templ_tpl in templs:
        templ,bestname,side_to_collect,side_used_in_fit,templ_grp = templ_tpl
        print('templ_tpl = ',templ,bestname, side_to_collect,side_used_in_fit, templ_grp)
        subdf = df[df['prefix'].str.match(templ)]
        #print('subdf len',len(subdf))
        if len(subdf) == 0:
            print('   skipping because prefix did not match templ = ',templ)
            continue

        assert df['grouping'].nunique() == 1
        assert df['interval_set'].nunique() == 1
        grouping = df.iloc[0]['grouping']
        iset = df.iloc[0]['interval_set']

        cond = subdf['parcel_group_names'].str.endswith('_L') | \
            subdf['parcel_group_names'].str.endswith('_R')
        if side_used_in_fit == 'single':
            subdf2 = subdf[cond]
        elif side_used_in_fit == 'both':
            subdf2 = subdf[~cond]
        #print('subdf2 len',len(subdf2))
        if len(subdf2) == 0:
            print('   skipping because did not find parcel_group_names endings consistent with side_used_in_fit = {}'.format(side_used_in_fit) )
            continue

        cond_exCB = (subdf2['brain_side_to_use'] == 'both') | \
            subdf2['brain_side_to_use'].str.endswith('exCB')
        cond2 = np.ones( len(subdf2) , dtype=bool)

        def getSideEff(side_to_collect, pgn, movesidelet, moveopsidelet):
            side_eff = None
            if side_to_collect == 'contralat_to_move':
                if pgn.startswith('Cerebellum'):
                    side_eff = movesidelet
                else:
                    side_eff = moveopsidelet
            elif side_to_collect == 'ipsilat_to_move':
                if pgn.startswith('Cerebellum'):
                    side_eff = moveopsidelet
                else:
                    side_eff = movesidelet
            return side_eff

        def CBsideOk(row):
            pgn = row['parcel_group_names']
            # if we have CB that starts is done on the same side with exCB, we should exclude it
            # because it was actully removed from the features so it is basically just LFP alone
            b = row['brain_side_to_use'].endswith('exCB')
            b = b and pgn.startswith('Cerebellum')
            b = b and pgn[-1] == row['brain_side_to_use'][0].upper()
            b = not b  # negate
            return b

        if side_to_collect == 'both':  # TODO: make it better here
            #subdf3 = subdf2
            if verbose:
                print('case1')
            cond2 = subdf2.apply(CBsideOk,1)
        else:
            if side_to_collect in ['left','right']:
                side_eff = side_to_collect
                sidelet = side_eff[0].upper()
                cond2 = subdf2['parcel_group_names'].str.endswith(f'_{sidelet}')
                if verbose:
                    print('case2, needs better implementation')
                    raise ValueError('not sure is fully implemented')
            else:
                if verbose:
                    print('case3')

                def lbd(row):
                    pgn = row['parcel_group_names']
                    movesidelet   = row['move_hand_side_letter']
                    moveopsidelet = row['move_hand_opside_letter']
                    side_eff = getSideEff(side_to_collect, pgn, movesidelet, moveopsidelet)
                    sidelet = side_eff[0].upper()
                    #b = True
                    # if we have CB that starts is done on the same side with exCB, we should exclude it
                    # because it was actully removed from the features so it is basically just LFP alone
                    b = CBsideOk(row)
                    return b and (sidelet == pgn[-1])

                cond2 = subdf2.apply(lbd,1)

        if exCB:
            cond2 &= cond_exCB
        subdf3 = subdf2[cond2]
        print('subdf3 len',len(subdf3))

        c = (subdf3['rawname'] == 'S01_off') &\
                (subdf3['parcel_group_names'].str.startswith('Cerebellum')  ) 
        lenCB = sum(c )
        # dirty hack needed because I have computed a bit too much stuff
        if lenCB > 1 and remove_bad:
            print(f'lenCB = {lenCB}  {subdf3.loc[c,["prefix","parcel_group_names","parcel_group_name_RC","bacc"]].values }')
            bad_prefixes = ['onlyH_act_only46', 'onlyH_act_only48'] + \
                ['onlyH_act_LFPand_only46', 'onlyH_act_LFPand_only48']

            subset_bad = set( subdf3['prefix'].unique() ) & set( bad_prefixes)
            pgns_bad = subdf3[subdf3['prefix' ].isin(bad_prefixes) ]['parcel_group_names'].unique()
            print(f'Removing bad prefixes {subset_bad} pgn = {pgns_bad}')
            subdf3 = subdf3[~subdf3['prefix'].isin(bad_prefixes)]

        subdf3= subdf3.reset_index()
        grpcols = ['grouping','interval_set','rawname']
        deb_cols = []
        deb_cols = ['prefix', 'parcel_group_names', 'brain_side_to_use','LFP_side_to_use']
        cols = grpcols + ['bacc','bacc_shuffled','improv','improv_shuffled','num'] + deb_cols
        grp = subdf3[cols].groupby(grpcols)

        grps += [grp]

        cts = list( set( grp.size() ) )
        print(cts)
        assert len(cts) == 1, cts
        assert cts[0] in [15,30],  ( cts[0], subdf3['prefix'].unique() )#, display(subdf3[cols]))
        #assert cts[0] == len( set(subdf3['prefix']) ), (cts, len( set(subdf3['prefix']) ))

        #g = grp.get_group(('merge_nothing', 'basic', 'S01_off'))
        g = grp.get_group((grouping, iset, 'S01_off'))
 

        prefixes_grp = g['prefix']
        pgn_grp      = g['parcel_group_names']
        t2p[templ_tpl] = prefixes_grp,pgn_grp

        bstu = g['brain_side_to_use']
        sbstu = set(bstu)
        n = bstu.nunique()
        #assert n == 1, sbstu
        lenCB = sum( pgn_grp.str.startswith('Cerebellum') )

        if verbose:
            print(cts, len(prefixes_grp), lenCB, Counter(bstu))
    #     if cts[0] > 31:
    #         break
        if 'best' in bestname:
            operation = 'max'
        elif 'worst' in bestname:
            operation = 'min'

        for operation_col in ['bacc', 'improv']:
            r = fillStatinfo(subdf3, grp, operation, operation_col).reset_index()
            r['bestname'] = bestname
            r['side_to_collect'] = side_to_collect
            r['side_used_in_fit'] = side_used_in_fit
            r['best_templ'] = templ

            stats += [r]

        print('')
    stats = pd.concat(stats).reset_index().drop(columns='index',axis=1)
    return stats

def addBestParcelGroups(output_per_raw, table_info_per_perf_type,
                        perf_tuple, score='special:min(sens,spec)',do_add=True,
                       remove_redundant_quasibest = True ):
    # modifies in place
    import re
    #import utils_postprocess_HPC as postp
    #score = defsp
    #score = 'spec'
    #score = 'bacc'
    perfkey = score

    from utils_postprocess_HPC import _extractPerfNumber

    templs = genBestPGTempls()


    templ_ind_sets = {}
    for ti,(templ,bestname,side_to_collect,side_used_in_fit,templ_grp) in enumerate(templs) :
        if templ_grp not in templ_ind_sets.keys():
            templ_ind_sets[templ_grp] = []
        templ_ind_sets[ templ_grp ] += [ti]
    #templ_ind_sets = [ [0,1], [2,3] ]

    prefs_per_pgn_pertk = {}
    if do_add:
        for k,d in table_info_per_perf_type[perf_tuple].items():
            prefs_per_pgn = {}
            prefs_per_pgn_pertk[k] = prefs_per_pgn
            # we may have in our collection results both when we fitted
            # Sensormitor from both sides AND separately Sensorimotor_L,
            # Sensorimotor_R
            for templ,bestname,side_to_collect,side_used_in_fit,templ_grp in templs:
                mn = 1e6
                m = 0 # max perf
                mkey,mkey_min = '',''
                mkey_nice,mkey_min_nice = '',''
                mpgn,mpgn_min = '',''
                i = 0
                for prefix in d:
                    try:
                        moc = output_per_raw[k[0]][prefix][k[1]][k[2]]
                    except KeyError as e:
                        continue

                    mr = re.match(templ,prefix)
                    if mr is None:
                        continue

                    corresp,all_info = loadRunCorresp(moc)
                    ind,pgn,nice_name = corresp.get(prefix, (None,None,None) )
                    if pgn is None or pgn == 'LFP':
                        continue
                    pgn_has_one_side = pgn.endswith('_L') or pgn.endswith('_R')
                    c1 = (side_used_in_fit == 'single' and pgn_has_one_side )
                    c2 = (side_used_in_fit == 'both' and (not pgn_has_one_side ) )
                    if not (c1 or c2):
                        continue

                    if pgn not in prefs_per_pgn:
                        prefs_per_pgn[pgn] = []
                    prefs_per_pgn[pgn] += [prefix]

                    rn = k[0]
                    subj = rn.split('_')[0]
                    mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
                    assert mainmoveside is not None
                    movesidelet = mainmoveside[0].upper()
                    moveopsidelet = utils.getOppositeSideStr( movesidelet )
                    #templ_eff = templ.    replace('%', moveopsidelet)
                    #templ_eff = templ_eff.replace('^', movesidelet)
                    #if side_to_collect == 'contralat_to_move' and \
                    #        (not (pgn_has_one_side) or not pgn.endswith(f'_{moveopsidelet}') ):
                    #    continue

                    if side_to_collect != 'both':
                        if side_to_collect in ['left','right']:
                            side_eff = side_to_collect
                        else:
                            if side_to_collect == 'contralat_to_move':
                                if pgn.startswith('Cerebellum'):
                                    side_eff = movesidelet
                                else:
                                    side_eff = moveopsidelet
                            if side_to_collect == 'ipsilat_to_move':
                                if pgn.startswith('Cerebellum'):
                                    side_eff = moveopsidelet
                                else:
                                    side_eff = movesidelet
                                #print(side_eff)

                        sidelet = side_eff[0].upper()
                        if pgn[-1] != sidelet:
                            continue


                    #if prefix == 'onlyH_act_only15':  #LFP
                    #    continue
                    #if prefix.find(templ  ) < 0:
                    #    continue

                    perf_cur = _extractPerfNumber(d[prefix],perfkey)
        #             if perfkey == defsp:
        #                 pc0 = d[prefix]['sens']
        #                 pc1 = d[prefix]['spec']
        #                 perf_cur = min(pc0,pc1)
        #             else:
        #                 perf_cur = d[prefix][perfkey]
                    if mn > perf_cur:
                        mn = perf_cur
                        mkey_min = prefix
                        mkey_min_nice = nice_name
                        mpgn_min = pgn
                    if m < perf_cur:
                        m = perf_cur
                        mkey = prefix

                        #corresp,all_info = loadRunCorresp(output_per_raw[k[0]][prefix][k[1]][k[2]])
                        #ind,pgn,nice_name = corresp[prefix]
                        #mkey_nice = nice_name
                        mkey_nice = nice_name
                        mpgn = pgn
                        #print(nice_name)
                    i += 1

                #print(i)
                if bestname.find('worst') >= 0:
                    d[bestname] = d[mkey_min].copy()
                    d[bestname]['name_best'] = mkey_min
                    d[bestname]['name_nice_best'] = mkey_min_nice
                    d[bestname]['parcel_group_name'] = mpgn_min
                elif bestname.find('best') >= 0:
                    d[bestname] = d[mkey].copy()
                    d[bestname]['name_best'] = mkey
                    d[bestname]['name_nice_best'] = mkey_nice
                    d[bestname]['parcel_group_name'] = mpgn
                else:
                    raise ValueError(f'Found nothing in {bestname}')

                print(k,bestname,mkey,'pgn=',mpgn,'nice=',mkey_nice, m)#,d[mkey][perfkey])
        #print(k,list(d['modLFP'].keys()))
    #print(list(table_info_per_perf_type[perf_tuple][('S07_off', 'merge_nothing', 'basic')].keys() ) )
    #%debug


    similarity_thr = 5e-2
    #  Handle case when LFP + best of single areas is NOT the same as best (LFP + single)
    for k,d in table_info_per_perf_type[perf_tuple].items():
        for tis in templ_ind_sets.values():
            name_nice_curset = []
            prefix_alt = ''
            templs_tpls_cur = [ templs[ti] for ti in tis ]
            #for ti in tis:
                #templ,bestname,side_to_collect,side_used_in_fit,templ_grp = templs[ti]
            for templ_tpl in templs_tpls_cur:
                templ,bestname,side_to_collect,side_used_in_fit,templ_grp = templ_tpl
                nn = d[bestname]['name_nice_best']
                name_nice_curset += [nn]
                if bestname.startswith('onlyH_act_only_best'):
                    prefix_best = bestname
                    prefix_alt = d[bestname]['name_best'].replace('_only','_LFPand_only')
                    if do_add:
                        prefs_cur = prefs_per_pgn_pertk[k][ d[bestname]['parcel_group_name'] ]
                        assert prefix_alt in prefs_cur, (prefix_alt, prefs_cur)
                elif bestname.startswith('onlyH_act_LFPand_best'):
                    prefix_LFP_best = bestname
                else:
                    print(f'not found best-like start in  {templ_tpl}')

            if not len(prefix_alt):
                print(f'skipping templates {templs_tpls_cur}')
                continue
            #print(name_nice_curset, prefix_alt)
            print(k,prefix_best, d[prefix_best]['parcel_group_name'])
            print('  ',prefix_LFP_best, d[prefix_LFP_best]['parcel_group_name'])

            corresp,all_info = loadRunCorresp(output_per_raw[k[0]][prefix_alt][k[1]][k[2]])
            ind,pgn,nice_name_alt = corresp[prefix_alt]

            c1 = prefix_LFP_best.find(prefix_best) < 0
            p1 = _extractPerfNumber(d[prefix_LFP_best], perfkey)
            p2 = _extractPerfNumber( d[prefix_alt], perfkey)
            pdiff = np.abs(p1  - p2)
            c2 = pdiff > similarity_thr

            name_nice1 = d[prefix_best]['name_nice_best']
            name_nice2 = d[prefix_LFP_best]['name_nice_best']

            prefix_quasibest = prefix_LFP_best.replace('onlyH_act_LFPand_best','onlyH_act_LFPand_quasibest')
            if (c1 and c2) or (not remove_redundant_quasibest):
                #print(pdiff)
                #assert nice_name.split()[-1].find( name_nice1.split()[-1] ) >= 0,  (nice_name_alt,  name_nice1)
                #assert nice_name_alt.find(name_nice1) >= 0, (nice_name_alt,  name_nice1)
                assert nice_name_alt.split()[2].find( name_nice1.split()[2] ) >= 0,  (nice_name_alt,  name_nice1)

                d[prefix_quasibest] = d[prefix_alt].copy()
                d[prefix_quasibest]['name_best'] = prefix_alt
                d[prefix_quasibest]['name_nice_best'] = nice_name_alt
                d[prefix_quasibest]['parcel_group_name'] = pgn
            else:
                if c1 and not c2:  # if areas are different but perf is similar
                    #d['onlyH_act_LFPand_best'] = kk
                    d[prefix_LFP_best] = d[prefix_alt].copy()
                    d[prefix_LFP_best]['name_best'] = prefix_alt
                    d[prefix_LFP_best]['name_nice_best'] = nice_name_alt
                    d[prefix_LFP_best]['parcel_group_name'] = pgn

                if prefix_quasibest in d and remove_redundant_quasibest:
                    del d[prefix_quasibest]

def plotOnePrefQuick(rawnames,table_info_per_perf_type, perf_to_use, pref = 'onlyH_actBB',
                     scores=['bacc','sens','spec'], title_type = 'subj'):
    subjs = list(sorted(set( [rn.split('_')[0] for rn in rawnames] ) ))
    #dict_keys(['sens_recalc', 'spec_recalc', 'descr', 'comment_from_runstrings', 'sens', 'spec', 'F1', 'acc', 'bacc', 'sens_red', 'spec_red', 'sens_red_recalc', 'spec_red_recalc', 'F1_red', 'num', 'num_red'])
    aa = table_info_per_perf_type[perf_to_use]
    tpls = sorted( list(aa.keys() ), key=lambda x: x[0] )

    #if not isinstance(prefs,list):
    #    prefs = [pref]

    nr = len(subjs)
    nc = len(scores)
    ww = 6; hh = 2
    fig,axs = plt.subplots(nr,nc,figsize=( nc*ww, nr*hh) ) #, sharex='col')
    axs = axs.reshape((nr,nc))
    for tpl in tpls:
        rn,grp,it = tpl
        subj,medcond = rn.split('_')
        medcond = medcond.upper()
        axi = subjs.index(subj)

        #mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
        #assert mainmoveside is not None
        #movesidelet = mainmoveside[0].upper()
        #moveopsidelet = utils.getOppositeSideStr( movesidelet )

        ##rpint(aa[tpl].keys() )
        ##print( aa[tpl]['onlyH_act'].keys() )
        #pref_eff = pref.replace('%', moveopsidelet)
        #pref_eff = pref_eff.replace('^', movesidelet)
        pref_eff = prefTempl2pref(pref, subj)

        for coli,score in enumerate(scores):
            val = aa[tpl][pref_eff][score]
            if score == 'bacc':
                print(pref, rn, score, f'{int(val * 100 )}%' )

            ax = axs[axi,coli]
            ax.barh([medcond ],[val * 100] )
            ax.set_xlim(0,100)
            if title_type == 'subj':
                ax.set_title(subj)
            elif title_type == 'rawname':
                ax.set_title(rn)

    # 'balanced acc'
            axs[-1,coli].set_xlabel(score)

def _old_collectPerformanceInfo2_(rawnames, prefixes, ndays_before = None,
                           n_feats_PCA=None,dim_PCA=None, nraws_used=None,
                           sources_type = None, printFilenames = False,
                           group_fn = 10, group_ind=0, subdir = '', old_file_format=False,
                           load_X=False, use_main_LFP_chan=False,
                           remove_large_items=1, list_only = False,
                           allow_multi_fn_same_prefix = False, use_light_files=True,
                            rawname_regex_full =False,
                           start_time=None,end_time=None ):
    '''
    rawnames can actually be just subject ids (S01  etc)
    red means smallest possible feat set as found by XGB

    label tuples is a list of tuples ( <newname>,<grouping name>,<int group name> )
    '''

    set_explicit_nraws_used_PCA = nraws_used is not None and isinstance(nraws_used,(int,str))
    set_explicit_n_feats_PCA = n_feats_PCA is not None
    set_explicit_dim_PCA = dim_PCA is not None

    #assert perf_to_use in [ 'perf', 'perfs_XGB']


    if set_explicit_nraws_used_PCA:
        regex_nrPCA = str(nraws_used)
    else:
        regex_nrPCA = '[0-9]+'
    if set_explicit_n_feats_PCA:
        regex_nfeats = str(n_feats_PCA)
    else:
        regex_nfeats = '[0-9]+'
    if set_explicit_dim_PCA:
        regex_pcadim = str(dim_PCA)
    else:
        regex_pcadim = '[0-9]+'

    from globvars import gp
    Ximputed_per_raw = {}
    good_bininds_per_raw = {}

    n_old = 0

    res = {}
    res_strongfeat = {}
    res_red = {}
    time_earliest, time_latest =  np.inf, 0
    output_per_raw = {}
    #TODO remove iteration alltogether
    if rawname_regex_full:
        rawnames__ = ['xx']
    else:
        rawnames__ = rawnames
    for rawgroup_idstr in rawnames__:
        #subres = {}
        #subres_red = {}
        #subres_strongfeat = {}
        if len(rawgroup_idstr) > 4:
            #subj,medcond,task  = utils.getParamsFromRawname(rawgroup_idstr)
            rawname_regexs = [f'({rawgroup_idstr})']
        else:
            #subj = rawgroup_idstr
            #assert len(subj) == 3
            #assert subj[0] == 'S'
            if rawname_regex_full:
                rawname_regexs = [ '([a-zS,_0-9]+)' ]
            else:
                rawname_regexs = [ f'({rawgroup_idstr})']

        #S99_on_move_parcel_aal_grp10-0_test_PCA_nr4_7chs_nfeats1128_pcadim29_skip32_wsz256.npz

        Ximputed_per_prefix = {}
        good_bininds_per_prefix = {}

        output_per_prefix = {}
        ################
        #for prefix_expr in prefixes:
        ################
        # TODO: collect all prefixes somehow. Maybe just init with 100 and
        # remember which ones I have found..
        # or do completely differetn cycle
        # or better first get all by prefixed remove old and then of those
        # extract all found prefixes, select oldest within prefix and
        # if I want specific prefixes, only keep them


        prefix_implicit = True
        #prefix_expr = '(\w+)_'
        prefix_expr = '([a-zA-Z0-9_,:]+)_'
        #regex = ('_{}_{}grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_' +\
        #    'pcadim({}).*wsz[0-9]+__\(.+,.+\)\.npz').\
        #    format(rawgroup_idstr[:3], sources_type_, group_fn, group_ind, prefix,
        #    regex_nrPCA, regex_nfeats, regex_pcadim)


#def genMLresFn(rawnames, sources_type, src_file_grouping_ind, src_grouping,
#            prefix, n_channels, nfeats_used,
#                pcadim, skip, windowsz,use_main_LFP_chan,
#               grouping_key,int_types_key, nr=None, regex_mode=False,
#               smart_rawn_grouping = False, rawname_format= 'subj',
#               custom_rawname_str=None):

        # we look for files for given rawname
        regex =  utils.genMLresFn(rawnames=rawname_regexs,
                sources_type=sources_type, src_file_grouping_ind=group_fn,
                src_grouping=group_ind, prefix=prefix_expr, n_channels='[0-9]+',
                nfeats_used = regex_nfeats, pcadim='[0-9]+',
                skip='[0-9]+', windowsz='[0-9]+', use_main_LFP_chan=use_main_LFP_chan,
                                grouping_key=r'([a-zA-Z0-9_&]+)',
                              int_types_key=r'([a-zA-Z0-9_&]+)',
                                    nr=nraws_used, regex_mode=1)

        if use_light_files:
            regex = '_!' + regex

        print('REGEX = ',regex)

        # just take latest
        dir_to_use = os.path.join(gv.data_dir, subdir)
        # here only regex param is improtant
        #fnfound, match_infos = utsne.findByPrefix(dir_to_use, rawgroup_idstr, prefix_expr, regex=regex, ret_aux=1)
        fnfound, match_infos = utsne.findByPrefix(dir_to_use,
            None, None, regex=regex, ret_aux=1)
        if len(fnfound) == 0:
            print('Nothing found :(((( for REGEX')
            continue
        else:
            print(f'In {dir_to_use} found {len(fnfound)} files matching regex')

        strs = []
        inds = []
        # dict of (prefix,int_grouping,intset) -> filename,mod_time
        fn_per_fntype = {}
        n_old = 0
        for fni in range(len(fnfound) ):
            fnf = fnfound[fni]
            fname_full = os.path.join(dir_to_use,fnf)

            mod_time = os.stat( fname_full ).st_mtime
            dt = datetime.fromtimestamp(mod_time)

            date_ok = True
            if start_time is not None:
                date_ok &= dt >= start_time
            if end_time is not None:
                date_ok &= dt <= end_time

            if not date_ok:
                n_old += 1
                continue
            else:
                if ndays_before is not None:
                    nh = getFileAge(fname_full)
                    nd = nh / 24
                    if nd > ndays_before:
                        #print('  !!! too old ({} days before), skipping {}'.format(nd,fnf) )
                        n_old += 1
                        continue
            # if we had'nt skiiped due to age
            match_info = match_infos[fni]
            mg = match_info.groups()
            #import pdb;pdb.set_trace()
            rawstrid = mg[0]
            rawels = rawstrid.split(',')
            subjs = list( set( [rawel[:3] for rawel in rawels] ) )
            #subjs_analyzed, subjs_analyzed_glob = \
            #    getRawnameListStructure(rawels, ret_glob=True)

            intset = mg[-1]
            int_grouping = mg[-2]
            prefix = mg[-3]

            mod_time = os.stat( fname_full ).st_mtime
            if printFilenames:
                print( dt.strftime(" Time: %d %b %Y %H:%M" ), ': ', fnf )
            time_earliest = min(time_earliest, mod_time)
            time_latest   = max(time_latest, mod_time)

            #s = '{}:{}:{}'.format(prefix,int_grouping,intset)
            s = (prefix,int_grouping,intset)
            # filter not-selected prefixed
            if prefixes is not None and prefix not in prefixes:
                print('skipping {} due to bad prefix'.format(fnf) )
                continue

            # here we only take the latest
            if s in fn_per_fntype:
                cur_tpl = (fnf,mod_time)
                if allow_multi_fn_same_prefix:
                    fn_per_fntype[s] += [ cur_tpl ]
                else:
                    fnf_other,mod_time_other = fn_per_fntype[s][0]
                    if mod_time > mod_time_other:  # if younger than keep it
                        fn_per_fntype[s] = [ cur_tpl ]
            else:
                fn_per_fntype[s] = [ ( fnf,mod_time ) ]

        if printFilenames:
            print( 'fn_per_fntype', fn_per_fntype.keys() )

        #################3
        for s in fn_per_fntype:
            tuples = fn_per_fntype[s]
            print(f'   {s}: {len( tuples ) } tuples')
        ####################

        if not list_only:
            print(f'!!!!!!!!!!   Start loading files for {rawgroup_idstr}')
            for s in fn_per_fntype:
                tuples = fn_per_fntype[s]
                #print(f'   {s}: {len( tuples ) } tuples')
                for tpli,tpl in enumerate(tuples ):
                    prefix,int_grouping,intset = s
                    prefix_eff = prefix[:]

                    if allow_multi_fn_same_prefix:
                        prefix_eff += f'#{tpli}'
                    if prefix_eff not in output_per_prefix:
                        output_per_prefix[prefix_eff] = {}

                    if int_grouping not in output_per_prefix[prefix_eff]:
                        output_per_prefix[prefix_eff][int_grouping] = {}

                    if intset in output_per_prefix[prefix_eff][int_grouping]:
                        raise ValueError('Already there!!!')

                    #fnf,mod_time = fn_per_fntype[s]
                    fnf,mod_time = tpl
                    fname_full = os.path.join(dir_to_use,fnf )

                    t0 = time()
                    f = np.load(fname_full, allow_pickle=1)

                    try:
                        if use_light_files:
                            res_cur = f['results_light'][()]
                        else:
                            res_cur = f['results_cur'][()]
                    except KeyError as e:
                        print(f'!!! Got error {e} for {fnf}, skiping')
                        break
                    res_cur['filename_full'] = fname_full

                    mod_time = os.stat( fname_full ).st_mtime
                    dt = datetime.fromtimestamp(mod_time)
                    res_cur['fname_mod_time'] = dt

                    if 'featsel_shap_res' in f and 'featsel_shap_res' not in res_cur:
                        print('Moving Shapley values')
                        res_cur['featsel_shap_res'] = f['featsel_shap_res'][()]

                    from utils_postprocess import removeLargeItems
                    if not use_light_files:
                        res_cur = removeLargeItems(res_cur)
                        output_per_prefix[prefix_eff]['feature_names_filtered'] = f['feature_names_filtered_pri'][()][0]
                    else:
                        output_per_prefix[prefix_eff]['feature_names_filtered'] = res_cur['feature_names_filtered']

                    output_per_prefix[prefix_eff][int_grouping][intset] = res_cur

                    ######################

                    if load_X:
                        Ximputed_per_prefix[prefix_eff] = f['X_imputed']
                        good_bininds_per_prefix[prefix_eff] =  f['bininds_good']

                    del f
                    t1 = time()
                    import gc; gc.collect()

                    tnow=time()
                    print(f'------- Loading and processing {fnf} took {tnow-t0:.2f}s, of it gc={tnow-t1:.2f}')

                    #if fnf.find('S07') >= 0:
                    #    import pdb;pdb.set_trace()
        else:
            output_per_prefix = None
            Ximputed_per_prefix = None
            good_bininds_per_prefix = None

        if load_X:
            Ximputed_per_raw[rawgroup_idstr]     =  Ximputed_per_prefix
            good_bininds_per_raw[rawgroup_idstr] =  good_bininds_per_prefix

        output_per_raw[rawgroup_idstr] = output_per_prefix

    if not np.isinf(time_earliest):
        print('Earliest file {}, latest file {}'.format(
            datetime.fromtimestamp(time_earliest).strftime("%d %b %Y %H:%M:%S" ),
            datetime.fromtimestamp( time_latest).strftime("%d %b %Y %H:%M:%S" ) ) )
    else:
        print('Found nothing :( ')

    print(f'In total found {n_old} old files')

    #feat_counts = {'full':res, 'red':res_red}
    #if output_per_raw_ is not None:
    #    output_per_raw = output_per_raw_
    #return feat_counts, output_per_raw
    ret = output_per_raw,Ximputed_per_raw, good_bininds_per_raw 

    return ret

def getFileAge(fname_full, ret_hours=True):
    #created = os.stat( fname_full ).st_ctime
    #dt = datetime.fromtimestamp(created)
    modified = os.stat( fname_full ).st_mtime
    dt = datetime.fromtimestamp(modified)
    today = datetime.today()
    tdelta = (today - dt)
    r = tdelta
    if ret_hours:
        nh = tdelta.total_seconds() / (60 * 60)
        r = nh
    return r

def listRecent(days = 5, hours = None, lookup_dir = None,
               start_time=None,end_time=None, verbose=1):
    if lookup_dir is None:
        lookup_dir = gv.data_dir
    lf = os.listdir(lookup_dir)
    final_list = []
    if verbose:
        print(f'INFO listRecent: Found {len(final_list)} in {lookup_dir}')
    for f in lf:
        date_ok = True
        modified = os.stat( pjoin(lookup_dir,f) ).st_mtime
        dt = datetime.fromtimestamp(modified)
        if start_time is not None:
            date_ok &= dt >= start_time
        if end_time is not None:
            date_ok &= dt <= end_time
        if not date_ok:
            continue
        if date_ok and days is None and hours is None:
            final_list += [f]
        else:
            # created earlier
            age_hours = getFileAge(pjoin(lookup_dir,f)  , 1)
    #        print(age_hours, f)
            age_days = age_hours / 24
            #print( age_days ,f )
            if hours is None and days is not None:
                if age_days < days:
                    final_list += [f]
            else:
                if days is None:
                    if age_hours < hours:
                        final_list += [f]
                else:
                    if age_hours < hours + days * 24:
                        final_list += [f]
    return final_list

def listRecentPrefixes(days = 5, hours = None, lookup_dir = None,
                       light_only=True, custom_rawname_regex=None,
                      start_time=None,end_time=None, recent_files=None ):
    import re
    if recent_files is not None:
        lf = recent_files
    else:
        lf = listRecent(days, hours, lookup_dir, start_time=start_time,
                        end_time=end_time)
    prefixes = []


    if custom_rawname_regex is None:
        regex = '_S.._.*grp[0-9\-]+_(.*)_ML'
    else:
        regex = '_'+custom_rawname_regex+ '_.*grp[0-9\-]+_(.*)_ML'
    if light_only:
        regex = '_!' + regex
    regex_c = re.compile(regex)

    for f in lf:
        out = re.match(regex_c, f)
        if out is None:
            continue
        prefix = out.groups()[-1]
        prefixes += [prefix]
    return list(sorted(set(prefixes) ) )

def collectFeatNums(output_per_raw, clf_types=None):
    feat_nums_perraw0 = {}
    feat_nums_perraw = {}
    feat_nums_red_perraw = {}
    feat_nums_red2_perraw = {}
    feat_numdict_perraw = {}

    if clf_types is None:
        clf_types = ['XGB', 'LDA']
    for rn in output_per_raw:
        #feat_nums_perprefix = {}
        feat_nums0 = {}
        feat_nums_per_prefix = {}
        feat_nums_red_per_prefix = {}
        feat_nums_red2_per_prefix = {}
        feat_numdict_per_prefix = {}
        for prefix,pg in output_per_raw[rn].items():
            feat_nums_red_pgs = {}
            feat_nums_red2_pgs = {}
            feat_nums_pgs = {}
            feat_numdict_pgs = {}
            for g,pitset in pg.items():
                if g == 'feature_names_filtered':
                    continue
                for it_set,mult_clf_results in pitset.items():
                    feat_numdict_pgs[(g,it_set)] = {}
                    for clf_type in clf_types:
                        feat_numdict_pgs[(g,it_set)][clf_type] = {}

                        if mult_clf_results is None:
                            continue
                        lda_anver = mult_clf_results.get('lda_analysis_versions',None)
                        if lda_anver is None:
                            lda_anver = mult_clf_results['LDA_analysis_versions']
                        if clf_type == 'XGB':
                            CV_aver_all = lda_anver['all_present_features']['CV_aver']
                            if 'ldaobj' in CV_aver_all:
                                nfeats = len( CV_aver_all['ldaobj'].scalings_ )
                            else:
                                nfeats =  CV_aver_all['nfeats']


                            #siXGB = mult_clf_results.get('strong_inds_XGB_fs',None )
                            siXGB = mult_clf_results.get('strong_inds_XGB',None )
                            if siXGB is not None:
                                nfeats_red = len(  siXGB )
                            else:
                                nfeats_red = -1


                            siXGB2 = mult_clf_results.get('strong_inds_XGB_fs',None )
                            if siXGB2 is not None:
                                nfeats_red2 = len(  siXGB2 )
                            else:
                                if ('best_inds_XGB_fs' not in mult_clf_results) and 'perfs_XGB_fs' in mult_clf_results:
                                    mult_clf_results['best_inds_XGB_fs'] =  mult_clf_results['perfs_XGB_fs'][-1]['featinds_present']
                                else:
                                    lda_version_name =  'strongest_features_XGB_fs_opinion'
                                    #print(lda_anver.keys() )
                                    lda_subver = lda_anver.get(lda_version_name, None)
                                    if lda_subver is not None:
                                        CV_aver_min = lda_subver['CV_aver']
                                        nfeats_red2 = CV_aver_min['nfeats']
                                    else:
                                        print('AAAAA')
                                        nfeats_red2 = -1

                            feat_numdict_pgs[(g,it_set)][clf_type]['all_features'] = nfeats
                            feat_numdict_pgs[(g,it_set)][clf_type]['red'] = nfeats_red
                            feat_numdict_pgs[(g,it_set)][clf_type]['fs_red'] = nfeats_red2

                        else:
                            for pt in lda_anver:
                                CV_aver = lda_anver[pt]['CV_aver']
                                if 'ldaobj' in CV_aver:
                                    nfeats = len( CV_aver['ldaobj'].scalings_ )
                                else:
                                    nfeats =  CV_aver['nfeats']

                                feat_numdict_pgs[(g,it_set)][clf_type][pt] = nfeats

                            #CV_aver_all = lda_anver['all_present_features']['CV_aver']
                            #if 'ldaobj' in CV_aver_all:
                            #    nfeats = len( CV_aver_all['ldaobj'].scalings_ )
                            #else:
                            #    nfeats =  CV_aver_all['nfeats']

                            #CV_aver_min = lda_anver['strongest_features_LDA_selMinFeatSet']['CV_aver']
                            #if 'ldaobj' in CV_aver_min:
                            #    nfeats_red = len( CV_aver_min['ldaobj'].scalings_ )
                            #else:
                            #    nfeats_red =  CV_aver_min['nfeats']

                            #nfeats_red2 = -1

                            #feat_nums_pgs[(g,it_set)]['LDA'] = nfeats_red
                            #feat_nums_pgs[(g,it_set)]['LDA'] = nfeats_red2
                            #nfeats_red = len(  CV_aver_min['ldaobj'].scalings_)
                        #X = mult_clf_results['transformed_imputed_CV']
                        #nfeats = len( mult_clf_results['MI_per_Feati'].feature_importances_ )
                        #if 'XGBobj' in mult_clf_results:
                        #    nfeats = len( mult_clf_results['XGBobj'].feature_importances_ )
                        #else:
                        #print(rn,prefix,g,it_set,X.shape)
                        #n = feat_nums.get(prefix,-1)
        #                 if n >= 0:
        #                     assert nfeats == n, (nfeats,n)
        #                 else:
        #                     nfeats = -1

                    feat_nums0[prefix] = nfeats
                    feat_nums_pgs[(g,it_set)] = nfeats
                    feat_nums_red_pgs[(g,it_set)] = nfeats_red
                    feat_nums_red2_pgs[(g,it_set)] = nfeats_red2
            feat_nums_red_per_prefix[prefix] = feat_nums_red_pgs
            feat_nums_red2_per_prefix[prefix] = feat_nums_red2_pgs
            feat_nums_per_prefix[prefix] = feat_nums_pgs
            feat_numdict_per_prefix[prefix] = feat_numdict_pgs
            #feat_nums_perprefix[prefix ] = feat_nums
        feat_nums_perraw0[rn] = feat_nums0
        feat_nums_perraw[rn] = feat_nums_per_prefix
        feat_nums_red_perraw[rn] = feat_nums_red_per_prefix
        feat_nums_red2_perraw[rn] = feat_nums_red2_per_prefix
        feat_numdict_perraw[rn] =   feat_numdict_per_prefix

    return feat_nums_perraw, feat_nums_perraw0, feat_nums_red_perraw, feat_nums_red2_perraw, feat_numdict_perraw

def confmat2bacc(C, mean_per_class=True, adjusted = False):
    if isinstance(C,list):
        Cs = C
        baccs = [ confmat2bacc(oneC, mean_per_class, adjusted) for oneC in Cs]
        assert mean_per_class  # when not a more accurate mean is needed
        score = np.mean(baccs)
        return score
    import warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1)  # this sum is normally = 1 for normalized confmat
    if np.any(np.isnan(per_class)):
        warnings.warn("y_pred contains classes not in y_true")
        per_class = per_class[~np.isnan(per_class)]

    if mean_per_class:
        score = np.mean(per_class)
    else:
        score = per_class
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

def analyzeRnstr(rn):
    mcs = []
    if ',' in rn:
        subjs_involved = []
        subrns = rn.split(',')
        mvss = []
        mcs = []
        for subrn in subrns:
            parts = subrn.split('_')
            subj = parts[0]
            if len(parts) > 1:
                mc = parts[1]
                mcs += [mc]
            mts = gv.gen_subj_info[subj].get('move_side',None)
            mvss += [mts]
            subjs_involved += [subj]
        #assert len( set(mvss) ) == 1, (rn, set(mvss) )
        #assert len( set(mcs) ) == 1
        mainmoveside = mvss[0]

        mc   = mcs[0]
    else:
        parts = rn.split('_')
        subj = parts[0]
        mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
        if len(parts) > 1:
            mc   = parts[1]
            mcs = [mc]
        subjs_involved = [subj]
        mvss = [mainmoveside]
    assert mainmoveside is not None
    movesidelet   = mainmoveside[0].upper()
    moveopsidelet = utils.getOppositeSideStr( movesidelet )

    mvsls  = [ s[0].upper() for s in mvss ]
    mvosls = [ utils.getOppositeSideStr(s)[0].upper() for s in mvss ]

    return  subjs_involved, mcs, mvsls, mvosls

def prepTableInfo3(output_per_raw, prefixes=None, 
        perf_to_use_list = [('perfs_XGB','perfs_XGB_red') ],
                  to_show = [('allsep','merge_nothing','basic')],
                  show_F1=False, use_CV_perf=True, rname_crop = slice(0,3), save_csv = True,
                  sources_type='parcel_aal', subdir='',
                   scores = ['sens','spec','F1'], return_df = False, df_inc_full_dat=False, recalc = False, use_tmpdir_to_load = False ):

    #perf_to_use is either 'perfs_XGB', 'perfs_XGB_red' or one of lda versions
    # Todo: pert_to_use_list -- is list of couples -- one and list ot feat
    # reductions

    #if feat_nums_perraw is None or feat_nums_red_perraw is None:

    colnames = ['prefix','grouping','interval_set','rawname', 'subject', 'medcond'] #, 'move_hand_side_letter', 'move_hand_opside_letter' ]
    colnames += ['clf_type','perf_name','perf_red_name']
    colnames += ['num','num_red','num_red2', 'bacc','acc''descr','comment_from_runstrings']
    for measure in ['sens', 'spec','F1']:
        for suff in ['','_red','_add']:
            rcns = ['']
            if recalc:
                rcns += ['_recalc']
            for recalc in rcns:
                colnames += [ measure + suff + recalc ]
    import pandas as pd
    df = pd.DataFrame(columns=colnames)
#label_types = [tpl[0] for tpl in to_show]
#print(label_types)
    table_info_per_perf_type = {}
    table_per_perf_type = {}

    assert output_per_raw is not None
    #rname_crop = slice(0,-5) # keeping medcond info
    #rname_crop = slice(0,3) # keeping medcond info

    if prefixes is None:
        k0 = list( output_per_raw.keys() ) [0]
        prefixes = list( sorted( list( output_per_raw[k0].keys() ) ) )


    # just to count
    tpll = multiLevelDict2TupleList(output_per_raw,4,3)

    rows = []
    ctr = 0
    for tpl in perf_to_use_list:
        clf_type,perf_to_use,perf_red_to_use = tpl[:3]
        if len(tpl) == 4:
            perf_add = tpl[3]
            print('perf tuple has length 4',perf_add)
        else:
            perf_add = None
        info_per_rn_pref ={}
        red_mode = False
        if perf_to_use.endswith('_red'):
            red_mode = True
        #table =  [ [''] +  prefix_labels_perraw[rawnames[0] ]
        #table =  [ [''] +  [ dct.get(prefix,prefix) for dct in prefix_labels_perraw] ]
        table =  [ [''] +  prefixes ]
        was_valid, was_red_valid = False, False
        for rn in output_per_raw:

            subjs_involved, mcs, mvsls, mvosls = analyzeRnstr(rn)
            movesidelet = mvsls[0]
            moveopsidelet = mvosls[0]

            for lt, it_grp, it_set in to_show:
            #for lt in label_types:
                info_per_pref = {}
                # raw and label type (grouping+int_types) name goes here
                # this will be a row name
                row_name = '{}_{}'.format(rn[rname_crop], lt)
                table_row = [row_name ]
                for prefix in prefixes:
                    print('start row',rn,it_grp,it_set,prefix)
                    info_cur = {}

                    #if len(subjs_involved) == 1:
                    info_cur['subject'] = list(set(subjs_involved))
                    info_cur['medcond'] = list(set(mcs))
                    info_cur['move_hand_side_letter'] = list(set(mvsls))
                    info_cur['move_hand_opside_letter'] = list(set(mvosls))
                    #sens,spec = res[rn].get(pref, (np.nan, np.nan))
                    r = output_per_raw[rn].get(prefix, None)
                    if (r is not None) and (it_grp not in r or it_set not in r[it_grp]):
                        r = None

                    perfs = None
                    perfs_red = None
                    num,num_red,num_red2 = None, None, None
                    perfs_add_CV=None
                    perfs_red_CV_recalc,perfs_CV_recalc=None,None
                    perf_add_cur = perf_add
                    corresp = None
                    all_runstr_info = None
                    if r is None:
                        print(f'  Warning: no prefix {prefix} for {rn}')
                        prefix_missing = True
                    else:
                        prefix_missing = False
                        mult_clf_results = r[it_grp][it_set]

                        if mult_clf_results is not None:
                            if not mult_clf_results.get('loaded',True):
                                fname_full = mult_clf_results['filename_full']
                                print(f'#### prepTableInfo3: Loading {ctr}/{len(tpll)}')

                                loadSingleRes(mult_clf_results,use_light_files=True,lighter_light=False,
                                        remove_large_items=True,
                                       use_tmpdir = use_tmpdir_to_load )

                                ctr += 1

                                #f = np.load(fname_full, allow_pickle=1)
                                #mult_clf_results.update( f['results_light'][()] )
                                #del f

                            corresp, all_runstr_info = loadRunCorresp(mult_clf_results)


                            class_label_names = mult_clf_results.get('class_label_names_ordered',None)
                            if recalc:
                                if class_label_names is not None:
                                    assert len(set(mvsls)) == 1
                                    fn = getLogFname(mult_clf_results)
                                    print(fn)
                                    lblind_trem = class_label_names.index(f'trem_{movesidelet}')
                                    #print('prepTableInfo3: warninig: Using fixed side for tremor: trem_L')
                                else:
                                    lblind_trem = 0

                            perfs_CV,perfs_noCV, perfs_red_CV,perfs_red_noCV  = None,None,None,None
                            perfs_CV_recalc,perfs_red_CV_recalc = None,None
                            perfs_add_CV_recalc = None,None,None
                            perf_shuffled = None
                            if clf_type == 'XGB':
                                XGB_anver = mult_clf_results['XGB_analysis_versions']
                                anver_cur = XGB_anver.get(perf_to_use)
                                if anver_cur is not None:
                                    if 'perf_aver' in anver_cur:
                                        perfs_CV   = anver_cur['perf_aver']
                                        perfs_noCV = anver_cur['perf_nocv']
                                        ps = anver_cur.get('perfs_CV', None)
                                    else:
                                        perfd = anver_cur['perf_dict']
                                        perfs_CV = perfd['perf_aver']
                                        perfs_noCV = perfd['perf_nocv']
                                        ps = perfd.get('perfs_CV', None)

                                        perf_shuffled = perfd['fold_type_shuffled'][-1]
                                    if recalc:
                                        perfs_CV_recalc = recalcPerfFromCV(ps,lblind_trem)

                                    confmats_cur = [p['confmat'] for p in ps]

                                    if perf_add_cur is not None:
                                        if perf_add_cur == 'across_subj':
                                            pdict = anver_cur['across'].get('subj',None)
                                            if pdict is None:
                                                import pdb; pdb.set_trace()
                                            if pdict is not None:
                                                perfs_add_CV = pdict['perf_aver']
                                                perfs_add_noCV = pdict['perf_nocv']
                                                ps = pdict['perfs_CV']
                                                if recalc:
                                                    perfs_add_CV_recalc = recalcPerfFromCV(ps,lblind_trem)
                                        elif perf_add_cur == 'across_medcond':
                                            pdict = anver_cur['across']['medcond']
                                            if pdict is not None:
                                                perfs_add_CV = pdict['perf_aver']
                                                perfs_add_noCV = pdict['perf_nocv']
                                                ps = pdict['perfs_CV']
                                                if recalc:
                                                    perfs_add_CV_recalc = recalcPerfFromCV(ps,lblind_trem)

                                                d = {}
                                                for perf in ps:
                                                    k = perf['generalization_pattern_from_fold']
                                                    d[k] = perf['balanced_accuracy']
                                                perfs_add_CV['genpat2bacc'] = d

                                                #pdict['perI#f'generalization_pattern_from_fold'
                                else:
                                    print(f'perf_to_use (={perf_to_use}): None!')

                                num_red_set = False
                                if perf_red_to_use == 'best_LFP' and 'XGB' in mult_clf_results['best_LFP']:
                                    chn_LFP = mult_clf_results['best_LFP']['XGB']['winning_chan']
                                    oo = mult_clf_results['XGB_analysis_versions'][f'all_present_features_only_{chn_LFP}']
                                    anver_red_cur =    oo['perf_dict']
                                    #perfs = pcm['perf_aver']
                                    num_red = len(oo['featis'] )
                                    num_red_set = True
                                else:
                                    anver_red_cur = mult_clf_results['XGB_analysis_versions'].get(perf_red_to_use,None)

                                if anver_red_cur is not None:
                                    if 'perf_aver' in anver_red_cur:
                                        perfs_red_CV   = anver_red_cur['perf_aver']
                                        perfs_red_noCV = anver_red_cur['perf_nocv']
                                        ps = anver_red_cur.get('perfs_CV', None)
                                        #print('1')
                                    else:
                                        perfs_red_CV = anver_red_cur['perf_dict']['perf_aver']
                                        perfs_red_noCV = anver_red_cur['perf_dict']['perf_nocv']
                                        ps = anver_red_cur['perf_dict'].get('perfs_CV', None)
                                    if recalc:
                                        perfs_red_CV_recalc = recalcPerfFromCV(ps,lblind_trem)
                                elif perf_red_to_use in ['interpret_EBM', 'interpret_DPEBM']:
                                    featsubset_name = 'all'
                                    EBM_dict = mult_clf_results['featsel_per_method'][perf_red_to_use][featsubset_name]
                                    perfs_red = EBM_dict['perf_dict']['perf_nocv']
                                    perfs_red_CV = EBM_dict['perf_dict']['perf_aver']
                                elif perf_red_to_use is None:
                                    perfs_red = None
                                    perfs_red_CV = None
                                else:
                                    print(f'perf_red_to_use (={perf_red_to_use}): None!')
                                    raise ValueError('perf_red_to_use is None')

                                if 'importances' in anver_cur:
                                    num = len(anver_cur['importances'] )
                                else:
                                    if perf_to_use == 'all_present_features':  # due to a small bug in selMinFeatSet
                                        num = len(anver_cur['sortinds'] )
                                    else:
                                        num = len(anver_cur['featinds_present'] )

                                if anver_red_cur is not None and not num_red_set:
                                    if 'importances' in anver_red_cur:
                                        num_red = len(anver_red_cur['importances'] )
                                    else:
                                        num_red = len(anver_red_cur['featinds_present'] )

                                if perf_red_to_use  == 'strongest_features_XGB_opinion':
                                    oo = mult_clf_results.get('PCA_XGBfeats',None)
                                    if oo is not None:
                                        num_red2 = oo.n_components_

                            elif clf_type == 'LDA':
                                #print("DDDDDDDDDDDDDDDDDDDDDDDDDD")
                                lda_anver = mult_clf_results.get('LDA_analysis_versions',None)
                                if lda_anver is None:
                                    lda_anver = mult_clf_results['lda_analysis_versions']

                                anver_cur = lda_anver.get(perf_to_use,None)
                                if anver_cur is not None:
                                    perfs_CV = anver_cur['CV']['CV_perfs']
                                    perfs_CV = np.mean(np.array([ perfs_CV[ip][:3] for ip in range(len(perfs_CV)) ]   ), axis=0)
                                    perfs_CV2 = anver_cur['CV_aver']['perfs']
                                    perfs_noCV = anver_cur['fit_to_all_data']['perfs']
                                    num =  anver_cur['CV_aver']['nfeats']
                                    ps  = anver_cur.get('perfs_CV', None)
                                else:
                                    print(f'perf_to_use (={perf_to_use}): None!')
                                if recalc:
                                    perfs_CV_recalc = recalcPerfFromCV(ps,lblind_trem)

                                anver_red_cur = lda_anver.get(perf_red_to_use,None)
                                if perf_red_to_use == 'best_LFP' and 'LDA' in mult_clf_results['best_LFP']:
                                    chn_LFP = mult_clf_results['best_LFP']['LDA']['winning_chan']
                                    anver_red_cur =     mult_clf_results['LDA_analysis_versions'][f'all_present_features_only_{chn_LFP}']
                                    num_red == len(anver_red_cur['featis'])
                                else:
                                    anver_red_cur = None

                                if anver_red_cur is not None:
                                    perfs_red_CV = anver_red_cur['CV']['CV_perfs']
                                    perfs_red_CV = np.mean(np.array([ perfs_red_CV[ip][:3] for ip in range(len(perfs_red_CV)) ]   ), axis=0)
                                    perfs_red_CV2 = anver_red_cur['CV_aver']['perfs']
                                    perfs_red_noCV = anver_red_cur['fit_to_all_data']['perfs']
                                    num_red = anver_red_cur['CV_aver']['nfeats']
                                    ps = anver_red_cur.get('perfs_CV', None)
                                else:
                                    print(f'perf_red_to_use (={perf_red_to_use}): None!')
                                    #raise ValueError('perf_red_to_use is None')
                                if recalc:
                                    perfs_red_CV_recalc = recalcPerfFromCV(ps,lblind_trem)

                                #num = numdict_cur[perf_to_use]
                                #num_red = numdict_cur[perf_red_to_use]
                            if clf_type in ['interpret_EBM', 'interpret_DPEBM']:
                                featsubset_name = 'all'
                                EBM_dict = mult_clf_results['featsel_per_method'][clf_type][featsubset_name]
                                perfs_red = EBM_dict['perf']
                                perfs_red_CV = EBM_dict['perf_aver']

                            if use_CV_perf:
                                perfs = perfs_CV
                                perfs_red = perfs_red_CV
                            else:
                                perfs = perfs_noCV
                                perfs_red = perfs_red_noCV

                    acc,bacc = np.nan,np.nan
                    if perfs is None:
                        print('Warning :',rn,prefix,lt)
                        sens,spec,F1 = np.nan, np.nan, np.nan
                    else:
                        #print([type( p) for p in perfs])
                        # sometimes perfs has confmat but sometimes not
                        if isinstance(perfs,(tuple,list,np.ndarray) ):
                            sens,spec,F1 = perfs[:3]
                        else:
                            sens,spec,F1 = perfs['sens'],perfs['spec'],perfs['F1']
                            acc,bacc    = perfs['accuracy'],perfs['balanced_accuracy']
                        if np.isnan(bacc):
                            bacc = confmat2bacc(confmats_cur)
                        was_valid = True

                    if perfs_red is None:
                        sens_red,spec_red,F1_red = np.nan, np.nan, np.nan
                        bacc_red = np.nan
                    else:
                        #print([type( p) for p in perfs_red])
                        #print('perfs_red = ', perfs_red)
                        if isinstance(perfs_red, (list,tuple,np.ndarray) ):
                            sens_red,spec_red,F1_red = perfs_red[:3]
                            bacc_red = np.nan
                        else:
                            sens_red,spec_red,F1_red = perfs_red['sens'],perfs_red['spec'],perfs_red['F1']
                            bacc_red = perfs_red['balanced_accuracy']
                        was_red_valid = True

                    if num is not None and num_red is not None:
                        assert num >= num_red, f'{rn},{lt},{it_grp},{it_set},{tpl}:{prefix}  {num},{num_red}'

                    if perfs_CV_recalc is not None:
                        if isinstance(perfs_CV_recalc, (list,tuple,np.ndarray) ):
                            info_cur['sens_recalc'] = perfs_CV_recalc[0]
                            info_cur['spec_recalc'] = perfs_CV_recalc[1]
                        else:
                            info_cur['sens_recalc'] = perfs_CV_recalc['sens']
                            info_cur['spec_recalc'] = perfs_CV_recalc['spec']
                    if corresp is not None and prefix in corresp:
                        info_cur['descr'] = corresp[prefix][-1]
                    else:
                        info_cur['descr'] = None
                    if all_runstr_info is not None:
                        info_cur['comment_from_runstrings'] = all_runstr_info.get('comment','')
                    else:
                        info_cur['comment_from_runstrings'] = None
                    info_cur['sens'] = sens
                    info_cur['spec'] = spec
                    info_cur['F1'] = F1
                    info_cur['acc']  = acc
                    info_cur['bacc'] = bacc
                    info_cur['sens_red'] = sens_red
                    info_cur['spec_red'] = spec_red
                    info_cur['bacc_red'] = bacc_red

                    info_cur['perf_dict_shuffled'] = perf_shuffled

                    info_cur['bacc_shuffled'] = _extractPerfNumber ( perf_shuffled, 'bacc' )
                    if perfs_red_CV_recalc is not None:
                        if isinstance(perfs_CV_recalc, (list,tuple,np.ndarray) ):
                            info_cur['sens_red_recalc'] = perfs_red_CV_recalc[0]
                            info_cur['spec_red_recalc'] = perfs_red_CV_recalc[1]
                        else:
                            info_cur['sens_red_recalc'] = perfs_red_CV_recalc['sens']
                            info_cur['spec_red_recalc'] = perfs_red_CV_recalc['spec']
                    else:
                        info_cur['sens_red_recalc'] = np.nan
                        info_cur['spec_red_recalc'] = np.nan
                    if perf_add_cur is not None and perfs_add_CV is not None:
                        if isinstance(perfs_add_CV, (list,tuple,np.ndarray) ):
                            print('bbbbb')
                            info_cur['sens_add'] = perfs_add_CV[0]
                            info_cur['spec_add'] = perfs_add_CV[1]
                            info_cur['F1_add'] = perfs_add_CV[2]
                        else:
                            info_cur['sens_add'] = perfs_add_CV['sens']
                            info_cur['spec_add'] = perfs_add_CV['spec']
                            info_cur['F1_add']   = perfs_add_CV['F1']
                            info_cur['bacc_add']   = perfs_add_CV['balanced_accuracy']
                            info_cur['perf_add_name'] = perf_add_cur

                            # remove full rawnames and put only medconds (otherwise outer join of pandas will make it look bad)
                            for k,v in perfs_add_CV.get('genpat2bacc',{}).items():
                                rns_ll = k.split('->')
                                medconds_ = []
                                for rns in rns_ll:
                                    rns = rns.split(',')
                                    import utils_preproc as upre
                                    r0,r = upre.getRawnameListStructure(rns, ret_glob=1)
                                    #print(r['medconds']) 
                                    medconds = list(r0.values())[0]['medconds'] 
                                    assert len(medconds) == 1 
                                    medconds_ += medconds
                                k = '_to_'.join(medconds_)
                                info_cur['bacc_' + k] = v
                                    #print(medconds[0])
                                #info_cur[k.replace('->','_to_')] = v

                        info_cur['sens_add_recalc'] = perfs_add_CV_recalc[0]
                        info_cur['spec_add_recalc'] = perfs_add_CV_recalc[1]
                    #print('sens_add',info_cur.get('sens_add',None))
                        #info_cur['F1_add_recalc']  = perfs_add_CV_recalc[2]


                    info_cur['F1_red'] = F1_red
                    info_cur['num'] = num
                    info_cur['num_red'] = num_red
                    if num_red2 is not None:
                        info_cur['num_red2'] = num_red2
                    #str_to_put = '{:.0f},{:.0f}'.format(100*sens,100*spec)
                    if show_F1:
                        str_to_put =  '{:.0f},{:.0f},{:.0f}'.format(100*sens,100*spec,100*F1)
                    else:
                        str_to_put =  '{:.0f},{:.0f}'.format(100*sens,100*spec)
                    try:
                        #if red_mode:
                        s = ' :{}/{}'.format(num_red,num)
                        #else:
                        #    s = ' :{}/{}'.format(num,num)
                        str_to_put = str_to_put + s

                    except KeyError as e:
                        print('AAAAA ',perf_to_use,rn,prefix,it_grp,it_set,e)
                    table_row += [ str_to_put  ]
                    print(rn,lt,prefix,str_to_put)

                    info_per_pref[prefix] = info_cur

                    info_cur['interval_set'] = it_set
                    info_cur['grouping'] = it_grp
                    info_cur['grouping_and_iset_label'] = lt
                    info_cur['rawname'] = rn
                    info_cur['prefix'] = prefix
                    info_cur['clf_type'] = clf_type
                    info_cur['perf_name'] = perf_to_use
                    info_cur['perf_red_name'] = perf_red_to_use

                    info_cur['numpts'] = len(mult_clf_results['class_labels_good_for_classif'])


                    #subj = rn.split('_')[0]
                    #mc   = rn.split('_')[1]
                    #mainmoveside = gv.gen_subj_info[subj].get('move_side',None)


                    pref_templ = None
                    for suff in ['LL','RR','LR','RL','BB','BL','BR', 'LB','RB']:
                        #assert len(set(mvsls)) == 1, (prefix,mvsls)
                        if prefix.endswith(suff) and len(set(mvsls)) == 1:
                            suff = prefix[-2:]
                            suff = suff.replace(movesidelet,'^')
                            suff = suff.replace(moveopsidelet,'%')
                            pref_templ = prefix[:-2] + suff

                    info_cur['prefix_templ'] = pref_templ
                    rows += [ info_cur ]
                    #df = df.append(info_cur,ignore_index=True)

                table += [table_row]
                info_per_rn_pref[ (rn,it_grp, it_set) ] = info_per_pref


            #rn,pref,lt

    #         table = np.array(table)
            if not was_valid:
                print('All nan, skipping')
                continue

            table_fname = "pptable_{}_{}_nr{}_nprefixes{}_nlts{}_{}_CV{}.csv".\
            format(sources_type, subdir, len(output_per_raw), len(prefixes),
                   len(to_show), perf_to_use, int(use_CV_perf) )
            table_fname_full = os.path.join(gv.dir_fig, table_fname)
            print(table_fname_full)

        table_per_perf_type[perf_to_use] = table
        table_info_per_perf_type[tpl] = info_per_rn_pref


    if return_df:
        df = pd.DataFrame(rows)
        return df, table_info_per_perf_type, table_per_perf_type
    else:
        return table_info_per_perf_type, table_per_perf_type



def plotFeatureImportance(ax, feature_names, shap_values,
                          mode='SHAP', explainer = None, nshow = 20,
                          color=None, alpha=0.7, sort=True):
    assert mode in ['XGB_gain', 'SHAP', 'XGB_Shapley', 'XGB_Shapley2', 'interpret_EBM', 'labind_vs_score' ], mode
    #print(mode)
    #print('Start plot feat importance')

    z = None
    aggregate = np.abs(shap_values)
    if mode.startswith('XGB_Shapley'):
        aggregate = np.mean(np.abs(shap_values[:, 0:-1]), axis=0)
        # sort by magnitude
        z = zip(aggregate, feature_names)
    elif mode == 'XGB_gain':
        z = zip(aggregate, feature_names)
        #z = [(x, y) for y, x in sorted(zip(aggregate, feature_names), reverse=True)]
        #z = list(zip(*z))
    elif mode in ['EBM','interpret_EBM', 'labind_vs_score']:
        if feature_names is None:
            #feature_names = global_exp['feature_names']
            feature_names = explainer['feature_names']
        if shap_values is None:
            aggregate = explainer._internal_obj['overall']['scores']
        z = zip(aggregate, feature_names)
        #z = [(x, y) for y, x in sorted(zip(aggregate, feature_names), reverse=True)]
        #z = list(zip(*z))

    if sort:
        z = sorted(z, reverse=True)
    z = [(x, y) for y, x in z]
    z = list(zip(*z))

    #print('fdfdfd ', len(z[0]), z[0][0] )
    z = ( z[0] [:nshow] , z[1] [:nshow] )


    ax.bar(z[0], z[1], color=color, alpha=alpha)
    #ax.set_xticks(rotation=90)
    ax.tick_params(axis='x', labelrotation=90 )
    #ax.tight_layout()

    ax.set_title(f'Scores (their abs vals) of type {mode}');

    return z

def mergeScores(scores,feature_names,collect_groups,
                feature_names_subset=None,
                feature_groups_names=None, aux=None,
                max_nfeats_to_sum=None):
    'here collect_groups can be either list of strings or list of lists (or mixed)'
    from featlist import selFeatsRegexInds
    # pool scores
    assert (isinstance(scores,list) or scores.ndim == 1 )
    assert len(scores) == len(feature_names), (len(scores), len(feature_names))
    if aux is not None:
        assert len(aux) == len(collect_groups), ( len(aux), len(collect_groups) )
    mean_scores, std_scores, max_scores = [],[],[]
    mean_signed_scores, std_signed_scores, max_signed_scores = [],[],[]
    sum_scores = []
    sum_signed_scores = []

    if feature_names_subset is not None:
        assert set( feature_names ) >= set(feature_names_subset), f'{set( feature_names )}, {set(feature_names_subset) }'
        #indices of subset features in the larger things
        #subinds = [ feature_names.index(f) for f in feature_names_subset  ]
    # lists of features to be pooled
    inds_lists = []
    names = []
    aux_res = []
    if feature_groups_names is not None:
        assert len(feature_groups_names) == len(collect_groups), ( len(feature_groups_names), len(collect_groups) )
        names_display = []
    else:
        names_display = None
    feature_names_l = feature_names.tolist()
    for cgri,cgr in enumerate(collect_groups):
        if feature_names_subset is not None:
            inds_small_set = selFeatsRegexInds(feature_names_subset,cgr)
            inds = [ feature_names_l.index(feature_names_subset[i]) for i in inds_small_set  ]
        else:
            inds = utsne.selFeatsRegexInds(feature_names,cgr)
        if len(inds):
            if isinstance(cgr,list):
                cgr_ = '||'.join(cgr)
            else:
                cgr_ = cgr

            if feature_groups_names is not None:
                names_display += [ f'{feature_groups_names[cgri]}:{len(inds)}' ]

            names += [ cgr_  ]
            aux_res += [aux[cgri] ]
            scca = np.abs( np.array(scores)[inds] )
            mean_scores +=  [ scca.mean()  ]
            std_scores +=   [ scca.std()  ]
            max_scores +=   [ scca.max()  ]
            sum_scores +=   [ np.sum(scca)  ]

            scc = np.array(scores)[inds]
            mean_signed_scores +=  [ scc.mean()  ]
            std_signed_scores +=   [ scc.std()  ]
            max_signed_scores +=   [ scc.max()  ]

            sumval = np.sum(scc)
            if (max_nfeats_to_sum is not None) and len(inds) > max_nfeats_to_sum:
                sumval = -5e-2
                print(names[-1], len(inds) )
                #import pdb; pdb.set_trace()
            sum_signed_scores +=   [ sumval  ]

            inds_lists +=   [inds]
        else:
            print(f'{cgr} gave zero features')

    #print(feature_names[inds_lists[-1] ])
    stats = {}
    stats['names'] = names
    stats['aux'] = aux_res
    stats['names_display'] = names_display
    stats['mean'] = mean_scores
    stats['sum_abs']  = sum_scores
    stats['std_abs']  = std_scores
    stats['max_abs']  = max_scores
    stats['mean'] = mean_signed_scores
    stats['sum']  = sum_signed_scores
    stats['std']  = std_signed_scores
    stats['max']  = max_signed_scores
    stats['inds_lists'] = inds_lists
    #return mean_scores,std_scores,max_scores
    return stats

def mergeScores2(scores,class_labels,lblind,
                 feature_names,collect_groups,
                feature_names_subset=None,feature_groups_names=None,
                 aux=None, max_nfeats_to_sum=20):
    '''
    uses full scores (not averaged over points)
    scores is for the fixed class for each data point
    here collect_groups can be either list of strings or list of lists (or mixed)'''
    from featlist import selFeatsRegexInds
    # pool scores
    assert scores.ndim == 2
    # raw scores contain bias term
    assert scores.shape[-1] - 1 == len(feature_names), (scores.shape[-1] - 1, len(feature_names))
    assert scores.shape[0] == len(class_labels), (scores.shape[0], len(class_labels) )
    if aux is not None:
        assert len(aux) == len(collect_groups), ( len(aux), len(collect_groups) )
    stats = {}
    stats['mean']       = []
    stats['std']        = []
    stats['max']        = []
    stats['absmin']     = []
    stats['sum']        = []
    stats['mean_abs']   = []
    stats['std_abs']    = []
    stats['max_abs']    = []
    stats['absmin_abs'] = []
    stats['sum_abs']    = []
    stats['names']      = []

    if feature_names_subset is not None:
        assert set( feature_names ) >= set(feature_names_subset), f'{set( feature_names )}, {set(feature_names_subset) }'
        #indices of subset features in the larger things
        #subinds = [ feature_names.index(f) for f in feature_names_subset  ]
    # lists of features to be pooled
    inds_lists = []
    names = []
    aux_res = []
    if feature_groups_names is not None:
        assert len(feature_groups_names) == len(collect_groups), ( len(feature_groups_names), len(collect_groups) )
        names_display = []
    else:
        names_display = None

    ptinds = np.where(class_labels == lblind)[0]
    scores_cur_full = scores[ptinds,0:-1]

    feature_names_l = feature_names.tolist()
    for cgri,cgr in enumerate(collect_groups):
        # indices of features
        if feature_names_subset is not None:
            inds_small_set = selFeatsRegexInds(feature_names_subset,cgr)
            inds = [ feature_names_l.index(feature_names_subset[i]) for i in inds_small_set  ]
        else:
            inds = utsne.selFeatsRegexInds(feature_names,cgr)
        if len(inds):
            if isinstance(cgr,list):
                cgr_ = '||'.join(cgr)
            else:
                cgr_ = cgr

            if feature_groups_names is not None:
                names_display += [feature_groups_names[cgri]]

            sc = scores_cur_full[:,inds]
            scores_feat_grouped_sum  = np.sum( sc, axis=1 ) #signed sum over features
            scores_feat_grouped_mean = np.mean(sc, axis=1 ) #signed
            scores_feat_grouped_std  = np.std (sc, axis=1 ) #signed
            scores_feat_grouped_max  = np.max (sc, axis=1 ) #signed
            scores_feat_grouped_min  = np.min (sc, axis=1 ) #signed


            scabs = np.abs(sc)
            scores_feat_grouped_sum_abs  = np.sum( scabs, axis=1 ) #UNsigned sum over features
            scores_feat_grouped_mean_abs = np.mean(scabs, axis=1 ) #UNsigned
            scores_feat_grouped_std_abs  = np.std (scabs, axis=1 ) #UNsigned
            scores_feat_grouped_max_abs  = np.max (scabs, axis=1 ) #UNsigned
            scores_feat_grouped_min_abs  = np.min (scabs, axis=1 ) #UNsigned
            #scores_cur = np.mean(sc, axis=0)
            #scores_cur = np.mean(sc, axis=0)  axis 0 -- ptind, axis 1 -- featis


            #now compute means over datapoints
            names += [cgr_]
            aux_res += [aux[cgri] ]
            # acorss data points
            stats['mean']   +=     [np.abs(scores_feat_grouped_mean).mean() ]
            stats['std']    +=     [np.abs(scores_feat_grouped_std).mean()  ]
            stats['max']    +=     [np.abs(scores_feat_grouped_max).mean()  ]
            stats['absmin'] +=     [np.abs(scores_feat_grouped_min).mean()  ]
            stats['sum']    +=     [np.abs(scores_feat_grouped_sum).mean()  ]
            #stats['mean'] [cgri] =  [ np.abs(scores_feat_grouped_sum).mean()  ]
            #stats['std']  [cgri] =   [ np.abs(scores_feat_grouped_std).mean()  ]
            #stats['max']  [cgri] =   [ np.abs(scores_feat_grouped_max).mean()  ]
            #stats['absmin']  [cgri] =   [ np.abs(scores_feat_grouped_min).mean()  ]
            #stats['sum']  [cgri] =   [ np.abs(scores_feat_grouped_sum).mean()  ]

            #assert  stats['mean'][cgri] <=  stats['max'][cgri] -- it actually can happen if mean is large negative

            stats['mean_abs']   += [   scores_feat_grouped_mean_abs.mean()   ]
            stats['std_abs']    += [   scores_feat_grouped_std_abs.mean()    ]
            stats['max_abs']    += [   scores_feat_grouped_max_abs.mean()    ]
            stats['absmin_abs'] += [    scores_feat_grouped_min_abs.mean() ]
            stats['sum_abs']    += [   scores_feat_grouped_sum_abs.mean()    ]


            stats['names'] += [ cgr_ ]

            #scc = np.array(scores)[inds]
            #mean_signed_scores +=  [ scc.mean()  ]
            #std_signed_scores +=   [ scc.std()  ]
            #max_signed_scores +=   [ scc.max()  ]
            #sum_signed_scores +=   [ np.sum(scc)  ]

            inds_lists +=   [inds]
        else:
            print(f'{cgr} gave zero features')

    #print(feature_names[inds_lists[-1] ])
    stats['aux'] = aux_res
    stats['names_display'] = names_display
    #stats['mean_signed'] = mean_signed_scores
    #stats['sum_signed']  = sum_signed_scores
    #stats['std_signed']  = std_signed_scores
    #stats['max_signed']  = max_signed_scores
    stats['inds_lists'] = inds_lists
    #return mean_scores,std_scores,max_scores
    return stats

def plotFeatImpStats(feat_types_all, scores_stats, fign='', axs=None,
                     bias=None, color=None, alpha=0.8, bar=True,
                     markersize = 10, show_max = True, show_sum=True,
                     show_std = True, marker_mean = 'o', marker_max = 'x',
                     plot_signed = True, skip_ax_inds = [],
                     same_sets_only = True):
    import matplotlib.pyplot as plt
    from plots import plotErrorBarStrings
    assert isinstance(scores_stats, dict)
    nr = 1; nc= 2;   # normally == len(scores_stats) - 2;
    if not show_sum:
        nc = 1
    ww = 3; hh = 5;
    if axs is None:
        fig,axs = plt.subplots(nr,nc, sharey='row', figsize = (nc*ww,nr*hh));
    else:
        assert len(axs) >= nc

    if plot_signed:
        # for a fixed data point we look at a signed sum
        # we really need max_abs here as well, otherwise max can be less than
        # mean
        #stat_keys = {'max':'max_abs', 'sum':'sum', 'mean':'mean', 'std':'std' }
        stat_keys = {'max':'max', 'sum':'sum', 'mean':'mean', 'std':'std' }
    else:
        stat_keys = {'max':'max_abs', 'sum':'sum_abs', 'mean':'mean_abs',
                     'std':'std_abs' }
    names = scores_stats.get('names_display',scores_stats['names'] )
    #for i,k in enumerate(ks):
    #    #print(k ,  scores_stats[k] )
    #    ax = axs[i]; ax.set_title(f'{fign} {k}')
        #assert len(feat_types_all) == len(scores_stats[k] ), ( len(feat_types_all), len(scores_stats[k] )  )
        #sc = scores_stats[k]
        #names_cur = names

        #if bias is not None :
        #    if k == 'mean':
        #        bias_cur = np.mean(bias)
        #    elif k == 'max':
        #        bias_cur = np.max(bias)
        #    elif k == 'std':
        #        bias_cur = np.std(bias)
        #    elif k == 'sum':
        #        bias_cur = np.sum(bias)
        #    nm = 'bias'

        #    mx = np.max(sc)
        #    if bias_cur > mx:
        #        div = 2 * bias_cur / mx
        #        bias_cur = mx / 2
        #        nm = nm + f'/{div:.4f}'

        #    names_cur = [nm] + names
        #    sc = [ bias_cur ] + list(sc)

    if color is None and 'aux' in scores_stats:
        color_cur = scores_stats['aux']
        if bias is not None:
            color_cur = [ [0,0,0]  ] + color_cur
    else:
        color_cur = color

    #print('color_cur =', color_cur)

    if show_sum:
        ax = axs[1]
    stat_key = stat_keys['sum']
    sc = scores_stats[stat_key]

    names_cur =  names
    sc =  list(sc)
    max_sum = np.max(sc)
    if bias is not None:
        nm = 'bias'
        bias_cur = np.sum(bias)
        mx = np.max(sc)
        if bias_cur > mx:
            div = 2 * bias_cur / mx
            bias_cur = mx / 2
            nm = nm + f'/{div:.4f}'

        names_cur = [nm] + names
        sc = [ bias_cur ] + list(sc)
    names_cur_sum = names_cur

    #print('sum names = ',names_cur)

    if show_sum:
        if bar:
            ax.barh(names_cur,np.array(sc), color=color_cur, alpha=alpha ); #ax.tick_params(axis='x', labelrotation=90 )
        else:
            #print('skip')
            #ax.plot(np.array(sc),names_cur, color=color_cur, alpha=alpha, lw=0,
            #        marker='o'); #ax.tick_params(axis='x', labelrotation=90 )
            if np.sum( np.array(list(names_cur[0])) == '/' ) == 2:
                import pdb; pdb.set_trace()

            #plotErrorBarStrings(ax,names_cur,sc,xerr=None,
            #    add_args={'fmt':marker_mean, 'color':color_cur,
            #              'alpha':alpha*0.9, 'markersize':markersize} )
            plotErrorBarStrings(ax,names_cur,sc,xerr=None,
                same_sets_only = same_sets_only,
                add_args={'marker':marker_mean, 'color':color_cur,
                        'alpha':alpha*0.9, 'markersize':markersize} )
        ax.set_title(stat_key)
    # TEMP, should be uncommented
    #ax.set_yticks([])

    #####################################

    ax = axs[0]
    stat_key = stat_keys['max']
    sc = scores_stats[stat_key]

    if bias is not None:
        nm = 'bias'
        bias_cur = np.max(bias)
        mx = np.max(sc)
        if bias_cur > mx:
            div = 2 * bias_cur / mx
            bias_cur = mx / 2
            nm = nm + f'/{div:.4f}'

    names_cur =  names
    sc =  list(sc)
    if bias is not None:
        names_cur = [nm] + names
        sc = [ np.nan ] + list(sc)
    max_max = np.max(sc)

    #if max_max < max_sum:
    #    import pdb; pdb.set_trace()

    names_cur_max = names_cur
    if show_max:
        if bar:
            ax.barh(names_cur,np.array(sc), color=color_cur, alpha=alpha ); #ax.tick_params(axis='x', labelrotation=90 )
        else:
            #print('skip')
            #ax.plot(np.array(sc), names_cur, color=color_cur, alpha=alpha,
            #        lw=0, marker='o'); #ax.tick_params(axis='x', labelrotation=90 )

            plotErrorBarStrings(ax,names_cur,sc,xerr=None,
                        same_sets_only = same_sets_only,
                                add_args={'marker':marker_max,
                    'color':color_cur, 'alpha':alpha*0.9,
                    'markersize':markersize * 1.2} )

    ################################

    stat_key = stat_keys['mean']
    stat_key_std = stat_keys['std']
    sc = scores_stats[stat_key]
    max_mean = np.max(sc)
    names_cur = names
    if bias is not None:
        nm = 'bias'
        bias_cur = np.max(bias)
        mx = np.max(sc)
        if bias_cur > mx:
            div = 2 * bias_cur / mx
            bias_cur = mx / 2
            nm = nm + f'/{div:.4f}'
        names_cur = [nm] + names
        sc = [ bias_cur ] + list(sc)

    names_cur_mean = names_cur
    assert tuple(names_cur_mean[1:]) == tuple(names_cur_sum[1:]), (names_cur_mean,names_cur_sum)
    assert tuple(names_cur_max[1:]) == tuple(names_cur_sum[1:]), (names_cur_max,names_cur_sum)

    #print('before plotting mean')
    ax = axs[0]
    xerr_to_show = None
    if show_std:
        xerr_to_show = [0] + list(scores_stats[stat_key_std] )
    if bar:
        ax.barh(names_cur,np.array(sc), color=color_cur,
                alpha=alpha*0.9, xerr= xerr_to_show );
        #ax.tick_params(axis='x', labelrotation=90 )
    else:
        plotErrorBarStrings(ax,names_cur,sc,xerr=xerr_to_show,
                same_sets_only = same_sets_only,
                add_args={'fmt':marker_mean, 'color':color_cur,
                'alpha':alpha*0.9, 'markersize':markersize} )
        #ax.errorbar(names_cur,np.array(sc), color=color_cur, alpha=alpha*0.9, xerr= [0] + list(scores_stats['std'] ), fmt='o' ); #ax.tick_params(axis='x', labelrotation=90 )
    keystr = [ stat_keys['mean'] ]
    if show_max:
        keystr += [stat_keys['max']]
    if show_std:
        keystr += [stat_keys['std']]
    keystr = ','.join(keystr)

    ax.set_title(keystr)

    for axi,ax in enumerate(axs):
        if axi not in skip_ax_inds :
            ax.grid()

    return max_mean, max_max, max_sum

def plotFeatSignifSHAP_list(pdf, outputs_grouped, fshs=['XGB_Shapley'],
                       figname_prefix='',
                       n_individ_feats_show=4, roi_labels = None,
                       chnames_LFP = None, body_side='L', hh=8, ww = None,
                       separate_by_band = False,
                        separate_by_band2 = True,
                            suptitle = None, suptitle_fontsize=20,
                      tickfontsize = 10, show_bias=False,
                            use_best_LFP=False, markersize=10, show_max = True,
                           merge_Hjorth = False,
                            Hjorth_diff_color=False,
                            show_std = True, average_over_subjects = True,
                            alpha = 0.8, alpha_over_subj = 1., use_full_scores=False,
                            featsel_on_VIF=True, show_abs_plots=False,
                            reconstruct_from_VIF = False,
                            marker_mean = 'o', marker_max = 'x',
                            axs=None, grand_average_per_feat_type=1,
                            perf_marker_size = 25,
                           cross_source_groups = False,
                           indivd_imp_xtick_pad =  -300  ):
    '''
    fshs -- names of featsel method to use
    '''
    if isinstance(fshs , str):
        fshs = [fshs]

    #fsh = 'XGB_Shapley'
    if len(fshs) == 1:
        colors_fsh = len(fshs) * [None]
        print('Setting empty main color')
    else:
        colors_fsh = ['blue','red','green', 'purple']


    import utils_postprocess_HPC as postp
    import matplotlib.pyplot as plt


    from collections.abc import Iterable


    #for rn,a in outputs_grouped:
    a = list(outputs_grouped.values())[0]
    (prefix,grp,int_type), mult_clf_output = a
    featsel_per_method             = mult_clf_output['featsel_per_method']
    featnames                      = mult_clf_output['feature_names_filtered']
    #class_labels_good_for_classif  = mult_clf_output['class_labels_good']
    class_labels_good_for_classif  = mult_clf_output['class_labels_good_for_classif']

    VIF_truncation = mult_clf_output.get('VIF_truncation',None)
    colinds_good_VIFsel  = VIF_truncation.get('colinds_good_VIFsel',None)
    if colinds_good_VIFsel is not None and featsel_on_VIF:
        featnames = np.array(featnames)[colinds_good_VIFsel]

    subskip_fit = mult_clf_output['pars'].get('subskip_fit',1)
    subskip_fit = int(subskip_fit)
    assert subskip_fit is not None


    #assert len(featnames_list) == len(fshs)
    #if isinstance(featnames_list, Iterable) and \
    #        (isinstance(featnames_list[0], str) ):
    #    featnames_list = [featnames_list]



    fspm_def = featsel_per_method[ fshs[0]  ]
    scores = fspm_def.get('scores',None)
    if scores is None:
        scores_per_class_def = fspm_def.get('scores_av',None)
    else:
        assert scores.shape[-1] - 1 == len(featnames), (scores.shape[-1] , len(featnames))
        scores_per_class_def = utsne.getScoresPerClass(class_labels_good_for_classif, scores)
    nscores = len(scores_per_class_def)

    #if isinstance(hh,str) and hh == 'auto':
    #    hh = len(featnames) / 200

    nr = nscores;
    if ww is None:
        ww = 2 + 3 + 5;
    if show_abs_plots:
        nc = 2 + 2 + 1 + 1; #nc= len(scores_stats) - 2;
        width_ratios = [1,1,1,1,0.4,3]
    else:
        nc = 2 + 1 + 1;
        width_ratios = [1,1,0.4,3]

    if axs is None:
        fig,axs = plt.subplots(nr,nc, figsize = (nc*ww,nr*hh),
                                gridspec_kw={'width_ratios': width_ratios} );
        plt.subplots_adjust(top=1-0.02)
    else:
        assert axs.shape == (nr,nc)


    cmap = plt.cm.get_cmap('tab20', 20)
    #cmap( (ri0 + i*ri1) % 20)

    from featlist import getFeatIndsRelToOnlyOneLFPchan
    from featlist import getChnamesFromFeatlist

    #biases = {}
    #maxs_list = {}
    outs = []

    outs_mb = []

    stats_per_all = []
    for rni,(rn,a) in enumerate(outputs_grouped.items() ):
        (prefix,grp,int_type), mult_clf_output = a
        featsel_per_method   = mult_clf_output['featsel_per_method']
        featnames            = mult_clf_output['feature_names_filtered'].copy()
        class_label_names    = mult_clf_output.get('class_label_names_ordered',None)

        VIF_truncation = mult_clf_output.get('VIF_truncation',None)
        colinds_good_VIFsel  = mult_clf_output['VIF_truncation'].get('colinds_good_VIFsel',None)
        #print('colinds_good_VIFsel ',colinds_good_VIFsel)

        assert not ( (colinds_good_VIFsel is not None) and use_best_LFP), 'requires more thinking'




        if colinds_good_VIFsel is not None and featsel_on_VIF:
            featnames_sub = np.array(featnames)[colinds_good_VIFsel]
        else:
            featnames_sub = featnames


        featnames_nice_sub = utils.nicenFeatNames(featnames_sub, {'kk':roi_labels},['kk'])

        class_labels_good_for_classif  = mult_clf_output['class_labels_good_for_classif']
        if class_label_names is None:
            class_labels_good  = mult_clf_output['class_labels_good']
            revdict = mult_clf_output['revdict']

            from sklearn import preprocessing
            lab_enc = preprocessing.LabelEncoder()
            # just skipped class_labels_good
            lab_enc.fit(class_labels_good)
            class_labels_good_for_classif = lab_enc.transform(class_labels_good)
            class_label_ids = lab_enc.inverse_transform( np.arange( len(set(class_labels_good_for_classif)) ) )
            class_label_names = [revdict[cli] for cli in class_label_ids]
            #print(class_label_names)

        if chnames_LFP is None:
            chnames_LFP = getChnamesFromFeatlist(featnames_sub, mod='LFP')


        if use_best_LFP:
            chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
            new_channel_name_templ='LFP007'
            #print(chn_LFP)
            feat_inds_curLFP, feat_inds_except_curLFP = \
                getFeatIndsRelToOnlyOneLFPchan(featnames,
                    chnpart=chn_LFP, chnames_LFP=chnames_LFP,
                           new_channel_name_templ=new_channel_name_templ,
                            mainLFPchan=chn_LFP)
            if colinds_good_VIFsel is not None:
                print('Intersecting VIF and curLFP inds')
                feat_inds_curLFP = np.intersect1d(feat_inds_curLFP,colinds_good_VIFsel)
            featnames_sub = np.array(featnames)[feat_inds_curLFP]
            chnames_LFP_cur = [new_channel_name_templ]
            #print(rn,chn_LFP, len(featnames) )
        else:
            chnames_LFP_cur = chnames_LFP

        #####
        exogs_list = VIF_truncation['exogs_list']
        VIF_truncation['VIF_search_worst']
        #VIF_truncation['X_for_VIF_shape'] = X_for_VIF.shape
        colinds_bad         = VIF_truncation[ 'colinds_bad_VIFsel']
        VIFsel_linreg_objs  = VIF_truncation['VIFsel_linreg_objs']
        VIFsel_featsets_list  = VIF_truncation[ 'VIFsel_featsets_list']


        for fshi,fsh in enumerate(fshs):
            fspm = featsel_per_method[fsh]
            #featnames = featnames_list[fshi]
            scores = fspm.get('scores',None)
            #print(scores.shape)
            if scores is None:
                if use_full_scores:
                    raise ValueError('we need full scores!')
                scores_per_class = fspm.get('scores_av',None)
                bias = fspm.get('scores_bias_av',None)

                if reconstruct_from_VIF:
                    scores_per_class_VIF = scores_per_class
                    scores_per_class_reconstructed = utsne.reconstructFullScoresFromVIFScores(scores_per_class_VIF,
                        len(featnames), colinds_bad,colinds_good_VIFsel,
                        VIFsel_featsets_list, VIFsel_linreg_objs, exogs_list )

                    featnames_sub = featnames
            else:
                if use_full_scores:
                    if len(set(class_labels_good_for_classif) ) == 2:
                        scores = scores[:,None,:]
                        scores = np.concatenate( [scores,scores], axis=1)
                    assert scores.ndim == 3, scores.shape

                    if reconstruct_from_VIF:
                        scores_full_per_class_VIF = scores
                        scores_full_reconstructed = utsne.reconstructFullScoresFromVIFScores(scores_full_per_class_VIF,
                            len(featnames), colinds_bad,colinds_good_VIFsel,
                            VIFsel_featsets_list, VIFsel_linreg_objs, exogs_list )

                        scores = scores_full_reconstructed
                        featnames_sub = featnames

                assert scores.shape[-1] - 1 == len(featnames_sub), (scores.shape[-1] , len(featnames_sub))
                # XGB doc: Note the final column is the bias term
                scores_per_class, bias = utsne.getScoresPerClass(class_labels_good_for_classif,
                                                                 scores, ret_bias=1)

                #if reconstruct_from_VIF:
                #    scores_per_class_VIF = scores_per_class
                #    scores_per_class_reconstructed = utsne.reconstructFullScoresFromVIFScores(scores_per_class_VIF,
                #        len(featnames), colinds_bad,colinds_good_VIFsel,
                #        VIFsel_featsets_list, VIFsel_linreg_objs, exogs_list )
                #print( fspm.keys(), scores.shape )
            print('scores.shape = ',scores.shape)
            #import pdb;pdb.set_trace()

            assert scores_per_class.shape[-1] == len(featnames_sub),  (scores_per_class.shape[-1], len(featnames_sub))

            ###############################################
            # make plots for every class label
            for lblind in range(scores_per_class.shape[0] ):
                # select points where true class is like the current one
                #ptinds = np.where(class_labels_good_for_classif == lblind)[0]
                #classid_enconded = lblind
                #scores_cur = np.mean(scores[ptinds,lblind,0:-1], axis=0)

                scores_cur = scores_per_class[lblind]
                if use_full_scores:
                    scores_full_cur = scores[:,lblind,:]
                label_str = class_label_names[lblind]

                ###############################




                    #display(feat_groups_all)
                clrs,feature_groups_names,feat_groups_all  = \
                    prepareFeatGroups(featnames_sub,body_side,
                                      roi_labels,cmap,
                                      chnames_LFP, separate_by_band,
                                      separate_by_band2,
                                      merge_Hjorth,
                                      Hjorth_diff_color,
                                      grand_average_per_feat_type,
                                      cross_source_groups)

                    #############################################

                #display(feat_groups_all)
                if use_full_scores:
                    feat_imp_stats = mergeScores2(scores_full_cur,
                                                  class_labels_good_for_classif,
                        lblind, featnames_sub, feat_groups_all,
                        feature_names_subset=None,
                        feature_groups_names=feature_groups_names, aux=clrs)
                else:
                    feat_imp_stats = mergeScores(scores_cur, featnames_sub,
                        feat_groups_all,
                        feature_names_subset=None,
                        feature_groups_names=feature_groups_names, aux=clrs)
                stats_per_all += [ (rn,fsh,lblind , feat_imp_stats ) ]
                ####################################

                if isinstance(rn,tuple):
                    rn_ = rn[0]
                else:
                    rn_ = rn
                #outs +=   [ (rn_,prefix,grp,int_type,fsh,scores_per_class,bias)  ]

                outs +=   [ (rn_,prefix,grp,int_type,fsh,featnames_nice_sub,label_str,scores_per_class[lblind],feat_imp_stats )  ]

                ####################################
                subaxs = axs[lblind,:]

                #if not show_bias:
                bias_to_plot = None
                #else:
                #    bias_to_plot = bias[lblind]

                maxs = []
                print( f'len(feat_groups_all) = {len(feat_groups_all)}, len(featnames_sub) = {len(featnames_sub)}')
                #import pdb;pdb.set_trace()
                #same_sets_only should be zero when I run giving axes with some stuff on them already,
                # especially if I use biases
                max0,max00,max1 = plotFeatImpStats(feat_groups_all, feat_imp_stats, axs= subaxs[:2],
                                bias=bias_to_plot, color=colors_fsh[fshi], bar=False,
                                 markersize=markersize, show_max = show_max,
                                 show_std = show_std, alpha=alpha,
                                 plot_signed=True,
                                 same_sets_only = 0,
                                 marker_mean= marker_mean,
                                 marker_max = marker_max)
                maxs +=  [max0,max00,max1]
                subaxs[0].tick_params(axis='y', labelsize=  tickfontsize)
                subaxs[0].set_title( f'{subaxs[0].get_title()}  {label_str}' )

                #fig.canvas.draw() # needed to set yticks

                if show_abs_plots:
                    max2,max22,max3 = plotFeatImpStats(feat_groups_all, feat_imp_stats, axs= subaxs[2:4],
                                    bias=bias_to_plot, color=colors_fsh[fshi], bar=False,
                                    markersize=markersize, show_max = show_max,
                                    show_std = show_std, alpha=alpha,plot_signed=False,
                                    same_sets_only = 0,
                                     marker_mean= marker_mean, marker_max = marker_max)
                    #subaxs[2].set_yticks([])
                    subaxs[2].set_title( f'{subaxs[2].get_title()}  {label_str}' )
                    maxs +=  [max2,max22,max3]


                rny = rn
                if isinstance(rn,tuple):
                    rny = rn[0]
                outs_mb += [(rni,rny,fsh,lblind,bias[lblind],maxs)]
                    #fig.canvas.draw() # needed to set yticks

                #subaxs[0].tick_params(axis='y', labelsize=  tickfontsize)
                #plt.tight_layout()
                #ax.set_title(  )
                #pdf.savefig()


                #plt.figure(figsize = (12,10))
                #ax = plt.gca()
                ax = subaxs[-1]
                #postp.plotFeatureImportance(ax, featnames_sub, scores[ptinds,lblind,:], 'XGB_Shapley')
                sort_individ_feats = fshi == 0
                plotFeatureImportance(ax, featnames_nice_sub,
                                            scores_per_class[lblind,:],
                                            'labind_vs_score',
                                            color=colors_fsh[fshi],
                                            sort = sort_individ_feats,
                                            nshow=n_individ_feats_show)
                ax.set_title( f'{figname_prefix}: ' + ax.get_title() + f'_lblind = {label_str} (lblind={lblind}):  {fsh}' )
                ax.tick_params(axis="x",direction="in", pad=indivd_imp_xtick_pad, labelsize=tickfontsize)

                ###############3

                perf_from_confmat = None
                if use_best_LFP:
                    #chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
                    pcm = mult_clf_output['XGB_analysis_versions'][f'all_present_features_only_{chn_LFP}']['perf_dict']
                    perfs = pcm['perf_aver']


                    #confmat = pcm.get('confmat', None)
                    ##print(pcm.keys())
                    #if confmat is None:
                    #    confmat = pcm.get('confmat_aver', None)
                    #    #print('using confmat_aver')
                    #perf_from_confmat = perfFromConfmat(confmat,lblind)

                    ps = pcm.get('perfs_CV', None)
                    perf_from_confmat = recalcPerfFromCV(ps,lblind)
                    #confmats = [p[-1] for p in ps]
                    #perf_from_confmat = perfFromConfmat(confmats,lblind)

                else:
                    pcm = mult_clf_output['XGB_analysis_versions']['all_present_features']['perf_dict']
                    perfs = pcm['perf_aver']

                ax = subaxs[-2]
                ax.set_xlim(50,101)
                ax.set_ylim(-1,101)
                if lblind == 0:
                    ax.scatter( [perfs[1] * 100], [ perfs[0] * 100 ], c='red',
                               marker = marker_mean, s=perf_marker_size )
                if perf_from_confmat is not None:
                    ax.scatter( [perf_from_confmat[1] * 100], [ perf_from_confmat[0] * 100 ], c='green',
                               marker = marker_mean, s=perf_marker_size)
                ax.set_xlabel('spec')
                ax.set_ylabel('sens')
                ax.set_title('performance')

                #plt.gcf().suptitle(f'{prefix}: lblind = {label_str} (lblind={lblind})')
                #pdf.savefig()
                #plt.close()

    assert len(fshs) == 1
    #for rn,a in outputs_grouped.items():
    #    for lblind in range(scores_per_class.shape[0] ):
    rni_,rn_,fsh_,lblind_,bias_,maxs_ = zip(*outs_mb)
    maxs_a = np.array(maxs_)
    biases_a = np.array(bias_)

    from plots import plotErrorBarStrings
    #print(outs_mb)

    for  i,(rni,rn,fsh,lblind,bias,maxs) in enumerate(outs_mb):
        if show_abs_plots:
            lm = 4
        else:
            lm = 2
        inds = np.where( (np.array(lblind_) == lblind) )[0]
        #\ & (np.array(rn_) == rn) )[0]
        assert len(inds) == len(outputs_grouped)
        biases_cur = biases_a[inds]
        #print('rnis ',np.array(rni_)[inds] )
        bm = np.max( biases_cur )
        mm = np.max( maxs_a[inds,:] )
        biasname = 'bias'
        if bm > mm:
            coef = mm / bm
            biases_cur = np.array(biases_cur) * coef
            biasname += f'/{coef:.3f}'
            print('bias correction to with coef',coef)

        for axi,ax in enumerate(axs[lblind,:lm] ):

            # this way the label will be indeed accurate for all columns
            #if axi == 0:
            #    if show_max:
            #        mm = np.max( maxs_a[inds,:2] )
            #    else:
            #        mm = np.max( maxs_a[inds,0] )
            #elif axi == 1:
            #    mm = np.max( maxs_a[inds,2] )
            #elif axi == 2:
            #    if show_max:
            #        mm = np.max( maxs_a[inds,2:4] )
            #    else:
            #        mm = np.max( maxs_a[inds,2] )
            #elif axi == 3:
            #    mm = np.max( maxs_a[inds,5] )

            bias_cur = biases_cur[rni]
            #print(biasname, bias_cur)
            plotErrorBarStrings(ax,[biasname],[bias_cur],xerr=None,
                same_sets_only = 0,
                add_args={'marker':marker_mean, 'color':'black',
                        'alpha':alpha*0.9, 'markersize':markersize} )

    #stats['names'] = names
    #stats['aux'] = aux_res
    #stats['names_display'] = names_display
    #stats['mean'] = mean_scores
    #stats['sum'] = sum_scores
    #stats['std'] = std_scores
    #stats['max'] = max_scores
    #stats['inds_lists'] = inds_lists

    # mutli-subjects on one? or across fshs?
    #for lblind in range(scores_per_class.shape[0] ):
    ## assume same number of scores in all subjects
    #    for fshi,fsh in enumerate(fshs):
    #        stats = [ feat_imp_stats for (rn,fsh_,lblind_ , feat_imp_stats ) in stats_per_all \
    #            if fsh_ == fsh and lblind_ == lblind]
    #        subaxs = axs[lblind,:]
    #        # make sure we have same order
    #        #for st in stats[1:]:
    #        #    assert tuple(st['names']) == tuple( stats[0]['names']   )
    #        #for st in stats[1:]:
    #        #    assert tuple(st['names_display']) == tuple( stats[0]['names_display']   )
    #        #stats_av = {}
    #        #for k,kv in stats[0].items():
    #        #    #print(k,type(kv) )
    #        #    #if isinstance(kv,np.ndarray) and kv.dtype == np.float:
    #        #    if k in ['names', 'names_display', 'inds_lists', 'aux']:
    #        #        stats_av[k] = kv
    #        #    else:
    #        #        tmp = np.array([ st[k] for st in stats ] )
    #        #        print(tmp.shape, tmp.dtype)
    #        #        stats_av[k] = np.mean(tmp  , axis=0 )
    #        #    #else:

    #        #plotFeatImpStats(feat_groups_all, stats_av, axs= subaxs[:2],
    #        #                bias=bias, color=colors_fsh[fshi], bar=False,
    #        #                    markersize=markersize * 1.5, show_max = show_max,
    #        #                    show_std = show_std, marker_mean='x', alpha=alpha_over_subj)

    #        for st in stats:
    #            plotFeatImpStats(feat_groups_all, st, axs= subaxs[:2],
    #                            bias=bias, color=colors_fsh[fshi], bar=False,
    #                                markersize=markersize * 1.5, show_max = show_max,
    #                                show_std = show_std, marker_mean='x', alpha=alpha_over_subj)


    plt.tight_layout()

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize)
    if pdf is not None:
        pdf.savefig()
        plt.close()

    return axs, outs

def plotFeatSignifEBM_list(pdf, outputs_grouped, fshs=['interpret_EBM'],
                       figname_prefix='',
                       n_individ_feats_show=4, roi_labels = None,
                       chnames_LFP = None, body_side='L', hh=8, ww = None,
                       separate_by_band = False,
                        separate_by_band2 = True,
                        suptitle = None, suptitle_fontsize=20,
                      tickfontsize = 10, show_bias=False,
                            use_best_LFP=False, markersize=10, show_max = True,
                           merge_Hjorth = False,
                            Hjorth_diff_color=False,
                            show_std = True, average_over_subjects = True,
                            alpha = 0.8, alpha_over_subj = 1., use_full_scores=False,
                            featsel_on_VIF=True, show_abs_plots=False,
                            reconstruct_from_VIF = False,
                            marker_mean = 'o', marker_max = 'x',
                            allow_dif_feat_group_sets = 0,
                            axs=None, grand_average_per_feat_type=1,
                            featsel_feat_subset_name='all', perf_marker_size = 25,
                            cross_source_groups = False, indivd_imp_xtick_pad=-300,
                          max_nfeats_to_sum = 20, legend_loc = 'lower right' ):
    '''
    fshs -- names of featsel method to use
    '''
    if isinstance(fshs , str):
        fshs = [fshs]

    #fsh = 'XGB_Shapley'
    if len(fshs) == 1:
        colors_fsh = len(fshs) * [None]
        print('Setting empty main color')
    else:
        colors_fsh = ['blue','red','green', 'purple']


    import utils_postprocess_HPC as postp
    import matplotlib.pyplot as plt


    from collections.abc import Iterable

    assert not reconstruct_from_VIF
    assert not use_full_scores

    #for rn,a in outputs_grouped:
    a = list(outputs_grouped.values())[0]
    (prefix,grp,int_type), mult_clf_output = a
    featsel_per_method             = mult_clf_output['featsel_per_method']
    featnames                      = mult_clf_output['feature_names_filtered']
    #class_labels_good_for_classif  = mult_clf_output['class_labels_good']
    class_labels_good_for_classif  = mult_clf_output['class_labels_good_for_classif']

    VIF_truncation = mult_clf_output.get('VIF_truncation',{})
    colinds_good_VIFsel  = VIF_truncation.get('colinds_good_VIFsel',None)
    if colinds_good_VIFsel is not None and featsel_on_VIF:
        featnames = np.array(featnames)[colinds_good_VIFsel]

    subskip_fit = mult_clf_output['pars'].get('subskip_fit',1)
    subskip_fit = int(subskip_fit)
    assert subskip_fit is not None


    #assert len(featnames_list) == len(fshs)
    #if isinstance(featnames_list, Iterable) and \
    #        (isinstance(featnames_list[0], str) ):
    #    featnames_list = [featnames_list]

    nscores = 1

    #if isinstance(hh,str) and hh == 'auto':
    #    hh = len(featnames) / 200

    nr = nscores;
    #if ww is None:
    #    ww = 2 + 3 + 5;
    #nc = 1 + 1 + 1;
    #width_ratios = [1,0.4,3]
    if ww is None:
        ww = 2 + 3 + 5;
    nc = 1 + 1 + 1 + 1;
    width_ratios = [1,1,0.4,3]

    if axs is None:
        fig,axs = plt.subplots(nr,nc, figsize = (nc*ww,nr*hh),
                                gridspec_kw={'width_ratios': width_ratios} );
        plt.subplots_adjust(top=1-0.02)
        axs = axs.reshape((nr,nc))
    else:
        assert axs.shape == (nr,nc),  ( axs.shape, (nr,nc)   )


    cmap = plt.cm.get_cmap('tab20', 20)
    #cmap( (ri0 + i*ri1) % 20)

    from featlist import getFeatIndsRelToOnlyOneLFPchan
    from featlist import getChnamesFromFeatlist

    outs = []

    stats_per_all = []
    for rn,a in outputs_grouped.items():
        (prefix,grp,int_type), mult_clf_output = a
        featsel_per_method   = mult_clf_output['featsel_per_method']
        featnames            = mult_clf_output['feature_names_filtered'].copy()
        class_label_names    = mult_clf_output.get('class_label_names_ordered',None)

        VIF_truncation = mult_clf_output.get('VIF_truncation',{})
        colinds_good_VIFsel  = VIF_truncation.get('colinds_good_VIFsel',None)
        #print('colinds_good_VIFsel ',colinds_good_VIFsel)

        assert not ( (colinds_good_VIFsel is not None) and use_best_LFP), 'requires more thinking'


        if colinds_good_VIFsel is not None and featsel_on_VIF:
            featnames_sub = np.array(featnames)[colinds_good_VIFsel]
        else:
            featnames_sub = featnames


        featnames_nice_sub = utils.nicenFeatNames(featnames_sub, {'kk':roi_labels},['kk'])

        class_labels_good_for_classif  = mult_clf_output['class_labels_good_for_classif']
        if class_label_names is None:
            class_labels_good  = mult_clf_output['class_labels_good']
            revdict = mult_clf_output['revdict']

            from sklearn import preprocessing
            lab_enc = preprocessing.LabelEncoder()
            # just skipped class_labels_good
            lab_enc.fit(class_labels_good)
            class_labels_good_for_classif = lab_enc.transform(class_labels_good)
            class_label_ids = lab_enc.inverse_transform( np.arange( len(set(class_labels_good_for_classif)) ) )
            class_label_names = [revdict[cli] for cli in class_label_ids]
            #print(class_label_names)

        if chnames_LFP is None:
            chnames_LFP = getChnamesFromFeatlist(featnames_sub, mod='LFP')


        if use_best_LFP:
            chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
            new_channel_name_templ='LFP007'
            #print(chn_LFP)
            feat_inds_curLFP, feat_inds_except_curLFP = \
                getFeatIndsRelToOnlyOneLFPchan(featnames,
                    chnpart=chn_LFP, chnames_LFP=chnames_LFP,
                           new_channel_name_templ=new_channel_name_templ,
                            mainLFPchan=chn_LFP)
            if colinds_good_VIFsel is not None:
                print('Intersecting VIF and curLFP inds')
                feat_inds_curLFP = np.intersect1d(feat_inds_curLFP,colinds_good_VIFsel)
            featnames_sub = np.array(featnames)[feat_inds_curLFP]
            chnames_LFP_cur = [new_channel_name_templ]
            #print(rn,chn_LFP, len(featnames) )
        else:
            chnames_LFP_cur = chnames_LFP

        #####
        #exogs_list = VIF_truncation['exogs_list']
        #VIF_truncation['VIF_search_worst']
        ##VIF_truncation['X_for_VIF_shape'] = X_for_VIF.shape
        #colinds_bad         = VIF_truncation[ 'colinds_bad_VIFsel']
        #VIFsel_linreg_objs  = VIF_truncation['VIFsel_linreg_objs']
        #VIFsel_featsets_list  = VIF_truncation[ 'VIFsel_featsets_list']


        for fshi,fsh in enumerate(fshs):
            fspm = featsel_per_method[fsh][featsel_feat_subset_name]
            #featnames = featnames_list[fshi]

            featnames_EBM = fspm['feature_names']
            non_interact_featis = np.where( [featnames_EBM[ind].find(' x ') < 0 \
                                for ind in range(len(featnames_EBM)) ])[0]
            featnaems_EBM_non_interact = np.array(featnames_EBM)[non_interact_featis]


            sortinds = None
            if featsel_feat_subset_name == 'VIFsel':
                assert set(featnaems_EBM_non_interact) == set(featnames_nice_sub), (featnames_nice_sub, featnaems_EBM_non_interact)
                sortinds = [featnames_nice_sub.index(featn)  for featn in featnaems_EBM_non_interact ]
                featnaems_EBM_non_interact = np.array(featnaems_EBM_non_interact)[sortinds]
                assert tuple(featnaems_EBM_non_interact) == tuple(featnames_nice_sub), (featnames_nice_sub, featnaems_EBM_non_interact)
                featnaems_EBM_non_interact_notnice = featnames_sub
            elif featsel_feat_subset_name == 'all':
                featnames_nice_all = utils.nicenFeatNames(featnames, {'kk':roi_labels},['kk'])
                assert set(featnaems_EBM_non_interact) == set(featnames_nice_all), (featnames_nice_all, featnaems_EBM_non_interact)
                sortinds = [featnames_nice_all.index(featn)  for featn in featnaems_EBM_non_interact ]
                featnaems_EBM_non_interact = np.array(featnaems_EBM_non_interact)[sortinds]
                assert tuple(featnaems_EBM_non_interact) == tuple(featnames_nice_all), (featnames_nice_all, featnaems_EBM_non_interact)
                featnaems_EBM_non_interact_notnice = featnames

            scores = fspm.get('scores',None)
            assert scores is not None
            scores = np.array(scores)[sortinds]
            scores_per_class = scores[None,:]

                #if reconstruct_from_VIF:
                #    scores_per_class_VIF = scores_per_class
                #    scores_per_class_reconstructed = utsne.reconstructFullScoresFromVIFScores(scores_per_class_VIF,
                #        len(featnames), colinds_bad,colinds_good_VIFsel,
                #        VIFsel_featsets_list, VIFsel_linreg_objs, exogs_list )
                #print( fspm.keys(), scores.shape )
            print('scores.shape = ',scores.shape)
            #import pdb;pdb.set_trace()

            ###############################################
            # make plots for every class label
            for lblind in range(scores_per_class.shape[0] ):
                # select points where true class is like the current one
                #ptinds = np.where(class_labels_good_for_classif == lblind)[0]
                #classid_enconded = lblind
                #scores_cur = np.mean(scores[ptinds,lblind,0:-1], axis=0)

                scores_cur = scores_per_class[lblind]
                if use_full_scores:
                    scores_full_cur = scores[:,lblind,:]
                label_str = class_label_names[lblind]

                ###############################

                    #display(feat_groups_all)
                clrs,feature_groups_names,feat_groups_all  = \
                    prepareFeatGroups(featnaems_EBM_non_interact,body_side,
                                      roi_labels,cmap,
                                      chnames_LFP, separate_by_band,
                                      separate_by_band2,
                                      merge_Hjorth,
                                      Hjorth_diff_color,
                                      grand_average_per_feat_type,
                                      cross_source_groups)


                #display(feat_groups_all)
                feat_imp_stats = mergeScores(scores_cur,
                    featnaems_EBM_non_interact_notnice,
                    feat_groups_all,
                    feature_names_subset=None,
                    feature_groups_names=feature_groups_names, aux=clrs,
                    max_nfeats_to_sum=max_nfeats_to_sum )
                stats_per_all += [ (rn,fsh,lblind , feat_imp_stats ) ]
                ####################################

                if isinstance(rn,tuple):
                    rn_ = rn[0]
                else:
                    rn_ = rn
                #outs +=   [ (rn_,prefix,grp,int_type,fsh,scores_per_class,bias)  ]

                outs +=   [ (rn_,prefix,grp,int_type,fsh,
                             featnaems_EBM_non_interact,label_str,
                             scores_per_class[lblind],feat_imp_stats )  ]

                ####################################
                subaxs = axs[lblind,:]
                #subaxs = axs[0,:]

                bias = None

                print( f'len(feat_groups_all) = {len(feat_groups_all)}, len(featnames_sub) = {len(featnames_sub)}')
                #import pdb;pdb.set_trace()
                plotFeatImpStats(feat_groups_all, feat_imp_stats, axs= subaxs[:2],
                                bias=None, color=colors_fsh[fshi], bar=False,
                                 markersize=markersize, show_max = show_max,
                                 show_std = show_std, alpha=alpha,
                                 show_sum=1,
                                 plot_signed=True,
                                 marker_mean= marker_mean,
                                 marker_max = marker_max,
                                 skip_ax_inds=[1],
                                 same_sets_only= not allow_dif_feat_group_sets)
                subaxs[0].tick_params(axis='y', labelsize=  tickfontsize)
                subaxs[0].set_title( f'{subaxs[0].get_title()}  {label_str}' )

                #fig.canvas.draw() # needed to set yticks

                    #fig.canvas.draw() # needed to set yticks

                #subaxs[0].tick_params(axis='y', labelsize=  tickfontsize)
                #plt.tight_layout()
                #ax.set_title(  )
                #pdf.savefig()


                #plt.figure(figsize = (12,10))
                #ax = plt.gca()
                ax = subaxs[-1]
                #postp.plotFeatureImportance(ax, featnames_sub, scores[ptinds,lblind,:], 'XGB_Shapley')
                sort_individ_feats = fshi == 0
                plotFeatureImportance(ax, featnaems_EBM_non_interact,
                                            scores_per_class[lblind,:],
                                            'labind_vs_score',
                                            color=colors_fsh[fshi],
                                            sort = sort_individ_feats,
                                            nshow=n_individ_feats_show)
                ax.set_title( f'{figname_prefix}: ' + ax.get_title() + f'_lblind = {label_str} (lblind={lblind}):  {fsh}' )
                ax.tick_params(axis="x",direction="in", pad=indivd_imp_xtick_pad, labelsize=tickfontsize)

                ###############3
                assert not use_best_LFP

                perf_from_confmat = None
                if use_best_LFP:
                    #chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
                    pcm = mult_clf_output['XGB_analysis_versions'][f'all_present_features_only_{chn_LFP}']['perf_dict']
                    perfs = pcm['perf_aver']


                    #confmat = pcm.get('confmat', None)
                    ##print(pcm.keys())
                    #if confmat is None:
                    #    confmat = pcm.get('confmat_aver', None)
                    #    #print('using confmat_aver')
                    #perf_from_confmat = perfFromConfmat(confmat,lblind)

                    ps = pcm.get('perfs_CV', None)
                    perf_from_confmat = recalcPerfFromCV(ps,lblind)
                    #confmats = [p[-1] for p in ps]
                    #perf_from_confmat = perfFromConfmat(confmats,lblind)

                else:
                    pcm = mult_clf_output['XGB_analysis_versions']['all_present_features']['perf_dict']
                    perfs = pcm['perf_aver']

                ax = subaxs[-2]
                ax.set_xlim(50,101)
                #ax.set_ylim(-1,101)
                ax.set_ylim(30,101)
                if lblind == 0:
                    ax.scatter( [perfs[1] * 100], [ perfs[0] * 100 ], c='red',
                               marker = marker_mean, s=perf_marker_size,
                               label='XGB_perf_recalc')
                if perf_from_confmat is not None:
                    ax.scatter( [perf_from_confmat[1] * 100], [ perf_from_confmat[0] * 100 ], c='green',
                               marker = marker_mean, s=perf_marker_size,
                               label='XGB_perf_recalc')

                perfs_actual = fspm['perf']
                ax.scatter( [perfs_actual[1] * 100],
                           [ perfs_actual[0] * 100 ], c='purple',
                            marker = marker_mean, s=perf_marker_size,
                           label='EMB_perf')

                ax.legend(loc=legend_loc)

                ax.set_xlabel('spec')
                ax.set_ylabel('sens')
                ax.set_title('performance')

    plt.tight_layout()

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize)
    if pdf is not None:
        pdf.savefig()
        plt.close()

    return axs, outs

def plotFeatSignifSHAP_ICA_list(pdf, outputs_grouped, fshs=['XGB_Shapley'],
                       figname_prefix='',
                       n_individ_feats_show=8, roi_labels = None,
                       chnames_LFP = None, body_side='L', hh=8, ww = None,
                       separate_by_band = False,
                        separate_by_band2 = True,
                            suptitle = None, suptitle_fontsize=20,
                      tickfontsize = 10, show_bias=False,
                            use_best_LFP=False, markersize=10, show_max = True,
                           merge_Hjorth = False,
                            Hjorth_diff_color=False,
                            show_std = True, average_over_subjects = True,
                            alpha = 0.8, alpha_over_subj = 1., use_full_scores=False,
                            featsel_on_VIF=True, show_abs_plots=False,
                            reconstruct_from_VIF = False,
                            marker_mean = 'o', marker_max = 'x',
                            axs=None, grand_average_per_feat_type=1,
                            perf_marker_size = 25,
                           cross_source_groups = False,
                           indivd_imp_xtick_pad =  -300  ):
    '''
    fshs -- names of featsel method to use
    '''
    if isinstance(fshs , str):
        fshs = [fshs]

    #fsh = 'XGB_Shapley'
    if len(fshs) == 1:
        colors_fsh = len(fshs) * [None]
        print('Setting empty main color')
    else:
        colors_fsh = ['blue','red','green', 'purple']


    import utils_postprocess_HPC as postp
    import matplotlib.pyplot as plt


    from collections.abc import Iterable


    #for rn,a in outputs_grouped:
    a = list(outputs_grouped.values())[0]
    (prefix,grp,int_type), mult_clf_output = a
    featsel_per_method             = mult_clf_output['featsel_per_method']
    #featnames                      = mult_clf_output['feature_names_filtered']
    featnames                      = mult_clf_output['featnames_for_fit']
    #class_labels_good_for_classif  = mult_clf_output['class_labels_good']
    class_labels_good_for_classif  = mult_clf_output['class_labels_good_for_classif']

    #VIF_truncation = mult_clf_output.get('VIF_truncation',None)
    #colinds_good_VIFsel  = VIF_truncation.get('colinds_good_VIFsel',None)
    #if colinds_good_VIFsel is not None and featsel_on_VIF:
    #    featnames = np.array(featnames)[colinds_good_VIFsel]

    subskip_fit = mult_clf_output['pars'].get('subskip_fit',1)
    subskip_fit = int(subskip_fit)
    assert subskip_fit is not None


    #assert len(featnames_list) == len(fshs)
    #if isinstance(featnames_list, Iterable) and \
    #        (isinstance(featnames_list[0], str) ):
    #    featnames_list = [featnames_list]



    fspm_def = featsel_per_method[ fshs[0]  ]
    scores = fspm_def.get('scores',None)
    if scores is None:
        scores_per_class_def = fspm_def.get('scores_av',None)
    else:
        assert scores.shape[-1] - 1 == len(featnames), (scores.shape[-1] , len(featnames))
        scores_per_class_def = utsne.getScoresPerClass(class_labels_good_for_classif, scores)
    nscores = len(scores_per_class_def)

    #if isinstance(hh,str) and hh == 'auto':
    #    hh = len(featnames) / 200

    nr = nscores;
    if ww is None:
        ww = 2 + 3 + 5;
    if show_abs_plots:
        nc = 2 + 2 + 1 + 1; #nc= len(scores_stats) - 2;
        width_ratios = [1,1,1,1,0.4,3]
    else:
        nc = 2 + 1 + 1;
        width_ratios = [1,1,0.4,3]

    if axs is None:
        fig,axs = plt.subplots(nr,nc, figsize = (nc*ww,nr*hh),
                                gridspec_kw={'width_ratios': width_ratios} );
        plt.subplots_adjust(top=1-0.02)
    else:
        assert axs.shape == (nr,nc)


    cmap = plt.cm.get_cmap('tab20', 20)
    #cmap( (ri0 + i*ri1) % 20)

    from featlist import getFeatIndsRelToOnlyOneLFPchan
    from featlist import getChnamesFromFeatlist

    #biases = {}
    #maxs_list = {}
    outs = []

    outs_mb = []

    stats_per_all = []
    for rni,(rn,a) in enumerate(outputs_grouped.items() ):
        (prefix,grp,int_type), mult_clf_output = a
        featsel_per_method   = mult_clf_output['featsel_per_method']
        #featnames            = mult_clf_output['feature_names_filtered'].copy()
        featnames                      = mult_clf_output['featnames_for_fit']
        class_label_names    = mult_clf_output.get('class_label_names_ordered',None)

        #VIF_truncation = mult_clf_output.get('VIF_truncation',None)
        #colinds_good_VIFsel  = mult_clf_output['VIF_truncation'].get('colinds_good_VIFsel',None)
        #print('colinds_good_VIFsel ',colinds_good_VIFsel)

        #assert not ( (colinds_good_VIFsel is not None) and use_best_LFP), 'requires more thinking'




        #if colinds_good_VIFsel is not None and featsel_on_VIF:
        #    featnames_sub = np.array(featnames)[colinds_good_VIFsel]
        #else:
        #    featnames_sub = featnames
        featnames_sub = featnames


        featnames_nice_sub = utils.nicenFeatNames(featnames_sub, {'kk':roi_labels},['kk'])

        class_labels_good_for_classif  = mult_clf_output['class_labels_good_for_classif']
        if class_label_names is None:
            class_labels_good  = mult_clf_output['class_labels_good']
            revdict = mult_clf_output['revdict']

            from sklearn import preprocessing
            lab_enc = preprocessing.LabelEncoder()
            # just skipped class_labels_good
            lab_enc.fit(class_labels_good)
            class_labels_good_for_classif = lab_enc.transform(class_labels_good)
            class_label_ids = lab_enc.inverse_transform( np.arange( len(set(class_labels_good_for_classif)) ) )
            class_label_names = [revdict[cli] for cli in class_label_ids]
            #print(class_label_names)

        #if chnames_LFP is None:
        #    chnames_LFP = getChnamesFromFeatlist(featnames_sub, mod='LFP')


        #if use_best_LFP:
        #    chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
        #    new_channel_name_templ='LFP007'
        #    #print(chn_LFP)
        #    feat_inds_curLFP, feat_inds_except_curLFP = \
        #        getFeatIndsRelToOnlyOneLFPchan(featnames,
        #            chnpart=chn_LFP, chnames_LFP=chnames_LFP,
        #                   new_channel_name_templ=new_channel_name_templ,
        #                    mainLFPchan=chn_LFP)
        #    #if colinds_good_VIFsel is not None:
        #    #    print('Intersecting VIF and curLFP inds')
        #    #    feat_inds_curLFP = np.intersect1d(feat_inds_curLFP,colinds_good_VIFsel)
        #    featnames_sub = np.array(featnames)[feat_inds_curLFP]
        #    chnames_LFP_cur = [new_channel_name_templ]
        #    #print(rn,chn_LFP, len(featnames) )
        #else:
        #    chnames_LFP_cur = chnames_LFP

        #####
        #exogs_list = VIF_truncation['exogs_list']
        #VIF_truncation['VIF_search_worst']
        ##VIF_truncation['X_for_VIF_shape'] = X_for_VIF.shape
        #colinds_bad         = VIF_truncation[ 'colinds_bad_VIFsel']
        #VIFsel_linreg_objs  = VIF_truncation['VIFsel_linreg_objs']
        #VIFsel_featsets_list  = VIF_truncation[ 'VIFsel_featsets_list']


        for fshi,fsh in enumerate(fshs):
            fspm = featsel_per_method[fsh]
            #featnames = featnames_list[fshi]
            scores = fspm.get('scores',None)
            #print(scores.shape)
            if scores is None:
                if use_full_scores:
                    raise ValueError('we need full scores!')
                scores_per_class = fspm.get('scores_av',None)
                bias = fspm.get('scores_bias_av',None)

                #if reconstruct_from_VIF:
                #    scores_per_class_VIF = scores_per_class
                #    scores_per_class_reconstructed = utsne.reconstructFullScoresFromVIFScores(scores_per_class_VIF,
                #        len(featnames), colinds_bad,colinds_good_VIFsel,
                #        VIFsel_featsets_list, VIFsel_linreg_objs, exogs_list )

                #    featnames_sub = featnames
            else:
                if use_full_scores:
                    if len(set(class_labels_good_for_classif) ) == 2:
                        scores = scores[:,None,:]
                        scores = np.concatenate( [scores,scores], axis=1)
                    assert scores.ndim == 3, scores.shape

                    #if reconstruct_from_VIF:
                    #    scores_full_per_class_VIF = scores
                    #    scores_full_reconstructed = utsne.reconstructFullScoresFromVIFScores(scores_full_per_class_VIF,
                    #        len(featnames), colinds_bad,colinds_good_VIFsel,
                    #        VIFsel_featsets_list, VIFsel_linreg_objs, exogs_list )

                    #    scores = scores_full_reconstructed
                    #    featnames_sub = featnames

                assert scores.shape[-1] - 1 == len(featnames_sub), (scores.shape[-1] , len(featnames_sub))
                # XGB doc: Note the final column is the bias term
                scores_per_class, bias = utsne.getScoresPerClass(class_labels_good_for_classif,
                                                                 scores, ret_bias=1)

                #if reconstruct_from_VIF:
                #    scores_per_class_VIF = scores_per_class
                #    scores_per_class_reconstructed = utsne.reconstructFullScoresFromVIFScores(scores_per_class_VIF,
                #        len(featnames), colinds_bad,colinds_good_VIFsel,
                #        VIFsel_featsets_list, VIFsel_linreg_objs, exogs_list )
                #print( fspm.keys(), scores.shape )
            print('scores.shape = ',scores.shape)
            #import pdb;pdb.set_trace()

            assert scores_per_class.shape[-1] == len(featnames_sub),  (scores_per_class.shape[-1], len(featnames_sub))

            ###############################################
            # make plots for every class label
            for lblind in range(scores_per_class.shape[0] ):
                # select points where true class is like the current one
                #ptinds = np.where(class_labels_good_for_classif == lblind)[0]
                #classid_enconded = lblind
                #scores_cur = np.mean(scores[ptinds,lblind,0:-1], axis=0)

                scores_cur = scores_per_class[lblind]
                if use_full_scores:
                    scores_full_cur = scores[:,lblind,:]
                label_str = class_label_names[lblind]

                ###############################




                    #display(feat_groups_all)

                clrs = [ cmap(clri) for clri in range(len(featnames_sub)) ]
                featnames_sub = np.array(featnames_sub)
                feat_groups_all =  [ f'^{fn}$' for fn in featnames_sub ]
                feat_groups_all      = np.array(feat_groups_all)
                feature_groups_names = np.array(featnames_sub)
                #clrs,feature_groups_names,feat_groups_all  = \
                #    prepareFeatGroups(featnames_sub,body_side,
                #                      roi_labels,cmap,
                #                      chnames_LFP, separate_by_band,
                #                      separate_by_band2,
                #                      merge_Hjorth,
                #                      Hjorth_diff_color,
                #                      grand_average_per_feat_type,
                #                      cross_source_groups)

                    #############################################

                #display(feat_groups_all)
                if use_full_scores:
                    feat_imp_stats = mergeScores2(scores_full_cur,
                                                  class_labels_good_for_classif,
                        lblind, featnames_sub, feat_groups_all,
                        feature_names_subset=None,
                        feature_groups_names=feature_groups_names, aux=clrs)
                else:
                    feat_imp_stats = mergeScores(scores_cur, featnames_sub,
                        feat_groups_all,
                        feature_names_subset=None,
                        feature_groups_names=feature_groups_names, aux=clrs)
                stats_per_all += [ (rn,fsh,lblind , feat_imp_stats ) ]
                ####################################

                if isinstance(rn,tuple):
                    rn_ = rn[0]
                else:
                    rn_ = rn
                #outs +=   [ (rn_,prefix,grp,int_type,fsh,scores_per_class,bias)  ]

                outs +=   [ (rn_,prefix,grp,int_type,fsh,featnames_nice_sub,label_str,scores_per_class[lblind],feat_imp_stats )  ]

                ####################################
                subaxs = axs[lblind,:]

                #if not show_bias:
                bias_to_plot = None
                #else:
                #    bias_to_plot = bias[lblind]

                maxs = []
                print( f'len(feat_groups_all) = {len(feat_groups_all)}, len(featnames_sub) = {len(featnames_sub)}')
                #import pdb;pdb.set_trace()
                #same_sets_only should be zero when I run giving axes with some stuff on them already,
                # especially if I use biases
                max0,max00,max1 = plotFeatImpStats(feat_groups_all, feat_imp_stats, axs= subaxs[:2],
                                bias=bias_to_plot, color=colors_fsh[fshi], bar=False,
                                 markersize=markersize, show_max = show_max,
                                 show_std = show_std, alpha=alpha,
                                 plot_signed=True,
                                 same_sets_only = 0,
                                 marker_mean= marker_mean,
                                 marker_max = marker_max)
                maxs +=  [max0,max00,max1]
                subaxs[0].tick_params(axis='y', labelsize=  tickfontsize)
                subaxs[0].set_title( f'{subaxs[0].get_title()}  {label_str}' )

                #fig.canvas.draw() # needed to set yticks

                if show_abs_plots:
                    max2,max22,max3 = plotFeatImpStats(feat_groups_all, feat_imp_stats, axs= subaxs[2:4],
                                    bias=bias_to_plot, color=colors_fsh[fshi], bar=False,
                                    markersize=markersize, show_max = show_max,
                                    show_std = show_std, alpha=alpha,plot_signed=False,
                                    same_sets_only = 0,
                                     marker_mean= marker_mean, marker_max = marker_max)
                    #subaxs[2].set_yticks([])
                    subaxs[2].set_title( f'{subaxs[2].get_title()}  {label_str}' )
                    maxs +=  [max2,max22,max3]


                rny = rn
                if isinstance(rn,tuple):
                    rny = rn[0]
                outs_mb += [(rni,rny,fsh,lblind,bias[lblind],maxs)]
                    #fig.canvas.draw() # needed to set yticks

                #subaxs[0].tick_params(axis='y', labelsize=  tickfontsize)
                #plt.tight_layout()
                #ax.set_title(  )
                #pdf.savefig()


                #plt.figure(figsize = (12,10))
                #ax = plt.gca()
                ax = subaxs[-1]
                #postp.plotFeatureImportance(ax, featnames_sub, scores[ptinds,lblind,:], 'XGB_Shapley')
                sort_individ_feats = fshi == 0
                plotFeatureImportance(ax, featnames_nice_sub,
                                            scores_per_class[lblind,:],
                                            'labind_vs_score',
                                            color=colors_fsh[fshi],
                                            sort = sort_individ_feats,
                                            nshow=n_individ_feats_show)
                ax.set_title( f'{figname_prefix}: ' + ax.get_title() + f'_lblind = {label_str} (lblind={lblind}):  {fsh}' )
                ax.tick_params(axis="x",direction="in", pad=indivd_imp_xtick_pad, labelsize=tickfontsize)

                ###############3

                perf_from_confmat = None
                #if use_best_LFP:
                #    #chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
                #    pcm = mult_clf_output['XGB_analysis_versions'][f'all_present_features_only_{chn_LFP}']['perf_dict']
                #    perfs = pcm['perf_aver']


                #    #confmat = pcm.get('confmat', None)
                #    ##print(pcm.keys())
                #    #if confmat is None:
                #    #    confmat = pcm.get('confmat_aver', None)
                #    #    #print('using confmat_aver')
                #    #perf_from_confmat = perfFromConfmat(confmat,lblind)

                #    ps = pcm.get('perfs_CV', None)
                #    perf_from_confmat = recalcPerfFromCV(ps,lblind)
                #    #confmats = [p[-1] for p in ps]
                #    #perf_from_confmat = perfFromConfmat(confmats,lblind)

                #else:
                #    pcm = mult_clf_output['XGB_analysis_versions']['all_present_features']['perf_dict']
                #    perfs = pcm['perf_aver']
                pcm = mult_clf_output['XGB_analysis_versions']['all_present_features']['perf_dict']
                perfs = pcm['perf_aver']

                ax = subaxs[-2]
                ax.set_xlim(50,101)
                ax.set_ylim(-1,101)
                if lblind == 0:
                    ax.scatter( [perfs[1] * 100], [ perfs[0] * 100 ], c='red',
                               marker = marker_mean, s=perf_marker_size )
                if perf_from_confmat is not None:
                    ax.scatter( [perf_from_confmat[1] * 100], [ perf_from_confmat[0] * 100 ], c='green',
                               marker = marker_mean, s=perf_marker_size)
                ax.set_xlabel('spec')
                ax.set_ylabel('sens')
                ax.set_title('performance')

                #plt.gcf().suptitle(f'{prefix}: lblind = {label_str} (lblind={lblind})')
                #pdf.savefig()
                #plt.close()

    assert len(fshs) == 1
    #for rn,a in outputs_grouped.items():
    #    for lblind in range(scores_per_class.shape[0] ):
    rni_,rn_,fsh_,lblind_,bias_,maxs_ = zip(*outs_mb)
    maxs_a = np.array(maxs_)
    biases_a = np.array(bias_)

    from plots import plotErrorBarStrings
    #print(outs_mb)

    for  i,(rni,rn,fsh,lblind,bias,maxs) in enumerate(outs_mb):
        if show_abs_plots:
            lm = 4
        else:
            lm = 2
        inds = np.where( (np.array(lblind_) == lblind) )[0]
        #\ & (np.array(rn_) == rn) )[0]
        assert len(inds) == len(outputs_grouped)
        biases_cur = biases_a[inds]
        #print('rnis ',np.array(rni_)[inds] )
        bm = np.max( biases_cur )
        mm = np.max( maxs_a[inds,:] )
        biasname = 'bias'
        if bm > mm:
            coef = mm / bm
            biases_cur = np.array(biases_cur) * coef
            biasname += f'/{coef:.3f}'
            print('bias correction to with coef',coef)

        for axi,ax in enumerate(axs[lblind,:lm] ):

            # this way the label will be indeed accurate for all columns
            #if axi == 0:
            #    if show_max:
            #        mm = np.max( maxs_a[inds,:2] )
            #    else:
            #        mm = np.max( maxs_a[inds,0] )
            #elif axi == 1:
            #    mm = np.max( maxs_a[inds,2] )
            #elif axi == 2:
            #    if show_max:
            #        mm = np.max( maxs_a[inds,2:4] )
            #    else:
            #        mm = np.max( maxs_a[inds,2] )
            #elif axi == 3:
            #    mm = np.max( maxs_a[inds,5] )

            bias_cur = biases_cur[rni]
            #print(biasname, bias_cur)
            plotErrorBarStrings(ax,[biasname],[bias_cur],xerr=None,
                same_sets_only = 0,
                add_args={'marker':marker_mean, 'color':'black',
                        'alpha':alpha*0.9, 'markersize':markersize} )

    #stats['names'] = names
    #stats['aux'] = aux_res
    #stats['names_display'] = names_display
    #stats['mean'] = mean_scores
    #stats['sum'] = sum_scores
    #stats['std'] = std_scores
    #stats['max'] = max_scores
    #stats['inds_lists'] = inds_lists

    # mutli-subjects on one? or across fshs?
    #for lblind in range(scores_per_class.shape[0] ):
    ## assume same number of scores in all subjects
    #    for fshi,fsh in enumerate(fshs):
    #        stats = [ feat_imp_stats for (rn,fsh_,lblind_ , feat_imp_stats ) in stats_per_all \
    #            if fsh_ == fsh and lblind_ == lblind]
    #        subaxs = axs[lblind,:]
    #        # make sure we have same order
    #        #for st in stats[1:]:
    #        #    assert tuple(st['names']) == tuple( stats[0]['names']   )
    #        #for st in stats[1:]:
    #        #    assert tuple(st['names_display']) == tuple( stats[0]['names_display']   )
    #        #stats_av = {}
    #        #for k,kv in stats[0].items():
    #        #    #print(k,type(kv) )
    #        #    #if isinstance(kv,np.ndarray) and kv.dtype == np.float:
    #        #    if k in ['names', 'names_display', 'inds_lists', 'aux']:
    #        #        stats_av[k] = kv
    #        #    else:
    #        #        tmp = np.array([ st[k] for st in stats ] )
    #        #        print(tmp.shape, tmp.dtype)
    #        #        stats_av[k] = np.mean(tmp  , axis=0 )
    #        #    #else:

    #        #plotFeatImpStats(feat_groups_all, stats_av, axs= subaxs[:2],
    #        #                bias=bias, color=colors_fsh[fshi], bar=False,
    #        #                    markersize=markersize * 1.5, show_max = show_max,
    #        #                    show_std = show_std, marker_mean='x', alpha=alpha_over_subj)

    #        for st in stats:
    #            plotFeatImpStats(feat_groups_all, st, axs= subaxs[:2],
    #                            bias=bias, color=colors_fsh[fshi], bar=False,
    #                                markersize=markersize * 1.5, show_max = show_max,
    #                                show_std = show_std, marker_mean='x', alpha=alpha_over_subj)


    plt.tight_layout()

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize)
    if pdf is not None:
        pdf.savefig()
        plt.close()

    return axs, outs


def prepareFeatGroups(featnames_sub,body_side, roi_labels,cmap, chnames_LFP=None,
                      separate_by_band=True, separate_by_band2=True,
                     merge_Hjorth=True, Hjorth_diff_color=True,
                     grand_average_per_feat_type=False, cross_source_groups=False ):
    '''
    returns
       clrs -- color per group
       feat_groups_all -- list of regexes
    '''
    import globvars as gv
    ftype_info = utils.collectFeatTypeInfo(featnames_sub)

    feat_groups_all = []
    feature_groups_names = []
    clrs = []

    clri = 0
    if grand_average_per_feat_type:
        feat_groups_basic = [f'^{ft}_.*' for ft in ftype_info['ftypes']]
        feat_groups_all+= feat_groups_basic
        feature_groups_names += feat_groups_basic
        clrs += [cmap(clri)] * len(feat_groups_basic)
    clri += 1

    ft = 'bpcorr'
    if 'bpcorr' in ftype_info['ftypes']:
        feat_groups_two_bands = [f'^{ft}_{fb1}_.*,{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
    #     feat_groups_two_bands = ['^bpcorr_gamma.*,tremor.*','^bpcorr_gamma.*,beta.*','^bpcorr_gamma.*,HFO.*',
    #                                 '^bpcorr_beta.*,tremor.*','^bpcorr_beta.*,gamma.*','^bpcorr_beta.*,HFO.*',
    #                                 '^bpcorr_tremor.*,beta.*','^bpcorr_tremor.*,gamma.*','^bpcorr_tremor.*,HFO.*']
        feat_groups_all += feat_groups_two_bands
        feature_groups_names += feat_groups_two_bands
        clrs += [cmap(clri)] * len(feat_groups_two_bands)

    from featlist import selFeatsRegexInds
    # self con
    ft = 'con'
    if ft in ftype_info['ftypes']:
        clri += 1  # here it makes sense since only one ft is used
        #        regex_same_LFP = r'.?.?corr.*(LFP.[0-9]+),.*\1.*'
        feat_groups_one_band = [ f'^{ft}_{fb}_(.*),'+ r'\1' for fb in ftype_info['fband_per_ftype'][ft] ]
        feat_groups_all += feat_groups_one_band
        feature_groups_names += [ f'{ft}_{fb}_self' for fb in ftype_info['fband_per_ftype'][ft] ]
        clrs += [cmap(clri)] * len(feat_groups_one_band)

    for ft in ['rbcorr', 'con']:
        if ft in ftype_info['ftypes']:
            #feat_groups_rbcorr_band = ['^rbcorr_tremor.*', '^rbcorr_beta.*',  '^rbcorr_gamma.*']
            feat_groups_one_band = [ f'^{ft}_{fb}_.*' for fb in ftype_info['fband_per_ftype'][ft] ]
            feat_groups_all += feat_groups_one_band
            feature_groups_names += feat_groups_one_band
            clrs += [cmap(clri)] * len(feat_groups_one_band)
    #feat_groups_all

    # allow HFO2 and high_beta  (allow numbers and one underscore in
    # the middle
    bnpattern = '[a-zA-Z0-9]+_?[a-zA-Z0-9]*'


    wasH = 0
    for ft in gv.noband_feat_types:
        if ft in ftype_info['ftypes']:
            wasH = True

    if wasH and merge_Hjorth:
        ft_templ = f'({"|".join(gv.noband_feat_types) })'
        a = [f'^{ft_templ}_LFP.*']
        feat_groups_all += a
        feature_groups_names += [  'Hjorth_LFP'  ]
        clrs += [cmap(clri)] * len(a)

    if wasH and not merge_Hjorth:
        for noband_type in gv.noband_feat_types:
        #ft_templ = f'({"|".join(gv.noband_feat_types) })'
            a = [f'^{noband_type}_LFP.*']
            feat_groups_all += a
            feature_groups_names += [  f'{noband_type}_LFP'  ]
            clrs += [cmap(clri)] * len(a)


    # Hjorth per parcel
    clri += 1
    from globvars import gp
    if roi_labels is not None:
        # main side (body)
        # get parcel indices of
        for grpn,parcel_list in gp.parcel_groupings_post.items():
            #feat_groups_cur = []
            #print(parcel_list)
            plws = utils.addSideToParcels(parcel_list, body_side)
            parcel_inds = [ roi_labels.index(parcel) for parcel in plws ]

            pas = '|'.join(map(str,parcel_inds) )
            chn_templ = f'msrc(R|L)_9_({pas})_c[0-9]+'

            if wasH and merge_Hjorth:
                if Hjorth_diff_color:
                    clri += 1
                ft_templ = f'({"|".join(gv.noband_feat_types) })'
                a = [f'^{ft_templ}_{chn_templ}']
                feat_groups_all += a
                feature_groups_names += [  f'Hjorth_{grpn}'  ]
                clrs += [cmap(clri)] * len(a)
                #clri += 1

            if wasH and not merge_Hjorth:
                if Hjorth_diff_color:
                    clri += 1
                #ft_templ = f'({"|".join(gv.noband_feat_types) })'
                for noband_type in gv.noband_feat_types:
                    a = [f'^{noband_type}_{chn_templ}']
                    feat_groups_all += a
                    feature_groups_names += [  f'{noband_type}_{grpn}'  ]
                    clrs += [cmap(clri)] * len(a)


    # con self per parcel
    clri += 1
    if roi_labels is not None:
        # main side (body)
        # get parcel indices of
        for grpn,parcel_list in gp.parcel_groupings_post.items():
            #feat_groups_cur = []
            #print(parcel_list)
            plws = utils.addSideToParcels(parcel_list, body_side)
            parcel_inds = [ roi_labels.index(parcel) for parcel in plws ]

            pas = '|'.join(map(str,parcel_inds) )
            chn_templ = f'msrc(R|L)_9_({pas})_c[0-9]+'
            ft = 'con'
            if ft in ftype_info['ftypes']:
                #        regex_same_LFP = r'.?.?corr.*(LFP.[0-9]+),.*\1.*'
                feat_groups_one_band = [ f'^{ft}_{fb}_{chn_templ},{chn_templ}' for fb in ftype_info['fband_per_ftype'][ft] ]
                feat_groups_all += feat_groups_one_band
                feature_groups_names += [ f'{ft}_{fb}_{grpn}_self' for fb in ftype_info['fband_per_ftype'][ft] ]
                clrs += [cmap(clri)] * len(feat_groups_one_band)


    # per LFP per band  and per parcel
    for lfpchn in chnames_LFP:
        clri += 1
        # now group per LFPch but with free source
        chn_templ = 'msrc(R|L)_9_[0-9]+_c[0-9]+'
        grpn = 'msrc*'

        # first per band only
        ft = 'bpcorr'
        if 'bpcorr' in ftype_info['ftypes']:

            if separate_by_band2:
                fbpairs = ftype_info['fband_pairs']
                fbpairs_dispnames = fbpairs
                # !! This assume LFP is always in the second place
                #if chnames_LFP is not None:
                #    for lfpchn in chnames_LFP:
                a = [f'^{ft}_{fb1}_{chn_templ},{fb2}_{lfpchn}' for fb1,fb2 in fbpairs]
                feat_groups_all += a
                feature_groups_names += [f'{ft}_{fb1}_{grpn},{fb2}_{lfpchn}' for fb1,fb2 in fbpairs_dispnames]
                clrs += [cmap(clri)] * len(a)
            #else:
            #    feat_groups_all += [f'^{ft}_{fb1}_{chn_templ},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
            #    feature_groups_names += [f'^{ft}_{fb1}_{grpn},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
        ft = 'rbcorr'
        # !! This assume LFP is always in the second place
        if ft in ftype_info['ftypes']:
            if separate_by_band2:
                fbsolos = ftype_info['fband_per_ftype'][ft]
                fbsolos_dispnames = fbsolos
                #if chnames_LFP is not None:
                #    for lfpchn in chnames_LFP:
                a = [ f'^{ft}_{fb}_{chn_templ},{fb}_{lfpchn}' for fb in fbsolos ]
                feat_groups_all += a
                feature_groups_names += [ f'{ft}_{fb}_{grpn},{fb}_{lfpchn}' for fb in fbsolos_dispnames ]
                clrs += [cmap(clri)] * len(a)
            #else:
            #    feat_groups_all += [ f'^{ft}_{fb}_{chn_templ},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
            #    feature_groups_names += [ f'^{ft}_{fb}_{grpn},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
        ft = 'con'
        # !! This assume LFP is always in the first place
        if ft in ftype_info['ftypes']:
            if separate_by_band2:
                fbsolos = ftype_info['fband_per_ftype'][ft]
                fbsolos_dispnames = fbsolos
                #if chnames_LFP is not None:
                #    for lfpchn in chnames_LFP:
                a = [ f'^{ft}_{fb}_{lfpchn},{chn_templ}' for fb in fbsolos ]
                feat_groups_all += a
                feature_groups_names += [ f'{ft}_{fb}_{lfpchn},{grpn}' for fb in fbsolos_dispnames ]
                clrs += [cmap(clri)] * len(a)

        #############################################

        fbpairs = [(bnpattern,bnpattern)]
        fbsolos = [bnpattern]
        fbpairs_dispnames = [('*','*')]
        fbsolos_dispnames = ['*']

        # now per parcel
        #clri += 1
        if roi_labels is not None:
            # main side (body)
            # get parcel indices of
            for grpn,parcel_list in gp.parcel_groupings_post.items():
                clri += 1
                #feat_groups_cur = []
                #print(parcel_list)
                plws = utils.addSideToParcels(parcel_list, body_side)
                parcel_inds = [ roi_labels.index(parcel) for parcel in plws ]

                pas = '|'.join(map(str,parcel_inds) )
                chn_templ = f'msrc(R|L)_9_({pas})_c[0-9]+'

                ft = 'bpcorr'
                if 'bpcorr' in ftype_info['ftypes']:
                    if separate_by_band:
                        fbpairs = ftype_info['fband_pairs']
                        fbpairs_dispnames = fbpairs
                    # !! This assume LFP is always in the second place
                    #if chnames_LFP is not None:
                    #    for lfpchn in chnames_LFP:
                    a = [f'^{ft}_{fb1}_{chn_templ},{fb2}_{lfpchn}' for fb1,fb2 in fbpairs]
                    feat_groups_all += a
                    feature_groups_names += [f'{ft}_{fb1}_{grpn},{fb2}_{lfpchn}' for fb1,fb2 in fbpairs_dispnames]
                    clrs += [cmap(clri)] * len(a)
                    #else:
                    #    feat_groups_all += [f'^{ft}_{fb1}_{chn_templ},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
                    #    feature_groups_names += [f'^{ft}_{fb1}_{grpn},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
                ft = 'rbcorr'
                # !! This assume LFP is always in the second place
                if ft in ftype_info['ftypes']:
                    if separate_by_band:
                        fbsolos = ftype_info['fband_per_ftype'][ft]
                        fbsolos_dispnames = fbsolos
                    #if chnames_LFP is not None:
                    #    for lfpchn in chnames_LFP:
                    a = [ f'^{ft}_{fb}_{chn_templ},{fb}_{lfpchn}' for fb in fbsolos ]
                    feat_groups_all += a
                    feature_groups_names += [ f'{ft}_{fb}_{grpn},{fb}_{lfpchn}' for fb in fbsolos_dispnames ]
                    clrs += [cmap(clri)] * len(a)
                    #else:
                    #    feat_groups_all += [ f'^{ft}_{fb}_{chn_templ},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
                    #    feature_groups_names += [ f'^{ft}_{fb}_{grpn},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
                ft = 'con'
                # !! This assume LFP is always in the first place
                if ft in ftype_info['ftypes']:
                    if separate_by_band:
                        fbsolos = ftype_info['fband_per_ftype'][ft]
                        fbsolos_dispnames = fbsolos
                    #if chnames_LFP is not None:
                    #    for lfpchn in chnames_LFP:
                    a = [ f'^{ft}_{fb}_{lfpchn},{chn_templ}' for fb in fbsolos ]
                    feat_groups_all += a
                    feature_groups_names += [ f'{ft}_{fb}_{lfpchn},{grpn}' for fb in fbsolos_dispnames ]
                    clrs += [cmap(clri)] * len(a)
                #else:
                #    feat_groups_all += [ f'^{ft}_{fb}_.*,{chn_templ}' for fb in ftype_info['fband_per_ftype'][ft] ]
                #    feature_groups_names += [ f'^{ft}_{fb}_.*,{grpn}' for fb in ftype_info['fband_per_ftype'][ft] ]

    # cross sources
    if (roi_labels is not None) and cross_source_groups:
        keyord = list(gp.parcel_groupings_post.keys())
        # main side (body)
        # get parcel indices of
        for grpni,grpn in enumerate(keyord):
            if grpni >= len(keyord) - 1:
                continue
            for grpn2 in keyord[grpni+1:] :
            #for grpn2,parcel_list2 in gp.parcel_groupings_post.items():
                clri += 1
                #feat_groups_cur = []
                #print(parcel_list)
                parcel_list1 = gp.parcel_groupings_post[grpn]
                parcel_list2 = gp.parcel_groupings_post[grpn2]

                plws = utils.addSideToParcels(parcel_list, body_side)
                parcel_inds = [ roi_labels.index(parcel) for parcel in plws ]

                plws2 = utils.addSideToParcels(parcel_list2, body_side)
                parcel_inds2 = [ roi_labels.index(parcel) for parcel in plws2 ]

                pas = '|'.join(map(str,parcel_inds) )
                chn_templ = f'msrc(R|L)_9_({pas})_c[0-9]+'

                pas2 = '|'.join(map(str,parcel_inds2) )
                chn_templ2 = f'msrc(R|L)_9_({pas2})_c[0-9]+'

                ft = 'bpcorr'
                if 'bpcorr' in ftype_info['ftypes']:
                    if separate_by_band:
                        fbpairs = ftype_info['fband_pairs']
                        fbpairs_dispnames = fbpairs
                    # !! This assume LFP is always in the second place
                    #if chnames_LFP is not None:
                    #    for lfpchn in chnames_LFP:
                    a = [f'^{ft}_{fb1}_{chn_templ},{fb2}_{chn_templ2}' for fb1,fb2 in fbpairs]
                    feat_groups_all += a
                    feature_groups_names += [f'{ft}_{fb1}_{grpn},{fb2}_{grpn2}' for fb1,fb2 in fbpairs_dispnames]
                    clrs += [cmap(clri)] * len(a)
                    #else:
                    #    feat_groups_all += [f'^{ft}_{fb1}_{chn_templ},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
                    #    feature_groups_names += [f'^{ft}_{fb1}_{grpn},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
                ft = 'rbcorr'
                # !! This assume LFP is always in the second place
                if ft in ftype_info['ftypes']:
                    if separate_by_band:
                        fbsolos = ftype_info['fband_per_ftype'][ft]
                        fbsolos_dispnames = fbsolos
                    #if chnames_LFP is not None:
                    #    for lfpchn in chnames_LFP:
                    a = [ f'^{ft}_{fb}_{chn_templ},{fb}_{chn_templ2}' for fb in fbsolos ]
                    feat_groups_all += a
                    feature_groups_names += [ f'{ft}_{fb}_{grpn},{fb}_{grpn2}' for fb in fbsolos_dispnames ]
                    clrs += [cmap(clri)] * len(a)
                    #else:
                    #    feat_groups_all += [ f'^{ft}_{fb}_{chn_templ},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
                    #    feature_groups_names += [ f'^{ft}_{fb}_{grpn},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
                ft = 'con'
                # !! This assume LFP is always in the first place
                if ft in ftype_info['ftypes']:
                    if separate_by_band:
                        fbsolos = ftype_info['fband_per_ftype'][ft]
                        fbsolos_dispnames = fbsolos
                    #if chnames_LFP is not None:
                    #    for lfpchn in chnames_LFP:
                    a = [ f'^{ft}_{fb}_{chn_templ},{chn_templ2}' for fb in fbsolos ]
                    feat_groups_all += a
                    feature_groups_names += [ f'{ft}_{fb}_{grpn},{grpn2}' for fb in fbsolos_dispnames ]
                    clrs += [cmap(clri)] * len(a)

    return clrs,feature_groups_names,feat_groups_all

def plotTableInfoBrain(table_info_per_perf_type, perf_tuple):
    raise ValueError('not implemented')
    from utils import vizGroup2
    # just to avoid error messages from pymode
    rec_info = None
    head_subj_ind = None
    multi_clf_output = None
    sind_str = None
    color_group_labels = None
    roi_lab_codes = None
    parcel_indices_all = None
    sizes_list = None
    srcgrp = None
    seed = 0
    import shutil



    if head_subj_ind is None:
        rncur = sind_str + '_off_hold'
    else:
        rncur = head_subj_ind + '_off_hold'
    sources_type=multi_clf_output['info']['sources_type']
    src_file_grouping_ind = multi_clf_output['info']['src_grouping_fn']
    #src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
    #                                                    sources_type,src_file_grouping_ind)
    #src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    from utils import genRecInfoFn
    src_rec_info_fn_full = genRecInfoFn(rncur,sources_type,src_file_grouping_ind)
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
    sgdn = 'all_raw'

    labels_dict = rec_info['label_groups_dict'][()]
    roi_labels_ = np.array(  labels_dict[sgdn] )
    roi_labels = ['unlabeled'] + list( roi_labels_[parcel_indices_all] )

    srcgroups_dict = rec_info['srcgroups_dict'][()]
    coords = rec_info['coords_Jan_actual'][()]


    clrs =  utils.vizGroup2(sind_str,coords,roi_labels,srcgrp, show=False,
                            alpha=.1, figsize_mult=1.5,msz=30, printLog=0,
                            color_grouping=roi_lab_codes,
                            color_group_labels= color_group_labels,
                            sizes=sizes_list, msz_mult=0.3, seed=seed)

def plotTableInfos2(table_info_per_perf_type, perf_tuple,
                      output_subdir='', alpha_bar = 0.7,
                    use_recalc_perf = True, prefixes_sorted = None,
                    prefix2final_name = None,
                    crop_rawname=slice(None,None),
                   sort_by_featnum = 0, set_nice_names = True ):
    import matplotlib.pyplot as plt

    info_per_rn_pref = table_info_per_perf_type[perf_tuple]
    rns = list( info_per_rn_pref.values() )
    nrpef = len( rns[0].keys() )

    nr = len(rns)
    nc = 3
    ww = 6; hh = 4 *  nrpef / 20
    fig,axs = plt.subplots(nr,nc, figsize = (ww*nc, hh*nr))
    axs = axs.reshape((nr,nc))

    pveclen = 2
    colors = ['blue', 'red', 'purple', 'green']
    color_full = colors[0]
    color_red = colors[1]
    color_red2 = colors[2]
    color_add = colors[3]
    str_per_pref_per_rowname_per_clftype = {}

    main_keys = [perf_tuple[1]]
    red_keys = [perf_tuple[2]]

    #if keys is None:
    #    #keys = list( sorted( table_info_per_perf_type.keys() ) )
    #    if perf_kind == 'XGB':
    #        keys = ['perfs_XGB', 'perfs_XGB_red' ]
    #    elif perf_kind == 'LDA':
    #        keys = ['all_present_features', 'strongest_features_LDA_selMinFeatSet']
    #    else:
    #        raise ValueError('wrong perf_kind')
    #else:
    #    assert perf_kind is None

    keys = perf_tuple[1],perf_tuple[2]


    pvec_summary_per_prefix_per_key = {}
    pvec_summary_red_per_prefix_per_key = {}


    # cycle to plot both perf
    #for ci,clf_type in enumerate( keys ):
    #pvec_summary_per_prefix = {}
    #pvec_summary_red_per_prefix = {}

    axind = 0

    str_per_pref_per_rowname = {}
    if pveclen == 3:
        #perftype = '(spec + sens + F1) / 3'
        perftype = 'min(spec,sens,F1)'
    elif pveclen == 2:
        perftype = 'min(spec,sens)'
    else:
        raise ValueError('wrong pveclen')

    for rowid_tuple,rowinfo in info_per_rn_pref.items():
        xs, xs_red, xs_red2 = [],[],[]
        ys, ys_red, ys_add = [],[],[]
        nums_red = []
        if prefixes_sorted is None:
            prefixes_sorted = list(sorted(rowinfo.keys()))
        else:
            # I want it to display same way as in jupyter, otherwise first goes below
            prefixes_sorted = prefixes_sorted[::-1]
        prefixes_wnums = []
        str_per_pref = {}
        for prefix in prefixes_sorted:
            prefinfo = rowinfo[prefix]
            if np.isnan(prefinfo['sens']):
                continue


            num = prefinfo.get('num',-1)
            num_red = prefinfo.get('num_red',-1)
            num_red2 = prefinfo.get('num_red2',-1)
            if num is None:
                num = -1
            if num_red is None:
                num_red = -1
            if num_red2 is None:
                num_red2 = -1
            xs += [ num]
            xs_red += [ num_red]
            xs_red2 += [ num_red2]

            order = ['sens', 'spec', 'F1']
            if use_recalc_perf:
                pvec = [prefinfo.get(os + '_recalc',np.nan) for os in order]
                pvec_red = [prefinfo.get(os + '_red' + '_recalc',np.nan) for os in order]
                pvec_add = [prefinfo.get(os + '_add' + '_recalc',np.nan) for os in order]
                #print(pvec_red)
            else:
                pvec = [prefinfo[os] for os in order]
                pvec_red = [prefinfo.get(os + '_red' , np.nan) for os in order]
                pvec_add = [prefinfo.get(os + '_add' , np.nan) for os in order]
            pvec = pvec[:pveclen]
            pvec_red = pvec_red[:pveclen]
            pvec_add = pvec_add[:pveclen]
            #pvec_red = [prefinfo['sens_red'], prefinfo['spec_red'] , prefinfo['F1_red']]
            assert pveclen in [2,3]
            str_to_put_ = utsne.sprintfPerfs(pvec)
            str_to_put_red = utsne.sprintfPerfs(pvec_red)
            str_to_put_add = utsne.sprintfPerfs(pvec_add)
            #if pveclen == 3:
            #    str_to_put_ =  '{:.0f}%,{:.0f}%,{:.0f}%'.format(100*pvec[0],100*pvec[1],100*pvec[2])
            #    str_to_put_red =  '{:.0f}%,{:.0f}%,{:.0f}%'.format(100*pvec_red[0],100*pvec_red[1],100*pvec_red[2])
            #elif pveclen == 2:
            #    pvec = [pvec[0], pvec[1] ]
            #    str_to_put_ =  '{:.0f}%,{:.0f}%'.format(100*pvec[0],100*pvec[1])
            #    str_to_put_red =  '{:.0f}%,{:.0f}%'.format(100*pvec_red[0],100*pvec_red[1])
            #else:
            #    raise ValueError('wrong pveclen')


            str_to_put = str_to_put_
            pvec = np.array(pvec)
            pvec_red = np.array(pvec_red)

            prefix_like = prefix
            if set_nice_names:
                descr = prefinfo['descr']
                if descr is not None:
                    prefix_like = descr

            if prefix2final_name is not None:
                prefix_like = prefix2final_name[prefix]

            #print(clf_type,str_to_put)
            prefixes_wnums += [prefix_like + f'# {num} : {str_to_put} (min-> {num_red} : {str_to_put_red})']

            #p = np.mean(pvec)
            p     = np.min(pvec)
            p_red = np.min(pvec_red)
            p_add = np.min(pvec_add)
            #ys += [prefinfo[perftype]]
            ys += [p]
            ys_red += [p_red]
            ys_add += [p_add]

            #print(ys_add)

        #print( prefixes_wnums )

        #print(ys_red)
        str_per_pref_per_rowname[rowid_tuple] = str_per_pref


        rowind_scatter_numnum = 0
        rowind_scatter = 1
        rowind_bars = 2

        rowid_tuple_to_show = ( rowid_tuple[0][crop_rawname],*rowid_tuple[1:] )

        ax = axs[axind,rowind_scatter_numnum]
        ax.set_title(rowid_tuple_to_show )

        ax.scatter(xs,xs_red, c = color_full)
        ax.set_xlabel('Number of features, full')
        ax.set_ylabel('Number of features, reduced')
        if len(xs):
            ax.plot([0, np.max(xs)], [0, np.max(xs)], ls='--')
        if len(xs_red):
            ax.set_ylim(0,np.max(xs_red)*1.1 )

            if np.max(xs_red2) > 0 :
                ax.plot([0, np.max(xs)], [0, np.max(xs_red2)],
                        ls='--', c=color_red2)


        #ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--')

        ####################################

        ax = axs[axind,rowind_scatter]
        ax.set_title(rowid_tuple_to_show)
        ax.scatter(xs,ys, c = color_full)
        ax.scatter(xs_red,ys_red, c = color_red)
        if len(xs_red2) and np.max(xs_red2) > 0 :
            ax.plot([0, np.max(xs_red2)], [0, np.max(ys_red)], ls='--', c=color_red2)
        ax.set_ylabel(perftype)
        ax.set_xlabel('Number of features')
        ax.set_ylim(0,1)
        #ax.set_xlabel('total feature number')
        ####################################
        ax = axs[axind,rowind_bars]
        if len(xs):
            ax.set_title(str(rowid_tuple_to_show)  + ';  order=' + ','.join(order[:pveclen] ) )
        ax.yaxis.tick_right()
        if sort_by_featnum:
            sis = np.argsort(xs)
        else:
            sis = np.arange(len(prefixes_wnums) )
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys)[sis], color = color_full,    alpha=alpha_bar)
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys_red)[sis],
                color = color_red, alpha=alpha_bar)
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys_add)[sis],
                color = color_add, alpha=alpha_bar)
        ax.set_xlabel(perftype)
        ax.set_xlim(0,1)
        #ax.tick_params(axs=)

        axind += 1
        #str_per_pref_per_rowname_per_clftype[clf_type] = str_per_pref_per_rowname

        #pvec_summary_per_prefix_per_key[clf_type] = pvec_summary_per_prefix
        #pvec_summary_red_per_prefix_per_key[clf_type] = pvec_summary_red_per_prefix

    plt.suptitle( str( perf_tuple ) + f' recalc perf {use_recalc_perf}', y=0.995, fontsize=14  )
    plt.tight_layout()
    #keystr = ','.join(keys)
    figfname = f'Performances_perf_tuple={perf_tuple}_pveclen={pveclen}.pdf'
    dirfig = pjoin(gv.dir_fig, output_subdir)
    if not os.path.exists(dirfig):
        os.mkdir(dirfig)
    figfname_full = pjoin(gv.dir_fig, output_subdir,figfname)
    plt.savefig(figfname_full)
    print(f'fig saved to  {figfname_full}')


def plotTableInfos_onlyBar(table_info_per_perf_type, perf_tuple,
                      output_subdir='', alpha_bar = 0.7,
                    use_recalc_perf = True, prefixes_sorted = None,
                           prefix2final_name = None,
                    crop_rawname='last',
                   sort_by_featnum = 0, show_add_info ='none', percents=True,
                           score= 'special:min(sens,spec)', per_medcond=False,
                           rawnames = None, expand_best = 0,
                          allow_missing_prefixes=0, hh =2, ww=5, label_fontsize = 14,
                           red = 'red' ):
    # rawnames argument is mainly to specify order
    import matplotlib.pyplot as plt

    info_per_rn_pref = table_info_per_perf_type[perf_tuple]
    rns = list( info_per_rn_pref.values() )
    nrpef = len( rns[0].keys() )

    if not per_medcond:
        nr = len(rns)
        nc = 1
    else:
        nc = 2
        nr = int( np.ceil( len(rns) / nc ) )
    fig,axs = plt.subplots(nr,nc, figsize = (ww*nc, hh*nr), sharex = 'col')
    axs = axs.reshape((nr,nc))

    if score.startswith('special'):
        pveclen = 2
    else:
        pveclen = 1
    colors = ['blue', 'red', 'purple', 'green']
    color_full = colors[0]
    color_red = colors[1]
    color_red2 = colors[2]
    color_add = colors[3]
    str_per_pref_per_rowname_per_clftype = {}

    main_keys = [perf_tuple[1]]
    red_keys = [perf_tuple[2]]

    #if keys is None:
    #    #keys = list( sorted( table_info_per_perf_type.keys() ) )
    #    if perf_kind == 'XGB':
    #        keys = ['perfs_XGB', 'perfs_XGB_red' ]
    #    elif perf_kind == 'LDA':
    #        keys = ['all_present_features', 'strongest_features_LDA_selMinFeatSet']
    #    else:
    #        raise ValueError('wrong perf_kind')
    #else:
    #    assert perf_kind is None

    keys = perf_tuple[1],perf_tuple[2]


    pvec_summary_per_prefix_per_key = {}
    pvec_summary_red_per_prefix_per_key = {}


    # cycle to plot both perf
    #for ci,clf_type in enumerate( keys ):
    #pvec_summary_per_prefix = {}
    #pvec_summary_red_per_prefix = {}

    axind = 0

    str_per_pref_per_rowname = {}
    if pveclen == 3:
        #perftype = '(spec + sens + F1) / 3'
        perftype = 'min(spec,sens,F1)'
    elif pveclen == 2:
        perftype = 'min(spec,sens)'
    else:
        perftype = score
    #else:
    #    raise ValueError('wrong pveclen')
    sind_strs = []
    for rowid_tuple,rowinfo in info_per_rn_pref.items():
        rn = rowid_tuple[0]
        sind_str,medcond = rn.split('_')
        sind_strs += [sind_str]
    sind_strs = list(sorted(set(sind_strs)))
    print('sind strs = ',sind_strs)

    #label_fontsize = None
    import pandas as pd
    df = pd.DataFrame( columns = ['rawname', 'barname', 'barname_pre', 'barname_pre_human', 'barname_sided',
        'shuffled', 'num', 'num_red', 'num_red2', 'p', 'p_red', 'p_add'] )

    for rowid_tuple,rowinfo in info_per_rn_pref.items():
        xs, xs_red, xs_red2 = [],[],[]
        ys, ys_red, ys_add = [],[],[]
        nums_red = []
        if prefixes_sorted is None:
            prefixes_sorted = list(sorted(rowinfo.keys()))
        prefixes_wnums = []
        str_per_pref = {}
        for prefix_pre in prefixes_sorted:
            rn = rowid_tuple[0]
            subj=rn.split('_')[0]
            prefix = prefTempl2pref(prefix_pre, subj)

            prefinfo = rowinfo.get(prefix,None)
            if prefinfo is None and allow_missing_prefixes:
                continue
            if np.isnan(prefinfo['sens']):
                continue

            d = {'rawname':rn,'barname_pre':prefix_pre, 'barname_sided':prefix }

            num = prefinfo.get('num',-1)
            num_red = prefinfo.get('num_red',-1)
            num_red2 = prefinfo.get('num_red2',-1)
            if num is None:
                num = -1
            if num_red is None:
                num_red = -1
            if num_red2 is None:
                num_red2 = -1
            xs += [ num]
            xs_red += [ num_red]
            xs_red2 += [ num_red2]

            d['num'] = num
            d['num_red'] = num_red
            d['num_red2'] = num_red2

            if score.startswith('special'):
                order = ['sens', 'spec', 'F1']
            else:
                order = [score]
            if use_recalc_perf:
                pvec = [prefinfo.get(os + '_recalc',np.nan) for os in order]
                pvec_red = [prefinfo.get(os + '_red' + '_recalc',np.nan) for os in order]
                pvec_add = [prefinfo.get(os + '_add' + '_recalc',np.nan) for os in order]
                #print(pvec_red)
            else:
                pvec = [prefinfo[os] for os in order]
                pvec_red = [prefinfo.get(os + '_red' , np.nan) for os in order]
                pvec_add = [prefinfo.get(os + '_add' , np.nan) for os in order]
            pvec = pvec[:pveclen]
            pvec_red = pvec_red[:pveclen]
            pvec_add = pvec_add[:pveclen]
            #pvec_red = [prefinfo['sens_red'], prefinfo['spec_red'] , prefinfo['F1_red']]
            #assert pveclen in [2,3]
            str_to_put_ = utsne.sprintfPerfs(pvec)
            str_to_put_red = utsne.sprintfPerfs(pvec_red)
            str_to_put_add = utsne.sprintfPerfs(pvec_add)
            #if pveclen == 3:
            #    str_to_put_ =  '{:.0f}%,{:.0f}%,{:.0f}%'.format(100*pvec[0],100*pvec[1],100*pvec[2])
            #    str_to_put_red =  '{:.0f}%,{:.0f}%,{:.0f}%'.format(100*pvec_red[0],100*pvec_red[1],100*pvec_red[2])
            #elif pveclen == 2:
            #    pvec = [pvec[0], pvec[1] ]
            #    str_to_put_ =  '{:.0f}%,{:.0f}%'.format(100*pvec[0],100*pvec[1])
            #    str_to_put_red =  '{:.0f}%,{:.0f}%'.format(100*pvec_red[0],100*pvec_red[1])
            #else:
            #    raise ValueError('wrong pveclen')


            str_to_put = str_to_put_
            pvec = np.array(pvec)
            pvec_red = np.array(pvec_red)

            #print(clf_type,str_to_put)
            prefix_like_orig = None
            if prefix2final_name is not None:
                prefix_like_orig = prefix2final_name[prefix_pre]
                bai = prefix_like_orig.find('best area')  # best area index
                wai = prefix_like_orig.find('worst area')  # worst area index
                if expand_best and bai >= 0:
                    print(prefix_like_orig,prefinfo['name_nice_best'])
                    #prefix_like = prefix_like[:bai] + \
                    #    prefinfo['name_nice_best'].split()[-1].split('+')[-1]
                    side_suff = [ '_L', '_R', '_B']
                    sided = False
                    #nnb = prefinfo['name_nice_best']
                    #for suff in side_suff:
                    #    if prefinfo['name_nice_best'].find(suff):
                    #        sided = True
                    #        nnb.split()
                    #area = prefinfo['name_nice_best'].split()[-1].split('+')[-1]
                    #prefix_like = prefix_like.replace('best area',area)
                    #prefix_like = prefix_like.replace('best area', prefinfo['name_nice_best'] )
                    prefix_like = prefix_like_orig.replace('best area', prefinfo['parcel_group_name'] )
                elif expand_best and wai >= 0:
                    print(prefix_like_orig,prefinfo['name_nice_best'])
                    side_suff = [ '_L', '_R', '_B']
                    sided = False
                    prefix_like = prefix_like_orig.replace('worst area', prefinfo['parcel_group_name'] )
                else:
                    prefix_like = prefix_like_orig
            else:
                prefix_like = prefix

            if show_add_info == 'num':
                prefixes_wnums += [prefix_like + f'. #features={num}']
            elif show_add_info == 'num_and_strperf':
                prefixes_wnums += [prefix_like + f'. #features={num} : {str_to_put}']
            elif show_add_info == 'none':
                prefixes_wnums += [prefix_like]
            else:
                raise ValueError('unk val of show_add_info = {show_add_info}')

            if red == 'red':
                p_red = np.min(pvec_red)
            elif red == 'shuffled':
                p_red = _extractPerfNumber ( prefinfo['perf_dict_shuffled'], score )

            #p = np.mean(pvec)
            p     = np.min(pvec)
            p_add = np.min(pvec_add)
            if percents:
                p *= 100
                p_red *= 100
                p_add *= 100
            #ys += [prefinfo[perftype]]
            ys += [p]
            ys_red += [p_red]
            ys_add += [p_add]

            d['barname'] = prefix_like
            d['barname_pre_human'] = prefix_like_orig
            d['p'] = p
            d['p_red'] = p_red
            d['p_add'] = p_add

            print('lalal ', d)
            df = df.append(d, ignore_index=1)

            #print(ys_add)
        # end of cycle over prefixed_sorted

        df.reset_index()

        print( 'prefixes_wnums = ',prefixes_wnums )
        print( 'ys_red =',ys_red )

        #print(ys_red)
        str_per_pref_per_rowname[rowid_tuple] = str_per_pref

        rowind_bars = 0
        rn = rowid_tuple[0]
        #rncrp = rn[crop_rawname]
        if crop_rawname == 'last':
            rncrp = rn.split('_')[-1]
        elif crop_rawname == 'no':
            rncrp = rn
        else:
            raise ValueError('unk crop')
        #rowid_tuple_to_show = (rncrp ,*rowid_tuple[1:] )
        rowid_tuple_to_show = rncrp.upper()

        if per_medcond:
            sind_str,medcond = rn.split('_')
            rowind_bars = ['off','on'].index(medcond)
            axind = sind_strs.index(sind_str)
        ax = axs[axind,rowind_bars]
        if len(xs):
            #ax.set_title(str(rowid_tuple_to_show)  + ';  order=' + ','.join(order[:pveclen] ) )
            ax.set_title(str(rowid_tuple_to_show) ) # + ';  order=' + ','.join(order[:pveclen] ) )
        ax.yaxis.tick_right()
        if sort_by_featnum:
            sis = np.argsort(xs)
        else:
            sis = np.arange(len(prefixes_wnums) )
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys)[sis], color = color_full,    alpha=alpha_bar)
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys_red)[sis],
                color = color_red, alpha=alpha_bar)
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys_add)[sis],
                color = color_add, alpha=alpha_bar)

        if per_medcond:
            if axind == (len(sind_strs) - 1):
                ax.set_xlabel(perftype, fontsize = label_fontsize)
            else:
                ax.set_xlabel('')
        else:
            ax.set_xlabel(perftype, fontsize = label_fontsize)

        ax.set_xlim(0,1)
        if percents:
            ax.set_xlim(0,100)
        #ax.tick_params(axs=)

        axind += 1
        #str_per_pref_per_rowname_per_clftype[clf_type] = str_per_pref_per_rowname

        #pvec_summary_per_prefix_per_key[clf_type] = pvec_summary_per_prefix
        #pvec_summary_red_per_prefix_per_key[clf_type] = pvec_summary_red_per_prefix
    # end of cycle over info_per_rn_pref

    plt.suptitle( str( perf_tuple ) + f' recalc perf {use_recalc_perf}', y=0.995, fontsize=14  )
    plt.tight_layout()
    #keystr = ','.join(keys)
    #figfname = f'Performances_perf_tuple={perf_tuple}_pveclen={pveclen}.pdf'
    #dirfig = pjoin(gv.dir_fig, output_subdir)
    #if not os.path.exists(dirfig):
    #    os.mkdir(dirfig)
    #plt.savefig(pjoin(gv.dir_fig, output_subdir,figfname))
    return axs, df


def plotTableInfos_onlyBarDf(df, barnames_pre=None, output_subdir='', alpha_bar = 0.7,
                    use_recalc_perf = True, prefixes_sorted = None,
                   prefix2final_name = None, crop_rawname='last',
                   sort_by_featnum = 0, show_add_info ='none', col1 = 'p',
                             col2 = 'p_red', col3 = None,
                   score= 'special:min(sens,spec)',
                   per_medcond=False, rawnames = None, expand_best =
                   0, remove_side_ending = True,
                   hh =2, ww=5, label_fontsize = 14, xlim=(0,100)
                   ):
    # rawnames argument is mainly to specify order
    import matplotlib.pyplot as plt

    #info_per_rn_pref = table_info_per_perf_type[perf_tuple]
    #rns = list( info_per_rn_pref.values() )
    #nrpef = len( rns[0].keys() )
    numrns = len( set( df['rawname']) )

    if not per_medcond:
        nr = numrns
        nc = 1
    else:
        nc = 2
        nr = int( np.ceil( numrns / nc ) )
    fig,axs = plt.subplots(nr,nc, figsize = (ww*nc, hh*nr), sharex = 'col')
    axs = axs.reshape((nr,nc))

    if score.startswith('special'):
        pveclen = 2
    else:
        pveclen = 1
    colors = ['blue', 'red', 'purple', 'green']
    color_full = colors[0]
    color_red = colors[1]
    color_red2 = colors[2]
    color_add = colors[3]
    str_per_pref_per_rowname_per_clftype = {}

    pvec_summary_per_prefix_per_key = {}
    pvec_summary_red_per_prefix_per_key = {}


    # cycle to plot both perf
    #for ci,clf_type in enumerate( keys ):
    #pvec_summary_per_prefix = {}
    #pvec_summary_red_per_prefix = {}

    axind = 0

    #str_per_pref_per_rowname = {}
    if pveclen == 3:
        #perftype = '(spec + sens + F1) / 3'
        perftype = 'min(spec,sens,F1)'
    elif pveclen == 2:
        perftype = 'min(spec,sens)'
    else:
        perftype = score
    #else:
    #    raise ValueError('wrong pveclen')
    sind_strs = []
    #for rowid_tuple,rowinfo in info_per_rn_pref.items():
    #    rn = rowid_tuple[0]
    #    sind_str,medcond = rn.split('_')
    #    sind_strs += [sind_str]
    #sind_strs = list(sorted(set(sind_strs)))

    sind_strs = list(sorted(set(df['subject']) ))
    print('sind strs = ',sind_strs)

    barnames_pre_available = list(sorted(set(df['barname_pre']) ))
    if barnames_pre is None:
        barnames_pre = barnames_pre_available
    else:
        assert set( barnames_pre ) < set(barnames_pre_available)
    print('barnames_pre' , barnames_pre)

    for rn in list(set(df['rawname'])):
    #for index,row in df.iterrow():
        xs, xs_red, xs_red2 = [],[],[]
        ys, ys_red, ys_add = [],[],[]

        prefixes_wnums = []
        str_per_pref = {}
        for barname_pre in barnames_pre:
            subdf = df[ (df['rawname'] == rn) & ( df['barname_pre'] == barname_pre ) ]
            assert len(subdf) == 1, (rn,barname_pre, len(subdf) )
            row = dict( subdf.iloc[0] )
            subj = str( row['subject'] )
            num      = int( row.get('num',-1)      )
            num_red  = int( row.get('num_red',-1)  )
            num_red2 = int( row.get('num_red2',-1) )

            xs += [ num]
            xs_red += [ num_red]
            xs_red2 += [ num_red2]

            str_to_put = f'{float(row["p"]):.0f}%'

            prefix_like      = str(row['barname'].values[0] )
            prefix_like_orig = str(row['barname_pre_human'].values[0] )
            if remove_side_ending:
                side_endings = ['_L','_R','_B']
                for se in side_endings:
                    if prefix_like.find(se) >= 0:
                        prefix_like = prefix_like.replace(se,'')

            #print(prefix_like,prefix_like_orig)

            if show_add_info == 'num':
                prefixes_wnums += [prefix_like + f'. #features={num}']
            elif show_add_info == 'num_and_strperf':
                prefixes_wnums += [prefix_like + f'. #features={num} : {str_to_put}']
            elif show_add_info == 'none':
                prefixes_wnums += [prefix_like]
            else:
                raise ValueError('unk val of show_add_info = {show_add_info}')
            ys     += [float(row[col1])]
            if col2 is not None:
                ys_red += [float(row[col2])]
            if col3 is not None:
                ys_add += [float(row[col3])]
        # end of cycle over prefixed_sorted

        print( 'prefixes_wnums = ',prefixes_wnums )
        print( 'ys_red =',ys_red )

        rowind_bars = 0
        if crop_rawname == 'last':
            rncrp = rn.split('_')[-1]
        elif crop_rawname == 'no':
            rncrp = rn
        else:
            raise ValueError('unk crop')
        rowid_tuple_to_show = rncrp.upper()

        ##########################################3
        if per_medcond:
            sind_str,medcond = rn.split('_')
            rowind_bars = ['off','on'].index(medcond)
            axind = sind_strs.index(sind_str)
        ax = axs[axind,rowind_bars]
        if len(xs):
            ax.set_title(str(rowid_tuple_to_show) ) # + ';  order=' + ','.join(order[:pveclen] ) )
        ax.yaxis.tick_right()
        if sort_by_featnum:
            sis = np.argsort(xs)
        else:
            sis = np.arange(len(prefixes_wnums) )
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys)[sis], color = color_full,    alpha=alpha_bar)
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys_red)[sis],
                color = color_red, alpha=alpha_bar)
        if len(ys_add):
            ax.barh(np.array(prefixes_wnums)[sis], np.array(ys_add)[sis],
                    color = color_add, alpha=alpha_bar)

        if per_medcond:
            if axind == (len(sind_strs) - 1):
                ax.set_xlabel(perftype, fontsize = label_fontsize)
            else:
                ax.set_xlabel('')
        else:
            ax.set_xlabel(perftype, fontsize = label_fontsize)

        #ax.set_xlim(0,1)
        ax.set_xlim(xlim)
        axind += 1
    # end of cycle over info_per_rn_pref

    plt.suptitle(  f' recalc perf {use_recalc_perf}', y=0.995, fontsize=14  )
    plt.tight_layout()
    return axs

def plotTableInfos_onlyBarDf2(df, dfstat, barnames_pre=None, output_subdir='', alpha_bar = 0.7,
                    use_recalc_perf = True,
                    crop_rawname='last',
                    sort_by_featnum = 0, show_add_info ='none', col1 = 'bacc',
                    col2 = 'bacc_shuffled', col3 = None, sorted_by = 'bacc',
                    per_medcond=False, rawnames = None, expand_best = 0,
                    endings_processing = 'replace_relative_side',
                    prefix2final_name = None,
                    hh =2, ww=5, label_fontsize = 14, xlim=(0,100),
                    xlabel = None):
    # rawnames argument is mainly to specify order
    import matplotlib.pyplot as plt
    #numrns = len( set( df['rawname']) )
    numrns = len( set(rawnames) & set(df['rawname']) )
    prefixes_all  = set(df['prefix']        )
    prefix_templs_all  = set(df['prefix_templ']        )
    bestnames_all = set(dfstat['bestname']  )

    if not per_medcond:
        nr = numrns
        nc = 1
    else:
        nc = 2
        nr = int( np.ceil( numrns / nc ) )
    fig,axs = plt.subplots(nr,nc, figsize = (ww*nc, hh*nr), sharex = 'col')
    axs = axs.reshape((nr,nc))

    colors = ['blue', 'red', 'purple', 'green']
    color_full = colors[0]
    color_red = colors[1]
    color_red2 = colors[2]
    color_add = colors[3]

    axind = 0
    sind_strs = list(sorted(set(df['subject']) ))
    print('sind strs = ',sind_strs)

    rows_output = []

    for rn in list(set(df['rawname'])):
    #for index,row in df.iterrow():
        xs, xs_red, xs_red2 = [],[],[]
        ys, ys_red, ys_add = [],[],[]

        prefixes_wnums = []
        str_per_pref = {}
        for barname_pre in barnames_pre:
            if barname_pre in bestnames_all:
                mode = 'bestname'
                dfcur = dfstat[dfstat['operation_col'] == sorted_by ]
                print(f'usindg dfstat for operation_col == {sorted_by}')
            elif barname_pre in prefixes_all:
                mode = 'prefix'
                dfcur = df
                print(f'using df for {mode}')
            elif barname_pre in prefix_templs_all:
                mode = 'prefix_templ'
                dfcur = df
                print(f'using df for {mode}')
            colname = mode

            subdf = dfcur[ (dfcur['rawname'] == rn) & ( dfcur[colname] == barname_pre ) ]

            assert len(subdf) == 1, (rn,barname_pre, len(subdf), mode )
            row = dict( subdf.iloc[0] )
            subj = str( row['subject'] )
            num      = int( row.get('num',-1)      )
            num_red  = int( row.get('num_red',-1)  )
            num_red2 = float( row.get('num_red2',-1) )

            xs += [ num]
            xs_red += [ num_red]
            xs_red2 += [ num_red2]

            p = float( row[col1] )
            if np.isnan(row[col1] ):
                print('          aaaaa')
                return None
            str_to_put = f'{p:.0f}%'

            prefix_like_orig      = prefix2final_name[barname_pre]
            assert prefix_like_orig is not None


            bai = prefix_like_orig.find('best area')  # best area index
            wai = prefix_like_orig.find('worst area')  # worst area index
            if expand_best and bai >= 0:
                assert row['parcel_group_names'] is not None, (barname_pre,row)
                print(prefix_like_orig,row['name_nice'])
                #prefix_like = prefix_like[:bai] + \
                #    prefinfo['name_nice_best'].split()[-1].split('+')[-1]
                side_suff = [ '_L', '_R', '_B']
                sided = False
                #nnb = prefinfo['name_nice_best']
                #for suff in side_suff:
                #    if prefinfo['name_nice_best'].find(suff):
                #        sided = True
                #        nnb.split()
                #area = prefinfo['name_nice_best'].split()[-1].split('+')[-1]
                #prefix_like = prefix_like.replace('best area',area)
                #prefix_like = prefix_like.replace('best area', prefinfo['name_nice_best'] )
                prefix_like = prefix_like_orig.replace('best area',
                                    row['parcel_group_names'] )
            elif expand_best and wai >= 0:
                print(prefix_like_orig,row['name_nice'])
                side_suff = [ '_L', '_R', '_B']
                sided = False
                prefix_like = prefix_like_orig.replace('worst area',
                                    row['parcel_group_names'] )
            else:
                prefix_like = prefix_like_orig



            print('prefix_like_orig = ', prefix_like_orig, 'prefix_like = ',prefix_like)

            row['barname_pre']      = barname_pre
            #rot['barname_pre'] = prefTempl2pref(barname_pre, subj)
            row['prefix_like_orig'] = prefix_like_orig
            row['prefix_like']      = prefix_like
            rows_output += [row]

            #prefix_like_orig = str(row['barname_pre_human'].values[0] )
            side_endings = ['_L','_R','_B']
            for se in side_endings:
                if prefix_like.find(se) >= 0:
                    if endings_processing == 'remove':
                        prefix_like = prefix_like.replace(se,'')
                    elif endings_processing == 'replace_relative_side':
                        sidelet = se[-1]
                        side_templ_symbol = sidelet2sideTempl(sidelet, subj)
                        prefix_like = prefix_like.replace(se,' ' + STS2ststr[side_templ_symbol] )


            #print(prefix_like,prefix_like_orig)

            if show_add_info == 'num':
                prefixes_wnums += [prefix_like + f'. #features={num}']
            elif show_add_info == 'num_and_strperf':
                prefixes_wnums += [prefix_like + f'. #features={num} : {str_to_put}']
            elif show_add_info == 'none':
                prefixes_wnums += [prefix_like]
            else:
                raise ValueError('unk val of show_add_info = {show_add_info}')
            ys         += [float(row[col1])]
            if col2 is not None:
                ys_red += [float(row[col2])]
            if col3 is not None:
                ys_add += [float(row[col3])]
        # end of cycle over prefixed_sorted

        print( 'prefix_likes = ',prefixes_wnums )
        print( 'ys =',ys )
        print( 'ys_red =',ys_red )

        ys = np.array(ys)
        ys_red = np.array(ys_red)

        rowind_bars = 0
        if crop_rawname == 'last':
            rncrp = rn.split('_')[-1]
        elif crop_rawname == 'no':
            rncrp = rn
        else:
            raise ValueError('unk crop')
        rowid_tuple_to_show = rncrp.upper()

        ##########################################3
        if per_medcond:
            sind_str,medcond = rn.split('_')
            rowind_bars = ['off','on'].index(medcond)
            axind = sind_strs.index(sind_str)
        ax = axs[axind,rowind_bars]
        if len(xs):
            ax.set_title(str(rowid_tuple_to_show) ) # + ';  order=' + ','.join(order[:pveclen] ) )
        ax.yaxis.tick_right()
        if sort_by_featnum:
            sis = np.argsort(xs)
        else:
            sis = np.arange(len(prefixes_wnums) )
        ax.barh(np.array(prefixes_wnums)[sis], ys[sis], color = color_full,    alpha=alpha_bar)
        ax.barh(np.array(prefixes_wnums)[sis], ys_red[sis],
                color = color_red, alpha=alpha_bar)
        if len(ys_add):
            ax.barh(np.array(prefixes_wnums)[sis], np.array(ys_add)[sis],
                    color = color_add, alpha=alpha_bar)

        if xlabel is None:
            xlabel = col1
        if per_medcond:
            if axind == (len(sind_strs) - 1):
                ax.set_xlabel(xlabel, fontsize = label_fontsize)
            else:
                ax.set_xlabel('')
        else:
            ax.set_xlabel(xlabel, fontsize = label_fontsize)

        #ax.set_xlim(0,1)
        ax.set_xlim(xlim)
        axind += 1
    # end of cycle over info_per_rn_pref
    dfout = pd.DataFrame( rows_output )

    plt.suptitle(  f' recalc perf {use_recalc_perf} col1={col1}, col2={col2}, sorted_by={sorted_by}', y=0.995, fontsize=14  )
    plt.tight_layout()
    return dfout, axs

def plotFeatNum2Perf(output_per_raw, perflists, prefixes=None, balance_level = 0.75, skip_plot=False, xlim=None ):
    import matplotlib.pyplot as plt
    if prefixes is None:
        prefixes = []
        for rn in output_per_raw:
            prefixes += list( output_per_raw[rn].keys() )

    rns = list(sorted(output_per_raw.keys()))

    nc = len(perflists)


    feat_names_per_prefix = {}
    for prefix in prefixes:
        nr = 0
        for rn in rns:
            pg = output_per_raw[rn].get(prefix,None)
            if pg is None:
                continue
            for g,pitset in pg.items():
                if g == 'feature_names_filtered':
                    continue
                for it_set,multi_clf_output in pitset.items():
                    nr += 1

        if not skip_plot:
            ww = 5; hh = 2
            fig,axs = plt.subplots(nr,nc, figsize=(nc*ww,nr*hh))
            plt.subplots_adjust(bottom=0.02, top=1-0.02)
            axs = axs.reshape((nr,nc))


        #nc = 1
        # left  = 0.125  # the left side of the subplots of the figure
        # right = 0.9    # the right side of the subplots of the figure
        # bottom = 0.1   # the bottom of the subplots of the figure
        # top = 0.9      # the top of the subplots of the figure
        # wspace = 0.2   # the amount of width reserved for blank space between subplots
        # hspace = 0.2   # the amount of height reserved


        feat_names_per_raw = {}

        rowind_per_perflist = {}
        for pl in perflists:
            rowind_per_perflist[pl] = 0
        for rn in rns:

            feat_names_per_pg_piset = {}
            #for prefix,pg in output_per_raw[rn].items():
            pg = output_per_raw[rn].get(prefix,None)
            if pg is None:
                continue
            for g,pitset in pg.items():
                if g == 'feature_names_filtered':
                    continue
                for it_set,multi_clf_output in pitset.items():
                    print(f'{rn} ----------------')

                    feat_names_per_perflist = {}
                    for pli,perflist in enumerate(perflists):
                        reatnames_res = None
                        pt = perflist
            #             cur = multi_clf_output[pt][-3]
            #             #print( cur['featinds_present'], cur['perf_aver'] )

            #             cur = multi_clf_output[pt][0]
            #             print( cur['featinds_present'][:3] , cur['perf_aver']  )

            #             cur = multi_clf_output[pt][-2]
            #             print( cur['featinds_present'][:3], cur['perf_aver'] )

                        ind_to_use = -1
                        cur = multi_clf_output[pt][ind_to_use]
                        featinds = cur['featinds_present']
                        if tuple(featinds[:5]) == tuple(range(5) ):   # dirty hack, better checking stop_now
                            print(rn,'changing to prev')
                            ind_to_use = -2
                            featinds = cur['featinds_present']
                            cur = multi_clf_output[pt][ind_to_use]
                        print('  ',len(featinds), cur['featinds_present'][:3], cur['perf_aver'] )
                        featnames = multi_clf_output['feature_names_filtered']

                        rng = range(1,len(multi_clf_output[pt]) + ind_to_use +1)
                        sens = [multi_clf_output[pt][ind_]['perf_aver'][0] for ind_ in rng]
                        spec = [multi_clf_output[pt][ind_]['perf_aver'][1] for ind_ in rng]
                        nums = [len(multi_clf_output[pt][ind_]['featinds_present']) for ind_ in rng]

                        # ind in the original array

                        ps = np.minimum( np.array(sens), np.array(spec) )
                        good_perf_inds = np.where( ps >  balance_level)[0]
                        if len(good_perf_inds):
                            # ind in original array
                            ind_balanced = good_perf_inds[0] + 1
                        else:
                            ind_balanced = None
                            print(f'Not reaching balanced level, max sens = {np.max(sens)*100:.2f}% :(')

                        best_sens = multi_clf_output[pt][0]['perf_aver'][0]
                        figtitle = f'{rn}--{g}--{it_set}: {perflist}'
                        if not skip_plot:
                            ax = axs[rowind_per_perflist[perflist],pli]
                            ax.plot(nums,sens,label='sens',c='b')
                            ax.plot(nums,spec,label='spec',c='brown')
                            ax.set_title(figtitle)
                            ax.axhline(y=best_sens,c='b',ls=':',label=f'best_sens={best_sens *100:.2f}%')
                        if ind_balanced is not None:
                            num_balanced = nums[ind_balanced-1]
                            if not skip_plot:
                                ax.axvline(x=num_balanced,c='r',ls=':',label=f'num_balanced={num_balanced}')

                            featinds = multi_clf_output[pt][ind_balanced]['featinds_present']#[ind]
                            #feat_names_per_pg_piset[(g,it_set)] =
                            featnames_res = featnames[featinds]
                            assert num_balanced == len(featnames_res)

                            print(f'{figtitle} num_balanced = {num_balanced}')
                        else:
                            featnames_res = None

                        if not skip_plot:
                            ax.legend(loc='lower right')
                            ax.set_ylim(-0.01,1.01)
                            if xlim is not None:
                                ax.set_xlim(xlim)
                        #ax2 = plt.gca().twiny()
                        #ax2.set_xticks(rng)

                        fip = multi_clf_output['perfs_XGB'][-1]['featinds_present']
                        fip_fs = multi_clf_output['perfs_XGB_fs'][-1]['featinds_present']
                        intersect_len, symdif_len = len( set(fip) & set(fip_fs) ), len( set(fip) ^ set(fip_fs) )
                        print(f'{figtitle}   len(fip)={len(fip)}, len(fip_fs)={len(fip_fs)}, intersect = {intersect_len}, symdif = {symdif_len}')
                        keyname= 'PCA_XGBfeats' #will change in future to PCA_XGBfeats
                        print(f'num PCA of XGB minfeat selected = {multi_clf_output[keyname].n_components_} ' )

                        #f = np.load( multi_clf_output['filename_full'], allow_pickle=True )
                        #print(len(featnames_res) )

                        rowind_per_perflist[perflist] += 1
                        feat_names_per_perflist[perflist] =featnames_res

                        #display(feat_names_per_perflist)
                    feat_names_per_pg_piset[(g,it_set)] = feat_names_per_perflist

            feat_names_per_raw[rn] = feat_names_per_pg_piset

            #import utils_postprocess as pp
            #pp.printDict(feat_names_per_raw, max_depth=6)


            #plt.suptitle(f'{prefix}_{pt}')

        if not skip_plot:
            plt.tight_layout()
            pt_str = ','.join(perflists)
            plt.savefig(pjoin(gv.dir_fig, f'nfeat_vs_sens_{prefix}_{pt_str}.pdf') )
            plt.close()


        feat_names_per_prefix[prefix] = feat_names_per_raw

    return feat_names_per_prefix


def plotImportantFeatLocations(sind_str, multi_clf_output,
                               featnames, head_subj_ind=None,
                               color_by_ftype=True, seed=1):
    # featnames here are NOT nice, but it is already a subset, not full set

    ### It should be like that (perhaps) but it is not :(
    # we will by default plot everything on the head of S01, it does not really
    # matter, just the brain shape changes

    import pymatreader

    #rncur = rawnames[0] + '_off_hold'
    #sind_str,mc,tk  = utils.getParamsFromRawname(rncur)
    if head_subj_ind is None:
        rncur = sind_str + '_off_hold'
    else:
        rncur = head_subj_ind + '_off_hold'
    sources_type=multi_clf_output['info']['sources_type']
    src_file_grouping_ind = multi_clf_output['info']['src_grouping_fn']
    #src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
    #                                                    sources_type,src_file_grouping_ind)
    #src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    src_rec_info_fn_full = utils.genRecInfoFn(rncur,sources_type,src_file_grouping_ind)
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)


    #print( list(rec_info.keys()) )

    labels_dict = rec_info['label_groups_dict'][()]
    srcgroups_dict = rec_info['srcgroups_dict'][()]
    coords = rec_info['coords_Jan_actual'][()]
    srcgrouping_names_sorted = rec_info['srcgroups_key_order'][()]
    sgdn = 'all_raw'

    ############

    featnames_nice = utils.nicenFeatNames(featnames,
                                        labels_dict,srcgrouping_names_sorted)
    feat_names_cur = np.array(featnames_nice)

    #print( feat_names_cur )

    #feat_names_cur_nice = featnames_nice[strong_inds]
    #feat_names_cur = featnames_filtered[strong_inds]

    import re
    src_chns_per_feat_nice = []
    src_chns_all_nice = []
    parcel_indices = []
    parcel_indices_all = []
    # for eact featname parse separately which sources play role
    for fni,fn in enumerate(featnames):
        p = 'msrc._[0-9]+_[0-9]+_c[0-9]+'
        source_chns_cur_featname = re.findall(p, fn)
        if len(source_chns_cur_featname) == 0:
            continue

        sides,groupis,parcelis,compis = \
            utils.parseMEGsrcChnamesShortList(source_chns_cur_featname)
        parcel_indices_all += parcelis
        parcel_indices += [parcelis]

        tmp = list(srcgrouping_names_sorted) * 10  # because we have 9 there
        nice_chns_cur_featname = utils.nicenMEGsrc_chnames(source_chns_cur_featname,labels_dict,tmp,
                                prefix='msrc_')
        nice_chns_cur_featname = list(set(nice_chns_cur_featname))
        src_chns_per_feat_nice += [nice_chns_cur_featname]
        src_chns_all_nice += nice_chns_cur_featname
        #print(fn,nice_chns_cur_featname)

    parcel_indices_all = list(set(parcel_indices_all))
    #parcel_indices_all

    roi_labels_ = np.array(  labels_dict[sgdn] )
    # need just for legend
    roi_labels = ['unlabeled'] + list( roi_labels_[parcel_indices_all] )

    srcgrp = np.zeros( srcgroups_dict[sgdn].shape, dtype=srcgroups_dict[sgdn].dtype)

    for pii,pi in enumerate(parcel_indices_all):
        srcgrp[srcgroups_dict[sgdn] == pi] = pii + 1 #list(roi_labels).index( rls[pii])

    ##########

    if color_by_ftype:
        from featlist import parseFeatNames
        r = parseFeatNames(featnames);
        tuples = zip(r['ftype'], r['fb1'], r['mod1'], r['fb2'], r['mod2'] )
        tuples = list(tuples)

        tuples_present = list( set(tuples) )
        color_group_labels = [str(tpl) for tpl in tuples_present]

        #codes_present = list(sorted(set(codes)) )
        codes = [0]*len(tuples)
        for ci in range(len(tuples)):
            codes[ci] = tuples_present.index( tuples[ci] )

        nice_ch1 = utils.nicenMEGsrc_chnames(r['ch1'], labels_dict,tmp, prefix='')
        nice_ch2 = utils.nicenMEGsrc_chnames(r['ch2'], labels_dict,tmp, prefix='', allow_empty=1)

        # set codes for each parcel present
        roi_lab_codes = [0] * len(roi_labels)
        lcr = len('_c0')  # crop len
        # over ROIs found in feature_names
        for rli,roi_lab in enumerate(roi_labels):
            roi_lab_codes[rli] = []
            for ci in range(len(tuples)):
                c1 = roi_lab == nice_ch1[ci][:-lcr]  # condition 1
                c2 = nice_ch2[ci] is not None  # condition 2
                if c2:
                    c2 = roi_lab == nice_ch2[ci][:-lcr]
                if c1 or c2:
                    #print(c1 or c2)
                    roi_lab_codes[rli] += [ codes[ci] ]
            #print(roi_lab_codes[rli])

        for rli,roi_lab in enumerate(roi_labels):
            if roi_lab != 'unlabeled':
                assert len( roi_lab_codes[rli] ) > 0
    else:
        color_group_labels = None
        codes = None



    ###############

    # I want to avoid repeating
    inds_same = np.where( np.array(r['ch1']) == np.array( r['ch2'] ) )[0]
    #inds_same
    inds_notsame = np.setdiff1d( np.arange(len(featnames)) , inds_same )

    a = np.array(r['ch1'])[inds_notsame].tolist() + np.array(r['ch2'])[inds_notsame].tolist()
    a += np.array(r['ch1'])[inds_same].tolist()
    a = utils.nicenMEGsrc_chnames(a, labels_dict,tmp, prefix='', allow_empty=1)
    a = np.array( [ael[:-3] for ael in a if ael is not None ] )

    #roi_labels = ['unlabeled']  # and more, just for example
    sizes_list = []
    for rli,roi_lab in enumerate(roi_labels):
        num_occur = np.sum( a == roi_lab )
        sizes_list += [num_occur]
        if roi_lab != 'unlabeled':
            assert num_occur > 0, roi_lab


    ############

    clrs =  utils.vizGroup2(sind_str,coords,roi_labels,srcgrp, show=False,
                            alpha=.1, figsize_mult=1.5,msz=30, printLog=0,
                            color_grouping=roi_lab_codes,
                            color_group_labels= color_group_labels,
                            sizes=sizes_list, msz_mult=0.3, seed=seed)

    plt.tight_layout()

def getConfmat(mult_clf_output,grouping,it,best_LFP=False,
        reaver_confmats=False,
        normalize_mode = 'true', keep_beh_state_sides = False,
        ret_pct = True, shuffled=False, across = None):
    # returns confmat (averaged over CV folds)  in percents
    if not best_LFP:
        pre_pcm = mult_clf_output['XGB_analysis_versions']['all_present_features']
        if across is not None:
            if across not in pre_pcm['across']:
                print(f'{across} not in pre_pcm["across"] (keys = {pre_pcm["across"].keys() } )')
                return None,None
            else:
                pcm  = pre_pcm['across'][across]
        else:
            pcm = pre_pcm['perf_dict']
    else:
        chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
        pre_pcm = mult_clf_output['XGB_analysis_versions'][f'all_present_features_only_{chn_LFP}']


        if across is not None:
            pcm  = pre_pcm['across'][across]['perf_dict']
        else:
            pcm = pre_pcm['perf_dict']

    if shuffled:
        perf_shuffled = pcm['fold_type_shuffled'][-1]
        pcm = perf_shuffled

    if reaver_confmats:
        ps = pcm.get('perfs_CV', None)
        if isinstance(ps[0],list):
            confmats_cur = [p[-1] for p in ps]
        else:
            confmats_cur = [p['confmat'] for p in ps]
        confmats_cur_normalized = [utsne.confmatNormalize(cm,normalize_mode) for cm in confmats_cur]
        confmat_normalized =  np.array(confmats_cur_normalized).mean(axis=0)
    else:
        confmat = pcm.get('confmat', None)
        if confmat is None:
            confmat = pcm.get('confmat_aver', None)
        assert confmat is not None
        confmat_normalized = utsne.confmatNormalize(confmat,normalize_mode)

    if ret_pct:
        confmat_normalized *= 100
    #confmats_normalized += [confmat_normalized]

    class_label_names              = mult_clf_output.get('class_label_names_ordered',None)
    if keep_beh_state_sides:
        class_label_names_ticks = class_label_names
    else :
        class_label_names_ticks = [cln.replace('_L','').replace('_R','') for cln in class_label_names]

    from globvars import gp

    if keep_beh_state_sides:
        canon_order = gp.int_types_basic_sided
    else:
        #canon_order = gp.int_types_basic

        int_types_basic = ['trem', 'notrem', 'hold', 'move']
        # needed for canonical ordering
        int_types_merge_mvt = ['trem', 'notrem', 'hold&move']
        int_types_merge_notrem = ['trem', 'notrem&hold&move']
        int_types_trem_vs_quiet = ['trem', 'notrem']

        interval_order_per_merge_it = {('merge_nothing','basic'):int_types_basic,
         ('merge_movements','basic'): int_types_merge_mvt,
         ('merge_all_not_trem','basic'): int_types_merge_notrem,
            ('merge_nothing','trem_vs_quiet'): int_types_trem_vs_quiet        ,
            ('merge_all_not_trem','trem_vs_quiet'): int_types_trem_vs_quiet        }

        if 'interval_order_per_merge_it' in vars(gp):
            canon_order = gp.interval_order_per_merge_it[(grouping,it)]
        else:
            canon_order = interval_order_per_merge_it[(grouping,it)]


    perm = [ class_label_names_ticks.index(intname) for intname in canon_order]
    assert tuple(np.array(class_label_names_ticks)[perm]) == tuple( canon_order)
    class_label_names_ticks  = canon_order

    confmat_normalized_reord = confmat_normalized[perm,:][:,perm]

    return confmat_normalized_reord, class_label_names_ticks



# old, now we incoroporate in df directly
def extractBehStateDur(tpll, grp, prefix, CID_most_recent, rawnames, prefixes, subdir):
    pref_print = prefix
    r = {}
    for tpl in sorted(tpll, key= lambda x: rawnames.index(x[0]) * 1000 + prefixes.index(x[1]) ):
        if tpl[2] != grp or tpl[1] != pref_print:
            continue
        cts = tpl[-1]['counts']
        tot = np.sum( list(cts.values()) )
        cts2 = cts.copy()
        s = ''
        for k,v in cts.items():
            k2 = k[:-2]
            #k2 = k
            stmp = f'{v / tot * 100:4.1f}%={v/32:5.1f}s'
            cts2[k2] = stmp
            s += f', {k2}: {stmp}'
        #print(tpl[:2],  cts2)
        print(f'{tpl[0]:7} {s[1:]}')
        #print(tpl[:2],  cts2['trem_L'])
        #k = ','.join( tpl[:-1] )
        k = tpl[:-1]
        r[k] = {}
        r[k]['counts'] = cts
        r[k]['infostr'] = s
        r[k]['total'] = tot / 32
        r[k]['bacc'] = tpl[-1]['XGB_analysis_versions']['all_present_features']['perf_dict']\
            ['perf_aver']['balanced_accuracy']

    if len(r):
        # import json
        # fn_full  = pjoin(gv.data_dir,subdir,'beh_states_durations.json')
        # with open(fn_full,'w') as f:
        #     json.dump(r,f)
        fn_full  = pjoin(gv.data_dir,subdir,f'beh_states_durations_CID{CID_most_recent}.npz')
        np.savez(fn_full, r)
        print('beh state duration saved to ',fn_full)
    else:
        print('zero len!')

    return r


def getMocFromRow(row : dict, output_per_raw: dict) -> dict:
    grp,it = row['grouping'],row['interval_set']
    try:
        moc = output_per_raw[row['rawname']][row['prefix']][grp][it]
    except KeyError as e:
        print(e)
        moc = None
    return moc

def getMocFromRowMultiOPR(row, pptype2res):
    '''
    pptype2res -- str -> output_per_raw
    '''
    ppt = row['pptype']
    #for ppt in pptype2res:
    #cond = df['pptype'] == ppt
    opr = pptype2res[ppt]
    
    moc = getMocFromRow(row,opr)
    return moc

def getTremorDetPerf(mult_clf_output,grouping,it, ret_pct=False,
        across=None, sidelet=None, allow_missing_tremor=False):
    # returns in percents
    if mult_clf_output is None:
        return np.nan
    cm,clnames = getConfmat(mult_clf_output,grouping,it,ret_pct=ret_pct,
            across=across)
    if cm is None:
        return np.nan
    else:
        lbl = 'trem'
        if sidelet is not None:
            lbl += '_' + sidelet
        if lbl not in clnames:
            if not allow_missing_tremor:
                raise ValueError('{} not in clnames = {}!'.format(lbl, clnames) )
            else:
                return np.nan
        else:
            ti = clnames.index(lbl)
            return cm[ti,ti]

def chanceLevelConfmat(countsd):
    # Let's say you have these counts for each class
    #counts = {"notrem": 300, "tremor": 200, "move": 400, "hold": 100}
    assert isinstance(countsd, odict)
    counts = odict(countsd)
    # Calculate the total number of instances
    total = sum(counts.values())

    # Create a dataframe for the confusion matrix
    n = len(countsd)
    confusion_matrix = pd.DataFrame(np.zeros((n,n)), index=counts.keys(), columns=counts.keys())

    # Fill each row with the proportions
    for class_name in confusion_matrix.index:
        for class_name2 in confusion_matrix.columns:
            confusion_matrix.loc[class_name,class_name2] = counts[class_name2] / total
    return confusion_matrix

def plotConfmats2(dfc, output_per_raw, normalize_mode = 'true', best_LFP=False, common_norm = True,
                reaver_confmats = 0, ww=3,hh=3, keep_beh_state_sides = True,
                 keep_subj_list_title = False, labelpad_cbar = 100, labelpad_x = 20, labelpad_y = 20,
                colorbar_axes_bbox = [0.80, 0.1, 0.045, 0.8], rename_class_names = None):
    '''
    plots mutiple confmats, for every row of dfc

    in this version of this function output_per_raw is ALL data and dfc is FILTERED
    normalize_mode == true means that we start from real positives (nut just correctly predicted)
    common_norm -- colorbar always has max 100 and min 0 (regardless of actual min and max vals)

    if best LFP then plot for the best channel
    '''
    #tpll = pp.multiLevelDict2TupleList(outputs_grouped,4,3)
    assert len(dfc)


    rawnames = list(sorted( dfc['rawname'].unique() ))
    # TODO: it will plot all confmats for all the outputs, I might want to restruct to just one
    #nc = int( np.ceil( np.sqrt( len(outputs_grouped) ) ) );
    #nr = len(outputs_grouped) // nc; #nc= len(scores_stats) - 2;
    #nr = len(outputs_grouped) // nc; #nc= len(scores_stats) - 2;
    nc = int( np.ceil( np.sqrt( len(rawnames) ) ) );
    nr = len(rawnames) / nc; #nc= len(scores_stats) - 2;
    nr = int(np.ceil(nr) )

    fig,axs = plt.subplots(nr,nc, figsize = (nc*ww + ww*0.5,nr*hh))#, gridspec_kw={'width_ratios': [1,1,3]} );
    if nr == 1 and nc == 1:
        axs = np.array([[axs]])

    #rawnames = list(sorted(outputs_per_raw.keys()) )

    axs = axs.flatten()
    for ax in axs:
        ax.set_visible(False)

    confmats_normalized = []
    for rowi, row in dfc.iterrows():
        #rn = tpl[0]
        #mult_clf_output = tpl[-1]
        rn = row['rawname']
        mult_clf_output = getMocFromRow(row, output_per_raw)

        if not best_LFP:
            pcm = mult_clf_output['XGB_analysis_versions']['all_present_features']['perf_dict']
        else:
            chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
            pcm = mult_clf_output['XGB_analysis_versions'][f'all_present_features_only_{chn_LFP}']['perf_dict']


        if reaver_confmats:
            ps = pcm.get('perfs_CV', None)
            if isinstance(ps[0],list):
                confmats_cur = [p[-1] for p in ps]
            else:
                confmats_cur = [p['confmat'] for p in ps]
            confmats_cur_normalized = [utsne.confmatNormalize(cm,normalize_mode) for cm in confmats_cur]
            confmat_normalized =  np.array(confmats_cur_normalized).mean(axis=0)*100
        else:
            confmat = pcm.get('confmat', None)
            if confmat is None:
                confmat = pcm.get('confmat_aver', None)
            assert confmat is not None
            confmat_normalized = utsne.confmatNormalize(confmat,normalize_mode) * 100
        confmats_normalized += [confmat_normalized]

    #     if normalize_mode == 'total':
    #         confmat_normalized = confmat / np.sum(confmat) * 100
    #     elif normalize_mode == 'col':
    #         confmat_normalized = confmat / np.sum(confmat, axis=1)[None,:] * 100


    # canonical order of interval names
    from globvars import gp
    if keep_beh_state_sides:
        canon_order = gp.int_types_basic_sided
    else:
        canon_order = gp.int_types_basic


    #if class_counts is not None:
    #else:
    #    cm_chance = None


    import matplotlib as mpl
    if common_norm:
        norm = mpl.colors.Normalize(vmin=0, vmax=100)
    else:
        mn,mx = np.nan,np.nan
        raise ValueError('Need to implement mn,mx computation!')
        norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
    #for axi,(ax,(rn,(spec_key,mult_clf_output) ) ) in enumerate(zip(axs, outputs_grouped.items() ) ):
    confmats_normalized_reord = []
    confmats_chance = []
    xts2 = []

    for rowi, row in dfc.iterrows():
        rn = row['rawname']
        print(f'Starting plotting confmat for {rn}')
        mult_clf_output = getMocFromRow(row, output_per_raw)

        class_label_names              = mult_clf_output.get('class_label_names_ordered',None)
        # if not in archive, regenerate
        if class_label_names is None:
            class_labels_good = mult_clf_output['class_labels_good']
            revdict = mult_clf_output['revdict']
            from sklearn import preprocessing
            lab_enc = preprocessing.LabelEncoder()
            # just skipped class_labels_good
            lab_enc.fit(class_labels_good)
            class_labels_good_for_classif = lab_enc.transform(class_labels_good)
            class_label_ids = lab_enc.inverse_transform( np.arange( len(set(class_labels_good_for_classif)) ) )
            class_label_names = [revdict[cli] for cli in class_label_ids]

        #        ## confmat_ratio[i,j] = ratio of true i-th predicted as j-th among total
        #confmat = confmats[axi]
        #confmat_normalized = utsne.confmatNormalize(confmat,normalize_mode) * 100
        confmat_normalized = confmats_normalized[rowi]


        class_counts = odict({})
        for class_name in canon_order:
            class_counts[class_name] = row[f'numpts_{class_name}']
        cm_chance = chanceLevelConfmat(class_counts) 
        cm_chance = cm_chance.values * 100 
        confmats_chance += [cm_chance]


        #sind_str = rn.split('_')[0]
        #axind =  sind_strs.index(sind_str)
        axind =  rawnames.index(rn)
        ax = axs[axind]
        ttl = rn
        if not keep_subj_list_title:
            end = rn.split('_')[-1]
            ll = len(end)
            #ttl = rn[:-ll-1].upper()
            #ttl = end.upper()
        ax.set_title(ttl.upper())


        if keep_beh_state_sides:
            class_label_names_ticks = class_label_names
        else :
            class_label_names_ticks = [cln[:-2] for cln in class_label_names]

        perm = [ class_label_names_ticks.index(intname) for intname in canon_order]
        assert tuple(np.array(class_label_names_ticks)[perm]) == tuple( canon_order)
        class_label_names_ticks  = canon_order

        # changing class names to publication-ready
        if rename_class_names is not None:
            class_label_names_ticks2 = []
            for cln in class_label_names_ticks:
                if keep_beh_state_sides:
                    cln_ = cln[:-2]
                else:
                    cln_ = cln

                if cln_ in rename_class_names:
                    cln_res = cln.replace( cln_, rename_class_names[cln_] )
                else:
                    cln_res = cln
                class_label_names_ticks2 += [cln_res]
            class_label_names_ticks = class_label_names_ticks2
        #print(rn, 'class_label_names =', class_label_names)
        #print(rn, 'class_label_names_ticks=', class_label_names_ticks)
        confmat_normalized_reord = confmat_normalized[perm,:][:,perm]
        confmats_normalized_reord += [confmat_normalized_reord]

        ax.set_visible(True)
        pc = ax.pcolor(confmat_normalized_reord, norm=norm)
         
        #cm_chance
        if cm_chance is not None:
            display(confmat_normalized_reord,cm_chance)
            diff_matrix = (confmat_normalized_reord - cm_chance)
            #sns.heatmap(original_matrix, annot=True, fmt=".0f", cmap='YlGnBu')
            #horshift= 0.2
            horshift= 0
            # Print the values of the difference matrix on the heatmap
            for i in range(diff_matrix.shape[0]):
                for j in range(diff_matrix.shape[1]):
                    mel0 = confmat_normalized_reord[i,j]
                    mel = diff_matrix[i, j] 
                    if mel >= 0:
                        sgn = '+'
                    else:
                        sgn = ''
                    ax.text(j + 0.5 + horshift, i + 0.5, 
                            f'{mel0:.0f}%\n({sgn}{mel:.0f}%)', 
                        ha='center', va='center', color='red', size=10)

        # setting ticks and labels nicely, not for all axes, only for the left and bottom ones
        rowi,coli = np.unravel_index(axind,(nr,nc))
        xts = ax.get_xticks()
        print(xts)

        mnxts,mxxts = np.min(xts), np.max(xts)
        shift  = ( (mxxts - mnxts) / 4) / 2
        #shift  = (xts[1] + xts[0]) / 2
        #shift = 0 #debug
        #xtsd = (shift)  + xts[:-1]
        #return ax
        if rowi == nr-1:
            xts2 = shift + np.linspace(xts[0],xts[-1],len(class_label_names) + 1 )
            xts2[-1] = xts[-1]
            ax.set_xticks(xts2 )
            ax.set_xticklabels( class_label_names_ticks + [''],rotation=90)
            ax.set_xlabel('Predicted', labelpad=labelpad_x)
        else:
            ax.set_xticks([])

        if coli == 0:
            xts2 = shift + np.linspace(xts[0],xts[-1],len(class_label_names) + 1 )
            xts2[-1] = xts[-1]
            #if len(xts2):
            ax.set_yticks(xts2)
            ax.set_yticklabels( class_label_names_ticks + [''])
            ax.set_ylabel('True', labelpad=labelpad_y)
        else:
            ax.set_yticks([])

        #ax.set_visible(True)
        del confmat_normalized, confmat_normalized_reord

    plt.subplots_adjust(left = 0.15, bottom=0.26, right=0.75, top=0.9)

    ##################################################

    confmat_chance_ = np.array(confmats_chance)

    confmat_normalized_ = np.array(confmats_normalized_reord)
    assert confmat_normalized_.size
    confmat_normalized_diags = np.array( [np.diag(np.diag(cm)) for cm in confmats_normalized_reord] )
    eyes = np.array( [np.eye(confmat_normalized_.shape[-1] ) for cm in confmats_normalized_reord], dtype=bool)
    confmat_normalized_offdiags = confmat_normalized_ - confmat_normalized_diags
    confmat_normalized_offdiags_largedval = confmat_normalized_offdiags + eyes * 1e5
    mx = np.max(confmat_normalized_)
    mn = np.min(confmat_normalized_)
    #mn_diag     = np.min ( confmat_normalized_diags  )
    #mn_off_diag = np.min (  confmat_normalized_offdiags_largedval  )
    #mx_off_diag = np.max (  confmat_normalized_offdiags  )


    confmat_chance_diags_els    = confmat_chance_[ np.where( eyes ) ]
    confmat_chance_offdiags_els = confmat_chance_[ np.where( ~eyes ) ]

    confmat_normalized_diags_els = confmat_normalized_[ np.where( eyes ) ]
    confmat_normalized_offdiags_els = confmat_normalized_[ np.where( ~eyes ) ]

    dcbt = {} # dict color bar ticks
    dcbt['max'] = mx
    dcbt['min'] = mn

    mn_diag     = np.min ( confmat_normalized_diags_els ); dcbt['min_diag'] = mn_diag
    mx_diag = np.max ( confmat_normalized_diags_els ); dcbt['max_diag'] = mx_diag
    dcbt['min_off_diag'] = np.min ( confmat_normalized_offdiags_els )
    dcbt['max_off_diag'] = np.max ( confmat_normalized_offdiags_els )

    me_diag = np.mean(confmat_normalized_diags_els); dcbt['mean_diag'] = me_diag
    me_off_diag = np.mean (  confmat_normalized_offdiags_els  ); dcbt['mean_off_diag'] = me_off_diag

    dcbt['max_chance'] = np.max(confmat_chance_)
    dcbt['min_chance'] = np.min(confmat_chance_)
    dcbt['min_chance_diag']     = np.min ( confmat_chance_diags_els )
    dcbt['max_chance_diag']     = np.max ( confmat_chance_diags_els )
    dcbt['min_chance_off_diag'] = np.min ( confmat_chance_offdiags_els )
    dcbt['max_chance_off_diag'] = np.max ( confmat_chance_offdiags_els )

    dcbt['mean_chance_diag']     = np.mean(confmat_chance_diags_els)
    dcbt['mean_chance_off_diag'] = np.mean (  confmat_chance_offdiags_els  )
    print(dcbt)

    ##################################################
    # create colorbar with tick
    cax = plt.axes(colorbar_axes_bbox)
    clrb = plt.colorbar(pc, cax=cax)
    #cax.set_ylabel(f'percent of {normalize_mode} points (in a CV fold)', labelpad=labelpad_cbar )
    cax.set_ylabel(f'percent of {normalize_mode} points', labelpad=labelpad_cbar )

    ax2 = clrb.ax.twinx()
    y0,y1 = cax.get_ybound()  # they are from 0 to 1
    #ticks       = [  mn_off_diag, mn_diag,  mx_off_diag, me_diag, me_off_diag]
    tick_labels0 = [ 'min_off_diag', 'min_diag',  'max_off_diag', 'mean_diag', 'mean_off_diag' ]
    if common_norm:
        #ticks       = [  mn_off_diag, mn_diag,  mx_off_diag, mx,mn ]
        tick_labels0 = [ 'min off diag', 'min diag',  'max off diag', 'max' , 'min' ]

    #ticks       = [  mn_off_diag, mn_diag,  mx_off_diag, mx,mn, me_diag, me_off_diag, mx_diag ]
    tick_labels0 = [ 'min_off_diag', 'min_diag',  'max_off_diag', 'max' , 'min','mean_diag', 'mean_off_diag', 'max_diag', 
            'mean_chance_off_diag', 'mean_chance_diag'  ]

    ticks = []
    tick_labels =[]
    for tl in tick_labels0:
        tick_labels += [tl.replace('_',' ' ) ]
        ticks += [dcbt[tl] ]


    desarr = np.array( ticks )
    #ax2.set_yticks( desarr/ (y1-y0) )
    ax2.set_yticks( desarr )
    ax2.set_yticklabels( tick_labels )
    ax2.set_ylim( y0,y1)

    print(mn,mx, mn_diag)
    return cax, clrb, confmats_normalized_reord, confmat_normalized_offdiags, confmat_normalized_diags_els
    #plt.tight_layout()

#def plotConfmats(outputs_grouped, normalize_mode = 'true', best_LFP=False, common_norm = True,
#                 ww=3,hh=3):
def plotConfmats(outputs_per_raw, normalize_mode = 'true', best_LFP=False, common_norm = True,
                reaver_confmats = 0, ww=3,hh=3, keep_beh_state_sides = True,
                 keep_subj_list_title = False, labelpad_cbar = 100, labelpad_x = 20, labelpad_y = 20,
                colorbar_axes_bbox = [0.80, 0.1, 0.045, 0.8], rename_class_names = None):
    '''
    normalize_mode == true means that we start from real positives (nut just correctly predicted)
    common_norm -- colorbar always has max 100 and min 0 (regardless of actual min and max vals)

    if best LFP then plot for the best channel
    '''
    #tpll = pp.multiLevelDict2TupleList(outputs_grouped,4,3)
    tpll = multiLevelDict2TupleList(outputs_per_raw,4,3)
    assert len(tpll)

    # TODO: it will plot all confmats for all the outputs, I might want to restruct to just one
    #nc = int( np.ceil( np.sqrt( len(outputs_grouped) ) ) );
    #nr = len(outputs_grouped) // nc; #nc= len(scores_stats) - 2;
    #nr = len(outputs_grouped) // nc; #nc= len(scores_stats) - 2;
    nc = int( np.ceil( np.sqrt( len(outputs_per_raw) ) ) );
    nr = len(outputs_per_raw) / nc; #nc= len(scores_stats) - 2;
    nr = int(np.ceil(nr) )

    fig,axs = plt.subplots(nr,nc, figsize = (nc*ww + ww*0.5,nr*hh))#, gridspec_kw={'width_ratios': [1,1,3]} );
    if nr == 1 and nc == 1:
        axs = np.array([[axs]])

    rawnames = list(sorted(outputs_per_raw.keys()) )

    axs = axs.flatten()
    for ax in axs:
        ax.set_visible(False)

    confmats_normalized = []
    for axi,(ax,tpl) in enumerate(zip(axs, tpll ) ):
        rn = tpl[0]
        mult_clf_output = tpl[-1]

        if not best_LFP:
            pcm = mult_clf_output['XGB_analysis_versions']['all_present_features']['perf_dict']
        else:
            chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
            pcm = mult_clf_output['XGB_analysis_versions'][f'all_present_features_only_{chn_LFP}']['perf_dict']


        if reaver_confmats:
            ps = pcm.get('perfs_CV', None)
            if isinstance(ps[0],list):
                confmats_cur = [p[-1] for p in ps]
            else:
                confmats_cur = [p['confmat'] for p in ps]
            confmats_cur_normalized = [utsne.confmatNormalize(cm,normalize_mode) for cm in confmats_cur]
            confmat_normalized =  np.array(confmats_cur_normalized).mean(axis=0)*100
        else:
            confmat = pcm.get('confmat', None)
            if confmat is None:
                confmat = pcm.get('confmat_aver', None)
            assert confmat is not None
            confmat_normalized = utsne.confmatNormalize(confmat,normalize_mode) * 100
        confmats_normalized += [confmat_normalized]

    #     if normalize_mode == 'total':
    #         confmat_normalized = confmat / np.sum(confmat) * 100
    #     elif normalize_mode == 'col':
    #         confmat_normalized = confmat / np.sum(confmat, axis=1)[None,:] * 100


    # canonical order of interval names
    from globvars import gp
    if keep_beh_state_sides:
        canon_order = gp.int_types_basic_sided
    else:
        canon_order = gp.int_types_basic


    import matplotlib as mpl
    if common_norm:
        norm = mpl.colors.Normalize(vmin=0, vmax=100)
    else:
        mn,mx = np.nan,np.nan
        raise ValueError('Need to implement mn,mx computation!')
        norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
    #for axi,(ax,(rn,(spec_key,mult_clf_output) ) ) in enumerate(zip(axs, outputs_grouped.items() ) ):
    confmats_normalized_reord = []
    xts2 = []
    for tpli,tpl in enumerate(tpll):
        rn = tpl[0]
        mult_clf_output = tpl[-1]

        class_label_names              = mult_clf_output.get('class_label_names_ordered',None)
        # if not in archive, regenerate
        if class_label_names is None:
            class_labels_good = mult_clf_output['class_labels_good']
            revdict = mult_clf_output['revdict']
            from sklearn import preprocessing
            lab_enc = preprocessing.LabelEncoder()
            # just skipped class_labels_good
            lab_enc.fit(class_labels_good)
            class_labels_good_for_classif = lab_enc.transform(class_labels_good)
            class_label_ids = lab_enc.inverse_transform( np.arange( len(set(class_labels_good_for_classif)) ) )
            class_label_names = [revdict[cli] for cli in class_label_ids]

        #        ## confmat_ratio[i,j] = ratio of true i-th predicted as j-th among total
        #confmat = confmats[axi]
        #confmat_normalized = utsne.confmatNormalize(confmat,normalize_mode) * 100
        confmat_normalized = confmats_normalized[tpli]

        #sind_str = rn.split('_')[0]
        #axind =  sind_strs.index(sind_str)
        axind =  rawnames.index(rn)
        ax = axs[axind]
        ttl = rn
        if not keep_subj_list_title:
            end = rn.split('_')[-1]
            ll = len(end)
            #ttl = rn[:-ll-1].upper()
            #ttl = end.upper()
        ax.set_title(ttl.upper())


        if keep_beh_state_sides:
            class_label_names_ticks = class_label_names
        else :
            class_label_names_ticks = [cln[:-2] for cln in class_label_names]

        perm = [ class_label_names_ticks.index(intname) for intname in canon_order]
        assert tuple(np.array(class_label_names_ticks)[perm]) == tuple( canon_order)
        class_label_names_ticks  = canon_order

        # changing class names to publication-ready
        if rename_class_names is not None:
            class_label_names_ticks2 = []
            for cln in class_label_names_ticks:
                if keep_beh_state_sides:
                    cln_ = cln[:-2]
                else:
                    cln_ = cln

                if cln_ in rename_class_names:
                    cln_res = cln.replace( cln_, rename_class_names[cln_] )
                else:
                    cln_res = cln
                class_label_names_ticks2 += [cln_res]
            class_label_names_ticks = class_label_names_ticks2
        #print(rn, 'class_label_names =', class_label_names)
        #print(rn, 'class_label_names_ticks=', class_label_names_ticks)
        confmat_normalized_reord = confmat_normalized[perm,:][:,perm]
        confmats_normalized_reord += [confmat_normalized_reord]

        pc = ax.pcolor(confmat_normalized_reord, norm=norm)

        # setting ticks and labels nicely, not for all axes, only for the left and bottom ones
        rowi,coli = np.unravel_index(axind,(nr,nc))
        xts = ax.get_xticks()
        #print(xts)
        mnxts,mxxts = np.min(xts), np.max(xts)
        shift  = ( (mxxts - mnxts) / 4) / 2
        #shift  = (xts[1] + xts[0]) / 2
        #shift = 0 #debug
        #xtsd = (shift)  + xts[:-1]
        if rowi == nr-1:
            xts2 = shift + np.linspace(xts[0],xts[-1],len(class_label_names) + 1 )
            xts2[-1] = xts[-1]
            ax.set_xticks(xts2 )
            ax.set_xticklabels( class_label_names_ticks + [''],rotation=90)
            ax.set_xlabel('Predicted', labelpad=labelpad_x)
        else:
            ax.set_xticks([])

        if coli == 0:
            xts2 = shift + np.linspace(xts[0],xts[-1],len(class_label_names) + 1 )
            xts2[-1] = xts[-1]
            #if len(xts2):
            ax.set_yticks(xts2)
            ax.set_yticklabels( class_label_names_ticks + [''])
            ax.set_ylabel('True', labelpad=labelpad_y)
        else:
            ax.set_yticks([])

        ax.set_visible(True)
        del confmat_normalized, confmat_normalized_reord

    plt.subplots_adjust(left = 0.15, bottom=0.26, right=0.75, top=0.9)

    ##################################################

    confmat_normalized_ = np.array(confmats_normalized_reord)
    assert confmat_normalized_.size
    confmat_normalized_diags = np.array( [np.diag(np.diag(cm)) for cm in confmats_normalized_reord] )
    eyes = np.array( [np.eye(confmat_normalized_.shape[-1] ) for cm in confmats_normalized_reord], dtype=bool)
    confmat_normalized_offdiags = confmat_normalized_ - confmat_normalized_diags
    confmat_normalized_offdiags_largedval = confmat_normalized_offdiags + eyes * 1e5
    mx = np.max(confmat_normalized_)
    mn = np.min(confmat_normalized_)
    #mn_diag     = np.min ( confmat_normalized_diags  )
    #mn_off_diag = np.min (  confmat_normalized_offdiags_largedval  )
    #mx_off_diag = np.max (  confmat_normalized_offdiags  )

    confmat_normalized_diags_els = confmat_normalized_[ np.where( eyes ) ]
    confmat_normalized_offdiags_els = confmat_normalized_[ np.where( ~eyes ) ]

    mn_diag     = np.min ( confmat_normalized_diags_els )
    mn_off_diag = np.min ( confmat_normalized_offdiags_els )
    mx_off_diag = np.max ( confmat_normalized_offdiags_els )

    me_diag = np.mean(confmat_normalized_diags_els)
    me_off_diag = np.mean (  confmat_normalized_offdiags_els  )

    ##################################################
    # create colorbar with tick
    cax = plt.axes(colorbar_axes_bbox)
    clrb = plt.colorbar(pc, cax=cax)
    #cax.set_ylabel(f'percent of {normalize_mode} points (in a CV fold)', labelpad=labelpad_cbar )
    cax.set_ylabel(f'percent of {normalize_mode} points', labelpad=labelpad_cbar )

    ax2 = clrb.ax.twinx()
    y0,y1 = cax.get_ybound()  # they are from 0 to 1
    ticks       = [  mn_off_diag, mn_diag,  mx_off_diag, me_diag, me_off_diag]
    tick_labels = [ 'min_off_diag', 'min_diag',  'max_off_diag', 'mean_diag', 'mean_off_diag' ]
    if common_norm:
        ticks       = [  mn_off_diag, mn_diag,  mx_off_diag, mx,mn ]
        tick_labels = [ 'min off diag', 'min diag',  'max off diag', 'max' , 'min' ]
    desarr = np.array( ticks )
    #ax2.set_yticks( desarr/ (y1-y0) )
    ax2.set_yticks( desarr )
    ax2.set_yticklabels( tick_labels )
    ax2.set_ylim( y0,y1)

    print(mn,mx, mn_diag)
    return cax, clrb, confmats_normalized_reord, confmat_normalized_offdiags, confmat_normalized_diags_els
    #plt.tight_layout()

def recalcPerfFromCV(perfs_CV,ind):
    #ps = pcm.get('perfs_CV', None)
    ps = perfs_CV
    if isinstance(ps[0], tuple):
        confmats_cur = [p[-1] for p in ps]
    elif isinstance(ps[0], dict):
        confmats_cur = [p['confmat'] for p in ps]
    else:
        raise ValueError(f'Wrong type {type(ps[0])}')
    return perfFromConfmat(confmats_cur,ind)
    #confmats_cur_normalized = [utsne.confmatNormalize(cm,normalize_mode) for cm in confmats_cur]
    #confmat_normalized =  np.array(confmats_cur_normalized).mean(axis=0)*100

def perfFromConfmat(confmat,ind):
    # second coord is
    if isinstance(confmat, np.ndarray):
        assert confmat.ndim == 2
        other_inds = np.setdiff1d( np.arange(confmat.shape[0] ), [ind] )
        # confmat[:,ind] -- whatever predicted to be ind
        prec = confmat[ind,ind] / np.sum( confmat[:,ind] )
        sens = confmat[ind,ind] / np.sum( confmat[ind,:] )

        #denom = np.sum( confmat[:,other_inds] )
        denom = np.sum( confmat[other_inds,:] )
        spec = np.sum(np.diag(confmat)[other_inds] ) / denom
        return sens,spec
    else:
        assert isinstance(confmat, list)
        ps = [ perfFromConfmat(cm,ind) for cm in confmat ]
        #oo = np.array(zip(*ps))
        #sens,spec = oo.mean(axis=1)
        sens,spec = np.array(ps).mean(axis=0)
        return sens,spec


#TP+FN -- sensitiv  -- total number of positives
#TP+FP -- precision -- over all predicted as positive

##################################################

    #nr = 1
    #nc = 2
    #ww = 6; hh = 4
    #fig,axs = plt.subplots(nr,nc, figsize = (ww*nc, hh*nr))
    #axs = axs.reshape((nr,nc))
    #axind = 0


    #prefixes_sorted #= list(sorted(rowinfo.keys()))
    #prefixes_wnums = []
    ##str_per_pref = {}

    ##pvec_dicts = [pvec_summary_per_prefix, pvec_summary_red_per_prefix]

    #for ci,clf_type in enumerate(keys):
    #    #pvec_dict_cur = pvec_dicts[ci]
    #    xs = []
    #    ys = []
    #    nums_red = []

    #    ktu = keys.index(clf_type)
    #    if  clf_type.endswith('_red') or clf_type.endswith('_selMinFeatSet'):
    #        ktu = 1 - keys.index(clf_type)
    #    ktu_inv = 1 - keys.index(keys[ktu] )
    #    for prefix in prefixes_sorted:
    #        prefinfo = rowinfo[prefix]

    #        num = prefinfo.get('num',-1)
    #        num_red = prefinfo.get('num_red',-1)
    #        xs += [ num]

    #        pvec = pvec_summary_per_prefix_per_key[keys[ktu] ][prefix]
    #        pvec_red = pvec_summary_red_per_prefix_per_key[ keys[ktu_inv] ][prefix]

    #        if pveclen == 3:
    #            str_to_put_ =  '{:.0f},{:.0f},{:.0f}'.format(100*pvec[0],100*pvec[1],100*pvec[2])
    #            str_to_put_red =  '{:.0f},{:.0f},{:.0f}'.format(100*pvec_red[0],100*pvec_red[1],100*pvec_red[2])
    #        elif pveclen == 2:
    #            str_to_put_ =  '{:.0f},{:.0f}'.format(100*pvec[0],100*pvec[1])
    #            str_to_put_red =  '{:.0f},{:.0f}'.format(100*pvec_red[0],100*pvec_red[1])
    #        else:
    #            raise ValueError('wrong pveclen')

    #        prefixes_wnums += [prefix + f'_{num} (min-> {num_red}) = ({str_to_put} -> {str_to_put_red})']

    #        if  clf_type.endswith('_red') or clf_type.endswith('_selMinFeatSet'):
    #            pvec_cur = pvec_summary_red_per_prefix_per_key[clf_type][prefix]
    #        else:
    #            pvec_cur = pvec_summary_per_prefix_per_key[clf_type][prefix]
    #        #p = np.mean(pvec)
    #        p = np.min(pvec_cur)
    #        #ys += [prefinfo[perftype]]
    #        ys += [p]
    #        #print(clf_type,prefix)
    #    #print(len(ys))

    #    ax = axs[axind,0]
    #    ax.set_title('Summary')
    #    ax.scatter(xs,ys, c = colors[ci])
    #    ax.set_ylabel(perftype)
    #    ax.set_xlabel('Number of features')
    #    ax.set_ylim(0,1)
    #    #ax.set_xlabel('total feature number')
    #    ####################################
    #    ax = axs[axind,1]
    #    ax.set_title('Summary')
    #    ax.yaxis.tick_right()
    #    sis = np.argsort(xs)
    #    ax.barh(np.array(prefixes_wnums)[sis], np.array(ys)[sis], color = colors[ci], alpha=0.8)
    #    ax.set_xlabel(perftype)
    #    ax.set_xlim(0,1)

    #    print(colors[ci])

    #plt.suptitle('Summary');

    #plt.tight_layout()
    #keystr = ','.join(keys)
    #figfname = f'PerfSummary_{keystr}_perf_kind={perf_kind}_pveclen={pveclen}.pdf'
    #plt.savefig(pjoin(gv.dir_fig, output_subdir,figfname))

def plotFeatHists(rawnames,featnames,featis,X_pri,Xconcat, bindict_per_rawn,
                  ivalis_tb_indarrays_merged,
                  xlim_common = (-5,5), nbins = 20, savefig = True):
    cmap = plt.cm.get_cmap('Set1', max(4,len(rawnames)) )
    from globvars import gp

    nr = len(featis); nc = len(rawnames) + 2 + len(gp.int_types_basic)
    ww = 4; hh = 2.5
    fig,axs = plt.subplots(nr,nc, figsize=(nc*ww,nr*hh))
    axs = axs.reshape((nr,nc))
    bt = None; top=None
    if len(featis) >= 8:
        bt = 0.01
        top = 0.99
    plt.subplots_adjust(bottom=bt,top=top, left=0.01,right=0.99)

    #xlim_common = None

    show_all = 0
    for axi,feati in enumerate(featis):
        featn = featnames[feati]
        print(featn)
        #ax = plt.gca()
        axdef = axs[axi,0]
        axconcat = axs[axi,1]
        for rawi in range(len(X_pri)):
            rawn = rawnames[rawi]
            axcur = axs[axi,rawi+2]

            dat = X_pri[rawi][:,feati]
            adat = np.abs(dat)
            q = np.quantile(adat,0.98)
            mask = adat < q

            # just pull all data together separately for each dataset on the
            # SAME axis
            ax = axdef
            ax.hist(dat[mask],alpha=0.5, density=1,bins=nbins, label=rawn, color=cmap(rawi))

            # just pull all data together separately for each dataset on
            # separate axes
            ax = axcur
            if show_all:
                ax.hist(dat[mask],alpha=0.5, density=1,bins=nbins, label=rawn)
            for iti,it in enumerate(gp.int_types_basic):
                itcur = it + '_L'
                binis = bindict_per_rawn[rawn]['beh_state'].get(itcur,None)
                if binis is None:
                    continue
                dat_it = dat[binis][adat[binis] < q]
                ax.hist(dat_it,alpha=0.5, density=1,bins=nbins, label=itcur, color=cmap(iti))

                axit = axs[axi,iti+len(rawnames) + 2]
                axit.hist(dat_it,alpha=0.5, density=1,bins=nbins,
                                     label=rawn, color=cmap(rawi))
                if (not it.startswith('notrem') ) or len(X_pri) <= 6:
                    axit.legend(loc='upper right')
                axit.set_title(itcur)

            axcur.legend(loc='upper right')
            axcur.set_title(rawn)

        for iti,it in enumerate(gp.int_types_basic):
            itcur = it + '_L'
            binis = ivalis_tb_indarrays_merged.get(itcur,None)
            dat_concat = Xconcat[binis,feati]
            adat_concat = np.abs(dat_concat)
            dat_concat = dat_concat[adat_concat  < np.quantile(adat_concat,0.98) ]
            axconcat.hist(dat_concat,alpha=0.5, density=1,bins=nbins, label=itcur, color=cmap(iti))

    #         for rawi in range(len(X_pri)):
    #             rawn = rawnames[rawi]
    #             axcur = axs[axi,rawi+6]
        axconcat.legend(loc='upper right')
        axconcat.set_title('concat')

        axdef.set_title(featn)
        if len(X_pri) <= 6:
            axdef.legend(loc='lower left')

    if xlim_common is not None:
        for ax in axs.flatten():
            ax.set_xlim(xlim_common)

    if savefig:
        figname = ','.join(rawnames) + f'_hists_n={nr}.pdf'
        plt.savefig(pjoin(gv.dir_fig, figname))
        plt.close()

def loadFullScores(outputs_grouped, crop_fname='auto', feat_subset_name=None,
                   force_reload = False, allow_missing=False):
    # adds full scores to the outputs
    import gc
    import os

    for rn,a in outputs_grouped.items():
        _, mult_clf_output = a
        #(prefix,grp,int_type)
        #tpl_cur = tpll[0]
        #mult_clf_output = tpl_cur[-1]
        featsel = mult_clf_output.get('featsel_per_method',{})
        scores = featsel.get('XGB_Shapley',{}).get('scores',None)
        # temp
#         if scores is not None:
#             del mult_clf_output['featsel_per_method']['XGB_Shapley']['scores']
#             scores = None

        if scores is None or force_reload:
            filename_fullsize = mult_clf_output['filename_full']
            pfsz = Path(filename_fullsize)
            if crop_fname == 'yes':
                do_crop = 1
            elif crop_fname == 'no':
                do_crop = 0
            elif crop_fname == 'auto':
                do_crop = pfsz.name.startswith('_!')

            if do_crop:
                filename_fullsize = pjoin(pfsz.parents[0], pfsz.name[2:])
            else:
                filename_fullsize = pjoin(pfsz.parents[0], pfsz.name)

            if not os.path.exists(filename_fullsize):
                print(f'!!!! {filename_fullsize} does not exist')
                return

            finfo = os.stat( filename_fullsize )
            print(finfo.st_size / (1024**2))
            f = np.load(filename_fullsize,allow_pickle=True)
            results_cur = f['results_cur'][()]

            mult_clf_output['pcaobj'] = f['pcaobj'][()]
            mult_clf_output['icaobj'] = f['icaobj'][()]
            mult_clf_output['XGBobj'] = results_cur['XGBobj']

            shp = results_cur['featsel_per_method']['XGB_Shapley']
            scores = shp.get('scores',None)
            found = False
            if scores is None:
                if not allow_missing:
                    assert feat_subset_name is not None
                if feat_subset_name in shp:
                    scores = shp[feat_subset_name].get('scores',None)
                if not allow_missing:
                    assert scores is not None
                if scores is not None:
                    found = True
                    print(scores.shape)

            #if 'feature_indices_used'

            if found:
                mult_clf_output['featsel_per_method']['XGB_Shapley']['scores'] = scores
            del f
            gc.collect()



def loadEBMExplainer(outputs_grouped, fs, force=False ):
    for rn,a in outputs_grouped.items():
        (prefix,grp,int_type), mult_clf_output = a
        loadEBMExplainer_(mult_clf_output, fs, force=force )

def loadEBMExplainer_(mult_clf_output, fs, force=False, cure_if_possible=True, full_scores = True ):
    pre = mult_clf_output['featsel_per_method']['interpret_EBM']
    if fs is None or (cure_if_possible and fs not in pre):
        clf_dict = pre
    else:
        clf_dict = pre[fs]


    if 'explainer' in clf_dict and not force:
        return
    filename_fullsize = mult_clf_output['filename_full']
    from pathlib import Path
    pfsz = Path(filename_fullsize)
    filename_fullsize = pjoin(pfsz.parents[0], pfsz.name[2:])
    #    finfo = os.stat( filename_fullsize )
    #    print(finfo.st_size / (1024**2))
    f = np.load(filename_fullsize,allow_pickle=True)

    results_cur =  f['results_cur'][()]

    pre2 = results_cur['featsel_per_method']['interpret_EBM']
    if fs is None or (cure_if_possible and fs not in pre2):
        clf_dict_full = pre2
    else:
        clf_dict_full = pre2[fs]
    #results_cur.keys()

    #clf_dict = EBM['info_per_cp'][('trem_L', 'hold_L&move_L')]
    #scores = clf_dict['scores'];
    #print(len(results_cur['feature_names_filtered']), len(EBM['feature_indices_used']) )
    #print('len(scores) = ',len(scores) )
    #explainer = clf_dict_full['explainer']


    if fs is None or (cure_if_possible and fs not in pre2):
        #clf_dict['explainer'] = explainer
        #if clf_dict
        print('scores in clf_dict','scores' in clf_dict, 'scores' in clf_dict_full)
        if set( pre2['feature_indices_used'] ) == set( results_cur['VIF_truncation'][ 'colinds_good_VIFsel'] ):
            fs = 'VIFsel'
        else:
            fs = 'all'
        mult_clf_output['featsel_per_method']['interpret_EBM'][fs] = clf_dict_full
        #mult_clf_output['featsel_per_method']['interpret_EBM']['VIFsel']['scores'] =
        #for kk in list(clf_dict.keys()):
        #    del mult_clf_output['featsel_per_method']['interpret_EBM'][kk]
    else:
        mult_clf_output['featsel_per_method']['interpret_EBM'][fs] = clf_dict_full
        #### clf_dict_full['ebm'].explain_local() # no, we need X!
    #print((rn,grp,int_type), utsne.sprintfPerfs(clf_dict['perf'] ) )

    del f
    del results_cur

def splitScoresEBM(scores,feature_names):
    # last will be the largest in return
    assert len( feature_names ) == len(scores)
    best_feat_name = feature_names [ np.argmax(scores)]
    #print(best_feat_name)
    sortinds = np.argsort(scores)
    not_interact = np.where([ feature_names[ind].find(' x ') < 0 \
                         for ind in  sortinds])[0]
    interact = np.where([ feature_names[ind].find(' x ') >= 0 \
                         for ind in  sortinds])[0]
    sortinds_interact = np.array(sortinds)[interact]
    sortinds_not_interact = np.array(sortinds)[not_interact]

    return np.array(scores)[sortinds_not_interact], np.array(feature_names)[sortinds_not_interact],\
        np.array(scores)[sortinds_interact], np.array(feature_names)[sortinds_interact]



def EBMlocExpl2scores(loc_expls, inc_interactions=False):
    assert loc_expls is not None
    from utils_postprocess_HPC import splitScoresEBM
    num_classes = len(loc_expls[0]['meta']['label_names'])

    featnames0 = loc_expls[0]['names']
    # here actual order of the features is not important for me
    inds_ni,fns_ni,_,_ = splitScoresEBM( np.arange(len(featnames0) ), featnames0  )

    #from scipy.stats import logistic
    from scipy.special import expit
    from scipy.special import logit

    scs = []
    fns = []
    intercepts = []
    true_labels = []
    predicted_labels = []
    for d in loc_expls:
        scores = np.array( d['scores'] )
        if num_classes == 2:
            # TODO: is it the correct formula?
            scores_neg = expit( 1 - logit(scores) )
            scores = np.vstack([scores,scores_neg])
        else:
            scores = scores.T   # id x nfeats
        featnames = d['names']

        assert tuple(featnames) == tuple(featnames0)

        if not inc_interactions:
            scores = scores[ :, inds_ni ]
            featnames = np.array(featnames)[ inds_ni]

        intercept_ind = 0
        assert d['extra']['names'][intercept_ind] == 'Intercept',\
            d['extra']['names'][intercept_ind]
        intercept = d['extra']['scores'][intercept_ind]
        true_label = d['perf']['actual']
        predicted_label = d['perf']['predicted']

        scs += [ scores]
        fns += [featnames]
        intercepts += [intercept]
        true_labels += [true_label]
        predicted_labels += [predicted_label]


    scores = np.array( scs)
    intercepts = np.array(intercepts)
    if num_classes == 2:
        intercepts = np.array( [ intercepts , -intercepts ] ).T
    print(scores.shape, intercepts.shape)
    res = np.concatenate([scores,intercepts[:,:,None]],axis=-1)
    return res,true_labels,predicted_labels, featnames

# confmats from EBM
def confinfo_from_EBM(tpll):
    Ms_full = []
    matnames = []
    matdicts = []
    for tpl in tpll:
        rn,prefix,g,it = tpl[:-1]
        d = tpl[-1]
        matnames += [f'{rn[-3:]}:{prefix}']
        revdict_lenc = d['revdict_lenc']
        r = d['featsel_per_method']['interpret_EBM']['all']
        Ms_full += [r['confmat_normalized']]

        matdicts += [r['perf_per_cp']]
    colnames = [None]*len(revdict_lenc)
    for k in sorted(revdict_lenc.keys() ):
        colnames[k] = revdict_lenc[k]

    main_colind = 0

    return Ms_full, matnames, matdicts, colnames, revdict_lenc, main_colind

# confmats from XGB
def confinfo_from_XGB(tpll):
    Ms_full = []
    matnames = []
    matdicts = []
    for tpl in tpll[:2]:
        rn,prefix,g,it = tpl[:-1]
        print(rn)
        d = tpl[-1]
        subrn = rn.split('_')[-1]
        matnames += [f'{subrn}:{prefix}']
        revdict_lenc = d['revdict_lenc']
        r = d['XGB_analysis_versions']['all_present_features']
        Ms_full += [r['perf_dict']['confmat_aver']]

        matdicts += [r['perf_per_cp']]
    colnames = [None]*len(revdict_lenc)
    for k in sorted(revdict_lenc.keys() ):
        colnames[k] = revdict_lenc[k]


    main_colind = 0
    return Ms_full, matnames, matdicts, colnames, revdict_lenc, main_colind

def loadRunCorresp(moc):
    if 'cmd' not in moc:
        return None,None
    runCID = dict( moc['cmd'][0] ).get('--runCID', None)
    corresp, all_info = None,None
    if runCID is not None:
        import json
        fn = pjoin(gv.code_dir,'run',f'___run_corresp_{runCID}.json')
        if not os.path.exists(fn):
            fn = pjoin(gv.code_dir,'run',f'___run_corresp_{runCID}.txt')
        with open(fn , 'r') as f:
            corresp_file = json.load( f )

        mod_time = os.stat( fn ).st_mtime
        dt = datetime.fromtimestamp(mod_time)

        all_info = {}
        if 'correspondance' in corresp_file:
            corresp = corresp_file['correspondance']
            all_info = corresp_file
        else:
            corresp = corresp_file
        all_info['date_created'] = dt
    else:
        print('loadRunCorresp: corresp file not found')
    return corresp, all_info

# care about backward-compat
def _extractPerfNumber( perf_dict, score):
    if 'perf_aver' in perf_dict and 'sens' not in perf_dict:
        perf_cur = perf_dict['perf_aver']
    else:
        perf_cur = perf_dict
    if isinstance(perf_cur, (tuple, list, np.ndarray) ):
        if score.startswith('special'):
            perf_one_number = min( perf_cur[0], perf_cur[1] )
        elif score == 'sens':
            perf_one_number = perf_cur[0]
        elif score == 'spec':
            perf_one_number = perf_cur[1]
        elif score == 'F1':
            perf_one_number = perf_cur[2]
    elif isinstance(perf_cur,dict):
        if score.startswith('special'):
            perf_one_number = min( perf_cur['sens'], perf_cur['spec'] )
        else:
            if score == 'bacc' and score not in perf_cur:
                score = 'balanced_accuracy'
            perf_one_number = perf_cur[score]
    else:
        raise ValueError( str(perf_cur) + str(type(perf_cur) ) )

    return perf_one_number



#base_perf_per_mode
def computeImprovementsPerParcelGroup(output_per_raw, base_perf_prefix,
                                      base_perf_low_prefix = None,
                                      mode = 'only',
                                      score = 'special:min_sens_spec',
                                      inv_exclude = True, printLog = False,
                                     ignore_base_prefix_missing = False  ):
    tpll = multiLevelDict2TupleList(output_per_raw,4,3)
    tpll_reshaped = list( zip(*tpll) ) # it is a tuple of lists

    n_chars = len('onlyH_act_')
    prefixes = set( tpll_reshaped[1] )
    #print( list(sorted(prefixes) ) )
    #prefixes2 = set( [ tpl[1] for tpl in tpll ]  )
    #assert prefixes == prefixes2, prefixes ^ prefixes2

    #perfs_per_medcond = {'on':[],'off':[]}
    perfs_per_medcond = {'on':{},'off':{}}
    perfs_base_per_medcond = {'on':np.nan,'off':np.nan}
    for prefix in prefixes:
        #corresp,all_info = loadRunCorresp(tpll[0][-1])
        #ind,pgn,nice_name = corresp[prefix]
        part = prefix[n_chars:]
        if not part.startswith(mode):
            #print(f'skipping {part} for {mode}, prefix={prefix}')
            continue

        cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == prefix )[0]
        if len(cur_prefix_inds) != 1:
            print( f'Wrong number of prefix {prefix} instances {len(cur_prefix_inds) }' )
            continue
        for cpi in cur_prefix_inds:
            output = tpll[cpi][-1]
            rn = tpll[cpi][0]
            medcond = rn.split('_')[-1]
            r = output['XGB_analysis_versions']['all_present_features']
            perfs_per_medcond[medcond][prefix] = \
                _extractPerfNumber (r['perf_dict'], score )
        #print(prefix,pgn, perf_cur)
    if mode == 'exclude':
        cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == base_perf_prefix )[0]
        assert len(cur_prefix_inds) <= 2
        for cpi in cur_prefix_inds:
            output = tpll[cpi][-1]
            rn = tpll[cpi][0]
            medcond = rn.split('_')[-1]
            r = output['XGB_analysis_versions']['all_present_features']
            perfs_base_per_medcond[medcond] = \
                _extractPerfNumber (r['perf_dict'], score )
            #if mode == 'exclude':
            #    perfs_base_per_medcond[medcond] = extractPerfNumber (r['perf_dict'] )
            #else:
            #    if base_perf_prefix is None:
            #        perfs_base_per_medcond[medcond] = 0.
            #    else:
    else:
        if base_perf_low_prefix is None:
            # I still need to set zeros for all the medconds present
            cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == base_perf_prefix )[0]
            if len(cur_prefix_inds) != 1:
                print('cur_prefix_inds =', [ tpll[ci][:-1] for ci in cur_prefix_inds ] )
                if ignore_base_prefix_missing:
                    perfs_base_per_medcond[medcond] = 0.
                else:
                    raise ValueError('cur_prefix_inds has wrong len')
            for cpi in cur_prefix_inds:
                output = tpll[cpi][-1]
                rn = tpll[cpi][0]
                medcond = rn.split('_')[-1]
                perfs_base_per_medcond[medcond] = 0.
        else:
            cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == base_perf_low_prefix )[0]
            assert (len(cur_prefix_inds) >0 ) and len(cur_prefix_inds) <= 2, \
                f'problem getting base perf { [ tpll[ci][:-1] for ci in cur_prefix_inds ] }'
            for cpi in cur_prefix_inds:
                output = tpll[cpi][-1]
                rn = tpll[cpi][0]
                medcond = rn.split('_')[-1]
                r = output['XGB_analysis_versions']['all_present_features']
                perfs_base_per_medcond[medcond] = \
                    _extractPerfNumber (r['perf_dict'], score )

    print(perfs_base_per_medcond)

    ######################################
    perfs_aver_per_medcond = {}
    for p in perfs_per_medcond:
        vs = []
        # I don't want to averge over LFP
        for pgn,v in perfs_per_medcond[p].items():
            if pgn != 'LFP':
                vs += [v]
        #list(perfs_per_medcond[p].values() )
        perfs_aver_per_medcond[p]  = np.mean( vs )

    #####################################


    impr_per_medcond_per_pgn = {'on':{}, 'off':{}}
    impr_wrtLFP_per_medcond_per_pgn = {'on':{}, 'off':{}}
    impr_wrt_base_per_medcond_per_pgn = {'on':{}, 'off':{}}
    for prefix in prefixes:
        part = prefix[n_chars:]
        if not part.startswith(mode):
            continue

        cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == prefix )[0]
        for cpi in cur_prefix_inds:
            corresp,all_info = loadRunCorresp(tpll[cpi][-1])
            ind,pgn,nice_name = corresp[prefix]

            output = tpll[cpi][-1]
            rn = tpll[cpi][0]
            medcond = rn.split('_')[-1]
            print(prefix,pgn)
            if pgn != 'LFP':
                if mode == 'only':
                    assert dict( output['cmd'][0] )['--parcel_group_names'] == pgn
                elif mode == 'exclude':
                    assert dict( output['cmd'][0] )['--parcel_group_names'] == '!'+pgn

            #output[]
            r = output['XGB_analysis_versions']['all_present_features']
            #perf_cur = r['perf_dict']['perf_aver']
            #perf_one_number = min( perf_cur[0], perf_cur[1] )
            perf_one_number = _extractPerfNumber (r['perf_dict'], score )
            improvement_wrt_base = perf_one_number  - perfs_base_per_medcond[medcond]   # prob (range is 0 to 1)
            improvement = perf_one_number  - perfs_aver_per_medcond[p]   # prob (range is 0 to 1)
            LFPkey =f'onlyH_act_{mode}15'
            if LFPkey in perfs_per_medcond[medcond]:
                improvement_wrt_LFP = perf_one_number - perfs_per_medcond[medcond][LFPkey]
            else:
                improvement_wrt_LFP = None

            #if mode == 'exclude':
            #    improvement = perf_one_number  - perfs_aver_per_medcond[p]   # prob (range is 0 to 1)

            if inv_exclude and mode == 'exclude':
                improvement_wrt_base = -improvement_wrt_base
                improvement = -improvement
                if improvement_wrt_LFP is not None:
                    improvement_wrt_LFP =- improvement_wrt_LFP

            impr_wrt_base_per_medcond_per_pgn[medcond][pgn] = improvement_wrt_base * 100  # now in pct
            impr_per_medcond_per_pgn[medcond][pgn] = improvement * 100  # now in pct
            if improvement_wrt_LFP is not None:
                impr_wrtLFP_per_medcond_per_pgn[medcond][pgn] = improvement_wrt_LFP * 100
            if printLog:
                print(prefix,pgn, medcond,improvement * 100)
    return impr_wrt_base_per_medcond_per_pgn, impr_per_medcond_per_pgn, impr_wrtLFP_per_medcond_per_pgn, perfs_aver_per_medcond

def sidelet2sideTempl(sidelet : str, subject : str) -> str:
    assert len(sidelet) == 1

    mainmoveside = gv.gen_subj_info[subject].get('move_side',None)
    assert mainmoveside is not None
    movesidelet = mainmoveside[0].upper()
    moveopsidelet = utils.getOppositeSideStr( movesidelet )

    if sidelet == movesidelet:
        return '%'
    elif sidelet == moveopsidelet:
        return '^'
    else:
        raise ValueError(f'wrong lettter {sidelet}')


def prefTempl2pref(pref_templ, subject):
    mainmoveside = gv.gen_subj_info[subject].get('move_side',None)
    assert mainmoveside is not None
    movesidelet = mainmoveside[0].upper()
    moveopsidelet = utils.getOppositeSideStr( movesidelet )
    pref = pref_templ.replace('%', moveopsidelet)
    pref = pref.replace('^', movesidelet)
    return pref

def prefixTempl2Prefix(prefix_templ,rawname):
    subj = rawname.split('_')[0]
    mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
    assert mainmoveside is not None
    movesidelet = mainmoveside[0].upper()
    moveopsidelet = utils.getOppositeSideStr( movesidelet )
    templ_eff = prefix_templ.    replace('%', moveopsidelet)
    templ_eff = templ_eff.replace('^', movesidelet)
    return templ_eff

def prefix2prefixTempl(prefix, rawname, LFP_side_to_use_final = None):
    subj = rawname.split('_')[0]
    mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
    assert mainmoveside is not None
    movesidelet = mainmoveside[0].upper()
    moveopsidelet = utils.getOppositeSideStr( movesidelet )

    tmp = prefix
    if LFP_side_to_use_final is not None:
        side = LFP_side_to_use_final
        sidelet = side[0].upper()

        tmp = tmp.replace('C',sidelet)  # different 

    tmp = tmp.replace(movesidelet,'^')
    tmp = tmp.replace(moveopsidelet,'%')

    return tmp


def _getPrefixInds(tpll,prefix_to_test):
    # it respects '%' and '^' wildcards
    prefix_inds = []
    for tpli,tpl in enumerate(tpll):
        rn,pref_cur = tpl[:2]
        subj = rn.split('_')[0]
        #mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
        #assert mainmoveside is not None
        #movesidelet = mainmoveside[0].upper()
        #moveopsidelet = utils.getOppositeSideStr( movesidelet )
        #pref = prefix_to_test.replace('%', moveopsidelet)
        #pref = pref.replace('^', movesidelet)

        pref = prefTempl2pref(prefix_to_test, subj)
        if pref == pref_cur:
            prefix_inds += [tpli]

    return prefix_inds

def computeImprovementsPerParcelGroup2(output_per_raw, base_perf_prefix,
                                      base_perf_low_prefix = None,
                                      mode = 'only',
                                      score = 'special:min_sens_spec',
                                      inv_exclude = True, printLog = False,
                                     ignore_base_prefix_missing = False,baseprefix = 'onlyH_act_'):
    # pgn -- parcel group name
    tpll = multiLevelDict2TupleList(output_per_raw,4,3)
    tpll_reshaped = list( zip(*tpll) ) # it is a tuple of lists

    n_chars = len(baseprefix) # after this part of the prefix should immediately follow mode string
    prefixes = list( set( tpll_reshaped[1] ) )
    #print( list(sorted(prefixes) ) )
    #prefixes2 = set( [ tpl[1] for tpl in tpll ]  )
    #assert prefixes == prefixes2, prefixes ^ prefixes2

    #perfs_per_medcond = {'on':[],'off':[]}
    perfs_per_medcond = {'on':{},'off':{}}
    perfs_base_per_medcond = {'on':np.nan,'off':np.nan}
    for prefix in prefixes:
        #corresp,all_info = loadRunCorresp(tpll[0][-1])
        #ind,pgn,nice_name = corresp[prefix]
        part = prefix[n_chars:]
        if not part.startswith(mode):
            #print(f'skipping {part} for {mode}, prefix={prefix}')
            continue

        cur_prefix_inds = _getPrefixInds(tpll,prefix)
        #cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == prefix )[0]
        if len(cur_prefix_inds) != 1:
            print( f'Wrong number of prefix {prefix} instances {len(cur_prefix_inds) }' )
            continue
        for cpi in cur_prefix_inds:
            output = tpll[cpi][-1]
            rn = tpll[cpi][0]
            medcond = rn.split('_')[-1]
            r = output['XGB_analysis_versions']['all_present_features']
            perfs_per_medcond[medcond][prefix] = \
                _extractPerfNumber (r['perf_dict'], score )
        #print(prefix,pgn, perf_cur)
    if mode == 'exclude':
        cur_prefix_inds = _getPrefixInds(tpll,base_perf_prefix)
        #cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == base_perf_prefix )[0]
        assert len(cur_prefix_inds) <= 2
        for cpi in cur_prefix_inds:
            output = tpll[cpi][-1]
            rn = tpll[cpi][0]
            medcond = rn.split('_')[-1]
            r = output['XGB_analysis_versions']['all_present_features']
            perfs_base_per_medcond[medcond] = \
                _extractPerfNumber (r['perf_dict'], score )
            #if mode == 'exclude':
            #    perfs_base_per_medcond[medcond] = extractPerfNumber (r['perf_dict'] )
            #else:
            #    if base_perf_prefix is None:
            #        perfs_base_per_medcond[medcond] = 0.
            #    else:
    else:
        if base_perf_low_prefix is None:
            # I still need to set zeros for all the medconds present
            #cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == base_perf_prefix )[0]
            cur_prefix_inds = _getPrefixInds(tpll,base_perf_prefix)
            if len(cur_prefix_inds) != 1:
                print('cur_prefix_inds =', [ tpll[ci][:-1] for ci in cur_prefix_inds ] )
                if ignore_base_prefix_missing:
                    perfs_base_per_medcond[medcond] = 0.
                else:
                    raise ValueError('cur_prefix_inds has wrong len')
            for cpi in cur_prefix_inds:
                output = tpll[cpi][-1]
                rn = tpll[cpi][0]
                medcond = rn.split('_')[-1]
                perfs_base_per_medcond[medcond] = 0.
        else:
            cur_prefix_inds = _getPrefixInds(tpll,base_perf_low_prefix)
            #cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == base_perf_low_prefix )[0]
            assert (len(cur_prefix_inds) >0 ) and len(cur_prefix_inds) <= 2, \
                f'problem getting base perf {base_perf_low_prefix} = { [ tpll[ci][:-1] for ci in cur_prefix_inds ] }'
            for cpi in cur_prefix_inds:
                output = tpll[cpi][-1]
                rn = tpll[cpi][0]
                medcond = rn.split('_')[-1]
                r = output['XGB_analysis_versions']['all_present_features']
                perfs_base_per_medcond[medcond] = \
                    _extractPerfNumber (r['perf_dict'], score )

    print(perfs_base_per_medcond)



    #####################################

    impr_wrt_base_per_medcond_per_pgn = {'on':{}, 'off':{}}
    for prefix in prefixes:
        # take the interesting part of the prefix
        part = prefix[n_chars:]
        if not part.startswith(mode):  # if it is a wrong prefix, skip
            continue

        ##############
        ############

        #cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == prefix )[0]
        cur_prefix_inds = _getPrefixInds(tpll, prefix)
        for cpi in cur_prefix_inds:
            tpl = tpll[cpi]
            output = tpl[-1]
            rn = tpl[0]
        #for tpl in tpll:
        #    rn,pref_cur = tpl[:2]

            corresp,all_info = loadRunCorresp(tpll[cpi][-1])
            ind,pgn,nice_name = corresp[prefix]

            output = tpl[-1]
            medcond = rn.split('_')[1]
            print(prefix,pgn)
            if pgn != 'LFP':
                if mode == 'only':
                    assert dict( output['cmd'][0] )['--parcel_group_names'] == pgn
                elif mode == 'exclude':
                    assert dict( output['cmd'][0] )['--parcel_group_names'] == '!'+pgn

            # all feats perf
            r = output['XGB_analysis_versions']['all_present_features']
            perf_one_number = _extractPerfNumber (r['perf_dict'], score )

            improvement_wrt_base = perf_one_number - perfs_base_per_medcond[medcond]   # prob (range is 0 to 1)

            #improvement = perf_one_number  - perfs_aver_per_medcond[p]   # prob (range is 0 to 1)
            #LFPkey =f'onlyH_act_{mode}15'
            #if LFPkey in perfs_per_medcond[medcond]:
            #    improvement_wrt_LFP = perf_one_number - perfs_per_medcond[medcond][LFPkey]
            #else:
            #    improvement_wrt_LFP = None

            if inv_exclude and mode == 'exclude':
                improvement_wrt_base = -improvement_wrt_base
            #    improvement = -improvement
            #    if improvement_wrt_LFP is not None:
            #        improvement_wrt_LFP =- improvement_wrt_LFP

            impr_wrt_base_per_medcond_per_pgn[medcond][pgn] = improvement_wrt_base * 100  # now in pct
            #impr_per_medcond_per_pgn[medcond][pgn] = improvement * 100  # now in pct
            #if improvement_wrt_LFP is not None:
            #    impr_wrtLFP_per_medcond_per_pgn[medcond][pgn] = improvement_wrt_LFP * 100
            #if printLog:
            #    print(prefix,pgn, medcond,improvement * 100)

    #  base_perf_low_prefix]
    for medcond in perfs_base_per_medcond:
        impr_wrt_base_per_medcond_per_pgn[medcond]['base_low'] = perfs_base_per_medcond[medcond]

    return impr_wrt_base_per_medcond_per_pgn

def plotTableInfoBrain(impr_per_medcond_per_pgn , medcond,
                       multi_clf_output, head_subj_ind=None,
                       inv_exclude=True, mode='only',
                       subdir='',
                       savefile_prefix = 'EXPORT_brain_map_area_strength_',
                       save_only = False):#, perf_tuple):
    from utils import vizGroup2
    from globvars import gp

    import pymatreader
    intensity_mult = 0.1
    intensity_mult = 1

    #rncur = rawnames[0] + '_off_hold'
    #sind_str,mc,tk  = utils.getParamsFromRawname(rncur)
    sind_str = 'S01'
    if head_subj_ind is None:
        rncur = sind_str + '_off_hold'
    else:
        rncur = head_subj_ind + '_off_hold'
    sources_type=multi_clf_output['info']['sources_type']
    src_file_grouping_ind = multi_clf_output['info']['src_grouping_fn']
    #src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
    #                                                    sources_type,src_file_grouping_ind)
    #src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    src_rec_info_fn_full = utils.genRecInfoFn(rncur,sources_type,src_file_grouping_ind)
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)


    labels_dict = rec_info['label_groups_dict'][()]
    srcgroups_dict = rec_info['srcgroups_dict'][()]
    coords = rec_info['coords_Jan_actual'][()]

    #if head_subj_ind is None:
    #    rncur = sind_str + '_off_hold'
    #else:
    #    rncur = head_subj_ind + '_off_hold'
    #sources_type=multi_clf_output['info']['sources_type']
    #src_file_grouping_ind = multi_clf_output['info']['src_grouping_fn']
    #src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
    #                                                    sources_type,src_file_grouping_ind)
    #src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    #rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
    sgdn = 'all_raw'

    roi_labels_ = np.array(  labels_dict[sgdn] )
    parcel_indices_all = np.arange(1,len(roi_labels_))
    roi_labels = ['unlabeled'] + list( roi_labels_[parcel_indices_all] )

    srcgrp = np.zeros( srcgroups_dict[sgdn].shape, dtype=srcgroups_dict[sgdn].dtype)
    for pii,pi in enumerate(parcel_indices_all):
        srcgrp[srcgroups_dict[sgdn] == pi] = pii + 1 #list(roi_labels).index( rls[pii])



    #############################
    from postprocess import updateSrcGroups
    srcgrp_new, brain_area_labels, intensities = updateSrcGroups(impr_per_medcond_per_pgn[medcond], roi_labels, srcgrp,
            use_both_sides = True, want_sided = True )
    #brain_area_labels = ['unlabeled'] + list( sorted( gp.parcel_groupings_post.keys() ) )
    #intensities = [np.nan] * len(brain_area_labels)
    #srcgrp_new = np.nan * np.ones( len(srcgrp) )
    #for pgn in impr_per_medcond_per_pgn[medcond]:
    #    if pgn in ['LFP'] or pgn.startswith('base_'):
    #        continue

    #    #TODO SIDE
    #    if pgn.endswith('_L') or pgn.endswith('_R'):
    #        pgn_eff = pgn[:-2]
    #        sided = True
    #    elif pgn.endswith('_B'):
    #        pgn_eff = pgn[:-2]
    #        sided = False
    #    else:
    #        pgn_eff = pgn
    #        sided = False
    #    parcel_labels = gp.parcel_groupings_post[pgn_eff] #without side information
    #    if pgn == 'Cerebellum':
    #        sidestr = '_R'
    #    else:
    #        sidestr = '_L'
    #    parcel_inds = [ roi_labels.index(pl + sidestr) for pl in parcel_labels ]
    #    #parcel_inds += [ roi_labels.index(pl + '_R') for pl in parcel_labels ]

    #    ind = brain_area_labels.index(pgn_eff)
    #    for pi in parcel_inds:
    #        srcgrp_new[srcgrp==pi]  = ind

    #    #brain_area_labels += [pgn]

    #    intensity_cur = impr_per_medcond_per_pgn[medcond][pgn] * intensity_mult
    #    #print(pgn,ind, intensity_cur)
    #    intensities[ind ]= intensity_cur #cmap(intensity_cur)  #* len(parcel_inds)
    #assert np.any( ~np.isnan( srcgrp_new ) ), srcgrp[ np.isnan( srcgrp_new ) ]
    #intensities = np.zeros(len(roi_labels))
    #intensities
    ###########################################
    #%matplotlib inline






    roi_lab_codes = [0] * len(roi_labels)
    color_group_labels = list( gp.parcel_groupings_post.keys()   )

    roi_lab_codes = None
    #color_group_labels = np.arange(len())


    cmap = plt.cm.get_cmap('inferno')


    savename = f'{savefile_prefix}medcond={medcond}_mode={mode}.npz'
    savename_full = pjoin(gv.data_dir,subdir,savename)

    def cvt(x):
        if np.isnan( float(x) ):
            return 'NaN'
        else:
            return str( int(x) )

    setdiff = set( map(str,range( len( brain_area_labels ) ) ) ) ^  set( map(cvt,srcgrp_new)  )
    setdiff = list(sorted(setdiff))
    # brain area labels, named
    ss = [ brain_area_labels[int(ind)] for ind in setdiff if ind != 'NaN' ]
    assert set(setdiff) == set(['0', 'NaN']), (setdiff, ss )


    info=dict(coords=coords,
        brain_area_labels=brain_area_labels,
        roi_labels=roi_labels,
        srcgrp0 = srcgroups_dict[sgdn],
        srcgrp = srcgrp,
        srcgrp_new=srcgrp_new,
        color_grouping=roi_lab_codes,
        intensities = intensities,
        color_group_labels= color_group_labels,
        impr_per_medcond_per_pgn=impr_per_medcond_per_pgn )

    np.savez(savename_full,info=info)
    print(f'Saved to {savename_full}')

    if save_only:
        return None, None, info
    #sind_str,


    # clrs =  utils.vizGroup2(sind_str,coords,roi_labels,srcgrp, show=False,
    #                         def_alpha=.1, figsize_mult=1.5,msz=30, printLog=0,
    #                         color_grouping=roi_lab_codes, intensities = intensities,
    #                         color_group_labels= color_group_labels,
    #                         sizes=None, msz_mult=0.3, seed=0, cmap=cmap)
    fig,axs, clrs, scatters = utils.vizGroup2(sind_str,coords,
                    brain_area_labels,srcgrp_new, show=False,
                    show_legend = False, def_alpha=.1, figsize_mult=1.5,msz=30,
                    printLog=0, color_grouping=roi_lab_codes, intensities =
                    intensities, color_group_labels= color_group_labels,
                    sizes=None, msz_mult=0.3, seed=0, cmap=cmap, projections =
                    ['top','side'])


    intensities = np.array(intensities)
    gm = ~np.isnan(np.array(intensities) )

    if sum( gm ) == 0:
        print('plotTableInfoBrain: WARNING: no valid intensitites!')
        return None,None,None

    mii,mai = np.min(intensities[gm]), np.max(intensities[gm])
    print(mii,mai)

    #bc = np.ones(4)
    #bc[:3] = 0.5
    #bc = tuple(bc)
    #axs[0].w_xaxis.set_pane_color(bc)

    # axs[1].w_xaxis.set_pane_color(bc)
    #plt.gcf().

    # I want to change labels but NOT locations -- does not work because
    # ticklabels for colorbar are set during render only
    clrb = plt.colorbar(scatters['top'])
    #ylabs = [tl.get_text() for tl in clrb.ax.get_yticklabels()]
    #print(ylabs)
    #ylabs2 = []
    #for yl in ylabs:
    #    if len(yl):
    #        x = float(yl)
    #        newyl = '{:.0f}'.format(x)
    #    else:
    #        newyl = yl
    #    ylabs2 += [newyl]
    ##ylabs_a = np.array( list( map(float,ylabs) ) ) * 100
    ##ylabs2 = map(lambda x: '{:.0f}'.format(x) , ylabs_a)
    #clrb.ax.set_yticklabels(ylabs2)


    refpt = 'base'

    impr_lfp = impr_per_medcond_per_pgn[medcond].get('LFP', np.nan)
    if refpt == 'base':
        if inv_exclude and mode == 'exclude':
            plt.title(f'H_act performance reduction per area removal,\nremoval of LFP={impr_lfp * intensity_mult:.0f}%')
        else:
            plt.title(f'H_act individual areas performance,\nLFP alone ={impr_lfp * intensity_mult:.0f}%')
    else:
        if inv_exclude and mode == 'exclude':
            axs[1].set_title(f'H_act {mode} areas relative performance -difference * {intensity_mult}, LFP={impr_lfp * intensity_mult:.0f}%')
        else:
            axs[1].set_title(f'H_act {mode} areas relative performance difference * {intensity_mult}, LFP={impr_lfp * intensity_mult:.0f}%')


    figname = f'brain_map_area_strength_medcond={medcond}_mode={mode}.pdf'
    #plt.title(figname[:-4]  )
    figname_full = pjoin(gv.dir_fig,subdir,figname)
    plt.savefig(figname_full)
    return axs, clrb, info
    #plt.colorbar();
    #plotTableInfoBrain(impr_per_medcond_per_pgn, output)


def getLogFname(mco,folder = '$OSCBAGDIS_DATAPROC_CODE/slurmout'):
    jobid = dict(mco['cmd'][0] )['--SLURM_job_id']
    fname = f'ML_{jobid}.out'
    folder = os.path.expandvars(folder)
    fname_full = pjoin(folder, fname)
    if not os.path.exists(fname_full):
        fname_full = pjoin(folder,'_backup', fname)
    return fname_full

def copyLogFname(mco, newfname = '_logfile_to_observe.out' ):
    # maybe add filename data or maybe sacct info (inc how much it took to run)
    fname_full = getLogFname(mco)
    import gv, shutil
    newname_full = pjoin(gv.code_dir, newfname)
    shutil.copy(fname_full, newname_full )
    print(f'copied to {newname_full}')

def printLogPart(mco, text_to_find = 'Start classif' ):
    fname_full = getLogFname(mco)

    with open(fname_full,'r') as f:
        lines = f.readlines()

    lineind = 0
    for linei,line in enumerate(lines):
        if line.startswith(text_to_find):
            lineind = linei

    print(lines[lineind, :] )

def filterOutputs(outputs_grouped,rns=None,
        prefs=None,grps=None,its=None):
    import utils
    # % -- main move side let
    outputs_res = {}
    if rns is None:
        rns = outputs_grouped.keys()
    for rn in rns:
        subj = rn.split('_')[0]
        #mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
        #assert mainmoveside is not None
        #movesidelet = mainmoveside[0].upper()
        #moveopsidelet = utils.getOppositeSideStr( movesidelet )

        # TODO: correctly processs copy from best LFP

        outputs_res[rn] = {}
        o_cur_rn = outputs_grouped[rn]
        prefs_cur = o_cur_rn.keys()
        if prefs is not None:
            prefs_cur = prefs
        for pref in prefs_cur:
            #pref = pref.replace('%', moveopsidelet)
            #pref = pref.replace('^', movesidelet)

            pref = prefTempl2pref(pref, subj)

            o_cur_pref = o_cur_rn.get(pref, None)
            if o_cur_pref is None:
                continue
            outputs_res[rn][pref] = {}
            gs_cur = o_cur_pref.keys()
            gs_cur = [g for g in gs_cur if g != 'feature_names_filtered']
            if grps is not None:
                gs_cur = grps
            for g in gs_cur:
                if g not in o_cur_pref:
                    continue
                outputs_res[rn][pref][g] = {}
                o_cur_g = o_cur_pref[g]
                its_cur = o_cur_g.keys()
                if its is not None:
                    its_cur = its
                for it in its_cur:
                    outputs_res[rn][pref][g][it] = o_cur_g[it]

    return outputs_res

def printOutputInfo(output_per_raw, datinfo = 'prefix', autosort = False, sort_dict = None):
    assert datinfo in ['prefix', 'rawname', 'grouping', 'interval_types']
    if isinstance(datinfo,str):
        datinfos = datinfo.split(',')
    elif isinstance(datinfo,list):
        datinfos = datinfo

    resinfos = {}
    for di in  datinfos:
        resinfos[di] = []

    assert not ( autosort and (sort_dict is not None) ), 'two sortings at the same time does not make sense'

    prefix_sort = None
    if sort_dict is not None:
        assert len(sort_dict) == 1
        #if di in sort_dict:
        k,v = list( sort_dict.items() )[0]
        if k != 'prefix':
            raise ValueError('not implemented')
        prefix_sort = v

    tpll = multiLevelDict2TupleList(output_per_raw,4,3, prefix_sort= prefix_sort)
    for tpl in tpll:
        rn,prefix,grouping,it = tpl[:-1]

        di = 'rawname'; d = rn
        if di in datinfos:
            if d not in resinfos[di]:
                resinfos[di] += [d]
        di = 'prefix'; d=prefix
        if di in datinfos:
            if d not in resinfos[di]:
                resinfos[di] += [d]
        di = 'grouping'; d = grouping
        if di in datinfos:
            if d not in resinfos[di]:
                resinfos[di] += [d]
        di = 'interval_types'; d = it
        if di in datinfos:
            if d not in resinfos[di]:
                resinfos[di] += [d]

    for di in resinfos:
        if autosort:
            resinfos[di] = list( sorted(resinfos[di]) )

    return resinfos

def accessMultiLevelDict(d,path):
    # how to distinguish value None from not found?
    sep  = '/'
    pparts = path.split(sep)
    #print(pparts, d.keys() )
    p0 = pparts[0]
    v0 = None
    if p0 in d:
        v0 = d[p0]
    else:
        raise ValueError(f'Not found! {p0} amonhg {d.keys()}')

    if len(pparts) == 1 or v0 is None:
        #return v0,''
        return v0
    else:
        #v,path_ret =  accessMultiLevelDict(v0, sep.join(pparts[1:] ) )
        v =  accessMultiLevelDict(v0, sep.join(pparts[1:] ) )
        return v
        #if v is None:
        #   sep.join(pparts[1:]

def walkMultiLevelDict(d, path):
    paths = []
    if not isinstance(d, dict):
        return path
    for k, v in d.iteritems():
        child_path = path + k + '/'
        if isinstance(v, str):
            paths.append(child_path + v)
        else:
            paths.extend(walkMultiLevelDict(v, child_path))
    return paths

def recurseMultiLevelDict(d, prefix=None, sep='/'):
    if prefix is None:
        prefix = []
    for key, value in d.items():
        if isinstance(value, dict):
            yield from recurseMultiLevelDict(value, prefix + [key])
        else:
            yield sep.join(prefix + [key, value])

def collectBestLFP(subdir = 'searchLFP_both_sides_oversample2_LFP256_allaritf' , 
        start_time_str = "26 January 2023 20:11:15",
        save_result = 1,  q_perm = 0.9, 
        savefile_rawname_format ='subj',
        output_per_raw = None):
    import globvars as gv
    import utils
    import utils_tSNE as utsne
    import utils_preproc as upre

    import os
    import sys
    import mne
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import multiprocessing as mpr
    import matplotlib as mpl
    import time
    import gc;
    import scipy.signal as sig
    import pandas as pd 

    import numpy as np
    import utils_postprocess_HPC as postp
    import pymatreader

    data_dir = gv.data_dir
    from os.path import join as pjoin

    light_only = True

    from dateutil import parser
    #start_time = parser.parse("19 Aug 2021 03:05:15")
    #start_time = parser.parse("14 April 2022 10:05:15")
    #start_time = parser.parse("20 April 2022 10:05:15")
    
    start_time = parser.parse(start_time_str)  #2022, 4, 22, 17, 12, 27, 37491)
    end_time = parser.parse("19 Aug 2041 21:21:45")
    #end_time = parser.parse("19 Aug 2021 03:41:45") # this one is just to make things faster
    ndaysBefore = None
    #subdir = 'nointerp'
    #subdir = 'nofeatsel'
    #subdir = 'searchLFP'
    #subdir = 'searchLFP_both_sides'
    #subdir = 'searchLFP_both_sides_oversample2'
    
    lookup_dir = pjoin(gv.data_dir,subdir)
    recent = postp.listRecent(days=ndaysBefore, lookup_dir= lookup_dir,
                              start_time=start_time, end_time=end_time)
    rncombinstrs = []
    print(f'Found {len(recent)} recent files')
    for lf in recent:
        st = 0
        if light_only:
            if not lf.startswith('_!'):
                continue
            st = 2
        if savefile_rawname_format == 'subj':
            rncombinstrs += [lf[st+1:st+4]]
        elif savefile_rawname_format in ['subj,medcond', 
                'subj,medcond_glob']:
            #print(lf[st+1:], lf[st+1:].split('_')[:2])
            rncombinstrs += [ '_'.join(  lf[st+1:].split('_')[:2] ) ]
        else:
            raise ValueError('not implemented')
    rncombinstrs = list(sorted(set(rncombinstrs)))
    print('recent rncombstrs infolved = ',rncombinstrs)

    import utils_postprocess_HPC as postp
    #Earliest file 19 Aug 2021 03:05:15, latest file 19 Aug 2021 21:21:45
    prefixes = postp.listRecentPrefixes(days = ndaysBefore, light_only=light_only, 
                                        lookup_dir= lookup_dir,
                                       start_time=start_time, end_time=end_time)
    print('recent prefixes = ',prefixes)



    from utils_postprocess_HPC import listComputedData
    r = listComputedData(subdir,prefixes,start_time, end_time, use_main_LFP_chan=0)
    if r is None:
        print("found nothing")
        return None
    print('listComputedData = ',r)

    # here load data in memory
    prefixes_to_load = prefixes
    sources_type = 'parcel_aal'  # or ''
    if output_per_raw is None: 
        r = postp.collectPerformanceInfo3(rncombinstrs,prefixes_to_load, nraws_used='[0-9]+',   
            start_time=start_time, end_time=end_time, use_main_LFP_chan=0,
            ndays_before=None, sources_type = sources_type, printFilenames=1,
            subdir=subdir, remove_large_items = 1,
            list_only=0, allow_multi_fn_same_prefix=0,
            use_light_files = light_only, rawname_regex_full=0)
        #output_per_raw,Ximp_per_raw,gis_per_raw = r
        output_per_raw,_,_ = r
    print('len(output_per_raw) =', len(output_per_raw))
    assert len(output_per_raw) > 0, "nothing got collected"
    import gc; gc.collect()

    # save

    import utils_postprocess as pp
    import json
    outputs_grouped_tpll = pp.multiLevelDict2TupleList(output_per_raw,4,3)

    cpd = 'sens,spec,F1'
    metrics = ['balanced_accuracy', cpd, 'sens', 'spec']
    from utils_tSNE import selBestLFP

    def extractPerf(p,metric_cur):
        if isinstance(p, float):
            if np.isnan(p):
                return p
        if metric_cur == cpd:
            v = np.array( [ p['sens'],p['spec'] ] )
        else:
            v = np.array( p[metric_cur] )
        return (v*100).tolist()

    best_LFP_dict = {}

    row = []
    rows_only = []
    rows_but = []
    for tpl in outputs_grouped_tpll:
        rncombinstr, prefix, grp, int_type, mult_clf_output = tpl
        blfp = mult_clf_output['best_LFP']
        assert 'XGB' in blfp, 'There was a problem during collection: "best_LFP" does not contain XGB'
        pd0 = blfp['XGB']['perf_drop']['only']
        chnames_LFP = list( pd0.keys() )
        if not chnames_LFP[0].startswith('LFP'):
            pd0 = list(pd0.values())[0].keys()
            chnames_LFP = list( pd0.keys() )

        cdk = ','.join([prefix,grp,int_type])
        if rncombinstr not in best_LFP_dict:
            best_LFP_dict[rncombinstr]= {}
        best_LFP_dict[rncombinstr][cdk] = {}

        subj = rncombinstr.split('_')[0]
        mainmoveside_cur = gv.gen_subj_info[ subj].get('move_side',None)
        movesidelet = mainmoveside_cur[0].upper()

        contralat_to_move_sidelet = utils.getOppositeSideStr( movesidelet )
        _,chnames_LFP_contralat_to_move = utsne.selFeatsRegex(None,chnames_LFP,[f'LFP{contralat_to_move_sidelet}'])

        d = {}
        d['filename_full' ]  = mult_clf_output['filename_full' ]
        for metric in metrics:
            print(f'   metric = {metric}')
            d[metric] ={}
            d[metric + '_shuffled'] = {}

            #if isinstance(rncombinstr,tuple):
            #    rncombinstr = rncombinstr[0]
            #kk, mult_clf_output = tpl
            #print(kk)
            #(prefix,grp,int_type) = kk
            #for clf_type in ['LDA','XGB']:
            #chnames_LFP_controlat_to_move
            pdrop_cl,winning_chan_cl = selBestLFP(mult_clf_output, 'XGB', chnames_LFP=chnames_LFP_contralat_to_move,
                                                  metric=metric, verbose=0)
            d[metric]['best_LFP_contralat_to_move']      = winning_chan_cl

            pdrop,winning_chan = selBestLFP(mult_clf_output, 'XGB', chnames_LFP=chnames_LFP, metric=metric, verbose=1)
            best_LFP = winning_chan
            #pdrop_ = dict( [(chn, (100*a).tolist()) for chn,a in list( pdrop.items() )] )
            d[metric]['best_LFP']      = best_LFP
            d[metric]['perf_drop_pct'] = pdrop

            # save actual values as well
            kn = f'all_present_features'
            pd = mult_clf_output['XGB_analysis_versions'][kn]['perf_dict']
            perf_aver = pd['perf_aver']
            d[metric][kn] = extractPerf(perf_aver,metric)
            perf_all_feats = d[metric][kn]

            perf_shuffled = pd['fold_type_shuffled'][-1]
            d[metric + '_shuffled'][kn ] = extractPerf(perf_shuffled,metric)

            if metric == 'balanced_accuracy':
                d[metric + '_perm_test_info'] = {}
                d[metric + '_perm_test_info'][kn] = list( pd['perm_test_info'] )

                d[metric + f'_perm_{q_perm:.1f}'] = {}
                perf_perm = np.quantile( pd['perm_test_info']['perm_scores'] , q_perm )
                d[metric + f'_perm_{q_perm:.1f}'][kn ] = perf_perm


            for chn in chnames_LFP:
                kn = f'all_present_features_only_{chn}'
                pd = mult_clf_output['XGB_analysis_versions'][kn]['perf_dict']
                perf_aver = pd['perf_aver']
                d[metric][kn] = extractPerf(perf_aver,metric)

                perf_shuffled = pd['fold_type_shuffled'][-1]
                d[metric + '_shuffled'][kn ] = extractPerf(perf_shuffled,metric)

                if metric == 'balanced_accuracy' and 'perm_test_info' in pd:
                    d[metric + '_perm_test_info'][kn] = list( pd['perm_test_info'])

                    perf_perm = np.quantile( pd['perm_test_info']['perm_scores'] , q_perm )
                    d[metric + f'_perm_{q_perm:.1f}'][kn ] = perf_perm

            for chn in chnames_LFP:
                kn = f'all_present_features_but_{chn}'
                pd = mult_clf_output['XGB_analysis_versions'][kn]['perf_dict']
                perf_aver = pd['perf_aver']
                d[metric][kn] = extractPerf(perf_aver,metric)

                perf_shuffled = pd['fold_type_shuffled'][-1]
                d[metric + '_shuffled'][kn ] = extractPerf(perf_shuffled,metric)

                if metric == 'balanced_accuracy' and 'perm_test_info' in pd:
                    d[metric + '_perm_test_info'][kn] = list( pd['perm_test_info'] )
                    perf_perm = np.quantile( pd['perm_test_info']['perm_scores'] , q_perm )
                    d[metric + f'_perm_{q_perm:.1f}'][kn ] = perf_perm

                dif = np.max( np.abs( np.array(perf_all_feats) - np.array(d[metric][kn] ) ) )
                if dif > 5:
                    print(tpl[:-1],metric,'but',chn,dif)

            d['total_num_feats'] = len( mult_clf_output['featnames_for_fit'] )
            d['total_num_datapoints'] = len( mult_clf_output['class_labels_good_for_classif'] )
            d['runCID'] = mult_clf_output['runCID']
            mtime = mult_clf_output.get('mod_time',None)
            mtime= datetime.fromtimestamp( mtime )
            d['fname_mod_time'] = mtime.strftime("%d %b %Y %H:%M")

            best_LFP_dict[rncombinstr][cdk] = d
            #if rncombinstr in best_LFP_dict:
            #
            #else:
            #    best_LFP_dict[rncombinstr] = d

            print(rncombinstr,prefix,f'true best={best_LFP}, best_cl={winning_chan_cl}, in paper={gv.gen_subj_info[subj]["lfpchan_used_in_paper"] }')

    fname_full = pjoin(gv.data_dir, subdir, f'best_LFP_info_both_sides_ext.json')
    if save_result:
        #fname = pjoin(gv.data_dir, subdir, f'best_LFP_info_both_sides.json')
        with open(fname_full, 'w') as f:
            json.dump(best_LFP_dict, f)
        import subprocess as sp
        r = sp.getoutput(f'python -m json.tool {fname_full}') 
        with open(fname_full,'w') as f:
            f.write(r)
        print('Saved to ' ,fname_full)
        #print(r)




    assert len(best_LFP_dict), 'best_LFP_dict is empty'
    return best_LFP_dict, output_per_raw
    #json.dumps(best_LFP_dict)
    #gv.code_dir




    #print(list(recurse(dirDict)))

def collectCalcResults(subdir, start_time, end_time = None, use_tmpdir_to_load = False, load=False,
        require_at_symbol_prefix = True, rawname_regex = None, rawname_before_string = None):
    # load -- hether I want to do actual load or just collect filenames (for deferred loading)
    import os, sys, mne, json, pymatreader, re, time, gc;
    import globvars as gv
    import utils
    import utils_tSNE as utsne
    import utils_preproc as upre
    import matplotlib.pyplot as plt
    import numpy as np
    import multiprocessing as mpr
    import matplotlib as mpl
    import scipy.signal as sig
    import pandas as pd 
    import utils_postprocess_HPC as postp

    data_dir = gv.data_dir
    from os.path import join as pjoin

    light_only = 1
    #light_only = 0
    ndaysBefore = None

    from dateutil import parser
    if end_time is None:
        end_time = parser.parse("30 Oct 2049 21:21:45")

    ndaysBefore = None
    lookup_dir = pjoin(gv.data_dir,subdir)
    print('list recent')
    recent = postp.listRecent(days=ndaysBefore, lookup_dir= lookup_dir,
                              start_time=start_time,
                                       end_time=end_time)
    print('len(recent)=',len(recent))
    rawnames = []
    if rawname_regex is None:
        rawname_regex = '([S0-9]+_[a-z]+)'
    re1 = re.compile('_\!_'+rawname_regex+'_.*')
    re2 = re.compile('_'+rawname_regex+'_.*')
    print('collect rawnames')
    for lf in recent:
        st = 0
        if light_only:
            if not lf.startswith('_!'):
                continue    
        if rawname_before_string is not None:
            ind = lf.find(rawname_before_string)
        else:
            ind = len(lf)
        if light_only:
            r = re.match(re1,lf[:ind])
        else:
            r = re.match(re2,lf[:ind])
        if r is None:
            print('None ',lf)
            continue
        cr = r.groups()[0]
        if cr not in rawnames:
            rawnames += [ cr ]
    rawnames = list(sorted(set(rawnames)))
    print('rawnames = ',rawnames)

    rawname_regex = '([S0-9]+_[a-z]+)'
    #a0 = re.findall('_\!_'+rawname_regex+'_.*',lf)
    #a1 = re.match('_\!_'+rawname_regex+'_.*',lf)
    #print(a0,a1.groups())#

    #print('list recent prefixes')

    #prefixes = postp.listRecentPrefixes(days = ndaysBefore, light_only=light_only,                                     
    #                                    custom_rawname_regex = rawname_regex, recent_files = recent)

    from utils_postprocess_HPC import listComputedData
    r = listComputedData(subdir,None,start_time, end_time, use_main_LFP_chan=1)
    print('listComputedData = ',r)
    if r is None:
        print("found nothing")
        return None
    print('listComputedData = ',r)

    rawnames_found, groupings_found, its_found, prefixes_found = r
    if len(rawnames_found ) == 0:
        return None


    prefixes = prefixes_found
    print('prefixes = ')
    display(prefixes)

    if require_at_symbol_prefix:
        assert np.any( [ p.find('@') >= 0 for p in  prefixes ] )

    ######################   Consistency

    #for grouping_to_check in ['merge_nothing']: #, 'merge_movements']:
    for grouping_to_check,it_to_check in [ ('merge_all_not_trem','basic'), ('merge_movements','basic'),
                                          ('merge_nothing','basic'), ('merge_nothing','trem_vs_quiet') ]:
        print('   ',grouping_to_check,it_to_check)
        r = checkPrefixCollectionConsistencty(subdir,prefixes,start_time, end_time,
                                              grouping_to_check, it_to_check,
                                              use_main_LFP_chan=1, light_only=1,
                                             prefixes_ignore  = [], preloaded=None , use_tmpdir_to_load = use_tmpdir_to_load)
        missing, preloaded = r
        print('missing=', missing)
    import gc; gc.collect()

    ##################    Load

    sources_type = 'parcel_aal'  # or ''
    #groupings_to_collect = ['merge_nothing']; interval_sets_to_collect = ['basic']
    groupings_to_collect = None; isets_to_collect = None
    prefixes_to_collect = None # = prefixes
    r = postp.collectPerformanceInfo3(None,prefixes_to_collect,
        interval_groupings=groupings_to_collect,
        interval_sets =  isets_to_collect,
        nraws_used='[0-9]+', sources_type = sources_type,
        printFilenames=1,
        ndays_before=ndaysBefore,
        use_main_LFP_chan=1,
        subdir=subdir, remove_large_items = 1,
        list_only=0, allow_multi_fn_same_prefix=0,
        use_light_files = light_only, rawname_regex_full=0,
        start_time=start_time,
        end_time=end_time, load=load, use_tmpdir_to_load=use_tmpdir_to_load)

    #nraws_used='(10,12,20,24)'
    #output_per_raw,Ximp_per_raw,gis_per_raw = r
    output_per_raw,_,_ = r
    print('len(output_per_raw) =', len(output_per_raw))
    import gc; gc.collect()


    #Audio(filename=sound_file, autoplay=True)
    import utils_postprocess as pp
    tpll = pp.multiLevelDict2TupleList(output_per_raw,4,3)

    z0 = [tpl[:-1] for tpl in tpll]
    #rns_ord, prefs_ord, grp_ord, it_ord = list (zip(*z0  ) )
    tpll_reshaped = np.array( list (zip(*z0  ) ) )

    ####################################

    import utils_postprocess as pp
    from datetime import datetime
    from utils_postprocess_HPC import loadRunCorresp
    CIDs,cretimes = [],[]
    mod_times = []
    nloaded = 0
    for tpl in tpll:
        moc = tpl[-1]
        corresp,all_info = loadRunCorresp(moc)
        if moc['loaded']:
            if 'cmd' not in moc:
                print(f'Cmd is not in {tpl[:-1]}')
                break
            runCID = dict( moc['cmd'][0] ).get('--runCID', None)
            moc['runstrings_creation_time'] = all_info['date_created']
            moc['runCID'] = runCID
            CIDs += [runCID]
            cretimes += [all_info['date_created']]

            #mod_time = os.stat( moc['filename_full'] ).st_mtime
            mod_time = moc['mod_time']
            dt = datetime.fromtimestamp(mod_time)
            moc['mod_time_dt'] = dt
            nloaded += 1

        mod_times += [moc['mod_time']] # timestamps only

    bv,bcoord,_ = plt.hist(mod_times)
    mn = datetime.fromtimestamp ( np.min(bcoord) )
    mx = datetime.fromtimestamp ( np.max(bcoord) )
    delta = mx-mn
    print(f'Time spread = {delta}, time min = {mn}, time max = {mx}')

    ################################   CID

    if not nloaded:
        print('Nothing is loaded :(')
    CIDs = list( map(int,CIDs) )
    CIDs_sorted = list( sorted( set( map(int,CIDs) ) ) )
    if len(CIDs_sorted) > 1:
        for CID in CIDs_sorted:
            inds = np.where( np.array(CIDs) == CID )[0]
            times = [tpll[i][-1]['mod_time'] for i in inds]
            times = list( sorted(times) )
            print('CID =  ',CID, 'len(inds)=',len(inds), times[0], times[-1] )
        CID_most_recent = list( sorted( map(int,CIDs) ) )[-1]
        print('most recent CID=', CID_most_recent )
    else:
        print("only one CID found")

    #################################   Completeness of loaded

    from utils_postprocess_HPC import checkTupleListTableCompleteness
    print('---- Total complentess')
    b,missing = checkTupleListTableCompleteness(tpll_reshaped)
    print(missing)

    print('---- per grouping complentess')
    for grps_cur in groupings_found:
        print(grps_cur)
        outputs_filtered = postp.filterOutputs(output_per_raw, rns=rawnames,
                            prefs=prefixes, grps =  [grps_cur] )
        b,missing =  checkTupleListTableCompleteness(outputs_filtered)
        print(missing)

    return output_per_raw

def extendDfMultiOPR(df, pptype2res, allow_missing_tremor = False):
    dfs = []
    for ppt,opr in pptype2res.items():
        print(f'  Extending DF for ppt={ppt}')
        # important to copy because inplace modfications follow 
        dfc = df.query(f'pptype == "{ppt}"').copy() 
        opr = pptype2res[ppt]
        dfc_mod = extendDf(dfc, opr, allow_missing_tremor=allow_missing_tremor)
        dfs += [dfc_mod]
    if len(dfs) > 1:
        r = pd.concat(dfs, ignore_index=1) 
    else:
        r = dfc_mod
    return r

def extendDf(df, output_per_raw, allow_missing_tremor = False):
    import re
    from utils_postprocess_HPC import getTremorDetPerf
    from utils_postprocess_HPC import getMocFromRow

    print('addRunCorrespCols')
    addRunCorrespCols(df, output_per_raw)

    def addTremorPerf(row):
        #print('a',row)
        grp,it = row['grouping'],row['interval_set']
        try:
            moc = getMocFromRow(row, output_per_raw)
            #moc = output_per_raw[row['rawname']][row['prefix']][grp][it]
            r = getTremorDetPerf(moc,grp,it, allow_missing_tremor=allow_missing_tremor)
        except KeyError:
            r = np.nan
        #row['tremor_det'] = r
        return r

    print('tremor detection performance')
    df['tremor_det_perf'] = df.apply(addTremorPerf,axis=1)
    #df.reset_index()

    def addPar(row):
        #print('a',row)
        try:
            moc = getMocFromRow(row, output_per_raw)
            r = moc['pars']
        except (KeyError,TypeError) as e:
            print('addPar: ',e)
            r = {}
        return r
    df['par'] = df.apply(addPar,axis=1)

    param_keys_to_extract = ('feat_types,n_permutations_permtest,rescale_feats,'
        'scale_feat_combine_type,'
        'baseline_int_type,grouping_best_LFP,int_types_best_LFP').split(',')
    for pk in param_keys_to_extract:
        df[pk] = df.apply(lambda x: x['par'].get(pk,None), 1)

    ###################
    def lbd(row):
        #print('a',row)
        try:
            moc = getMocFromRow(row, output_per_raw)
            r = moc['feat_pars_pri']
        except (KeyError,TypeError) as e:
            print('add feat_pars_pri: ',e)
            r = None
        return r
    df['feat_pars_pri'] = df.apply(lbd,axis=1)

    feat_param_keys_to_extract = ('scale_data_combine_type,baseline_int_type,'
            'prescale_data,rescale_feats').split(',')
    for pk in feat_param_keys_to_extract:
        def lbd(x): 
            fp = x['feat_pars_pri']
            if fp is None:
                return None
            else:
                return fp[0].get(pk,None)
        df['featf:' + pk] = df.apply(lbd, 1)

    df['num'] = pd.to_numeric(df['num'])
    df['num'] = df['num'].map(lambda x: int(x) if ( not (np.isnan(x) or (x is None) ) ) else 0    )
    df['numpts'] = pd.to_numeric(df['numpts'])
    df['sens'] = pd.to_numeric(df['sens'])
    df['spec'] = pd.to_numeric(df['spec'])

    df['g_is'] = df.apply(lambda x: (x['grouping'],x['interval_set']),1)


    df['subject'] = df['rawname'].apply(lambda x: x.split('_')[0]) 
    def f(s):
        els = s.split('_')
        if len(els) > 1:
            return els[1]
        elif len(els) == 1:
            return 'off,on'
        else:
            return None

    df['medcond'] = df['rawname'].apply(f) 
    df['move_hand_side_letter'] = df['move_hand_side_letter'].apply(lambda x: x[0] if isinstance(x,list) else x) 
    df['move_hand_opside_letter'] = df['move_hand_opside_letter'].apply(lambda x: x[0] if isinstance(x,list) else x) 


    print('addParsColumns')
    df = addParsColumns(df, output_per_raw)
    df['subskip_fit'] = pd.to_numeric(df['subskip_fit'])

    ##### Add counts columns
    def lbd(row):
        moc = getMocFromRow(row, output_per_raw)
        if moc is None:
            return None, None
        if 'counts' not in moc:
            return -1,-1
        cts = moc['counts']

        tot = np.sum( list(cts.values()) )
        cts2 = cts.copy()
        s = ''

        subskip_fit = row['subskip_fit']
        total = tot / subskip_fit  # in sec

        #for k,v in cts.items():
        #    k2 = k[:-2]
        #    #k2 = k
        #    stmp = f'{v / tot * 100:4.1f}%={v/32:5.1f}s'
        #    cts2[k2] = stmp
        #    s += f', {k2}: {stmp}'
        #    # 256

        return cts, total

    # these numpers are taken from label_good which HAVE NOT been subskipped
    df[['points_per_beh_state', 'total_len_sec']] = df.apply(lbd,1,result_type='expand')

    from utils_tSNE import countClassLabels
    def lbd(row):
        moc = getMocFromRow( row , output_per_raw)
        if moc is None:
            return { 'uuu':np.nan}
        y = featnames = moc['class_labels_good_for_classif']
        revdict = moc['revdict_lenc']
        numpoints_per_class_id = countClassLabels(y, class_ids_grouped=None, revdict=revdict)
        #print(sidelet)
        #print(numpoints_per_class_id)
        return numpoints_per_class_id

    df['points_per_beh_state2'] = df.apply(lbd,1)
    df['numpts2'] = df['points_per_beh_state2'].apply(lambda x: np.sum( list( x.values() ) ) ,1)

    ##########################################o

    print('pctpts')
    from globvars import gp
    for it in  gp.int_types_basic:
        def lbd(d):        
            if d is None:
                return None
            for k,v in d.items():
                if k.startswith(it):
                    #res = d[k] / 32
                    res = d[k] 
                    return int( res )
        df[f'numpts_{it}'] = df['points_per_beh_state2'].apply(lbd)
        
        def lbd(row):
            n = row[f'numpts_{it}']
            if n is None:
                n = np.nan
            npts_pct = n / row['numpts2']
            return npts_pct * 100
        df[f'pctpts_{it}'] = df.apply(lbd,1)
        
    from featlist import selFeatsRegexInds
    from utils_postprocess_HPC import prefTempl2pref, prefix2prefixTempl

    print('mainLFPside')
    regex = re.compile(".*LFP(.).*")
    def lbd(row):
        moc = getMocFromRow( row , output_per_raw)
        if moc is None:
            return None,None
        featnames = moc['feature_names_filtered']
        inds = selFeatsRegexInds(featnames, '.*LFP.*')
        sidelet = None
        templ = None

        if len(inds):
            assert len(inds) in [1,3], len(inds)
            r = re.match(regex, featnames[inds[0]])
            sidelet = r.groups()[0]
            
            templ = prefix2prefixTempl(sidelet, row['subject'])
        #print(sidelet)
        return sidelet, templ

    df[['mainLFPside','mainLFPside_templ']] = df.apply(lbd,1,result_type = 'expand')
    #return sidelet

    ########################################
    def lbd(row):
        moc = getMocFromRow(row, output_per_raw)
        if moc is None:
            return None
    #     pars = moc['pars']
    #     s = f' ../run/run_ML.py'
    #     for p,v in pars.items():
    #         if p in ['iniAdd', 'search_best_LFP', 'code_ver' ]:
    #             continue  
    #         s += f' --{p} {v}'        
    #     s += ' --exit_after artif_processed'

        s = moc['LFP_side_to_use_final']
        return s

    #df['runs'] = df.apply(lbd,1)
    #df['runs']

    df['LFP_side_to_use'] = df.apply(lbd,1)


    ###########################################
    def lbd(row):
        moc = getMocFromRow(row, output_per_raw)
        if moc is None:
            return None
        rn = row['rawname']
        subj = row['subject']
        prefix = row['prefix']
        if isinstance(subj, list):
            subj = subj[0]
        subjs_involved, mcs, mvsls, mvosls = analyzeRnstr(rn)

        mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
        moveopside =  utils.getOppositeSideStr( mainmoveside ) 

        movesidelet = mainmoveside[0].upper()
        moveopsidelet = utils.getOppositeSideStr( movesidelet )

        pars = moc['pars']

        #onlyH_mob_subskip8@left_exCB-copy_from_search_LFP

        pref_templ = None
        pref_templ0 = None
        atsymbol = prefix.find('@') >= 0
        if atsymbol:
            dashi = prefix.find('-')
            side_brain = prefix[:dashi]
            side_LFP   = prefix[dashi + 1:]

            suff = side_LFP
            suff = suff.replace(mainmoveside,'^')
            side_LFP2_0 = suff.replace(moveopside,'%')
            if side_LFP == "copy_from_search_LFP":
                assert pars['LFP_side_to_use'] == 'copy_from_search_LFP'
                side = moc['LFP_side_to_use_final']
                sidelet = side[0].upper()

                #suff = side_LFP
                #suff = suff.replace(side_LFP,sidelet)
                suff = sidelet
                suff = suff.replace(movesidelet,'^')
                side_LFP2 = suff.replace(moveopsidelet,'%')
                #print('side_LFP2 = ',side_LFP2)

            else:
                side_LFP2 = side_LFP2_0

            if side_LFP2 in ['L','R' ]:
                print(side_LFP, side_LFP2_0, side_LFP2)
                raise ValueError('rrr')

            suff = side_brain
            suff = suff.replace(mainmoveside,'^')
            brain_side2 = suff.replace(moveopside,'%')

            brain_side2 = brain_side2.replace('both','B')

            pref_templ = f"{brain_side2}-{side_LFP2}"
            pref_templ0 = f"{brain_side2}-{side_LFP2_0}"
                #pref_templ = prefix[:-2] + suff
        else:
            for suff in ['LL','RR','LR','RL','BB','BL','BR', 'LB','RB']:
                #assert len(set(mvsls)) == 1, (prefix,mvsls)
                if prefix.endswith(suff) and len(set(mvsls)) == 1:
                    suff = prefix[-2:]
                    suff = suff.replace(movesidelet,'^')
                    suff = suff.replace(moveopsidelet,'%')
                    pref_templ = prefix[:-2] + suff
                    
                    pref_templ0 = pref_templ
            for suff in ['BC','LC','RC']:
                if prefix.endswith(suff) and len(set(mvsls)) == 1:
                    assert pars['LFP_side_to_use'] == 'copy_from_search_LFP'
                    side = moc['LFP_side_to_use_final']
                    sidelet = side[0].upper()

                    suff = prefix[-2:]
                    suff = suff.replace('C',sidelet)  # different 
                    suff = suff.replace(movesidelet,'^')
                    suff = suff.replace(moveopsidelet,'%')
                    pref_templ = prefix[:-2] + suff

                    # same but not touching C
                    suff = prefix[-2:]
                    suff = suff.replace(movesidelet,'^')
                    suff = suff.replace(moveopsidelet,'%')
                    pref_templ0 = prefix[:-2] + suff

        #if pref_templ is None:
        #    print( prefix )
        return pref_templ,pref_templ0

    df[['prefix_templ','prefix_templ0']] = df.apply(lbd,1,
            result_type= 'expand')

    # need base_pref_start_dict
    #df = addImprovColDfAll(df, inplace
    #df = addImprovCol

    qval = 0.9
    global rowinfostr_fails
    rowinfostr_fails = []
    def lbd(row):
        moc = getMocFromRow(row, output_per_raw)
        if moc is None:
            global rowinfostr_fails
            rowinfostr = f'{row["rawname"]} {row["grouping"]} {row["interval_set"]} {row["pptype"]} {row["prefix"] }'
            print(f'For this row we got no Moc :( {rowinfostr}')
            #row
            rowinfostr_fails += [ rowinfostr ]
            #raise ValueError('aaa')
            return None
        anver = moc['XGB_analysis_versions']    
        pti = anver['all_present_features']['perf_dict']['perm_test_info']
        #print(pti['score_noperm'])
        perm_scores = pti['perm_scores']
        v = np.quantile( perm_scores, q=qval)
        return v
    print('rowinfostr_fails = ',rowinfostr_fails)

    df[f'bacc_perm_{qval:.1f}'] = df.apply(lbd,1)


    return df

###
