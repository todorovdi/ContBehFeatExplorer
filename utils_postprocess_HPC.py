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

# compared to 2 it adds warid to tuple keys
def collectPerformanceInfo3(rawnames, prefixes, ndays_before = None,
                           n_feats_PCA=None,dim_PCA=None, nraws_used=None,
                           sources_type = None, printFilenames = False,
                           group_fn = 10, group_ind=0, subdir = '', old_file_format=False,
                           use_main_LFP_chan=False,
                           remove_large_items=1, list_only = False,
                           allow_multi_fn_same_prefix = False, use_light_files=True,
                            rawname_regex_full =False,
                           start_time=None,end_time=None,
                           lighter_light = False ):
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
        return None
    else:
        print(f'In {dir_to_use} found {len(fnfound)} files matching regex')

    strs = []
    inds = []
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
            print('skipping {} due to bad prefix'.format(fnf) )
            continue

        if rawnames is not None and rawstrid not in rawnames:
            print('skipping {} due to bad rawname'.format(fnf) )
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
        print( 'fn_per_fntype keys are', fn_per_fntype.keys() )

    #################3
    for s in fn_per_fntype:
        tuples = fn_per_fntype[s]
        print(f'   {s}: {len( tuples ) } tuples')
    ####################

    if not list_only:
        for s in fn_per_fntype:
            print(f'!!!!!!!!!!   Start loading files for {s}')
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

                t0 = time()
                f = np.load(fname_full, allow_pickle=1)

                if use_light_files:
                    res_cur = f['results_light'][()]
                else:
                    res_cur = f['results_cur'][()]
                res_cur['filename_full'] = fname_full

                if 'featsel_shap_res' in f and 'featsel_shap_res' not in res_cur:
                    print('Moving Shapley values')
                    res_cur['featsel_shap_res'] = f['featsel_shap_res'][()]

                from utils_postprocess import removeLargeItems
                if (not use_light_files) or lighter_light:
                    if 'class_labels_good' in f:
                        res_cur['class_labels_good'] = f['class_labels_good']
                    else:
                        print('class_labels_good is not in the archive!')
                    res_cur = removeLargeItems(res_cur)

                    if remove_large_items:
                        newfn = fnf
                        if not use_light_files:
                            newfn = '_!' + fnf
                        fname_light = pjoin( dir_to_use, newfn)
                        print('resaving LIGHT file ',fname_light)
                        np.savez(fname_light, results_light=res_cur)

                ######################

                del f
                t1 = time()
                import gc; gc.collect()

                tnow=time()
                print(f'------- Loading and processing {fnf} took {tnow-t0:.2f}s, of it gc={tnow-t1:.2f}')

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
                output_per_raw[rawstrid][prefix_eff]['feature_names_filtered'] = res_cur['feature_names_filtered']

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

    return output_per_raw,Ximputed_per_raw, good_bininds_per_raw


def collectPerformanceInfo2(rawnames, prefixes, ndays_before = None,
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

    return output_per_raw,Ximputed_per_raw, good_bininds_per_raw

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
               start_time=None,end_time=None):
    if lookup_dir is None:
        lookup_dir = gv.data_dir
    lf = os.listdir(lookup_dir)
    final_list = []
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
                      start_time=None,end_time=None ):
    import re
    lf = listRecent(days, hours, lookup_dir, start_time=start_time,
                    end_time=end_time)
    prefixes = []
    for f in lf:
        if custom_rawname_regex is None:
            regex = '_S.._.*grp[0-9\-]+_(.*)_ML'
        else:
            regex = '_'+custom_rawname_regex+ '_.*grp[0-9\-]+_(.*)_ML'
        if light_only:
            regex = '_!' + regex
        out = re.match(regex, f)
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


def prepTableInfo2(output_per_raw, prefixes=None, perf_to_use_list = [('perfs_XGB','perfs_XGB_red') ],
                  to_show = [('allsep','merge_nothing','basic')],
                  show_F1=False, use_CV_perf=True, rname_crop = slice(0,3), save_csv = True,
                  sources_type='parcel_aal', subdir=''):

    #perf_to_use is either 'perfs_XGB', 'perfs_XGB_red' or one of lda versions
    # Todo: pert_to_use_list -- is list of couples -- one and list ot feat
    # reductions

    #if feat_nums_perraw is None or feat_nums_red_perraw is None:
    r = collectFeatNums(output_per_raw)
    feat_nums_perraw, feat_nums_perraw0, \
        feat_nums_red_perraw, feat_nums_red2_perraw,\
        feat_numdict_perraw = r

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

    for tpl in perf_to_use_list:
        clf_type,perf_to_use,perf_red_to_use = tpl
        info_per_rn_pref ={}
        red_mode = False
        if perf_to_use.endswith('_red'):
            red_mode = True
        #table =  [ [''] +  prefix_labels_perraw[rawnames[0] ]
        #table =  [ [''] +  [ dct.get(prefix,prefix) for dct in prefix_labels_perraw] ]
        table =  [ [''] +  prefixes ]
        was_valid, was_red_valid = False, False
        for rn in output_per_raw:
            for lt, it_grp, it_set in to_show:
            #for lt in label_types:
                info_per_pref = {}
                # raw and label type (grouping+int_types) name goes here
                # this will be a row name
                row_name = '{}_{}'.format(rn[rname_crop], lt)
                table_row = [row_name ]
                for prefix in prefixes:
                    info_cur = {}
                    #sens,spec = res[rn].get(pref, (np.nan, np.nan))
                    r = output_per_raw[rn].get(prefix, None)
                    if (r is not None) and (it_grp not in r or it_set not in r[it_grp]):
                        r = None

                    perfs = None
                    if r is None:
                        print(f'  Warning: no prefix {prefix} for {rn}')
                    else:
                        mult_clf_results = r[it_grp][it_set]
                        if mult_clf_results is not None:

                            numdict_cur_ = feat_numdict_perraw[rn][prefix][(it_grp,it_set)]
                            numdict_cur = numdict_cur_[clf_type]

                            perfs_CV,perfs_noCV, perfs_red_CV,perfs_red_noCV  = None,None,None,None
                            num,num_red = None, None
                            if clf_type == 'XGB':
                                if perf_to_use == 'perfs_XGB':
                                #if perf_to_use in ['perfs_XGB','perfs_XGB_red']:
                                    #if 'perfs_XGB' in mult_clf_results and mult_clf_results['perfs_XGB'] is not None:
                                    if perf_to_use in mult_clf_results and mult_clf_results[perf_to_use] is not None:
                                        perfs_XGB = mult_clf_results[perf_to_use]
                                        ind = 0
                                        perfs_noCV = perfs_XGB[ind]['perf_nocv']
                                        perfs_CV = perfs_XGB[ind]['perf_aver']
                                        if perf_red_to_use == 'perfs_XGB_red':
                                            perfs_XGB = mult_clf_results[perf_to_use]
                                            num_red = numdict_cur['red']
                                        elif perf_red_to_use == 'perfs_XGB_fs_red':
                                            perfs_XGB = mult_clf_results['perfs_XGB_fs']
                                            num_red = numdict_cur['fs_red']
                                        else:
                                            raise ValueError(f'Undef! perf_red_to_use={perf_red_to_use}')
                                        ind = -1
                                        perfs_red_noCV = perfs_XGB[ind]['perf_nocv']
                                        perfs_red_CV = perfs_XGB[ind]['perf_aver']
                                        #if perf_to_use == 'perfs_XGB':
                                        #    ind = 0
                                        #elif perf_to_use == 'perfs_XGB_red':
                                        #    ind = -1
                                        num = numdict_cur['all_features']
                                else:
                                    XGB_anver = mult_clf_results['XGB_analysis_versions']
                                    anver_cur = XGB_anver.get(perf_to_use)
                                    if anver_cur is not None:
                                        perfs_CV = anver_cur['perf_aver']
                                        perfs_noCV = anver_cur['perf_nocv']
                                    else:
                                        print(f'perf_to_use (={perf_to_use}): None!')

                                    anver_red_cur = XGB_anver.get(perf_red_to_use)
                                    if anver_red_cur is not None:
                                        perfs_red_CV = anver_red_cur['perf_aver']
                                        perfs_red_noCV = anver_red_cur['perf_nocv']
                                    else:
                                        print(f'perf_to_use (={perf_to_use}): None!')
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
                                else:
                                    print(f'perf_to_use (={perf_to_use}): None!')

                                anver_red_cur = lda_anver.get(perf_red_to_use,None)
                                if anver_red_cur is not None:
                                    perfs_red_CV = anver_red_cur['CV']['CV_perfs']
                                    perfs_red_CV = np.mean(np.array([ perfs_red_CV[ip][:3] for ip in range(len(perfs_red_CV)) ]   ), axis=0)
                                    perfs_red_CV2 = anver_red_cur['CV_aver']['perfs']
                                    perfs_red_noCV = anver_red_cur['fit_to_all_data']['perfs']
                                else:
                                    print(f'perf_red_to_use (={perf_red_to_use}): None!')

                                num = numdict_cur[perf_to_use]
                                num_red = numdict_cur[perf_red_to_use]

                            if use_CV_perf:
                                perfs = perfs_CV
                                perfs_red = perfs_red_CV
                            else:
                                perfs = perfs_noCV
                                perfs_red = perfs_red_noCV

                    if perfs is None:
                        print('Warning :',rn,prefix,lt)
                        sens,spec,F1 = np.nan, np.nan, np.nan
                    else:
                        #print([type( p) for p in perfs])
                        # sometimes perfs has confmat but sometimes not
                        sens,spec,F1 = perfs[0],perfs[1],perfs[2]
                        was_valid = True

                    if perfs_red is None:
                        sens_red,spec_red,F1_red = np.nan, np.nan, np.nan
                    else:
                        #print([type( p) for p in perfs_red])
                        sens_red,spec_red,F1_red = perfs_red[0],perfs_red[1],perfs_red[2]
                        was_red_valid = True

                    info_cur['sens'] = sens
                    info_cur['spec'] = spec
                    info_cur['F1'] = F1
                    info_cur['sens_red'] = sens_red
                    info_cur['spec_red'] = spec_red
                    info_cur['F1_red'] = F1_red
                    info_cur['num'] = num
                    info_cur['num_red'] = num_red
                    #str_to_put = '{:.0f},{:.0f}'.format(100*sens,100*spec)
                    if show_F1:
                        str_to_put =  '{:.0f},{:.0f},{:.0f}'.format(100*sens,100*spec,100*F1)
                    else:
                        str_to_put =  '{:.0f},{:.0f}'.format(100*sens,100*spec)
                    try:
                        if red_mode:
                            s = ' :{}/{}'.format(num_red,num)
                        else:
                            s = ' :{}/{}'.format(num,num)
                        str_to_put = str_to_put + s

                    except KeyError as e:
                        print('AAAAA ',perf_to_use,rn,prefix,it_grp,it_set,e)
                    table_row += [ str_to_put  ]
                    print(rn,lt,prefix,str_to_put)

                    info_per_pref[prefix] = info_cur

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

    return table_info_per_perf_type, table_per_perf_type

def prepTableInfo3(output_per_raw, prefixes=None, perf_to_use_list = [('perfs_XGB','perfs_XGB_red') ],
                  to_show = [('allsep','merge_nothing','basic')],
                  show_F1=False, use_CV_perf=True, rname_crop = slice(0,3), save_csv = True,
                  sources_type='parcel_aal', subdir=''):

    #perf_to_use is either 'perfs_XGB', 'perfs_XGB_red' or one of lda versions
    # Todo: pert_to_use_list -- is list of couples -- one and list ot feat
    # reductions

    #if feat_nums_perraw is None or feat_nums_red_perraw is None:

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
            for lt, it_grp, it_set in to_show:
            #for lt in label_types:
                info_per_pref = {}
                # raw and label type (grouping+int_types) name goes here
                # this will be a row name
                row_name = '{}_{}'.format(rn[rname_crop], lt)
                table_row = [row_name ]
                for prefix in prefixes:
                    info_cur = {}
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
                    if r is None:
                        print(f'  Warning: no prefix {prefix} for {rn}')
                        prefix_missing = True
                    else:
                        prefix_missing = False
                        mult_clf_results = r[it_grp][it_set]
                        if mult_clf_results is not None:
                            class_label_names = mult_clf_results.get('class_label_names_ordered',None)
                            if class_label_names is not None:
                                lblind_trem = class_label_names.index('trem_L')
                                print('prepTableInfo3: warninig: Using fixed side for tremor: trem_L')
                            else:
                                lblind_trem = 0

                            perfs_CV,perfs_noCV, perfs_red_CV,perfs_red_noCV  = None,None,None,None
                            perfs_CV_recalc,perfs_red_CV_recalc = None,None
                            if clf_type == 'XGB':
                                XGB_anver = mult_clf_results['XGB_analysis_versions']
                                anver_cur = XGB_anver.get(perf_to_use)
                                if anver_cur is not None:
                                    if 'perf_aver' in anver_cur:
                                        perfs_CV   = anver_cur['perf_aver']
                                        perfs_noCV = anver_cur['perf_nocv']
                                        ps = anver_cur.get('perfs_CV', None)
                                    else:
                                        perfs_CV = anver_cur['perf_dict']['perf_aver']
                                        perfs_noCV = anver_cur['perf_dict']['perf_nocv']
                                        ps = anver_cur['perf_dict'].get('perfs_CV', None)
                                    perfs_CV_recalc = recalcPerfFromCV(ps,lblind_trem)

                                    if perf_add is not None:
                                        if perf_add == 'across_subj':
                                            pdict = anver_cur['across']['subj']
                                            if pdict is not None:
                                                perfs_add_CV = pdict['perf_aver']
                                                perfs_add_noCV = pdict['perf_nocv']
                                                ps = pdict['perfs_CV']
                                                perfs_add_CV_recalc = recalcPerfFromCV(ps,lblind_trem)
                                        elif perf_add == 'across_medcond':
                                            pdict = anver_cur['across']['medcond']
                                            if pdict is not None:
                                                perfs_add_CV = pdict['perf_aver']
                                                perfs_add_noCV = pdict['perf_nocv']
                                                ps = pdict['perfs_CV']
                                                perfs_add_CV_recalc = recalcPerfFromCV(ps,lblind_trem)
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
                                    perfs_red_CV_recalc = recalcPerfFromCV(ps,lblind_trem)
                                elif perf_red_to_use in ['interpret_EBM', 'interpret_DPEBM']:
                                    featsubset_name = 'all'
                                    EBM_dict = mult_clf_results['featsel_per_method'][perf_red_to_use][featsubset_name]
                                    perfs_red = EBM_dict['perf']
                                    perfs_red_CV = EBM_dict['perf_dict']['perf_aver']
                                else:
                                    print(f'perf_red_to_use (={perf_to_use}): None!')

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

                    if perfs is None:
                        print('Warning :',rn,prefix,lt)
                        sens,spec,F1 = np.nan, np.nan, np.nan
                    else:
                        #print([type( p) for p in perfs])
                        # sometimes perfs has confmat but sometimes not
                        sens,spec,F1 = perfs[:3]
                        was_valid = True

                    if perfs_red is None:
                        sens_red,spec_red,F1_red = np.nan, np.nan, np.nan
                    else:
                        #print([type( p) for p in perfs_red])
                        sens_red,spec_red,F1_red = perfs_red[:3]
                        was_red_valid = True

                    if num is not None and num_red is not None:
                        assert num >= num_red, f'{rn},{lt},{it_grp},{it_set},{tpl}:{prefix}  {num},{num_red}'

                    if perfs_CV_recalc is not None:
                        info_cur['sens_recalc'] = perfs_CV_recalc[0]
                        info_cur['spec_recalc'] = perfs_CV_recalc[1]
                    info_cur['sens'] = sens
                    info_cur['spec'] = spec
                    info_cur['F1'] = F1
                    info_cur['sens_red'] = sens_red
                    info_cur['spec_red'] = spec_red
                    if perfs_red_CV_recalc is not None:
                        info_cur['sens_red_recalc'] = perfs_red_CV_recalc[0]
                        info_cur['spec_red_recalc'] = perfs_red_CV_recalc[1]
                    else:
                        info_cur['sens_red_recalc'] = np.nan
                        info_cur['spec_red_recalc'] = np.nan
                    if perf_add is not None and perfs_add_CV is not None:
                        info_cur['sens_add'] = perfs_add_CV[0]
                        info_cur['spec_add'] = perfs_add_CV[1]
                        info_cur['F1_add'] = perfs_add_CV[2]
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

    #sc = scores_stats['std']
    #ax = axs[0]
    #xerr = sc
    #names_cur = [nm] + names
    #sc = [ 0 ] + list(sc)
    #ax.barh(names_cur,np.array(sc), color=color_cur, alpha=1., xerr=xerr ); #ax.tick_params(axis='x', labelrotation=90 )


#def plotFeatSignifSHAP(pdf,featsel_per_method, fshs, featnames_list,
#                       class_labels_good_for_classif,
#                       class_label_names,
#                       featnames_subset = None, figname_prefix='',
#                       n_individ_feats_show=20, roi_labels = None,
#                       chnames_LFP = None, body_side='L', hh=8,
#                       separate_by_band = False, suptitle = None, suptitle_fontsize=20,
#                      tickfontsize = 10,
#                       marker_mean = 'o', marker_max = 'x', axs=None):
#    '''
#    fshs -- names of featsel method to use
#    '''
#    import utils_postprocess_HPC as postp
#    import matplotlib.pyplot as plt
#
#    if isinstance(fshs , str):
#        fshs = [fshs]
#
#    from collections.abc import Iterable
#    if isinstance(featnames_list, Iterable) and \
#            (isinstance(featnames_list[0], str) ):
#        featnames_list = [featnames_list]
#
#    assert len(featnames_list) == len(fshs)
#
#    #fsh = 'XGB_Shapley'
#    if len(fshs) == 1:
#        colors_fsh = len(fshs) * [None]
#    else:
#        colors_fsh = ['blue','red','green', 'purple']
#
#    fspm_def = featsel_per_method[ fshs[0]  ]
#    scores = fspm_def.get('scores',None)
#    if scores is None:
#        scores_per_class_def = fspm_def.get('scores_av',None)
#    else:
#        assert scores.shape[-1] - 1 == len(featnames_list[0]), (scores.shape[-1] , len(featnames_list[0]))
#        scores_per_class_def = utsne.getScoresPerClass(class_labels_good_for_classif, scores)
#    nscores = len(scores_per_class_def)
#
#    nr = nscores; nc = 3; #nc= len(scores_stats) - 2;
#    ww = 3 + 5;
#    if axs is None:
#        fig,axs = plt.subplots(nr,nc, figsize = (nc*ww,nr*hh), gridspec_kw={'width_ratios': [1,1,3]} );
#        plt.subplots_adjust(top=1-0.02)
#    else:
#        assert axs.shape == (nr,nc)
#
#
#    cmap = plt.cm.get_cmap('tab20', 20)
#    #cmap( (ri0 + i*ri1) % 20)
#
#
#    for fshi,fsh in enumerate(fshs):
#        fspm = featsel_per_method[fsh]
#        featnames = featnames_list[fshi]
#        scores = fspm.get('scores',None)
#        if scores is None:
#            scores_per_class = fspm.get('scores_av',None)
#            bias = fspm.get('scores_bias_av',None)
#
#        else:
#            assert scores.shape[-1] - 1 == len(featnames), (scores.shape[-1] , len(featnames))
#            # XGB doc: Note the final column is the bias term
#            scores_per_class, bias = utsne.getScoresPerClass(class_labels_good_for_classif, scores, ret_bias=1)
#            print( fspm.keys(), scores.shape )
#
#
#        # make plots for every class label
#        for lblind in range(scores_per_class.shape[0] ):
#            # select points where true class is like the current one
#            #ptinds = np.where(class_labels_good_for_classif == lblind)[0]
#            #classid_enconded = lblind
#            #scores_cur = np.mean(scores[ptinds,lblind,0:-1], axis=0)
#
#            scores_cur = scores_per_class[lblind]
#            label_str = class_label_names[lblind]
#
#            ###############################
#            ftype_info = utils.collectFeatTypeInfo(featnames)
#
#            feat_groups_all = []
#            feature_groups_names = []
#            clrs = []
#
#            clri = 0
#            feat_groups_basic = [f'^{ft}_.*' for ft in ftype_info['ftypes']]
#            feat_groups_all+= feat_groups_basic
#            feature_groups_names += feat_groups_basic
#            clrs += [cmap(clri)] * len(feat_groups_basic)
#
#            clri += 1
#
#            ft = 'bpcorr'
#            if 'bpcorr' in ftype_info['ftypes']:
#                feat_groups_two_bands = [f'^{ft}_{fb1}_.*,{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
#            #     feat_groups_two_bands = ['^bpcorr_gamma.*,tremor.*','^bpcorr_gamma.*,beta.*','^bpcorr_gamma.*,HFO.*',
#            #                                 '^bpcorr_beta.*,tremor.*','^bpcorr_beta.*,gamma.*','^bpcorr_beta.*,HFO.*',
#            #                                 '^bpcorr_tremor.*,beta.*','^bpcorr_tremor.*,gamma.*','^bpcorr_tremor.*,HFO.*']
#                feat_groups_all += feat_groups_two_bands
#                feature_groups_names += feat_groups_two_bands
#                clrs += [cmap(clri)] * len(feat_groups_two_bands)
#
#            for ft in ['rbcorr', 'con']:
#                if ft in ftype_info['ftypes']:
#                    #feat_groups_rbcorr_band = ['^rbcorr_tremor.*', '^rbcorr_beta.*',  '^rbcorr_gamma.*']
#                    feat_groups_one_band = [ f'^{ft}_{fb}_.*' for fb in ftype_info['fband_per_ftype'][ft] ]
#                    feat_groups_all += feat_groups_one_band
#                    feature_groups_names += feat_groups_one_band
#                    clrs += [cmap(clri)] * len(feat_groups_one_band)
#            #feat_groups_all
#
#            # allow HFO2 and high_beta  (allow numbers and one underscore in
#            # the middle
#            bnpattern = '[a-zA-Z0-9]+_?[a-zA-Z0-9]*'
#
#
#            wasH = 0
#            for ft in gv.noband_feat_types:
#                if ft in ftype_info['ftypes']:
#                    wasH = True
#
#            if wasH:
#                ft_templ = f'({"|".join(gv.noband_feat_types) })'
#                a = [f'^{ft_templ}_LFP.*']
#                feat_groups_all += a
#                feature_groups_names += [  '^Hjorth_LFP'  ]
#                clrs += [cmap(clri)] * len(a)
#
#
#            clri += 1
#            from globvars import gp
#            if roi_labels is not None:
#                # main side (body)
#                # get parcel indices of
#                for grpn,parcel_list in gp.parcel_groupings_post.items():
#                    #feat_groups_cur = []
#                    #print(parcel_list)
#                    plws = utils.addSideToParcels(parcel_list, body_side)
#                    parcel_inds = [ roi_labels.index(parcel) for parcel in plws ]
#
#                    pas = '|'.join(map(str,parcel_inds) )
#                    chn_templ = f'msrc(R|L)_9_({pas})_c[0-9]+'
#
#                    if wasH:
#                        ft_templ = f'({"|".join(gv.noband_feat_types) })'
#                        a = [f'^{ft_templ}_{chn_templ}']
#                        feat_groups_all += a
#                        feature_groups_names += [  f'^Hjorth_{grpn}'  ]
#                        clrs += [cmap(clri)] * len(a)
#                        #clri += 1
#
#
#            if chnames_LFP is None:
#                chnames_LFP = ['.*']
#            for lfpchn in chnames_LFP:
#
#
#                clri += 1
#                separate_by_band2 = True
#                # now group per LFPch but with free source
#                chn_templ = 'msrc(R|L)_9_[0-9]+_c[0-9]+'
#                grpn = 'msrc*'
#
#                ft = 'bpcorr'
#                if 'bpcorr' in ftype_info['ftypes']:
#
#                    if separate_by_band2:
#                        fbpairs = ftype_info['fband_pairs']
#                        fbpairs_dispnames = fbpairs
#                    # !! This assume LFP is always in the second place
#                    #if chnames_LFP is not None:
#                    #    for lfpchn in chnames_LFP:
#                    a = [f'^{ft}_{fb1}_{chn_templ},{fb2}_{lfpchn}' for fb1,fb2 in fbpairs]
#                    feat_groups_all += a
#                    feature_groups_names += [f'^{ft}_{fb1}_{grpn},{fb2}_{lfpchn}' for fb1,fb2 in fbpairs_dispnames]
#                    clrs += [cmap(clri)] * len(a)
#                    #else:
#                    #    feat_groups_all += [f'^{ft}_{fb1}_{chn_templ},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
#                    #    feature_groups_names += [f'^{ft}_{fb1}_{grpn},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
#                ft = 'rbcorr'
#                # !! This assume LFP is always in the second place
#                if ft in ftype_info['ftypes']:
#                    if separate_by_band2:
#                        fbsolos = ftype_info['fband_per_ftype'][ft]
#                        fbsolos_dispnames = fbsolos
#                    #if chnames_LFP is not None:
#                    #    for lfpchn in chnames_LFP:
#                    a = [ f'^{ft}_{fb}_{chn_templ},{fb}_{lfpchn}' for fb in fbsolos ]
#                    feat_groups_all += a
#                    feature_groups_names += [ f'^{ft}_{fb}_{grpn},{fb}_{lfpchn}' for fb in fbsolos_dispnames ]
#                    clrs += [cmap(clri)] * len(a)
#                    #else:
#                    #    feat_groups_all += [ f'^{ft}_{fb}_{chn_templ},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
#                    #    feature_groups_names += [ f'^{ft}_{fb}_{grpn},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
#                ft = 'con'
#                # !! This assume LFP is always in the first place
#                if ft in ftype_info['ftypes']:
#                    if separate_by_band2:
#                        fbsolos = ftype_info['fband_per_ftype'][ft]
#                        fbsolos_dispnames = fbsolos
#                    #if chnames_LFP is not None:
#                    #    for lfpchn in chnames_LFP:
#                    a = [ f'^{ft}_{fb}_{lfpchn},{chn_templ}' for fb in fbsolos ]
#                    feat_groups_all += a
#                    feature_groups_names += [ f'^{ft}_{fb}_{lfpchn},{grpn}' for fb in fbsolos_dispnames ]
#                    clrs += [cmap(clri)] * len(a)
#
#                #############################################
#
#                fbpairs = [(bnpattern,bnpattern)]
#                fbsolos = [bnpattern]
#                fbpairs_dispnames = [('*','*')]
#                fbsolos_dispnames = ['*']
#
#                clri += 1
#                if roi_labels is not None:
#                    # main side (body)
#                    # get parcel indices of
#                    for grpn,parcel_list in gp.parcel_groupings_post.items():
#                        #feat_groups_cur = []
#                        #print(parcel_list)
#                        plws = utils.addSideToParcels(parcel_list, body_side)
#                        parcel_inds = [ roi_labels.index(parcel) for parcel in plws ]
#
#                        pas = '|'.join(map(str,parcel_inds) )
#                        chn_templ = f'msrc(R|L)_9_({pas})_c[0-9]+'
#
#                        ft = 'bpcorr'
#                        if 'bpcorr' in ftype_info['ftypes']:
#                            if separate_by_band:
#                                fbpairs = ftype_info['fband_pairs']
#                                fbpairs_dispnames = fbpairs
#                            # !! This assume LFP is always in the second place
#                            #if chnames_LFP is not None:
#                            #    for lfpchn in chnames_LFP:
#                            a = [f'^{ft}_{fb1}_{chn_templ},{fb2}_{lfpchn}' for fb1,fb2 in fbpairs]
#                            feat_groups_all += a
#                            feature_groups_names += [f'^{ft}_{fb1}_{grpn},{fb2}_{lfpchn}' for fb1,fb2 in fbpairs_dispnames]
#                            clrs += [cmap(clri)] * len(a)
#                            #else:
#                            #    feat_groups_all += [f'^{ft}_{fb1}_{chn_templ},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
#                            #    feature_groups_names += [f'^{ft}_{fb1}_{grpn},{fb2}_.*' for fb1,fb2 in ftype_info['fband_pairs']]
#                        ft = 'rbcorr'
#                        # !! This assume LFP is always in the second place
#                        if ft in ftype_info['ftypes']:
#                            if separate_by_band:
#                                fbsolos = ftype_info['fband_per_ftype'][ft]
#                                fbsolos_dispnames = fbsolos
#                            #if chnames_LFP is not None:
#                            #    for lfpchn in chnames_LFP:
#                            a = [ f'^{ft}_{fb}_{chn_templ},{fb}_{lfpchn}' for fb in fbsolos ]
#                            feat_groups_all += a
#                            feature_groups_names += [ f'^{ft}_{fb}_{grpn},{fb}_{lfpchn}' for fb in fbsolos_dispnames ]
#                            clrs += [cmap(clri)] * len(a)
#                            #else:
#                            #    feat_groups_all += [ f'^{ft}_{fb}_{chn_templ},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
#                            #    feature_groups_names += [ f'^{ft}_{fb}_{grpn},.*' for fb in ftype_info['fband_per_ftype'][ft] ]
#                        ft = 'con'
#                        # !! This assume LFP is always in the first place
#                        if ft in ftype_info['ftypes']:
#                            if separate_by_band:
#                                fbsolos = ftype_info['fband_per_ftype'][ft]
#                                fbsolos_dispnames = fbsolos
#                            #if chnames_LFP is not None:
#                            #    for lfpchn in chnames_LFP:
#                            a = [ f'^{ft}_{fb}_{lfpchn},{chn_templ}' for fb in fbsolos ]
#                            feat_groups_all += a
#                            feature_groups_names += [ f'^{ft}_{fb}_{lfpchn},{grpn}' for fb in fbsolos_dispnames ]
#                            clrs += [cmap(clri)] * len(a)
#                        #else:
#                        #    feat_groups_all += [ f'^{ft}_{fb}_.*,{chn_templ}' for fb in ftype_info['fband_per_ftype'][ft] ]
#                        #    feature_groups_names += [ f'^{ft}_{fb}_.*,{grpn}' for fb in ftype_info['fband_per_ftype'][ft] ]
#
#                #display(feat_groups_all)
#
#
#            #display(feat_groups_all)
#            feat_imp_stats = mergeScores(scores_cur, featnames,
#                                         feat_groups_all,
#                                         feature_names_subset=featnames_subset,
#                                         feature_groups_names=feature_groups_names, aux=clrs)
#            ####################################
#
#
#            ####################################
#            subaxs = axs[lblind,:]
#            subaxs[1].set_yticklabels([])
#            #subaxs[2].set_yticklabels([])
#
#            plotFeatImpStats(feat_groups_all, feat_imp_stats, axs= subaxs[:2],
#                             bias=bias, color=colors_fsh[fshi] )
#            subaxs[0].tick_params(axis='y', labelsize=  tickfontsize)
#            #subaxs[0].tick_params(axis='y', labelsize=  tickfontsize)
#            #plt.tight_layout()
#            #ax.set_title(  )
#            #pdf.savefig()
#
#
#            #plt.figure(figsize = (12,10))
#            #ax = plt.gca()
#            ax = subaxs[-1]
#            #postp.plotFeatureImportance(ax, featnames, scores[ptinds,lblind,:], 'XGB_Shapley')
#            sort_individ_feats = fshi == 0
#            plotFeatureImportance(ax, featnames,
#                                        scores_per_class[lblind,:],
#                                        'labind_vs_score', color=colors_fsh[fshi],
#                                        sort = sort_individ_feats, nshow=n_individ_feats_show)
#            ax.set_title( f'{figname_prefix}: ' + ax.get_title() + f'_lblind = {label_str} (lblind={lblind}):  {fsh}' )
#            ax.tick_params(axis="x",direction="in", pad=-300)
#
#
#            #plt.gcf().suptitle(f'{prefix}: lblind = {label_str} (lblind={lblind})')
#            #pdf.savefig()
#            #plt.close()
#
#    plt.tight_layout()
#
#    if suptitle is not None:
#        plt.suptitle(suptitle, fontsize=suptitle_fontsize)
#    if pdf is not None:
#        pdf.savefig()
#        plt.close()




#featsel_per_method, fshs, featnames_list,
#class_labels_good_for_classif,
#class_label_names,

#outputs_grouped   #
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
    from utils import vizGroup2

    labels_dict = rec_info['label_groups_dict'][()]
    srcgroups_dict = rec_info['srcgroups_dict'][()]
    coords = rec_info['coords_Jan_actual'][()]

    if head_subj_ind is None:
        rncur = sind_str + '_off_hold'
    else:
        rncur = head_subj_ind + '_off_hold'
    sources_type=multi_clf_output['info']['sources_type']
    src_file_grouping_ind = multi_clf_output['info']['src_grouping_fn']
    src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
                                                        sources_type,src_file_grouping_ind)
    src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
    sgdn = 'all_raw'
    roi_labels_ = np.array(  labels_dict[sgdn] )
    roi_labels = ['unlabeled'] + list( roi_labels_[parcel_indices_all] )


    clrs =  utils.vizGroup2(sind_str,coords,roi_labels,srcgrp, show=False,
                            alpha=.1, figsize_mult=1.5,msz=30, printLog=0,
                            color_grouping=roi_lab_codes,
                            color_group_labels= color_group_labels,
                            sizes=sizes_list, msz_mult=0.3, seed=seed)

def plotTableInfos2(table_info_per_perf_type, perf_tuple,
                      output_subdir='', alpha_bar = 0.7,
                    use_recalc_perf = True, prefixes_sorted = None,
                    crop_rawname=slice(None,None),
                   sort_by_featnum = 0 ):
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

            #print(clf_type,str_to_put)
            prefixes_wnums += [prefix + f'# {num} : {str_to_put} (min-> {num_red} : {str_to_put_red})']

            #p = np.mean(pvec)
            p     = np.min(pvec)
            p_red = np.min(pvec_red)
            p_add = np.min(pvec_add)
            #ys += [prefinfo[perftype]]
            ys += [p]
            ys_red += [p_red]
            ys_add += [p_add]

            #print(ys_add)

        print( prefixes_wnums )

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
    plt.savefig(pjoin(gv.dir_fig, output_subdir,figfname))


def plotTableInfos_onlyBar(table_info_per_perf_type, perf_tuple,
                      output_subdir='', alpha_bar = 0.7,
                    use_recalc_perf = True, prefixes_sorted = None,
                           prefix2final_name = None,
                    crop_rawname='last',
                   sort_by_featnum = 0 ):
    import matplotlib.pyplot as plt

    info_per_rn_pref = table_info_per_perf_type[perf_tuple]
    rns = list( info_per_rn_pref.values() )
    nrpef = len( rns[0].keys() )

    nr = len(rns)
    nc = 1
    ww = 12; hh = 4 *  nrpef / 20
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

            #print(clf_type,str_to_put)
            if prefix2final_name is not None:
                prefix_like = prefix2final_name[prefix]
            else:
                prefix_like = prefix
            prefixes_wnums += [prefix_like + f'# {num} : {str_to_put}']

            #p = np.mean(pvec)
            p     = np.min(pvec)
            p_red = np.min(pvec_red)
            p_add = np.min(pvec_add)
            #ys += [prefinfo[perftype]]
            ys += [p]
            ys_red += [p_red]
            ys_add += [p_add]

            #print(ys_add)

        print( prefixes_wnums )

        #print(ys_red)
        str_per_pref_per_rowname[rowid_tuple] = str_per_pref

        rowind_bars = 0
        rn = rowid_tuple[0]
        #rncrp = rn[crop_rawname]
        if crop_rawname == 'last':
            rncrp = rn.split('_')[-1]
        else:
            rncrp = rn[crop_rawname]
        #rowid_tuple_to_show = (rncrp ,*rowid_tuple[1:] )
        rowid_tuple_to_show = rncrp.upper()

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
    #figfname = f'Performances_perf_tuple={perf_tuple}_pveclen={pveclen}.pdf'
    #dirfig = pjoin(gv.dir_fig, output_subdir)
    #if not os.path.exists(dirfig):
    #    os.mkdir(dirfig)
    #plt.savefig(pjoin(gv.dir_fig, output_subdir,figfname))


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
    src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
                                                        sources_type,src_file_grouping_ind)
    src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
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

def plotConfmats(outputs_grouped, normalize_mode = 'true', best_LFP=False, common_norm = True,
                 ww=3,hh=3):
    '''
    normalize_mode == true means that we start from real positives (nut just correctly predicted)
    '''
    nc = int( np.ceil( np.sqrt( len(outputs_grouped) ) ) );
    nr = len(outputs_grouped) // nc; #nc= len(scores_stats) - 2;
    #print(nr,nc)
    fig,axs = plt.subplots(nr,nc, figsize = (nc*ww + ww*0.5,nr*hh))#, gridspec_kw={'width_ratios': [1,1,3]} );
    if nr == 1 and nc == 1:
        axs = np.array([[axs]])
    #plt.subplots_adjust(top=1-0.02)
    #normalize_mode = 'total'

    axs = axs.flatten()
    #mn,mn_diag,mn_off_diag = 1e10,1e10,1e10
    #mx_off_diag,mx = -1,-1
    #confmats = []
    confmats_normalized = []
    for axi,(ax,(rn,(spec_key,mult_clf_output) ) ) in enumerate(zip(axs, outputs_grouped.items() ) ):
        #print(axi,k)
        if not best_LFP:
            pcm = mult_clf_output['XGB_analysis_versions']['all_present_features']['perf_dict']
        else:
            chn_LFP = mult_clf_output['best_LFP']['XGB']['winning_chan']
            pcm = mult_clf_output['XGB_analysis_versions'][f'all_present_features_only_{chn_LFP}']['perf_dict']

        reaver_confmats = 1
        if reaver_confmats:
            ps = pcm.get('perfs_CV', None)
            confmats_cur = [p[-1] for p in ps]
            confmats_cur_normalized = [utsne.confmatNormalize(cm,normalize_mode) for cm in confmats_cur]
            confmat_normalized =  np.array(confmats_cur_normalized).mean(axis=0)*100
        else:
            confmat = pcm.get('confmat', None)
            #print(pcm.keys())
            if confmat is None:
                confmat = pcm.get('confmat_aver', None)
            assert confmat is not None
            #confmats += [confmat]
            confmat_normalized = utsne.confmatNormalize(confmat,normalize_mode) * 100
        confmats_normalized += [confmat_normalized]

    #     if normalize_mode == 'total':
    #         confmat_normalized = confmat / np.sum(confmat) * 100
    #     elif normalize_mode == 'col':
    #         confmat_normalized = confmat / np.sum(confmat, axis=1)[None,:] * 100

    confmat_normalized_ = np.array(confmats_normalized)
    confmat_normalized_diags = np.array( [np.diag(np.diag(cm)) for cm in confmats_normalized] )
    eyes = np.array( [np.eye(confmat_normalized_.shape[-1] ) for cm in confmats_normalized], dtype=bool)
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


    import matplotlib as mpl
    if common_norm:
        norm = mpl.colors.Normalize(vmin=0, vmax=100)
    else:
        norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
    for axi,(ax,(rn,(spec_key,mult_clf_output) ) ) in enumerate(zip(axs, outputs_grouped.items() ) ):

        class_label_names              = mult_clf_output.get('class_label_names_ordered',None)
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
        confmat_normalized = confmats_normalized[axi]

        ax = axs[axi]
        ax.set_title(rn)
        pc = ax.pcolor(confmat_normalized, norm=norm)

        rowi,coli = np.unravel_index(axi,(nr,nc))

        xts = ax.get_xticks()
        shift  = (xts[1] - xts[0]) / 2
        xtsd = (shift)  + xts[:-1]
        if rowi == nr-1:
            ax.set_xticks(shift +np.linspace(xts[0],xts[-1],len(class_label_names) ) )
            ax.set_xticklabels( class_label_names,rotation=90)
            ax.set_xlabel('predicted')
        else:
            ax.set_xticks([])

        if coli == 0:
            ax.set_yticks(shift +np.linspace(xts[0],xts[-1],len(class_label_names) ))
            ax.set_yticklabels( class_label_names)
            ax.set_ylabel('true')
        else:
            ax.set_yticks([])

        del confmat_normalized

    plt.subplots_adjust(left = 0.15, bottom=0.26, right=0.75, top=0.9)
    cax = plt.axes([0.80, 0.1, 0.045, 0.8])
    clrb = plt.colorbar(pc, cax=cax)
    cax.set_ylabel(f'percent of _{normalize_mode}_ points (in a CV fold)', labelpad=90 )

    ax2 = clrb.ax.twinx()
    y0,y1 = cax.get_ybound()  # they are from 0 to 1
    ticks       = [  mn_off_diag, mn_diag,  mx_off_diag, me_diag, me_off_diag]
    tick_labels = [ 'min_off_diag', 'min_diag',  'max_off_diag', 'mean_diag', 'mean_off_diag' ]
    if common_norm:
        ticks       += [  mn_off_diag, mn_diag,  mx_off_diag, mx,mn]
        tick_labels += [ 'min_off_diag', 'min_diag',  'max_off_diag', 'max' , 'min' ]
    desarr = np.array( ticks )
    #ax2.set_yticks( desarr/ (y1-y0) )
    ax2.set_yticks( desarr )
    ax2.set_yticklabels( tick_labels )
    ax2.set_ylim( y0,y1)

    print(mn,mx, mn_diag)
    return cax,clrb
    #plt.tight_layout()

def recalcPerfFromCV(perfs_CV,ind):
    #ps = pcm.get('perfs_CV', None)
    ps = perfs_CV
    confmats_cur = [p[-1] for p in ps]
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
    from pathlib import Path
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

def computeImprovementsPerParcelgroup(output_per_raw, mode = 'only',
                                      inv_exclude = True, printLog = False):
     #exclude
    #mode = 'exclude'

    import utils_postprocess as pp
    tpll = pp.multiLevelDict2TupleList(output_per_raw,4,3)
    tpll_reshaped = list( zip(*tpll) )
    len(tpll_reshaped)

    runCID = dict( tpll[0][-1]['cmd'][0] )['--runCID']
    import json
    with open( pjoin(gv.code_dir,'run',f'___run_corresp_{runCID}.txt'), 'r') as f:
        corresp = json.load( f )
    #___run_corresp_16381692938201.txt

    n_chars = len('onlyH_act_')

    #perfs_per_medcond = {'on':[],'off':[]}
    perfs_per_medcond = {'on':{},'off':{}}
    for prefix in corresp:
        ind,pgn,nice_name = corresp[prefix]
        part = prefix[n_chars:]
        if not part.startswith(mode):
            continue

        cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == prefix )[0]
        for cpi in cur_prefix_inds:
            output = tpll[cpi][-1]
            rn = tpll[cpi][0]
            medcond = rn.split('_')[-1]
            r = output['XGB_analysis_versions']['all_present_features']
            perf_cur = r['perf_dict']['perf_aver']
            perf_one_number = min( perf_cur[0], perf_cur[1] )
            perfs_per_medcond[medcond][prefix] = perf_one_number
        #print(prefix,pgn, perf_cur)

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
    for prefix in corresp:
        ind,pgn,nice_name = corresp[prefix]
        part = prefix[n_chars:]
        if not part.startswith(mode):
            continue

        cur_prefix_inds = np.where( np.array(tpll_reshaped[1]) == prefix )[0]
        for cpi in cur_prefix_inds:
            output = tpll[cpi][-1]
            rn = tpll[cpi][0]
            medcond = rn.split('_')[-1]
            if pgn != 'LFP':
                if mode == 'only':
                    assert dict( output['cmd'][0] )['--parcel_group_names'] == pgn
                elif mode == 'exclude':
                    assert dict( output['cmd'][0] )['--parcel_group_names'] == '!'+pgn

            #output[]
            r = output['XGB_analysis_versions']['all_present_features']
            perf_cur = r['perf_dict']['perf_aver']
            perf_one_number = min( perf_cur[0], perf_cur[1] )
            improvement = perf_one_number  - perfs_aver_per_medcond[p]   # prob (range is 0 to 1)
            improvement_wrt_LFP = perf_one_number - perfs_per_medcond[medcond][f'onlyH_act_{mode}15']

            #if mode == 'exclude':
            #    improvement = perf_one_number  - perfs_aver_per_medcond[p]   # prob (range is 0 to 1)

            if inv_exclude and mode == 'exclude':
                improvement = -improvement
                improvement_wrt_LFP =- improvement_wrt_LFP

            impr_per_medcond_per_pgn[medcond][pgn] = improvement * 100  # now in pct
            impr_wrtLFP_per_medcond_per_pgn[medcond][pgn] = improvement_wrt_LFP * 100
            if printLog:
                print(prefix,pgn, medcond,improvement * 100)
    return impr_per_medcond_per_pgn, impr_wrtLFP_per_medcond_per_pgn, perfs_aver_per_medcond

def plotTableInfoBrain(impr_per_medcond_per_pgn , medcond, multi_clf_output, head_subj_ind=None, inv_exclude=True, mode='only',
                       subdir=''):#, perf_tuple):
    from utils import vizGroup2
    from globvars import gp

    import pymatreader

    #rncur = rawnames[0] + '_off_hold'
    #sind_str,mc,tk  = utils.getParamsFromRawname(rncur)
    sind_str = 'S01'
    if head_subj_ind is None:
        rncur = sind_str + '_off_hold'
    else:
        rncur = head_subj_ind + '_off_hold'
    sources_type=multi_clf_output['info']['sources_type']
    src_file_grouping_ind = multi_clf_output['info']['src_grouping_fn']
    src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
                                                        sources_type,src_file_grouping_ind)
    src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)


    labels_dict = rec_info['label_groups_dict'][()]
    srcgroups_dict = rec_info['srcgroups_dict'][()]
    coords = rec_info['coords_Jan_actual'][()]

    if head_subj_ind is None:
        rncur = sind_str + '_off_hold'
    else:
        rncur = head_subj_ind + '_off_hold'
    sources_type=multi_clf_output['info']['sources_type']
    src_file_grouping_ind = multi_clf_output['info']['src_grouping_fn']
    src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
                                                        sources_type,src_file_grouping_ind)
    src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)
    sgdn = 'all_raw'

    roi_labels_ = np.array(  labels_dict[sgdn] )
    parcel_indices_all = np.arange(1,len(roi_labels_))
    roi_labels = ['unlabeled'] + list( roi_labels_[parcel_indices_all] )

    srcgrp = np.zeros( srcgroups_dict[sgdn].shape, dtype=srcgroups_dict[sgdn].dtype)
    for pii,pi in enumerate(parcel_indices_all):
        srcgrp[srcgroups_dict[sgdn] == pi] = pii + 1 #list(roi_labels).index( rls[pii])



    #############################
    brain_area_labels = ['unlabeled'] + list( sorted( gp.parcel_groupings_post.keys() ) )
    intensities = [np.nan] * len(brain_area_labels)
    srcgrp_new = np.nan * np.ones( len(srcgrp) )
    for pgn in impr_per_medcond_per_pgn[medcond]:
        if pgn == 'LFP':
            continue

        parcel_labels = gp.parcel_groupings_post[pgn] #without side information
        if pgn == 'Cerebellum':
            sidestr = '_R'
        else:
            sidestr = '_L'
        parcel_inds = [ roi_labels.index(pl + sidestr) for pl in parcel_labels ]
        #parcel_inds += [ roi_labels.index(pl + '_R') for pl in parcel_labels ]

        ind = brain_area_labels.index(pgn)
        for pi in parcel_inds:
            srcgrp_new[srcgrp==pi]  = ind

        #brain_area_labels += [pgn]

        intensity_cur = impr_per_medcond_per_pgn[medcond][pgn] / 10
        #print(pgn,ind, intensity_cur)
        intensities[ind ]= intensity_cur #cmap(intensity_cur)  #* len(parcel_inds)
    #intensities = np.zeros(len(roi_labels))
    #intensities
    ###########################################
    #%matplotlib inline






    roi_lab_codes = [0] * len(roi_labels)
    color_group_labels = list( gp.parcel_groupings_post.keys()   )

    roi_lab_codes = None
    #color_group_labels = np.arange(len())


    cmap = plt.cm.get_cmap('inferno')

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
    mii,mai = np.min(intensities[gm]), np.max(intensities[gm])
    print(mii,mai)

    #bc = np.ones(4)
    #bc[:3] = 0.5
    #bc = tuple(bc)
    #axs[0].w_xaxis.set_pane_color(bc)

    # axs[1].w_xaxis.set_pane_color(bc)
    #plt.gcf().
    plt.colorbar(scatters['top'])



    impr_lfp = impr_per_medcond_per_pgn[medcond]['LFP']
    if inv_exclude and mode == 'exclude':
        plt.title(f'H_act {mode} areas relative performance -difference / 10, LFP={impr_lfp/10:.2f}')
    else:
        plt.title(f'H_act {mode} areas relative performance difference / 10, LFP={impr_lfp/10:.2f}')

    figname_full = pjoin(gv.dir_fig,subdir,f'brain_map_area_strength_medcond={medcond}_mode={mode}.pdf')
    plt.savefig(figname_full)
    #plt.colorbar();
    #plotTableInfoBrain(impr_per_medcond_per_pgn, output)

def getLogFname(mco,folder = '$OSCBAGDIS_DATAPROC_CODE/slurmout'):
    jobid = dict(mco['cmd'][0] )['--SLURM_job_id']
    fname = f'ML_{jobid}.out'
    folder = os.path.expandvars(folder)
    fname_full = pjoin(folder, fname)
    return fname_full

def copyLogFname(mco, newfname = '_logfile_to_observe.out' ):
    # maybe add filename data or maybe sacct info (inc how much it took to run)
    fname_full = getLogFname(mco)
    import gv
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
