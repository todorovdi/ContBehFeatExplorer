import utils
import utils_tSNE as utsne
import os
from datetime import datetime
import globvars as gv
import numpy as np
from os.path import join as pjoin
from time import time
import pandas as pd

def collectPerformanceInfo2(rawnames, prefixes, ndays_before = None,
                           n_feats_PCA=None,dim_PCA=None, nraws_used=None,
                           sources_type = None, printFilenames = False,
                           group_fn = 10, group_ind=0, subdir = '', old_file_format=False,
                           load_X=False, use_main_LFP_chan=False,
                           remove_large_items=1, list_only = False,
                           allow_multi_fn_same_prefix = False, use_light_files=True):
    '''
    rawnames can actually be just subject ids (S01  etc)
    red means smallest possible feat set as found by XGB

    label tuples is a list of tuples ( <newname>,<grouping name>,<int group name> )
    '''

    set_explicit_nraws_used_PCA = nraws_used is not None and isinstance(nraws_used,int)
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

    res = {}
    res_strongfeat = {}
    res_red = {}
    time_earliest, time_latest =  np.inf, 0
    output_per_raw = {}
    for rawname_ in rawnames:
        #subres = {}
        #subres_red = {}
        #subres_strongfeat = {}
        if len(rawname_) > 4:
            subj,medcond,task  = utils.getParamsFromRawname(rawname_)
        else:
            subj = rawname_
            assert len(subj) == 3
            assert subj[0] == 'S'

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
        #    format(rawname_[:3], sources_type_, group_fn, group_ind, prefix,
        #    regex_nrPCA, regex_nfeats, regex_pcadim)

        # we look for files for given rawname
        regex =  utils.genMLresFn([ rawname_ ],
                                    sources_type, group_fn, group_ind,
                prefix_expr, '[0-9]+', regex_nfeats,
                '[0-9]+', '[0-9]+', '[0-9]+', use_main_LFP_chan,
                                    '(\w+)','(\w+)',
                                    nr=nraws_used, regex_mode=1)

        if use_light_files:
            regex = '_!' + regex

        print('REGEX = ',regex)

        # just take latest
        dir_to_use = os.path.join(gv.data_dir, subdir)
        # here only regex param is improtant
        #fnfound, match_infos = utsne.findByPrefix(dir_to_use, rawname_, prefix_expr, regex=regex, ret_aux=1)
        fnfound, match_infos = utsne.findByPrefix(dir_to_use, None, None, regex=regex, ret_aux=1)
        if len(fnfound) == 0:
            print('Nothing found :(((( for REGEX')
            continue


        strs = []
        inds = []
        # dict of (prefix,int_grouping,intset) -> filename,mod_time
        fn_per_fntype = {}
        n_old = 0
        for fni in range(len(fnfound) ):
            fnf = fnfound[fni]
            fname_full = os.path.join(dir_to_use,fnf)
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
            intset = mg[-1]
            int_grouping = mg[-2]
            prefix = mg[-3]

            mod_time = os.stat( fname_full ).st_mtime
            if printFilenames:
                dt = datetime.fromtimestamp(mod_time)
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
            print(f'!!!!!!!!!!   Start loading files for {rawname_}')
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


                    if use_light_files:
                        res_cur = f['results_light'][()]
                    else:
                        res_cur = f['results_cur'][()]
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

                    #if remove_large_items:
                    #    for lda_anver in res_cur['lda_analysis_versions'].values():
                    #        keys_to_clean = ['X_transformed', 'ldaobj', 'ldaobjs']
                    #        for ktc in keys_to_clean:
                    #            if ktc in lda_anver:
                    #                del lda_anver[ktc]
                    #        #del lda_anver['ldaobj']
                    #    if 'Xconcat_good_cur' in res_cur:
                    #        del res_cur['Xconcat_good_cur']
                    #    del res_cur['transformed_imputed']
                    #    del res_cur['transformed_imputed_CV']
                    #    for pt in ['perfs_XGB','perfs_XGB_fs' ]:
                    #        for i in range(len(res_cur[pt]) ):
                    #            if 'clf_obj' in res_cur[pt]:
                    #                del res_cur[pt]['clf_obj']
                    #            #if 'fold_type_shuffled' in res_cur[pt]:
                    #            #    del res_cur[pt]['clf_obj']

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
        else:
            output_per_prefix = None
            Ximputed_per_prefix = None
            good_bininds_per_prefix = None

        if load_X:
            Ximputed_per_raw[rawname_]     =  Ximputed_per_prefix
            good_bininds_per_raw[rawname_] =  good_bininds_per_prefix

        output_per_raw[rawname_] = output_per_prefix

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
    created = os.stat( fname_full ).st_ctime
    dt = datetime.fromtimestamp(created)
    modified = os.stat( fname_full ).st_mtime
    dt = datetime.fromtimestamp(modified)
    today = datetime.today()
    tdelta = (today - dt)
    r = tdelta
    if ret_hours:
        nh = tdelta.total_seconds() / (60 * 60)
        r = nh
    return r

def listRecent(days = 5, hours = None, lookup_dir = None):
    if lookup_dir is None:
        lookup_dir = gv.data_dir
    lf = os.listdir(lookup_dir)
    final_list = []
    for f in lf:
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

def listRecentPrefixes(days = 5, hours = None, lookup_dir = None, light_only=True):
    import re
    lf = listRecent(days, hours, lookup_dir)
    prefixes = []
    for f in lf:
        regex = '_S.._.*grp[0-9\-]+_(.*)_ML'
        if light_only:
            regex = '_!' + regex
        out = re.match(regex, f)
        if out is None:
            continue
        prefix = out.groups()[0]
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
                    perfs_red = None
                    num,num_red,num_red2 = None, None, None
                    if r is None:
                        print(f'  Warning: no prefix {prefix} for {rn}')
                    else:
                        mult_clf_results = r[it_grp][it_set]
                        if mult_clf_results is not None:

                            perfs_CV,perfs_noCV, perfs_red_CV,perfs_red_noCV  = None,None,None,None
                            if clf_type == 'XGB':
                                XGB_anver = mult_clf_results['XGB_analysis_versions']
                                anver_cur = XGB_anver.get(perf_to_use)
                                if anver_cur is not None:
                                    if 'perf_aver' in anver_cur:
                                        perfs_CV   = anver_cur['perf_aver']
                                        perfs_noCV = anver_cur['perf_nocv']
                                    else:
                                        perfs_CV = anver_cur['perf_dict']['perf_aver']
                                        perfs_noCV = anver_cur['perf_dict']['perf_nocv']
                                else:
                                    print(f'perf_to_use (={perf_to_use}): None!')

                                anver_red_cur = XGB_anver.get(perf_red_to_use)
                                if anver_red_cur is not None:
                                    if 'perf_aver' in anver_red_cur:
                                        perfs_red_CV   = anver_red_cur['perf_aver']
                                        perfs_red_noCV = anver_red_cur['perf_nocv']
                                    else:
                                        perfs_red_CV = anver_red_cur['perf_dict']['perf_aver']
                                        perfs_red_noCV = anver_red_cur['perf_dict']['perf_nocv']
                                else:
                                    print(f'perf_to_use (={perf_to_use}): None!')

                                if 'importances' in anver_cur:
                                    num = len(anver_cur['importances'] )
                                else:
                                    if perf_to_use == 'all_present_features':  # due to a small bug in selMinFeatSet
                                        num = len(anver_cur['sortinds'] )
                                    else:
                                        num = len(anver_cur['featinds_present'] )

                                if 'importances' in anver_red_cur:
                                    num_red = len(anver_red_cur['importances'] )
                                else:
                                    num_red = len(anver_red_cur['featinds_present'] )

                                if perf_red_to_use  == 'strongest_features_XGB_opinion':
                                    num_red2 = mult_clf_results['PCA_XGBfeats'].n_components_

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

                                num =  anver_cur['CV_aver']['nfeats']
                                num_red = anver_red_cur['CV_aver']['nfeats']

                                #num = numdict_cur[perf_to_use]
                                #num_red = numdict_cur[perf_red_to_use]

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

                    if num is not None and num_red is not None:
                        assert num >= num_red, f'{rn},{lt},{it_grp},{it_set},{tpl}:{prefix}  {num},{num_red}'

                    info_cur['sens'] = sens
                    info_cur['spec'] = spec
                    info_cur['F1'] = F1
                    info_cur['sens_red'] = sens_red
                    info_cur['spec_red'] = spec_red
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
                          mode='SHAP', explainer = None, nshow = 20):
    assert mode in ['XGB_gain', 'SHAP', 'XGB_Shapley', 'EBM']
    #print('Start plot feat importance')

    aggregate = shap_values
    if mode == 'XGB_Shapley':
        aggregate = np.mean(np.abs(shap_values[:, 0:-1]), axis=0)
        # sort by magnitude
        z = [(x, y) for y, x in sorted(zip(aggregate, feature_names), reverse=True)]
        z = list(zip(*z))
    elif mode == 'XGB_gain':
        z = [(x, y) for y, x in sorted(zip(aggregate, feature_names), reverse=True)]
        z = list(zip(*z))
    elif mode == 'EBM':
        if feature_names is None:
            #feature_names = global_exp['feature_names']
            feature_names = explainer['feature_names']
        if shap_values is None:
            aggregate = explainer._internal_obj['overall']['scores']
        z = [(x, y) for y, x in sorted(zip(aggregate, feature_names), reverse=True)]
        z = list(zip(*z))

    #print('fdfdfd ', len(z[0]), z[0][0] )
    z = ( z[0] [:nshow] , z[1] [:nshow] )


    ax.bar(z[0], z[1])
    #ax.set_xticks(rotation=90)
    ax.tick_params(axis='x', labelrotation=90 )
    #ax.tight_layout()

    ax.set_title(f'Scores of type {mode}');


def mergeScores(scores,feature_names,collect_groups, scores_type='EBM'):
    # pool scores
    assert (isinstance(scores,list) or scores.ndim == 1 )
    mean_scores, std_scores, max_scores = [],[],[]
    # lists of features to be pooled
    inds_lists = []
    names = []
    for cgr in collect_groups:
        inds = utsne.selFeatsRegexInds(feature_names,cgr)
        if len(inds):
            names += [cgr]
            mean_scores +=  [ np.array(scores)[inds].mean()  ]
            std_scores +=   [ np.array(scores)[inds].std()  ]
            max_scores +=   [ np.array(scores)[inds].max()  ]
            inds_lists +=   [inds]

    #print(feature_names[inds_lists[-1] ])
    stats = {}
    stats['names'] = names
    stats['mean'] = mean_scores
    stats['std'] = std_scores
    stats['max'] = max_scores
    stats['inds_lists'] = inds_lists
    #return mean_scores,std_scores,max_scores
    return stats

def plotFeatImpStats(feat_types_all, scores_stats, fign='', axs=None):
    import matplotlib.pyplot as plt
    assert isinstance(scores_stats, dict)
    nr = 1; nc= len(scores_stats) - 2; ww = 3; hh = 5;
    if axs is None:
        fig,axs = plt.subplots(nr,nc, sharey='row', figsize = (nc*ww,nr*hh));
    else:
        assert len(axs) >= nc

    ks = set( scores_stats.keys() ) - set( ['names', 'inds_lists']  )
    ks = sorted(ks)
    names = scores_stats['names']
    for i,k in enumerate(ks):
        #print(k ,  scores_stats[k] )
        ax = axs[i]; ax.set_title(f'{fign} {k}')
        #assert len(feat_types_all) == len(scores_stats[k] ), ( len(feat_types_all), len(scores_stats[k] )  )
        ax.barh(names,np.array(scores_stats[k])); #ax.tick_params(axis='x', labelrotation=90 )


def plotFeatSignifSHAP(pdf,featsel_per_method, fsh, featnames,
                       class_labels_good_for_classif, class_label_names,prefix):
    import utils_postprocess_HPC as postp
    import matplotlib.pyplot as plt

    #fsh = 'XGB_Shapley'
    fspm = featsel_per_method[fsh]
    scores = fspm['scores']
    print( fspm.keys(), scores.shape )

    assert scores.shape[-1] - 1 == len(featnames), (scores.shape[-1] , len(featnames))


    nr = scores.shape[1]; nc = 4; #nc= len(scores_stats) - 2;
    ww = 3 + 5; hh = 8;
    fig,axs = plt.subplots(nr,nc, figsize = (nc*ww,nr*hh), gridspec_kw={'width_ratios': [1,1,1,3]} );

    scores_pre_class = utsne.getScoresPerClass(class_labels_good_for_classif, scores)

    for lblind in range(scores.shape[1] ):
        # select points where true class is like the current one
        #ptinds = np.where(class_labels_good_for_classif == lblind)[0]
        #classid_enconded = lblind
        #scores_cur = np.mean(scores[ptinds,lblind,0:-1], axis=0)

        scores_cur = scores_pre_class[lblind]
        label_str = class_label_names[lblind]

        ###############################
        ftype_info = utils.collectFeatTypeInfo(featnames)

        feat_groups_all = []
        feat_groups_basic = [f'^{ft}_.*' for ft in ftype_info['ftypes']]
        feat_groups_all+= feat_groups_basic

        ft = 'bpcorr'
        if 'bpcorr' in ftype_info['ftypes']:
            feat_groups_two_bands = [f'^{ft}_{fb1}.*,{fb2}.*' for fb1,fb2 in ftype_info['fband_pairs']]
        #     feat_groups_two_bands = ['^bpcorr_gamma.*,tremor.*','^bpcorr_gamma.*,beta.*','^bpcorr_gamma.*,HFO.*',
        #                                 '^bpcorr_beta.*,tremor.*','^bpcorr_beta.*,gamma.*','^bpcorr_beta.*,HFO.*',
        #                                 '^bpcorr_tremor.*,beta.*','^bpcorr_tremor.*,gamma.*','^bpcorr_tremor.*,HFO.*']
            feat_groups_all += feat_groups_two_bands

        for ft in ['rbcorr', 'con']:
            if ft in ftype_info['ftypes']:
                #feat_groups_rbcorr_band = ['^rbcorr_tremor.*', '^rbcorr_beta.*',  '^rbcorr_gamma.*']
                feat_groups_one_band = [ f'^{ft}_{fb}.*' for fb in ftype_info['fband_per_ftype'][ft] ]
                feat_groups_all += feat_groups_one_band
        #feat_groups_all

        feat_imp_stats = postp.mergeScores(scores_cur, featnames, feat_groups_all)
        ####################################


        ####################################
        subaxs = axs[lblind,:]
        subaxs[1].set_yticklabels([])
        subaxs[2].set_yticklabels([])

        postp.plotFeatImpStats(feat_groups_all, feat_imp_stats, axs= subaxs[:3])
        plt.gcf().suptitle(f'{prefix}: lblind = {label_str} (lblind={lblind})')
        #plt.tight_layout()
        #ax.set_title(  )
        #pdf.savefig()


        #plt.figure(figsize = (12,10))
        #ax = plt.gca()
        ax = subaxs[-1]
        #postp.plotFeatureImportance(ax, featnames, scores[ptinds,lblind,:], 'XGB_Shapley')
        postp.plotFeatureImportance(ax, featnames, scores[lblind,:], 'interpret_EBM')
        ax.set_title( f'{prefix}: ' + ax.get_title() + f'_lblind = {label_str} (lblind={lblind})' )
        #plt.tight_layout()
        #pdf.savefig()
        #plt.close()

    plt.tight_layout()
    pdf.savefig()
    plt.close()


def plotTableInfos2(table_info_per_perf_type, perf_tuple,
                      output_subdir=''):
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
    colors = ['blue', 'red', 'purple']
    color_full = colors[0]
    color_red = colors[1]
    color_red2 = colors[2]
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

    for rowname,rowinfo in info_per_rn_pref.items():
        xs, xs_red, xs_red2 = [],[],[]
        ys, ys_red = [],[]
        nums_red = []
        prefixes_sorted = list(sorted(rowinfo.keys()))
        prefixes_wnums = []
        str_per_pref = {}
        for prefix in prefixes_sorted:
            prefinfo = rowinfo[prefix]


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

            pvec = [prefinfo['spec'] , prefinfo['sens'] , prefinfo['F1']]
            pvec_red = [prefinfo['spec_red'] , prefinfo['sens_red'] , prefinfo['F1_red']]
            if pveclen == 3:
                str_to_put_ =  '{:.0f}%,{:.0f}%,{:.0f}%'.format(100*pvec[0],100*pvec[1],100*pvec[2])
                str_to_put_red =  '{:.0f}%,{:.0f}%,{:.0f}%'.format(100*pvec_red[0],100*pvec_red[1],100*pvec_red[2])
            elif pveclen == 2:
                pvec = [pvec[0], pvec[1] ]
                str_to_put_ =  '{:.0f}%,{:.0f}%'.format(100*pvec[0],100*pvec[1])
                str_to_put_red =  '{:.0f}%,{:.0f}%'.format(100*pvec_red[0],100*pvec_red[1])
            else:
                raise ValueError('wrong pveclen')


            str_to_put = str_to_put_
            pvec = np.array(pvec)
            pvec_red = np.array(pvec_red)

            #print(clf_type,str_to_put)
            prefixes_wnums += [prefix + f'# {num} : {str_to_put} (min-> {num_red} : {str_to_put_red})']

            #p = np.mean(pvec)
            p = np.min(pvec)
            p_red = np.min(pvec_red)
            #ys += [prefinfo[perftype]]
            ys += [p]
            ys_red += [p_red]

        str_per_pref_per_rowname[rowname] = str_per_pref


        rowind_scatter_numnum = 0
        rowind_scatter = 1
        rowind_bars = 2


        ax = axs[axind,rowind_scatter_numnum]
        ax.set_title(rowname)

        ax.scatter(xs,xs_red, c = color_full)
        ax.set_xlabel('Number of features, full')
        ax.set_ylabel('Number of features, reduced')
        ax.plot([0, np.max(xs)], [0, np.max(xs)], ls='--')
        ax.set_ylim(0,np.max(xs_red)*1.1 )

        if np.max(xs_red2) > 0 :
            ax.plot([0, np.max(xs)], [0, np.max(xs_red2)], ls='--', c=color_red2)


        #ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--')

        ####################################

        ax = axs[axind,rowind_scatter]
        ax.set_title(rowname)
        ax.scatter(xs,ys, c = color_full)
        ax.scatter(xs_red,ys_red, c = color_red)
        if np.max(xs_red2) > 0 :
            ax.plot([0, np.max(xs_red2)], [0, np.max(ys_red)], ls='--', c=color_red2)
        ax.set_ylabel(perftype)
        ax.set_xlabel('Number of features')
        ax.set_ylim(0,1)
        #ax.set_xlabel('total feature number')
        ####################################
        ax = axs[axind,rowind_bars]
        ax.set_title(rowname)
        ax.yaxis.tick_right()
        sis = np.argsort(xs)
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys)[sis], color = color_full, alpha=0.8)
        ax.barh(np.array(prefixes_wnums)[sis], np.array(ys_red)[sis], color = color_red, alpha=0.8)
        ax.set_xlabel(perftype)
        ax.set_xlim(0,1)
        #ax.tick_params(axs=)

        axind += 1
        #str_per_pref_per_rowname_per_clftype[clf_type] = str_per_pref_per_rowname

        #pvec_summary_per_prefix_per_key[clf_type] = pvec_summary_per_prefix
        #pvec_summary_red_per_prefix_per_key[clf_type] = pvec_summary_red_per_prefix

    plt.suptitle( str( perf_tuple )  )
    plt.tight_layout()
    #keystr = ','.join(keys)
    figfname = f'Performances_perf_tuple={perf_tuple}_pveclen={pveclen}.pdf'
    plt.savefig(pjoin(gv.dir_fig, output_subdir,figfname))

def plotFeatNum2Perf(output_per_raw, perflists, prefixes=None, balance_level = 0.75 ):
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

        ww = 5; hh = 2
        fig,axs = plt.subplots(nr,nc, figsize=(nc*ww,nr*hh))
        plt.subplots_adjust(bottom=0.02, top=1-0.02)
        axs = axs.reshape((nr,nc))

        feat_names_per_perflist = {}

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
        featlist_per_rn={}
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

                    for pli,perflist in enumerate(perflists):
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
                        if tuple(featinds[:5]) == tuple(range(5) ):
                            print(rn,'changing to prev')
                            ind_to_use = -2
                            featinds = cur['featinds_present']
                            cur = multi_clf_output[pt][ind_to_use]
                        print('  ',len(featinds), cur['featinds_present'][:3], cur['perf_aver'] )
                        featnames = multi_clf_output['feature_names_filtered']
                        featlist_per_rn[rn] = featnames[featinds]

                        rng = range(1,len(multi_clf_output[pt]) + ind_to_use +1)
                        sens = [multi_clf_output[pt][ind_]['perf_aver'][0] for ind_ in rng]
                        spec = [multi_clf_output[pt][ind_]['perf_aver'][1] for ind_ in rng]
                        nums = [len(multi_clf_output[pt][ind_]['featinds_present']) for ind_ in rng]

                        # ind in the original array

                        ps = np.minimum( np.array(sens), np.array(spec) )
                        good_perf_inds = np.where( ps >  balance_level)[0]
                        if len(good_perf_inds):
                            ind_balanced = good_perf_inds[0] + 1
                        else:
                            ind_balanced = None
                            print(f'Not reaching balanced level, max sens = {np.max(sens)*100:.2f}% :(')

                        ax = axs[rowind_per_perflist[perflist],pli]
                        ax.plot(nums,sens,label='sens',c='b')
                        ax.plot(nums,spec,label='spec',c='brown')
                        figtitle = f'{rn}--{g}--{it_set}: {perflist}'
                        ax.set_title(figtitle)
                        best_sens = multi_clf_output[pt][0]['perf_aver'][0]
                        ax.axhline(y=best_sens,c='b',ls=':',label=f'best_sens={best_sens *100:.2f}%')
                        if ind_balanced is not None:
                            num_balanced = nums[ind_balanced-1]
                            ax.axvline(x=num_balanced,c='r',ls=':',label=f'num_balanced={num_balanced}')

                            featinds = multi_clf_output[pt][ind_balanced]['featinds_present']#[ind]
                            #feat_names_per_pg_piset[(g,it_set)] =
                            reatnames_res = featnames[featinds]
                        else:
                            reatnames_res = None

                        ax.legend(loc='lower right')
                        ax.set_ylim(-0.01,1.01)
                        #ax2 = plt.gca().twiny()
                        #ax2.set_xticks(rng)

                        fip = multi_clf_output['perfs_XGB'][-1]['featinds_present']
                        fip_fs = multi_clf_output['perfs_XGB_fs'][-1]['featinds_present']
                        intersect_len, symdif_len = len( set(fip) & set(fip_fs) ), len( set(fip) ^ set(fip_fs) )
                        print(f'len(fip)={len(fip)}, len(fip_fs)={len(fip_fs)}, intersect = {intersect_len}, symdif = {symdif_len}')
                        keyname= 'PCA_XGBfeats' #will change in future to PCA_XGBfeats
                        print(f'num PCA of XGB minfeat selected = {multi_clf_output[keyname].n_components_} ' )

                        #f = np.load( multi_clf_output['filename_full'], allow_pickle=True )

                        rowind_per_perflist[perflist] += 1
                        feat_names_per_perflist[perflist] =reatnames_res
                feat_names_per_pg_piset[(g,it_set)] = feat_names_per_perflist

            feat_names_per_raw[rn] = feat_names_per_pg_piset


            #plt.suptitle(f'{prefix}_{pt}')

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

    import matplotlib.pyplot as plt
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


    print( list(rec_info.keys()) )

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

        roi_lab_codes = [0] * len(roi_labels)

        lcr = len('_c0')
        # over ROIs found in feature_names
        for rli,roi_lab in enumerate(roi_labels):
            roi_lab_codes[rli] = []
            for ci in range(len(tuples)):
                c1 = roi_lab == nice_ch1[ci][:-lcr]
                c2 = nice_ch2[ci] is not None
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
