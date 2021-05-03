import utils
import utils_tSNE as utsne
import os
from datetime import datetime
import globvars as gv
import numpy as np

def collectPerformanceInfo(rawnames, prefixes, ndays_before = None,
                           n_feats_PCA=None,dim_PCA=None, nraws_used=None,
                           sources_type = None, printFilenames = False,
                           group_fn = 10, group_ind=0, subdir = '', old_file_format=False,
                           load_X=False, use_main_LFP_chan=False):
    '''
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
        for prefix_expr in prefixes:
            #if output_per_raw_ is None:
            lda_output_pg = None
        ################
        # TODO: collect all prefixes somehow. Maybe just init with 100 and
        # remember which ones I have found..
        # or do completely differetn cycle
        # or better first get all by prefixed remove old and then of those
        # extract all found prefixes, select oldest within prefix and
        # if I want specific prefixes, only keep them

            if old_file_format:
                prefix_implicit = False
                prefix = prefix_expr
                if sources_type  is not None and len(sources_type) > 0:
                    sources_type_  = sources_type + '_'
                else:
                    sources_type_ = ''

                regex = '{}_{}grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_pcadim({}).*wsz[0-9]+\.npz'.\
                    format(rawname_, sources_type_, group_fn, group_ind, prefix_expr,
                    regex_nrPCA, regex_nfeats, regex_pcadim)
            else:
                if prefix_expr is None or prefix_expr.find('*') >= 0:
                    prefix_implicit = True
                    prefix_expr = '(\w+)_'
                else:
                    prefix_implicit = False
                    if prefix_expr[-1] != '_':
                        prefix = prefix_expr
                        prefix_expr += '_'
                    else:
                        prefix = prefix_expr[:-1]
                #regex = ('_{}_{}grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_' +\
                #    'pcadim({}).*wsz[0-9]+__\(.+,.+\)\.npz').\
                #    format(rawname_[:3], sources_type_, group_fn, group_ind, prefix,
                #    regex_nrPCA, regex_nfeats, regex_pcadim)

                regex =  utils.genMLresFn([ rawname_ ],
                                          sources_type, group_fn, group_ind,
                        prefix_expr, '[0-9]+', regex_nfeats,
                        '[0-9]+', '[0-9]+', '[0-9]+', use_main_LFP_chan,
                                          '(\w+)','(\w+)',
                                          nr=nraws_used, regex_mode=1)

            print('REGEX = ',regex)

            # just take latest
            dir_to_use = os.path.join(gv.data_dir, subdir)
            fnfound, match_infos = utsne.findByPrefix(dir_to_use, rawname_, prefix_expr, regex=regex, ret_aux=1)
            if len(fnfound) > 1:
                time_getter = lambda tpl: os.stat( os.path.join(dir_to_use,tpl[0] ) ).st_mtime
                fm = list( zip(fnfound,match_infos) )
                fm.sort(key=time_getter)

                for fnf,mi in fm:
                    modified = os.stat( os.path.join(dir_to_use,fnf) ).st_mtime
                    dt = datetime.fromtimestamp(modified)
                    print( dt.strftime("%d %b %Y %H:%M" ), modified )

                print( 'For {} found not single fnames (unsorted) {}'.format(rawname_,fnfound) )
                fnfound = [ fm[-1][0] ]
                match_info = fm[-1][-1]
            elif len(fnfound) == 1:
                match_info = match_infos[0]

            #print(match_info, match_info.groups() )

            if len(fnfound) == 1:
                fnf = fnfound[0]
                fname_full = os.path.join(dir_to_use,fnf)
                created_last = os.stat( fname_full ).st_ctime
                if printFilenames:
                    dt = datetime.fromtimestamp(created_last)
                    print( dt.strftime("%d %b %Y %H:%M" ), fnf )

                time_earliest = min(time_earliest, created_last)
                time_latest   = max(time_latest, created_last)

                if ndays_before is not None:
                    nh = getFileAge(fname_full)
                    nd = nh / 24
                    if nd > ndays_before:
                        print('  !!! too old ({} days before), skipping {}'.format(nd,fnf) )
                        continue

                mg = match_info.groups()
                int_grouping = mg[-2]
                intset = mg[-1]
                if prefix_implicit:
                    prefix = mg[-3]

                    print('re deduced ', int_grouping,intset,prefix)

                fname_PCA_full = os.path.join(dir_to_use,fnfound[0] )
                f = np.load(fname_PCA_full, allow_pickle=1)
                #PCA_info = f['info'][()]

                if old_file_format:
                    lda_output_pg = f['lda_output_pg'][()]
                    if 'feature_names_filtered' in f :
                        lda_output_pg['feature_names_filtered'] = f['feature_names_filtered']
                    else:
                        print( list(f.keys() ) )
                        lda_output_pg['feature_names_filtered'] = f['feature_names_all']

                    output_per_prefix[prefix] = lda_output_pg
                else:
                    #lda_output_pg = f['results_cur'][()]
                    #lda_output_pg['feature_names_filtered'] = f['feature_names_filtered_pri'][()][0]

                    res_cur = f['results_cur'][()]


                    if prefix not in output_per_prefix:
                        output_per_prefix[prefix] = {}
                    if int_grouping not in output_per_prefix[prefix]:
                        output_per_prefix[prefix][int_grouping] = {}
                    if intset in output_per_prefix[prefix][int_grouping]:
                        raise ValueError('Already there!!!')
                    else:
                        output_per_prefix[prefix][int_grouping][intset] = res_cur

                    output_per_prefix[prefix]['feature_names_filtered'] = f['feature_names_filtered_pri'][()][0]
                #if save_X:
                #    keys = ['X_imputed','bininds_good']
                #    for k in keys:
                #        lda_output_pg[k] = f[k]

                Ximputed_per_prefix[prefix] = f['X_imputed']
                good_bininds_per_prefix[prefix] =  f['bininds_good']

                # saving for output


        Ximputed_per_raw[rawname_]     =  Ximputed_per_prefix
        good_bininds_per_raw[rawname_] =  good_bininds_per_prefix
            #else:
            #    a = output_per_raw_.get(rawname_,{})
            #    lda_output_pg = a.get(prefix,None)
            #if lda_output_pg is None:
            #    continue

        #    skipThis = False
        #    lda_output_per_lti = [0]*len(label_tuples)
        #    # over desired tuples
        #    for lti,(lbl,gr,it  ) in enumerate(label_tuples):
        #        # if not all groups are present, then skip entirely
        #        if gr not in lda_output_pg:
        #            skipThis = True
        #            #print('skip ',gr, lda_output_pg.keys() )
        #            break
        #        # saveing desired
        #        lda_output_per_lti[lti] = lda_output_pg[gr][it]
        #    #return None,None  # for debug

        #    if skipThis:
        #        print('---- Skipping {} _ {} because not all necessary gropings are there'.format(rawname_,prefix) )
        #        continue

        #    if use_CV_perf:
        #        perfsXGB_subind = -1  # index in the tuple
        #    else:
        #        perfsXGB_subind = -2

        #    #perfs += [ (i,inds, perf_nocv,res_aver)   ]

        #    subsubres = {}
        #    subsubres_red = {}
        #    subsubres_strongfeat = {}
        #    # over desired tuples
        #    for lti,(lbl,gr,it  ) in enumerate(label_tuples):
        #        lda_output_cur = lda_output_per_lti[lti]
        #        if  lda_output_cur is None:
        #            continue
        #        if perf_to_use == 'perfs_XGB':
        #            if perf_to_use not in label_tuples:
        #                continue
        #            subsubres[lbl] = lda_output_cur[perf_to_use][0][perfsXGB_subind]
        #            subsubres_red[lbl] = lda_output_cur[perf_to_use][-1][perfsXGB_subind]
        #        else:
        #            lda_anver = lda_output_cur['lda_analysis_versions']
        #            if use_CV_perf:
        #                k = 'CV_aver'
        #            else:
        #                k = 'fit_to_all_data'
        #            subsubres[lbl] = lda_anver[perf_to_use][k]['perfs']
        #            #try:
        #            #    subsubres_red[lbl] = lda_anver['best_PCA-derived_features_0.6'][k]['perfs']
        #            #except KeyError as e:
        #            #    subsubres_red[lbl] = np.nan,np.nan,np.nan

        #        #_, best_inds_XGB , perf_nocv, res_aver =   perfs_XGB[perf_ind] o2['perfs_XGB']
        #            #si = o1['strongest_inds_pc'][0][0]
        #        feature_names_all = lda_output_pg['feature_names_filtered']

        #    #for lti,(lbl,gr,it  ) in enumerate(label_tuples):
        #    #    lda_output_cur = lda_output_per_lti[lti]
        #    #    if lda_output_cur is None:
        #    #        continue
        #        if perf_to_use == 'perf':
        #            subsubres_strongfeat[lbl]        = [lda_output_cur['strongest_inds_pc'][0]]
        #        elif perf_to_use == 'perfs_XGB':
        #            subsubres_strongfeat[lbl]        = lda_output_cur['strong_inds_XGB'][-num_strong_feats_to_show:]

        #    for ln in subsubres_strongfeat:
        #        si = subsubres_strongfeat[ln]
        #        str_to_use = ''
        #        for sii in si[::-1]:
        #            sn = feature_names_all[sii]
        #            str_to_use += sn + ';'
        #        str_to_use = str_to_use[:-1]
        #        subsubres_strongfeat[ln] = str_to_use

        #    subres[prefix] = subsubres
        #    subres_red[prefix] = subsubres_red
        #    subres_strongfeat[prefix] = subsubres_strongfeat
        #res[rawname_] = subres
        #res_red[rawname_] = subres_red
        #res_strongfeat[rawname_] = subres_strongfeat

        output_per_raw[rawname_] = output_per_prefix

    if not np.isinf(time_earliest):
        print('Earliest file {}, latest file {}'.format(
            datetime.fromtimestamp(time_earliest).strftime("%d %b %Y %H:%M:%S" ),
            datetime.fromtimestamp( time_latest).strftime("%d %b %Y %H:%M:%S" ) ) )
    else:
        print('Found nothing :( ')

    #feat_counts = {'full':res, 'red':res_red}
    #if output_per_raw_ is not None:
    #    output_per_raw = output_per_raw_
    #return feat_counts, output_per_raw

    return output_per_raw,Ximputed_per_raw, good_bininds_per_raw

def collectPerformanceInfo2(rawnames, prefixes, ndays_before = None,
                           n_feats_PCA=None,dim_PCA=None, nraws_used=None,
                           sources_type = None, printFilenames = False,
                           group_fn = 10, group_ind=0, subdir = '', old_file_format=False,
                           load_X=False, use_main_LFP_chan=False):
    '''
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
        prefix_expr = '(\w+)_'
        #regex = ('_{}_{}grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_' +\
        #    'pcadim({}).*wsz[0-9]+__\(.+,.+\)\.npz').\
        #    format(rawname_[:3], sources_type_, group_fn, group_ind, prefix,
        #    regex_nrPCA, regex_nfeats, regex_pcadim)

        regex =  utils.genMLresFn([ rawname_ ],
                                    sources_type, group_fn, group_ind,
                prefix_expr, '[0-9]+', regex_nfeats,
                '[0-9]+', '[0-9]+', '[0-9]+', use_main_LFP_chan,
                                    '(\w+)','(\w+)',
                                    nr=nraws_used, regex_mode=1)

        print('REGEX = ',regex)

        # just take latest
        dir_to_use = os.path.join(gv.data_dir, subdir)
        fnfound, match_infos = utsne.findByPrefix(dir_to_use, rawname_, prefix_expr, regex=regex, ret_aux=1)
        if len(fnfound) == 0:
            print('Nothing found :(((( for REGEX')
            continue


        strs = []
        inds = []
        fn_per_fntype = {}
        for fni in range(len(fnfound) ):
            fnf = fnfound[fni]
            fname_full = os.path.join(dir_to_use,fnf)
            if ndays_before is not None:
                nh = getFileAge(fname_full)
                nd = nh / 24
                if nd > ndays_before:
                    print('  !!! too old ({} days before), skipping {}'.format(nd,fnf) )
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
                print( dt.strftime("%d %b %Y %H:%M" ), fnf )
            time_earliest = min(time_earliest, mod_time)
            time_latest   = max(time_latest, mod_time)

            #s = '{}:{}:{}'.format(prefix,int_grouping,intset)
            s = (prefix,int_grouping,intset)
            if prefixes is not None and prefix not in prefixes:
                print('skipping {} due to bad prefix'.format(fnf) )
                continue

            if s in fn_per_fntype:
                fnf_other,mod_time_other = fn_per_fntype[s]
                if mod_time > mod_time_other:  # if younger than keep it
                    fn_per_fntype[s] = fnf,mod_time
            else:
                fn_per_fntype[s] = fnf,mod_time

        if printFilenames:
            print( 'fn_per_fntype', fn_per_fntype.keys() )
        for s in fn_per_fntype:
            prefix,int_grouping,intset = s
            if prefix not in output_per_prefix:
                output_per_prefix[prefix] = {}

            if int_grouping not in output_per_prefix[prefix]:
                output_per_prefix[prefix][int_grouping] = {}

            if intset in output_per_prefix[prefix][int_grouping]:
                raise ValueError('Already there!!!')

            fnf,mod_time = fn_per_fntype[s]

            fname_full = os.path.join(dir_to_use,fnf )
            f = np.load(fname_full, allow_pickle=1)
            res_cur = f['results_cur'][()]
            res_cur['filename_full'] = fname_full

            if 'featsel_shap_res' in f and 'featsel_shap_res' not in res_cur:
                print('Moving Shapley values')
                res_cur['featsel_shap_res'] = f['featsel_shap_res'][()]

            remove_large_items = 1
            if remove_large_items:
                for lda_anver in res_cur['lda_analysis_versions']:
                    del lda_anver['X_transformed']
                    #del lda_anver['ldaobj']
                if 'Xconcat_good_cur' in res_cur:
                    del res_cur['Xconcat_good_cur']
                del res_cur['transformed_imputed']
                del res_cur['transformed_imputed_CV']

            output_per_prefix[prefix][int_grouping][intset] = res_cur

            output_per_prefix[prefix]['feature_names_filtered'] = f['feature_names_filtered_pri'][()][0]
            ######################

            Ximputed_per_prefix[prefix] = f['X_imputed']
            good_bininds_per_prefix[prefix] =  f['bininds_good']

            del f

        Ximputed_per_raw[rawname_]     =  Ximputed_per_prefix
        good_bininds_per_raw[rawname_] =  good_bininds_per_prefix

        output_per_raw[rawname_] = output_per_prefix

    if not np.isinf(time_earliest):
        print('Earliest file {}, latest file {}'.format(
            datetime.fromtimestamp(time_earliest).strftime("%d %b %Y %H:%M:%S" ),
            datetime.fromtimestamp( time_latest).strftime("%d %b %Y %H:%M:%S" ) ) )
    else:
        print('Found nothing :( ')

    #feat_counts = {'full':res, 'red':res_red}
    #if output_per_raw_ is not None:
    #    output_per_raw = output_per_raw_
    #return feat_counts, output_per_raw

    return output_per_raw,Ximputed_per_raw, good_bininds_per_raw


def getFileAge(fname_full, ret_hours=True):
    created = os.stat( fname_full ).st_ctime
    dt = datetime.fromtimestamp(created)
    today = datetime.today()
    tdelta = (today - dt)
    r = tdelta
    if ret_hours:
        nh = tdelta.seconds / ( 60 * 60 )
        r = nh
    return nh


