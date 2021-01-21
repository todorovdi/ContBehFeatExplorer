import utils
import utils_tSNE as utsne
import os
from datetime import datetime
import globvars as gv
import numpy as np

def collectPerformanceInfo(rawnames, prefixes, label_tuples, ndays_before = None,
                           n_feats_PCA=None,dim_PCA=None, nraws_used_PCA=None,
                           num_strong_feats_to_show = 3, perf_to_use = 'perfs_XGB',
                           use_CV_perf = True, sources_type = None, printFilenames = False,
                           group_fn = 10, group_ind=0, output_per_raw_= None):
    '''
    red means smallest possible feat set as found by XGB

    label tuples is a list of tuples ( <newname>,<grouping name>,<int group name> )
    '''

    set_explicit_nraws_used_PCA = nraws_used_PCA is not None
    set_explicit_n_feats_PCA = n_feats_PCA is not None
    set_explicit_dim_PCA = dim_PCA is not None

    #assert perf_to_use in [ 'perf', 'perfs_XGB']

    if use_CV_perf:
        iii = -1  # index in the tuple
    else:
        iii = -2

    if set_explicit_nraws_used_PCA:
        regex_nrPCA = str(nraws_used_PCA)
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

    res = {}
    res_strongfeat = {}
    res_red = {}
    time_earliest, time_latest =  np.inf, 0
    output_per_raw = {}
    for rawname_ in rawnames:
        subres = {}
        subres_red = {}
        subres_strongfeat = {}
        subj,medcond,task  = utils.getParamsFromRawname(rawname_)

        #S99_on_move_parcel_aal_grp10-0_test_PCA_nr4_7chs_nfeats1128_pcadim29_skip32_wsz256.npz

        output_per_prefix = {}
        for prefix in prefixes:
            if output_per_raw_ is None:
                lda_output_pg = None
                if sources_type  is not None and len(sources_type) > 0:
                    sources_type_  = sources_type + '_'
                else:
                    sources_type_ = ''
                regex = '{}_{}grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_pcadim({}).*wsz[0-9]+\.npz'.\
                format(rawname_, sources_type_, group_fn, group_ind, prefix,
                    regex_nrPCA, regex_nfeats, regex_pcadim,
                    )
                print(regex)

                # just take latest
                fnfound = utsne.findByPrefix(gv.data_dir, rawname_, prefix, regex=regex)
                if len(fnfound) > 1:
                    time_getter = lambda fnf: os.stat( os.path.join(gv.data_dir,fnf) ).st_mtime
                    fnfound.sort(key=time_getter)

                    for fnf in fnfound:
                        created = os.stat( os.path.join(gv.data_dir,fnf) ).st_ctime
                        dt = datetime.fromtimestamp(created)
                        print( dt.strftime("%d %b %Y %H:%M:%S" ), created )

                    print( 'For {} found not single fnames {}'.format(rawname_,fnfound) )
                    fnfound = [ fnfound[-1] ]

                if len(fnfound) == 1:
                    fnf = fnfound[0]
                    fname_full = os.path.join(gv.data_dir,fnf)
                    created_last = os.stat( fname_full ).st_ctime
                    if printFilenames:
                        dt = datetime.fromtimestamp(created_last)
                        print( dt.strftime("%d %b %Y %H:%M:%S" ), fnf )

                    time_earliest = min(time_earliest, created_last)
                    time_latest   = max(time_latest, created_last)

                    if ndays_before is not None:
                        nh = getFileAge(fname_full)
                        nd = nh / 24
                        if nd > ndays_before:
                            print('  !!! too old ({} days before), skipping {}'.format(nd,fnf) )
                            continue

                    fname_PCA_full = os.path.join(gv.data_dir,fnfound[0] )
                    f = np.load(fname_PCA_full, allow_pickle=1)
                    PCA_info = f['info'][()]

                    lda_output_pg = f['lda_output_pg'][()]
                    lda_output_pg['feature_names_all'] = f['feature_names_all']
                    # saving for output
                    output_per_prefix[prefix] = lda_output_pg
            else:
                a = output_per_raw_.get(rawname_,{})
                lda_output_pg = a.get(prefix,None)
                #print('fddd ',rawname_, prefix, a.keys(), lda_output_pg.keys() )
            if lda_output_pg is None:
                #print('fdfdfdffdffffdsfsdfsd')
                continue

            skipThis = False
            oss = [0]*len(label_tuples)
            # over desired tuples
            for tsi,(lbl,gr,it  ) in enumerate(label_tuples):
                # if not all groups are present, then skip entirely
                if gr not in lda_output_pg:
                    skipThis = True
                    #print('skip ',gr, lda_output_pg.keys() )
                    break
                # saveing desired
                oss[tsi] = lda_output_pg[gr][it]
            #return None,None  # for debug

            if skipThis:
                print('---- Skipping {} _ {} because not all necessary gropings are there'.format(rawname_,prefix) )
                continue
#             o1 = lda_output_pg['merge_nothing']['basic']
#             o2 = lda_output_pg['merge_all_not_trem']['basic']
#             o3 = lda_output_pg['merge_all_not_trem']['trem_vs_quiet']


#             use_red_feat_set = False
#             if use_red_feat_set:
#                 perf_ind = -1
#             else:
#                 perf_ind = 0
            #print('LAPALA ',oss[0].keys() )

            subsubres = {}
            subsubres_red = {}
            # over desired tuples
            for tsi,(lbl,gr,it  ) in enumerate(label_tuples):
                if perf_to_use == 'perfs_XGB':
                    if perf_to_use not in oss[tsi]:
                        skipThis = True
                        break
                    subsubres[lbl] = oss[tsi][perf_to_use][0][iii]
                    subsubres_red[lbl] = oss[tsi][perf_to_use][-1][iii]
                else:
                    lda_anver = oss[tsi]['lda_analysis_versions']
                    if use_CV_perf:
                        k = 'CV_aver'
                    else:
                        k = 'fit_to_all_data'
                    subsubres[lbl] = lda_anver[perf_to_use][k]['perfs']

                if skipThis:
                    print('Skipping ')
                    continue

    #             if perf_to_use == 'perfs_XGB':
    #                 if perf_to_use not in o1:
    #                     continue
    #                 subsubres['allsep']        = o1[perf_to_use][perf_ind][iii]
    #                 subsubres['trem_vs_all']   = o2[perf_to_use][perf_ind][iii]
    #                 subsubres['trem_vs_quiet'] = o3[perf_to_use][perf_ind][iii]
    #             elif perf_to_use == 'perf':
    #                 subsubres['allsep']        = o1[perf_to_use]
    #                 subsubres['trem_vs_all']   = o2[perf_to_use]
    #                 subsubres['trem_vs_quiet'] = o3[perf_to_use]

            #_, best_inds_XGB , perf_nocv, res_aver =   perfs_XGB[perf_ind] o2['perfs_XGB']
                #si = o1['strongest_inds_pc'][0][0]
                feature_names_all = lda_output_pg['feature_names_all']

                subsubres_strongfeat = {}
                for tsi,(lbl,gr,it  ) in enumerate(label_tuples):
                    if perf_to_use == 'perf':
                        subsubres_strongfeat[lbl]        = [oss[tsi]['strongest_inds_pc'][0]]
                    elif perf_to_use == 'perfs_XGB':
                        subsubres_strongfeat[lbl]        = oss[tsi]['strong_inds_XGB'][-num_strong_feats_to_show:]
    #             if perf_to_use == 'perf':
    #                 subsubres_strongfeat['allsep']        = [o1['strongest_inds_pc'][0]]
    #                 subsubres_strongfeat['trem_vs_all']   = [o2['strongest_inds_pc'][0]]
    #                 subsubres_strongfeat['trem_vs_quiet'] = [o3['strongest_inds_pc'][0]]
    #             elif perf_to_use == 'perfs_XGB':
    #                 subsubres_strongfeat['allsep']        = o1['strong_inds_XGB'][-num_strong_feats_to_show:]
    #                 subsubres_strongfeat['trem_vs_all']   = o2['strong_inds_XGB'][-num_strong_feats_to_show:]
    #                 subsubres_strongfeat['trem_vs_quiet'] = o3['strong_inds_XGB'][-num_strong_feats_to_show:]

                    #print(subsubres_strongfeat)

                for ln in subsubres_strongfeat:
                    si = subsubres_strongfeat[ln]
                    str_to_use = ''
                    for sii in si[::-1]:
                        sn = feature_names_all[sii]
                        str_to_use += sn + ';'
                    str_to_use = str_to_use[:-1]
                    subsubres_strongfeat[ln] = str_to_use

                subres[prefix] = subsubres
                subres_red[prefix] = subsubres_red
                subres_strongfeat[prefix] = subsubres_strongfeat
        res[rawname_] = subres
        res_red[rawname_] = subres_red
        res_strongfeat[rawname_] = subres_strongfeat

        output_per_raw[rawname_] = output_per_prefix

    if not np.isinf(time_earliest):
        print('Earliest file {}, latest file {}'.format(
            datetime.fromtimestamp(time_earliest).strftime("%d %b %Y %H:%M:%S" ),
            datetime.fromtimestamp( time_latest).strftime("%d %b %Y %H:%M:%S" ) ) )
    else:
        print('Found nothing :( ')

    feat_counts = {'full':res, 'red':res_red}
    if output_per_raw_ is not None:
        output_per_raw = output_per_raw_
    return feat_counts, output_per_raw


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


def collectPerformanceInfo2(output_per_raw, label_tuples,
                           num_strong_feats_to_show = 3, perf_to_use = 'perfs_XGB',
                           use_CV_perf = True):
    '''
    label tuples is a list of tuples ( <newname>,<grouping name>,<int group name> )
    '''

    set_explicit_nraws_used_PCA = nraws_used_PCA is not None
    set_explicit_n_feats_PCA = n_feats_PCA is not None
    set_explicit_dim_PCA = dim_PCA is not None

    assert perf_to_use in [ 'perf', 'perfs_XGB']

    if use_CV_perf:
        iii = -1  # index in the tuple
    else:
        iii = -2

    if set_explicit_nraws_used_PCA:
        regex_nrPCA = str(nraws_used_PCA)
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

    res = {}
    res_strongfeat = {}
    res_red = {}
    time_earliest, time_latest =  np.inf, 0
    output_per_raw = {}
    for rawname_ in rawnames:
        subres = {}
        subres_red = {}
        subres_strongfeat = {}
        subj,medcond,task  = utils.getParamsFromRawname(rawname_)

        #S99_on_move_parcel_aal_grp10-0_test_PCA_nr4_7chs_nfeats1128_pcadim29_skip32_wsz256.npz

        output_per_prefix = {}
        for prefix in prefixes:
            if sources_type  is not None and len(sources_type) > 0:
                sources_type_  = sources_type + '_'
            else:
                sources_type_ = ''
            regex = '{}_{}grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_pcadim({}).*'.\
            format(rawname_, sources_type_, group_fn, group_ind, prefix,
                   regex_nrPCA, regex_nfeats, regex_pcadim,
                   )
            print(regex)

            # just take latest
            fnfound = utsne.findByPrefix(gv.data_dir, rawname_, prefix, regex=regex)
            if len(fnfound) > 1:
                time_getter = lambda fnf: os.stat( os.path.join(gv.data_dir,fnf) ).st_mtime
                fnfound.sort(key=time_getter)

                for fnf in fnfound:
                    created = os.stat( os.path.join(gv.data_dir,fnf) ).st_ctime
                    dt = datetime.fromtimestamp(created)
                    print( dt.strftime("%d %b %Y %H:%M:%S" ), created )

                print( 'For {} found not single fnames {}'.format(rawname_,fnfound) )
                fnfound = [ fnfound[-1] ]

            if len(fnfound) == 1:
                if printFilenames:
                    fnf = fnfound[0]
                    created = os.stat( os.path.join(gv.data_dir,fnf) ).st_ctime
                    dt = datetime.fromtimestamp(created)
                    print( dt.strftime("%d %b %Y %H:%M:%S" ), fnf )

                created_last = os.stat( os.path.join(gv.data_dir,fnfound[0]) ).st_ctime
                time_earliest = min(time_earliest, created_last)
                time_latest   = max(time_latest, created_last)

                fname_PCA_full = os.path.join(gv.data_dir,fnfound[0] )
                f = np.load(fname_PCA_full, allow_pickle=1)
                PCA_info = f['info'][()]

                lda_output_pg = f['lda_output_pg'][()]
                # saving for output
                output_per_prefix[prefix] = lda_output_pg

                skipThis = False
                oss = [0]*len(label_tuples)
                # over desired tuples
                for tsi,(lbl,gr,it  ) in enumerate(label_tuples):
                    # skipping undesired groups
                    if gr not in lda_output_pg:
                        skipThis = True
                        break
                    # saveing desired
                    oss[tsi] = lda_output_pg[gr][it]

                if skipThis:
                    print('---- Skipping {} _ {} because not all necessary gropings are there'.format(rawname_,prefix) )
                    continue
    #             o1 = lda_output_pg['merge_nothing']['basic']
    #             o2 = lda_output_pg['merge_all_not_trem']['basic']
    #             o3 = lda_output_pg['merge_all_not_trem']['trem_vs_quiet']


    #             use_red_feat_set = False
    #             if use_red_feat_set:
    #                 perf_ind = -1
    #             else:
    #                 perf_ind = 0

                subsubres = {}
                subsubres_red = {}
                # over desired tuples
                for tsi,(lbl,gr,it  ) in enumerate(label_tuples):
                    if perf_to_use == 'perfs_XGB':
                        if perf_to_use not in oss[tsi]:
                            skipThis = True
                            break
                        subsubres[lbl] = oss[tsi][perf_to_use][0][iii]
                        subsubres_red[lbl] = oss[tsi][perf_to_use][-1][iii]
                    else:
                        subsubres[lbl] = oss[tsi][perf_to_use]

                if skipThis:
                    print('Skipping ')
                    continue

    #             if perf_to_use == 'perfs_XGB':
    #                 if perf_to_use not in o1:
    #                     continue
    #                 subsubres['allsep']        = o1[perf_to_use][perf_ind][iii]
    #                 subsubres['trem_vs_all']   = o2[perf_to_use][perf_ind][iii]
    #                 subsubres['trem_vs_quiet'] = o3[perf_to_use][perf_ind][iii]
    #             elif perf_to_use == 'perf':
    #                 subsubres['allsep']        = o1[perf_to_use]
    #                 subsubres['trem_vs_all']   = o2[perf_to_use]
    #                 subsubres['trem_vs_quiet'] = o3[perf_to_use]

            #_, best_inds_XGB , perf_nocv, res_aver =   perfs_XGB[perf_ind] o2['perfs_XGB']
                feature_names_all = f['feature_names_all']
                #si = o1['strongest_inds_pc'][0][0]

                subsubres_strongfeat = {}
                for tsi,(lbl,gr,it  ) in enumerate(label_tuples):
                    if perf_to_use == 'perf':
                        subsubres_strongfeat[lbl]        = [oss[tsi]['strongest_inds_pc'][0]]
                    elif perf_to_use == 'perfs_XGB':
                        subsubres_strongfeat[lbl]        = oss[tsi]['strong_inds_XGB'][-num_strong_feats_to_show:]
    #             if perf_to_use == 'perf':
    #                 subsubres_strongfeat['allsep']        = [o1['strongest_inds_pc'][0]]
    #                 subsubres_strongfeat['trem_vs_all']   = [o2['strongest_inds_pc'][0]]
    #                 subsubres_strongfeat['trem_vs_quiet'] = [o3['strongest_inds_pc'][0]]
    #             elif perf_to_use == 'perfs_XGB':
    #                 subsubres_strongfeat['allsep']        = o1['strong_inds_XGB'][-num_strong_feats_to_show:]
    #                 subsubres_strongfeat['trem_vs_all']   = o2['strong_inds_XGB'][-num_strong_feats_to_show:]
    #                 subsubres_strongfeat['trem_vs_quiet'] = o3['strong_inds_XGB'][-num_strong_feats_to_show:]

                    #print(subsubres_strongfeat)

                for ln in subsubres_strongfeat:
                    si = subsubres_strongfeat[ln]
                    str_to_use = ''
                    for sii in si[::-1]:
                        sn = feature_names_all[sii]
                        str_to_use += sn + ';'
                    str_to_use = str_to_use[:-1]
                    subsubres_strongfeat[ln] = str_to_use

                subres[prefix] = subsubres
                subres_red[prefix] = subsubres_red
                subres_strongfeat[prefix] = subsubres_strongfeat
        res[rawname_] = subres
        res_red[rawname_] = subres_red
        res_strongfeat[rawname_] = subres_strongfeat

        output_per_raw[rawname_] = output_per_prefix

    if not np.isinf(time_earliest):
        print('Earliest file {}, latest file {}'.format(
            datetime.fromtimestamp(time_earliest).strftime("%d %b %Y %H:%M:%S" ),
            datetime.fromtimestamp( time_latest).strftime("%d %b %Y %H:%M:%S" ) ) )
    else:
        print('Found nothing :( ')

    feat_counts = {'full':res, 'red':res_red}
    return feat_counts, output_per_raw
