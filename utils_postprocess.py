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

                regex = r'{}_{}grp{}-{}_{}_PCA_nr({})_[0-9]+chs_nfeats({})_pcadim({}).*wsz[0-9]+\.npz'.\
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
                           load_X=False, use_main_LFP_chan=False,
                           remove_large_items=1 ):
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


            if remove_large_items:
                for lda_anver in res_cur['LDA_analysis_versions']:
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
    import os
    from os.path import join as pjoin
    if lookup_dir is None:
        lookup_dir = gv.data_dir
    lf = os.listdir(gv.data_dir)
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

def listRecentPrefixes(days = 5, hours = None, lookup_dir = None):
    import re
    lf = listRecent(days, hours, lookup_dir)
    prefixes = []
    for f in lf:
        out = re.match('_S.._.*grp[0-9\-]+_(.*)_ML', f)
        prefix = out.groups()[0]
        prefixes += [prefix]
    return list(sorted(set(prefixes) ) )


def total_size(o, handlers={}, verbose=False, minRepSz = None, printNotFound = 0):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

        minRepSz -- size in bytes after which we'll print the object

    """
    import sys
    from itertools import chain

    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    #deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o,up=None):
        #if id(o) in seen:       # do not double count the same object
        #    return 0
        seen.add(id(o))
        #s = sys.getsizeof(o, default_size)
        if isinstance(o,np.ndarray):
            s = o.nbytes
        else:
            s = sys.getsizeof(o)

        if verbose:
            print(s, type(o), repr(o)  ) #, file=stderr)

        #import calcResStruct as cRS

        found = 0
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                sz = sum(map(sizeof, handler(o)))
                if minRepSz is not None:
                    if sz > minRepSz: # and not type(up) == cRS.calcResStruct:
                        print(sz,o)
                s += sz
                found = 1
                break
            elif hasattr(o, '__dict__'):
                sz = sum(map( (lambda x: sizeof(x, up=o) ), dict_handler(o.__dict__)))
                s += sz
                found = 1
                break
         #if not printNotFound:
         #    if o is not None and\
         #            not isinstance(o,str) and\
         #            not isinstance(o,np.ndarray) and not isinstance(o,float) \
         #            and not isinstance(o,int):# and not isinstance wrapper_descriptor:
         #        print(type(o),o,up)

        return s

    return sizeof(o)

def extractLightInfo(f):
    res_cur = f['results_cur'][()]

    res_cur['feature_names_filtered'] = f['feature_names_filtered_pri'][()][0]
    res_cur['class_labels_good'] = f['class_labels_good']

    if 'pars' not in res_cur:
        res_cur['pars'] = f['pars'][()]
    elif isinstance(res_cur['pars'], np.array):
        res_cur['pars'] = res_cur['pars'][()]

    return removeLargeItems(res_cur)

def removeLargeItems(res_cur, keep_featsel='all',
                     remove_full_scores=True, verbose=0):
    featsel_methods = list(res_cur['featsel_per_method'] )
    for fsh in featsel_methods:
        if isinstance(keep_featsel,list) and fsh not in keep_featsel:
            del res_cur['featsel_per_method'][fsh]
            continue

        # over feature subset names
        for ffsn in res_cur['featsel_per_method'][fsh]:
            fspm_cur = res_cur['featsel_per_method'][fsh][ffsn]
            if remove_full_scores:
                class_labels_good = res_cur['class_labels_good']
                revdict = res_cur['revdict']
                from sklearn import preprocessing
                lab_enc = preprocessing.LabelEncoder()
                # just skipped class_labels_good
                if isinstance(fspm_cur,dict) and ('scores' in fspm_cur) and ( fsh not in [ 'interpret_EBM', 'interpret_DPEBM' ] ):
                    if 'scores_av' not in fspm_cur:
                        scores = fspm_cur['scores']

                        subskip_fit = round( (len( class_labels_good )  )/ scores.shape[0] )
                        if 'pars' in res_cur:
                            assert int( res_cur['pars']['subskip_fit'] ) == subskip_fit

                        lab_enc.fit(class_labels_good )
                        class_ids = lab_enc.transform(class_labels_good[::subskip_fit])

                        scores_av, bias = utsne.getScoresPerClass(class_ids,scores, ret_bias=1)
                        res_cur['featsel_per_method'][fsh][ffsn]['scores_av'] = scores_av

                        res_cur['featsel_per_method'][fsh][ffsn]['scores_bias_av'] = bias
                    del res_cur['featsel_per_method'][fsh][ffsn]['scores']

            if fsh not in ['interpret_EBM' , 'interpret_DPEBM', 'XGB_Shapley']:
                continue
            if not isinstance(fspm_cur,dict):
                continue
            print(fsh, ffsn, 'fspm_cur.keys() = ', fspm_cur.keys())
            #for ts in ['explainer', 'explainer_loc', 'ebmobj', 'ebm_mergeobj']:

            if 'perf_dict' in fspm_cur.keys():
                if 'clf_objs' in fspm_cur['perf_dict']:
                    del fspm_cur['perf_dict']['clf_objs']
            for ts in ['explainer', 'explainer_loc', 'ebmobj', 'ebm_mergedobj', 'expl_datas']:
                if ts in fspm_cur.keys():
                    del fspm_cur[ts]
                info_per_cp = fspm_cur.get('info_per_cp',None)
                if info_per_cp is not None:
                    for info_cur in info_per_cp.values():
                        if ts in info_cur:
                            del info_cur[ts]

                if fsh in ['interpret_EBM', 'interpret_DPEBM' ] and (ts not in fspm_cur):
                    for fsn,info_cur_ in fspm_cur.items():
                        if isinstance(info_cur_,dict) and ts in info_cur_:
                            del info_cur_[ts]
                        info_per_cp = fspm_cur.get('info_per_cp',None)
                        if info_per_cp is not None:
                            for info_cur in info_per_cp.values():
                                if ts in info_cur:
                                    del info_cur[ts]


    if ('best_inds_XGB_fs' not in res_cur) and 'perfs_XGB_fs' in res_cur:
        res_cur['best_inds_XGB_fs'] =  res_cur['perfs_XGB_fs'][-1]['featinds_present']

    if 'LDA_analysis_versions' in res_cur:
        for lda_anver in res_cur['LDA_analysis_versions'].values():
            keys_to_clean = ['X_transformed', 'ldaobj', 'ldaobjs']
            for subver in lda_anver.values():
                for ktc in keys_to_clean:
                    if ktc in subver:
                        if ktc == 'ldaobj':
                            subver['nfeats'] = len( subver[ktc].scalings_ )
                        elif ktc == 'ldaobjs':
                            subver['nfeats'] = [ len( ldaobj.scalings_ ) for ldaobj in subver[ktc] ]

                        if verbose:
                            print('delted ',ktc)
                        del subver[ktc]
        #del lda_anver['ldaobj']
    if 'Xconcat_good_cur' in res_cur:
        if verbose:
            print('delted Xconcat_good_cur')
        del res_cur['Xconcat_good_cur']
    try:
        del res_cur['transformed_imputed']
        del res_cur['transformed_imputed_CV']


        del res_cur['ldaobj_avCV']
        del res_cur['ldaobjs_CV']
        del res_cur[ 'ldaobj']

    except KeyError as e:
        print('already removed ',e)
    for pt in ['perfs_XGB','perfs_XGB_fs', 'perfs_XGB_fs_boruta' ]:
        vv = res_cur.get(pt,None)
        if vv is not None:
            for i in range(len(vv) ):
                sub = vv[i]
                if 'args' in sub:
                    if 'X' in sub['args']:
                        del sub['args']['X']
                    if 'class_labels' in sub['args']:
                        del sub['args']['class_labels']
                    if 'clf' in sub['args']:
                        del sub['args']['clf']
                if 'clf_obj' in sub:
                    if verbose:
                        print('delted clf_obj')
                    del sub['clf_obj']
                if 'clf_objs' in sub:
                    if verbose:
                        print('delted clf_obj')
                    del sub['clf_objs']

    XGB_anvers = res_cur.get('XGB_analysis_versions',{} )
    for aname,sub in XGB_anvers.items():
        if 'args' in sub:
            if 'X' in sub['args']:
                del sub['args']['X']
            if 'class_labels' in sub['args']:
                del sub['args']['class_labels']
            if 'clf' in sub['args']:
                del sub['args']['clf']
        if 'clf_obj' in sub:
            if verbose:
                print('delted clf_obj')
            del sub['clf_obj']
        if 'clf_objs' in sub:
            if verbose:
                print('delted clf_obj')
            del sub['clf_objs']

    return res_cur

def printSizeInfo(res_cur,depthcur=0,depthleft=0, units=1024**2, minsize=1):
    if isinstance(res_cur, (int,float,str)):
        return
    if not ( isinstance(res_cur, dict) or hasattr( res_cur, '__dict__') ):
        print('ret')
        print( f'{total_size(res_cur) / units:.4f}' )
        return
    if hasattr( res_cur, '__dict__') :
        res_cur = res_cur.__dict__
    sz = 0
    keys = list(res_cur.keys() )
    sz_per_key = [0] * len(keys)
    for ik,kk in enumerate(keys):
        item = res_cur[kk]
        s = total_size(item, minRepSz=None)
        sz_per_key[ik]  = s
        sz += s
        #print(s, kk  )
    indent = ''.join( [' '] * depthcur * 2 )
    sz2 = total_size(res_cur)

    print(f'  Total {indent}{sz} bytes = {sz / units:.4f} Mb' + f' or {indent}{sz2} bytes = {sz2 / units:.4f} Mb')
    #print()

    print(f'{indent}Sorted subparts')
    for k,s in  sorted( zip(keys,sz_per_key), key=lambda x: x[1], reverse=1 ) :
        if s/units >= minsize:
            print(f'{indent}{s/ units:.4f} Mb -- size of {k}' )
    return sz_per_key


from collections.abc import Iterable
def printDict(d,max_depth=3, depth_cur=0,print_leaves = False, indent_nchars=2):
    # tool for exploring dictionaries with high degree of nestedness and large
    # (to print) leaves
    if hasattr(d,'__dict__'):
        d = d.__dict__
    indent = ''.join(' '*depth_cur * indent_nchars)
    if not isinstance(d,Iterable):
        if print_leaves:
            print(indent,d)
        return
    if depth_cur > max_depth:
        return
    if isinstance(d,dict):
        for k,item in d.items():
            s = ''
            if isinstance(item,Iterable):
                s = f'  {len(item)}'
            print(f'{indent}{k}' + s)
            printDict(item,max_depth=max_depth,
                depth_cur=depth_cur+1,print_leaves=print_leaves)
    else:
        for item in d:
            printDict(item,max_depth=max_depth,
                depth_cur=depth_cur+1,print_leaves=print_leaves)

def getStrongCorrelPairs(C,strong_correl_level = 0.7):
    absC = np.abs(C)

    C_nocenter = absC - np.diag(np.diag(absC))
    C_nocenter = np.triu(C_nocenter) # since it is symmetric

    C_flat = C_nocenter.flatten()
    sinds = np.argsort(C_flat)
    hist, bin_edges = np.histogram(C_nocenter.flatten(), bins=20, density=False)
    strong_correl_inds = np.where( C_flat > strong_correl_level )[0]
    tuples = [np.unravel_index(i,C_nocenter.shape) for i in strong_correl_inds]
    return tuples
    #for ind in
    #coords = np.unravel_index(i, C_nocenter.shape)

def getSynonymList(tuples, ret_dict = False):
    d = {}
    for a,b in tuples:
        if a in d:
            d[a] += [b]
        else:
            d[a] = [a,b]
    if ret_dict:
        return d
    else:
        return list(d.values())

def getNotSyn(C_subset,strong_correl_level):

    pairs = getStrongCorrelPairs(C_subset,strong_correl_level)
    print(f'getNotSyn: Num correl pairs {len(pairs)}, it makes {100 * len(pairs)/C_subset.size} % of total pair num')

    synlist = getSynonymList(pairs)

    # indices in fip_fs not in the original array
    allinds = np.arange(C_subset.shape[0])
    inds_to_rem = []
    for syns in synlist:
        inds_to_rem += syns[1:]
    nonsyn_feat_inds = allinds[~np.in1d(allinds,inds_to_rem)]
    return nonsyn_feat_inds

def selBestColumns(M, q_thr):
    # searches for columns that have in at least one row entry larger than q_thr - quantile in this row
    # returns column indices
    assert M.ndim == 2
    col_inds= []
    q = np.quantile(M,q_thr,axis=1)
    #print(q)
    for i in range(M.shape[0] ):
        cur_inds = np.where( M[i] > q[i] )[0]
        col_inds += cur_inds.tolist()
    col_inds = np.unique( col_inds )
    return col_inds


# test
#A = [[0, 0.1, 10, 8, 0, 0, 0],
#[0, 0.1, 0, 8, 0, 10, 0] ]
#A = np.array(A)
#print(A, A.shape)
#
#r = selBestColumns(A,0.8)
#tuple(r) == tuple( np.array([2, 3, 5]) )


def printSynInfo(synlist,indlist,featnames,ftypes_print = ['bpcorr'],minlen_print = 2,
                 minlen_print_feanames = 2):
    from featlist import parseFeatNames

    nums_difs=[]
    lens = []


    for syni,syns in enumerate(synlist):
        indslist_orig = np.array(indlist)[ syns ]
        #print(len(indslist_orig))
        featnames_syns_cur = np.array(featnames)[indslist_orig]
        r = parseFeatNames(featnames_syns_cur)
        ftypes = list( set( r['ftype'] ) )
        if ftypes_print is not None:
            ftype_print_allowed = (ftypes[0] in ftypes_print) and ( len(ftypes) == 1 )
        else:
            ftype_print_allowed = True
        len_cur = len(syns)
        if len_cur >= minlen_print and ftype_print_allowed:
            print(f'index = {syni}, total num = { len_cur} ')
        num_difs = 0
        for k,vals in r.items():
            lcur = len(set(vals))
            if lcur > 1:
                if len_cur >= minlen_print and ftype_print_allowed:
                    print(f'    num different {k:5} = {lcur}')
                num_difs += 1
        if len_cur >= minlen_print and ftype_print_allowed:
            print(f'  num_difs was = {num_difs}')
        nums_difs += [num_difs]
        lens += [len_cur]

        if len_cur >= minlen_print_feanames and ftype_print_allowed:
            print(featnames_syns_cur)


def multiLevelDict2TupleList(d,min_depth=0,max_depth=99999, cur_depth = 0, prefix_sort = None):
    '''
    max_depth  -- depth after which I return dict instead of leafs
    min_dpeht  -- depth before which I don't save leaves
    '''
    #if max_depth < 0:
    #    return []
    r = []
    for k,subd in d.items():
        #print(type(subd),k, f'min_depth={min_depth}, max_depth={max_depth}, cur_depth={cur_depth} ')
        if isinstance(subd,dict):
            if cur_depth < max_depth:
                dl = multiLevelDict2TupleList(subd, min_depth, max_depth, cur_depth + 1 )
                for u in dl:
                    r += [(k, *u) ]
                #print('fd0')
            else:
                r += [(k, subd) ]
                #print('fd1')
        else:
            # putting leafs
            if cur_depth >= min_depth:# and cur_depth <= max_depth + 1:
                r += [(k,subd)]
                #print('leaf')
            continue

    if prefix_sort is not None:
        assert isinstance(prefix_sort, list)
        prefix_ind = 1
        # in case we sort only some prefixes
        all_prefixes = set( [tpl[prefix_ind] for tpl in r] )
        rest_prefixes = set(all_prefixes) - set(prefix_sort)
        rest_prefixes = list(sorted(rest_prefixes))
        prefixes_all_sorted = prefix_sort + rest_prefixes
        r = sorted( r, key = lambda x: prefixes_all_sorted.index( x[1] ) )
        r = list(r)
    return r


def groupOutputs(output_per_raw, prefixes = None, label_groupings=None,
                 int_types_sets=None, filter_by_spec=None, printLog=False):
    '''
    convert multilevel dict with specific possible keys to a list of tuples
    with dictionaries at the end of each tuple
    note that the keys in the returned dict have type = tuple
    '''
    output_sepc_order = ['subjname','prefix','label_grouping','int_types_set']


    output_per_raw_tpll = multiLevelDict2TupleList(output_per_raw,4,3)

    outputs_grouped = {}
    if filter_by_spec is None:
        filter_by_spec = {}
        filter_by_spec['prefix'] = prefixes
        filter_by_spec['label_grouping'] = label_groupings
        filter_by_spec['int_types_set'] = int_types_sets
    else:
        assert prefixes is None
        assert label_groupings is None
        assert int_types_sets is None
    keys = list(sorted( set(output_sepc_order) - set(filter_by_spec.keys()) ))
    for oo in output_per_raw_tpll:
        bad = False
        spec_cur = []
        for spec_name,vals in filter_by_spec.items():
            if vals is None:
                continue
            spec_ind = output_sepc_order.index(spec_name)
            if oo[spec_ind] not in vals:
                bad = True
                continue
            spec_cur += [oo[spec_ind]]
        if not bad:
            kvs = []
            for k in keys:
                spec_ind = output_sepc_order.index(k)
                kvs += [oo[spec_ind]]
            #key = ','.join(kvs)
            key = tuple(kvs)
            if printLog:
                print(oo[:-1], 'turns into ',key,spec_cur)
            if key not in outputs_grouped:
                outputs_grouped[key] = []
            outputs_grouped[key] += [ tuple(spec_cur), oo[-1] ]
    return outputs_grouped


#def getBestLFP_clToMove(best_LFP_dict,subj,metric='balanced_accuracy',
def getBestLFPfromDict(best_LFP_dict,subj,metric='balanced_accuracy',
                        grp = 'merge_nothing', it = 'basic',
                        prefix_type='modLFP_onlyH_act',
                        brain_side='contralat_to_move',
                        disjoint=True, exCB=False, drop_type='only'):
    # disjoint either positive (bool) or negative (=subskip value)
    # return LFP channel name as a string
    import globvars as gb
    mainmoveside = gv.gen_subj_info[subj].get('move_side',None)
    maintremside = gv.gen_subj_info[subj].get('tremor_side',None)
    if brain_side == 'contralat_to_move':
        assert mainmoveside is not None
        side = utils.getOppositeSideStr( mainmoveside )
    elif brain_side in ['left','right','both']:
        side = brain_side
    elif brain_side in ['left_exCB', 'right_exCB', 'left_onlyCB', 'right_onlyCB']:
        side = brain_side.split('_')[0]  # becasue STN is not in Cerebellum
    elif brain_side.startswith('both'):
        side = 'both'
    else:
        raise ValueError(f'wrong side {brain_side}')

    if brain_side.startswith('both'):
        assert not exCB   # just because of runstrings I used to produce best LFP json
    side_det_str = f'brain{side}'
    #movesidelet = mainmoveside_cur[0].upper()
    #contralat_to_move_sidelet = utils.getOppositeSideStr( movesidelet )
    #side_det_str = f'brain{contralat_to_move_side}'
    if exCB:
        side_det_str += '_exCB'
    if disjoint > 0:
        side_det_str += '_disjoint'
    elif disjoint < 0:
        subskip = -disjoint
        subskip2str = {8:'_disjoint',4:'_semidisjoint', 1:'' }
        side_det_str += subskip2str[subskip]

    if brain_side == 'contralat_to_move':
        best_kind = 'best_LFP_contralat_to_move'
    else:
        best_kind = 'best_LFP'

    g = f'{prefix_type}_{side_det_str},{grp},{it}'
    best = best_LFP_dict[subj][g][metric][best_kind]

    if drop_type != 'both':
        best = best[drop_type]

    if side != 'both':
        assert best[3] == side[0].upper(), (best,side)  # otherwise we'll get 0 features
    return best
