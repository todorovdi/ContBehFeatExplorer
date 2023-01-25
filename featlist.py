import numpy as np
import re
import globvars as gv
from collections.abc import Iterable
import utils

def selFeatsRegexInds(names, regexs, unique=1):
    '''
    names
    regexs list
    return indices of names that match at least one of the regexes
    '''
    import re
    if isinstance(regexs,str):
        regexs = [regexs]


    regexs_c = len(regexs) * [0]
    for i,pattern in enumerate(regexs):
        regexs_c[i] = re.compile(pattern)

    inds = []
    for namei,name in enumerate(names):
        #for pattern in regexs:
        #    r = re.match(pattern, name)
        for pattern_c in regexs_c:
            r = pattern_c.match(name)
            if r is not None:
                inds += [namei]
                if unique:
                    break

    return inds

def parseFeatNames(fns,n_jobs=1):
    assert isinstance(fns,Iterable) and not isinstance(fns,str)
    pr = parseFeatName(fns, n_jobs=n_jobs)
    ftype, fb1,fb2,ch1,ch2,mod1,mod2 = [0]*len(fns),\
        [0]*len(fns),[0]*len(fns),[0]*len(fns),\
        [0]*len(fns),[0]*len(fns),[0]*len(fns)

    for (p,ind) in pr:
        #inds  += [ind]
        #ftype += [p['ftype'] ]
        #fb1   += [p['fb1'  ] ]
        #fb2   += [p['fb2'  ] ]
        #ch1   += [p['ch1'  ] ]
        #ch2   += [p['ch2'  ] ]
        #mod1  += [p['mod1' ] ]
        #mod2  += [p['mod2' ] ]
        ftype[ind] = p['ftype']
        fb1  [ind] = p['fb1'  ]
        fb2  [ind] = p['fb2'  ]
        ch1  [ind] = p['ch1'  ]
        ch2  [ind] = p['ch2'  ]
        mod1 [ind] = p['mod1' ]
        mod2 [ind] = p['mod2' ]



    d = {}
    d['ftype'] = ftype
    d['fb1'] = fb1
    d['fb2'] = fb2
    d['ch1'] = ch1
    d['ch2'] = ch2
    d['mod1'] = mod1
    d['mod2'] = mod2
    return d

def getFreqsFromParseRes(parse_result):
    r = parse_result
    relevant_freqs = list( set(r['fb1'] + r['fb2'] ) )
    relevant_freqs = [ rq for rq in relevant_freqs if rq is not None]
    return relevant_freqs


def parseFeatName(fn,  addarg = None, n_jobs=1):

    if isinstance(fn,Iterable) and not isinstance(fn,str):
        from joblib import Parallel, delayed, cpu_count
        if n_jobs  == -1:
            n_jobs = cpu_count()
            n_jobs = min(n_jobs, len(fn) )
        if n_jobs > 1:
            res = Parallel(n_jobs=n_jobs)(delayed(parseFeatName)\
                ( fn[i],i ) for i in range(len(fn) ) )
        else:
            return [ parseFeatName( fn[i],i ) for i in range(len(fn) ) ]
        return res
    d = {}

    Hmode = False
    rH = gv.common_regexs['match_feat_beg_H'].match(fn)
    if rH is None:
        r = gv.common_regexs['match_feat_beg_notH'].match(fn)
        assert r is not None
    else:
        Hmode = True
        r = rH
        ch1 = r.groups()[1]
    ftype = r.groups()[0]

    if ftype in gv.wband_feat_types:
        fbregex = '[a-zA-Z0-9]{3,9}'
        #regex = ftype + f'_({fbregex})_(.*)$'
        #r = re.match(regex,fn)
        #print(regex, fn)

        regex = f'^({fbregex})_(.*)$'
        r = re.match(regex, fn[len(ftype) +1 :]  )
        if r is not None and rH is None:
            fb = r.groups()[0]
            ch1 = r.groups()[1]

    fb1_ = None
    fb2_ = None
    ch2 = None
    mod2 = None

    if ftype in gv.bichan_feat_types:
        if ftype in gv.bichan_bifreq_feat_types:
            #from time import time
            #t0 = time()
            #r = re.match(f'^{ftype}' + '_([a-zA-Z0-9]+)_(.+)*,([a-zA-Z0-9]+)_(.+)$',fn)
            regex_c = gv.common_regexs[ 'match_band_ch_band_ch_beg']
            r = regex_c.match(fn[len(ftype) +1 :])

            #print(time()- t0)

            fb1_ = r.groups()[0]
            ch1 = r.groups()[1]
            fb2_ = r.groups()[2]
            ch2 = r.groups()[3]
        elif ftype == 'con':
            r = re.match(f'^{ftype}_{fb}_' +
                         '(\w+)*,(\w+)$',fn)
            fb1_ = fb
            ch1 = r.groups()[0]
            fb2_ = fb
            ch2 = r.groups()[1]
        else:
            raise ValueError(f'Wrong ! ftype ={ftype} ')

    crop_LFP = 3
    crop_msrc = 4

    if ch1.startswith('LFP'):
        mod1 = ch1[:crop_LFP]
    elif ch1.startswith('msrc'):
        mod1 = ch1[:crop_msrc]
    if ch2 is not None:
        if ch2.startswith('LFP'):
            mod2 = ch2[:crop_LFP]
        elif ch2.startswith('msrc'):
            mod2 = ch2[:crop_msrc]

    d['ftype'] = ftype
    d['fb1'] = fb1_
    d['fb2'] = fb2_
    d['ch1'] = ch1
    d['ch2'] = ch2
    d['mod1'] = mod1
    d['mod2'] = mod2

    if addarg is not None:
        return d,addarg
    else:
        return d

def collectFeatTypeInfo(feature_names, keep_sides=0, ext_info = True,
                        collect_ftype_inds=False):
    import globvars as gv
    ftypes = []
    fbands = []
    fbands_first, fbands_second = [],[]
    fband_pairs = []
    fband_per_ftype = {}
    mod_per_ftype = {}
    tuples = []

    ftype_inds = {}


    bichan_feat_info =  {}
    for ft in gv.bichan_bifreq_feat_types:
        bichan_feat_info[ft] = None
    for bfik in bichan_feat_info:
        bichan_feat_info[bfik] = {'band_LHS':[], 'band_RHS':[] , 'mod_LHS':[],
                                  'mod_RHS':[], 'band_mod_LHS':[],
                                  'band_mod_RHS':[] }
        for k in bichan_feat_info[bfik]:
            bichan_feat_info[bfik][k] = []

    crop_LFP = 3
    crop_msrc = 4
    if keep_sides:
        crop_LFP += 1
        crop_msrc += 1

    bpcorr_left  = []
    bpcorr_right = []
    rbcorr_left  = []
    rbcorr_right = []

    bpcorr_left_mod  = []
    bpcorr_right_mod = []
    rbcorr_left_mod  = []
    rbcorr_right_mod = []
    for fni,fn in enumerate(feature_names):
        Hmode = False
        rH = re.match('(H_[a-z]{1,5})_',fn)
        if rH is None:
            r = re.match('([a-zA-Z0-9]+)_',fn)
            Hmode = True
        else:
            r = rH
        ftype = r.groups()[0]
        ftypes += [ftype]

        if collect_ftype_inds:
            ftype_inds[ftype] = ftype_inds.get(ftype,[] ) + [fni]

        r = re.match(ftype + '_([a-zA-Z0-9]+)_',fn)
        if r is not None and rH is None:
            fb = r.groups()[0]
            fbands += [fb]
            fband_per_ftype[ftype] = fband_per_ftype.get(ftype,[]) + [fb]

        if rH is not None:
            r = re.match(ftype + '_([a-zA-Z0-9_]+)$',fn)
            chn = r.groups()[0]
            if chn.startswith('LFP'):
                mod = chn[:crop_LFP]
            if chn.startswith('msrc'):
                mod = chn[:crop_msrc]
            if ftype not in mod_per_ftype:
                mod_per_ftype[ftype] = [mod]
            else:
                mod_per_ftype[ftype] += [mod]

            if ext_info:
                tuples += [ (ftype,None,mod, None,None ) ]

        if ext_info:
            if ftype in bichan_feat_info:
                r = re.match(ftype + '_([a-zA-Z0-9]+)_.*,([a-zA-Z0-9]+)_',fn)
                fb1 = r.groups()[0]
                fb2 = r.groups()[1]
                fband_pairs += [(fb1,fb2)]
                fbands_first  += [fb1]
                fbands_second += [fb2]

                fband_per_ftype[ftype] = fband_per_ftype.get(ftype,[] ) + [fb2]  # fb1 is aleady there


                r = re.match(ftype + '_([a-zA-Z0-9]+)_(.+),([a-zA-Z0-9]+)_(.+)$',fn)
                fb1_ = r.groups()[0]
                mod1 = r.groups()[1]
                fb2_ = r.groups()[2]
                mod2 = r.groups()[3]
                assert fb1_ == fb1
                assert fb2_ == fb2

                if mod1.startswith('LFP'):
                    mod1 = mod1[:crop_LFP]
                if mod2.startswith('LFP'):
                    mod2 = mod2[:crop_LFP]
                if mod1.startswith('msrc'):
                    mod1 = mod1[:crop_msrc]
                if mod2.startswith('msrc'):
                    mod2 = mod2[:crop_msrc]


                if ftype not in mod_per_ftype:
                    mod_per_ftype[ftype] = [mod1]
                    mod_per_ftype[ftype] = [mod2]
                else:
                    mod_per_ftype[ftype] += [mod1]
                    mod_per_ftype[ftype] += [mod2]

                bichan_feat_info[ftype]['band_LHS']     += [fb1 ]
                bichan_feat_info[ftype]['band_RHS']     += [fb2 ]
                bichan_feat_info[ftype]['mod_LHS']      += [mod1]
                bichan_feat_info[ftype]['mod_RHS']      += [mod2]
                bichan_feat_info[ftype]['band_mod_LHS'] += [(fb1,mod1)]
                bichan_feat_info[ftype]['band_mod_RHS'] += [(fb2,mod2)]

                if ext_info:
                    tuples += [ (ftype, fb1,mod1, fb2,mod2 ) ]

    for k1,v1 in bichan_feat_info.items():
        for k2,v2 in v1.items():
            #print(k1,k2,len(v2) )
            bichan_feat_info[k1][k2] = list( set(v2) )

    for k1,v1 in fband_per_ftype.items():
        fband_per_ftype[k1] = list( sorted( set(v1) ))

    for k1,v1 in mod_per_ftype.items():
        mod_per_ftype[k1] =list( sorted( set(v1) ))



    info = {}
    info['ftypes'] = list(sorted( set(ftypes)) )
    info['ftype_inds'] = ftype_inds
    info['fbands'] = list(sorted( set(fbands + fbands_second)) )
    info['fbands_first'] = list(sorted( set(fbands_first)) )
    info['fbands_second'] = list(sorted(set(fbands_second)) )
    info['fband_pairs']= list(set(fband_pairs))
    info['mod_per_ftype']= mod_per_ftype
    info['fband_per_ftype']= fband_per_ftype

    # remove those that were not found
    for ftype in set( bichan_feat_info.keys() ) - set(ftypes):
        del bichan_feat_info[ftype]
    info['bichan_feat_info'] = bichan_feat_info

    return info

def getBadInds_doubleBandFeats(feature_names_all,
                               fbands_per_mod, fbands_def, printLog=0,
                               parsed_featnames=None):
    if fbands_per_mod is not None and len(fbands_per_mod)  == 2:
        for mod in fbands_per_mod:
            assert len(fbands_per_mod[mod] )

        if parsed_featnames is None:
            print('getBadInds_doubleBandFeats: reparsing!')
            parsed_featnames = parseFeatNames(feature_names_all)

        # I assume that we want first modality to go with first band(s) and
        # second with second. EVEN if order of modalities is different.
        # I just make sure that first bands go with first modality (not
        # modality position)


        data_modalities = list( fbands_per_mod.keys() )
        #mod1 = data_modalities[0]
        #mod2 = data_modalities[1]
        mod1 = data_modalities[0]
        mod2 = data_modalities[1]

        #fbnames_bad1 = set(fbands_def) - set(fbands_per_mod[ mod1 ] )
        #fbnames_bad2 = set(fbands_def) - set(fbands_per_mod[ mod2 ] )

        mask1_first = np.array(parsed_featnames['mod1']) == mod1
        mask2_first = np.array(parsed_featnames['mod1']) == mod2

        mask1_second = np.array(parsed_featnames['mod2']) == mod1
        mask2_second = np.array(parsed_featnames['mod2']) == mod2
        #mask_mod1_somewhere = mask11  | mask21
        #mask_mod2_somewhere = mask12  | mask22

        mask  =  np.zeros(len(feature_names_all), dtype = bool )
        fa1 = np.array(parsed_featnames['fb1'] )
        fa2 = np.array(parsed_featnames['fb2'] )
        for fb1 in fbands_per_mod[mod1]:
            fb1_first   = ( fa1  == fb1 )
            fb1_second  = ( fa2  == fb1 )
            for fb2 in fbands_per_mod[mod2]:
                fb2_first   = ( fa1  == fb2 )
                fb2_second  = ( fa2  == fb2 )
                mask_cur  = ( fb1_first & mask1_first ) & (fb2_second * mask2_second)
                mask_cur  |= ( fb1_second & mask1_second ) & (fb2_first * mask2_first)

                print(f'{mod1}:{fb1},  {mod2}:{fb2}, {sum(mask_cur)}' )

                mask |= mask_cur
        #for fb2 in fbands_per_mod[1]:
        #    for fb1 in fbands_per_mod[0]:
        #        mask_cur  = ( np.array(parsed_featnames['fb1'] ) == fb1 ) & \
        #            ( np.array(parsed_featnames['fb2'] ) == fb2 )
        #        mask &= mask_cur

        #print('Removing bands {mod1}={fbnames_bad1}, {mod2}={fbnames_bad2}')
        #regexs = []
        #ftype_regex = '[a-zA-Z0-9]{1,6}'
        #for fb1 in fbnames_bad1:
        #    for fb2 in fbands_def:
        #        regex = '^' + ftype_regex + f'_{fb1}_{mod1}.+*,{fb2}_{mod2}.+$'
        #        regexs += [regex]
        #for fb1 in fbands_def:
        #    for fb2 in fbnames_bad2:
        #        regex = '^' + ftype_regex + f'_{fb1}_{mod1}.+*,{fb2}_{mod2}.+$'
        #        regexs += [regex]
        #print(sum(mask), len(feature_names_all) )

        inds_bad_ = np.where( np.logical_not(mask) )[0]

        #regexs = []
        #inds_bad_fbands = selFeatsRegexInds(feature_names_all, regexs, unique=1)

        #inds_bad_ = inds_bad_fbands

        if printLog:
            print('fbands_to_use: removing ',
                feature_names_all[ list(inds_bad_) ] )

        return inds_bad_
        #if len(fbands_per_mod[0] ) < len(fbands_def):
    else:
        return []

def filterFeats(feature_names_all, chnames_LFP, LFP_related_only, parcel_types,
                remove_crossLFP,
                cross_couplings_only, self_couplings_only, fbands_to_use,
                features_to_use, fbands_per_mod, feat_types_all,
                data_modalities, data_modalities_all,
                msrc_inds, parcel_group_names,
                roi_labels,srcgrouping_names_sorted, src_file_grouping_ind,
                fbands_def, fband_names_fine_inc_HFO,
                use_lfp_HFO,
                use_main_LFP_chan, mainLFPchan, mainLFPchan_new_name_templ,
                brain_side_to_use, LFP_side_to_use,
                remove_corr_self_couplings=1,
                verbose=0):

    #print(locals())
    from globvars import gp
    import re
    bad_inds = set([] )

    if verbose >= 1:
        print('------- filterStats start -------- ')

    from time import time
    t0 = time()

    print('Start parsing feat names')
    parsed_featnames = parseFeatNames(feature_names_all)

    #t1 = time()
    #dif = t1 - t0;
    #print(f'parseFeatNames took {dif:.3f} secs')

    featnames = np.array(feature_names_all)



    # remove those that don't have LFP in their names i.e. purely MEG features
    if LFP_related_only:
        regexes = ['.*LFP.*']  # LFP to LFP (all bands) and LFP to msrc (all bands)
        inds_good_ = selFeatsRegexInds(feature_names_all,regexes)
        inds_bad_ = set(range(len(feature_names_all))) - set(inds_good_)
        bad_inds.update(inds_bad_)
        if verbose>0:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'LFP_related_only: after removing {len(inds_bad_)} '
                  f'remain {len(remaining) }') )

            if verbose>1:
                print('LFP_related_only: removing ',  featnames[ list( inds_bad_) ] )


            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('LFP_related_only: remaining ',  featnames[ list( remaining_fis) ] )

            #print(utils.collectFeatTypeInfo(featnames[list(remaining)] ) )


    all_parcels= roi_labels[gp.src_grouping_names_order[src_file_grouping_ind]]
    # both and all_available mean different things but in both cases
    # we don't want to remove anything
    if brain_side_to_use not in ['all_available', 'both']:
        assert brain_side_to_use in [ 'left', 'right',
                                     'left_exCB', 'right_exCB', 'both_onlyCB',
                                     'right_onlyCB', 'left_onlyCB']
        if  brain_side_to_use != 'both_onlyCB':
            sidelet = brain_side_to_use[0].upper()
            opsidelet = utils.getOppositeSideStr(sidelet)
        else:
            # in this case it is not really importatn
            sidelet = 'L'
            opsidelet = 'R'
        # this is to prohibit for cross-side side couplings if we computed them
        if brain_side_to_use.endswith('CB'):
            CB_curside = f'Cerebellum_{sidelet}'
            CB_opside = f'Cerebellum_{opsidelet}'
            CBcsi = all_parcels.index(CB_curside)
            CBosi = all_parcels.index(CB_opside)
            #CB_opside = f'Cerebellum_{sidelet}'
            if brain_side_to_use.endswith('exCB'):
                opsrcre1 = '.*msrc'+opsidelet+'_[0-9]+_(?!' + str(CBosi) +  ').*_.*'  # not CB with opsidelet will be prohib
                opsrcre2 = '.*msrc'+sidelet+'_[0-9]+_' + str(CBcsi) + '_.*'    # CB with sidelet will be prohib
                regexes = [opsrcre1, opsrcre2]
            else:
                if brain_side_to_use == 'both_onlyCB':
                    opsrcre1 = '.*msrc'+opsidelet+'_[0-9]+_(?!' + str(CBosi) +  ').*_.*'  # not CB with opsidelet will be prohib
                    opsrcre2 = '.*msrc'+sidelet  +'_[0-9]+_(?!' + str(CBcsi) +  ').*_.*'  # not CB with   sidelet will be prohib
                    regexes = [opsrcre1, opsrcre2]
                elif brain_side_to_use in ['left_onlyCB' ,'right_onlyCB']:
                    opsrcre1 = '.*msrc'+sidelet+'_[0-9]+_(?!' + str(CBosi) +  ').*_.*'  # not CB with   sidelet will be prohib
                    regexes = [opsrcre1]
        else:
            opsrcre = '.*msrc'+opsidelet+'.*'
            regexes = [opsrcre]
        inds_bad_  = selFeatsRegexInds(feature_names_all,regexes)
        bad_inds.update(inds_bad_)

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'brain_side_to_use: after removing {len(inds_bad_)} '
                  f'remain {len(remaining) }') )
            if verbose>1:
                print('brain_side_to_use: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('brain_side_to_use: remaining ',  featnames[ list( remaining_fis) ] )

    # both and all_available mean different things but in both cases
    # we don't want to remove anything
    if LFP_side_to_use not in ['all_available', 'both']:
        assert LFP_side_to_use in [ 'left', 'right' ], LFP_side_to_use
        sidelet = LFP_side_to_use[0].upper()
        opsidelet = utils.getOppositeSideStr(sidelet)
        # this is to prohibit for cross-side side couplings if we computed them
        opLFPre = '.*LFP'+opsidelet+'.*'

        regexes = [opLFPre]
        inds_bad_  = selFeatsRegexInds(feature_names_all,regexes)
        bad_inds.update(inds_bad_)

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'LFP_side_to_use: after removing {len(inds_bad_)} '
                  f'remain {len(remaining) }') )
            if len(remaining) == 0:
                print(f'LFP_side_to_use = {LFP_side_to_use} killed everyone!')
            if verbose>1:
                print('LFP_side_to_use: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('LFP_side_to_use: remaining ',  featnames[ list( remaining_fis) ] )

    # parcel_group_names can take only few values
    # parcel_types is a list of actual parcel names or !parcel_name
    # remove features involving parcels not of the desired type
    if len(parcel_types) == 1 and parcel_types[0] == 'all' and len(parcel_group_names) == 0:
        print('Using all parcels from the file')
    else:
        assert set(roi_labels.keys() ) == set(srcgrouping_names_sorted )

        # sided
        all_parcels= roi_labels[gp.src_grouping_names_order[src_file_grouping_ind]]

        motorlike_parcels = gp.parcel_groupings_post['Sensorimotor']
        #motorlike_parcels = gp.areas_list_aal_my_guess

        bad_parcels = []
        if 'motor-related' in parcel_group_names:
            assert 'not_motor-related' not in parcel_group_names
            # since parcel_types can be wihout side
            # over all parcels
            for pcl1 in all_parcels:
                pcl1_sideless = pcl1[:-2]
                if pcl1_sideless not in motorlike_parcels:
                    bad_parcels += [pcl1_sideless + '_L' ]
                    bad_parcels += [pcl1_sideless + '_R' ]
                # over motor without side
                #for pcl2 in motorlike_parcels:
                #    # if
                #    if pcl1.find(pcl2) < 0:
                #        bad_parcels += [pcl1]
                        #print(pcl2)
            #print('all_parcels=',all_parcels)
        if 'not_motor-related' in parcel_group_names:
            # since parcel_types can be wihout side
            assert 'motor-related' not in parcel_group_names
            for pcl1_sideless in motorlike_parcels:
                bad_parcels += [pcl1_sideless + '_L' ]
                bad_parcels += [pcl1_sideless + '_R' ]
            #for pcl1 in all_parcels:
            #    for pcl2 in motorlike_parcels:
            #        if pcl1.find(pcl2) >= 0:
            #            bad_parcels += [pcl1]

        good_parcels_cur = []
        bad_parcels_cur = []
        for pgn in parcel_group_names:
            if pgn.startswith('!'):
                negmode = True
                pgneff = pgn[1:]
            else:
                negmode = False
                pgneff = pgn

            sided = False
            if pgn.endswith('_L') or pgn.endswith('_R'):
                pgneff = pgn[:-2]
                sided = True
            if pgn.endswith('_B'):
                pgneff = pgn[:-2]
                sided = False

            if pgneff in gp.parcel_groupings_post:
                curlabel_sublist = gp.parcel_groupings_post[pgneff]
                if negmode:
                    for pcl1_sideless in curlabel_sublist:
                        if sided:
                            bad_parcels_cur += [pcl1_sideless + pgn[-2:] ]
                        else:
                            bad_parcels_cur += [pcl1_sideless + '_L' ]
                            bad_parcels_cur += [pcl1_sideless + '_R' ]
                else:
                    for pcl1_sideless in curlabel_sublist:
                        if sided:
                            good_parcels_cur += [pcl1_sideless + pgn[-2:] ]
                        else:
                            good_parcels_cur += [pcl1_sideless + '_L' ]
                            good_parcels_cur += [pcl1_sideless + '_R' ]
                    #for pcl1 in all_parcels:
                    #    pcl1_sideless = pcl1[:-2]
                    #    if pcl1_sideless not in curlabel_sublist:
                    #        bad_parcels += [pcl1_sideless + '_L' ]
                    #        bad_parcels += [pcl1_sideless + '_R' ]
        assert not (len(bad_parcels_cur) and len(good_parcels_cur) )
        if len(good_parcels_cur):
            bad_parcels += list(  set(all_parcels)   - set(good_parcels_cur ) )
        else:
            bad_parcels += bad_parcels_cur

        #print('bad_parcels=',set(bad_parcels) )

        if 'all' in parcel_types:
            good_parcels = all_parcels
            bad_parcels2 = []
        else:
            # we may have a list of parcel names (with or without side) directly
            # since parcel_types can be wihout side
            good_parcels = []
            bad_parcels_add = []
            for pcl in parcel_types:
                if pcl.endswith('_L') or pcl.endswith('_R'):
                    if not pcl.startswith('!'):
                        good_parcels += [pcl]
                    else:
                        bad_parcels_add += [pcl]
                else:
                    if pcl.endswith('_B'):
                        pcl = pcl[:-2]
                    if not pcl.startswith('!'):
                        good_parcels += [pcl + '_L' ]
                        good_parcels += [pcl + '_R' ]
                    else:
                        bad_parcels_add += [pcl[1:] + '_L' ]
                        bad_parcels_add += [pcl[1:] + '_R' ]
            if len(good_parcels):
                bad_parcels2 = ( set(all_parcels) - set(good_parcels) ) | set(bad_parcels_add)
            else:
                bad_parcels2 = set(bad_parcels_add)
        #for pcl1 in all_parcels:
        #    for pcl2 in parcel_types:
        #        if pcl1.find(pcl2) < 0:
        #            bad_parcels2 += [pcl1]
        #import pdb; pdb.set_trace()
        print('size of set(bad_parcels2)= ',len(set(bad_parcels2) ) )

        bad_parcels += bad_parcels2
        #desired_parcels_inds
        #bad_parcel_inds = set( range(len(all_parcels) ) ) -

        temp = set(bad_parcels)
        assert len(temp) > 0, parcel_types
        if verbose:
            print(f'final Parcels to remove {len(temp)} of {len(all_parcels)} :',temp)
        bad_parcel_inds = [i for i, p in enumerate(all_parcels) if p in temp]
        assert len(bad_parcel_inds) > 0, bad_parcels

        final_src_grouping = 9
        regexes = []
        for bpi in bad_parcel_inds:
            regex_parcel_cur = '.*src._[0-9]+_{}_.*'.format( bpi)
            regexes += [regex_parcel_cur]
        inds_bad_parcels = selFeatsRegexInds(feature_names_all,regexes)

        inds_bad_ = inds_bad_parcels
        bad_inds.update(inds_bad_)

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'parcel_group_names: after removing {len(inds_bad_)} '
                  f'remain {len(remaining) }') )
            if verbose>1:
                print('parcel_group_names: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('parcel_group_names: remaining ',  featnames[ list( remaining_fis) ] )

            #print('remaining = ',featnames[list(remaining)] )

        #print('test exit', len(inds_bad_parcels) ); sys.exit(1) ;
    if remove_corr_self_couplings:
    # this essintially means variance (well, I shift it by global mean, not by
    # local, but still) of band-passed signal
        regex_same_LFP = r'.?.?corr.*(LFP.[0-9]+),.*\1.*'
        regex_same_src = r'.?.?corr.*(msrc._[0-9]+_[0-9]+_c[0-9]+),.*\1.*'
        regexs = [regex_same_LFP, regex_same_src]
        inds_self_coupling = selFeatsRegexInds(feature_names_all,regexs)

        inds_bad_ = inds_self_coupling
        bad_inds.update(inds_bad_)

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'remove_corr_self_couplings: after removing {len(inds_bad_)} '
                  f'remain {len(remaining) }') )
            if verbose>1:
                print('remove_corr_self_couplings: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('remove_corr_self_couplings: remaining ',  featnames[ list( remaining_fis) ] )

    # we normally should not even compute those during feat preparation!
    if remove_crossLFP:
        # we want to keep same LFP but remove cross
        regex_same_LFP = r'.*(LFP.[0-9]+),.*\1.*'
        inds_same_LFP = selFeatsRegexInds(feature_names_all,[regex_same_LFP])

        regex_biLFP = r'.*(LFP.[0-9]+),.*(LFP.[0-9]+).*'
        inds_biLFP = selFeatsRegexInds(feature_names_all,[regex_biLFP])

        inds_notsame_LFP = set(inds_biLFP) - set(inds_same_LFP)
        if len(inds_notsame_LFP):

            #same LFP are fine, it is just power
            #print( np.array(feature_names_all)[list(inds_notsame_LFP)] )

            inds_bad_ = inds_notsame_LFP
            bad_inds.update(inds_bad_)

            if verbose:
                remaining = set( range(len(feature_names_all) ) ) - bad_inds
                print(( f'remove_crossLFP: after removing {len(inds_bad_)} '
                    f'remain {len(remaining) }') )
                if verbose>1:
                    print('remove_crossLFP: removing  ',
                          featnames[ list(inds_bad_) ] )

    if cross_couplings_only:
        regex_same_LFP = r'.*(LFP.[0-9]+),.*\1.*'
        regex_same_src = r'.*(msrc._[0-9]+_[0-9]+_c[0-9]+),.*\1.*'
        regex_same_Hs = [ r'H_act.*', r'H_mob.*', r'H_compl.*'   ]
        regexs = [regex_same_LFP, regex_same_src] + regex_same_Hs
        inds_self_coupling = selFeatsRegexInds(feature_names_all,regexs)


        if len(inds_bad_):
            inds_bad_ = inds_self_coupling
            #print('Removing self-couplings of LFP and msrc {}'.format( inds_self_coupling) )
            bad_inds.update(inds_bad_ )

            if verbose:
                remaining = set( range(len(feature_names_all) ) ) - bad_inds
                print(( f'cross_couplings_only: after removing {len(inds_bad_)} '
                    f'remain {len(remaining) }') )
                if verbose > 1:
                    print('cross_couplings_only: removing ',
                        featnames[ list(inds_bad_) ] )

                if verbose>2:
                    remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                    remaining_fis = list( sorted(remaining_fis) )
                    print('cross_couplings_only: remaining ',  featnames[ list( remaining_fis) ] )


    if self_couplings_only:
        regex_same_LFP = r'.*(LFP.[0-9]+),.*\1.*'
        regex_same_src = r'.*(msrc._[0-9]+_[0-9]+_c[0-9]+),.*\1.*'
        regex_same_Hs = [ r'H_act.*', r'H_mob.*', r'H_compl.*'   ]
        regexs = [regex_same_LFP, regex_same_src] + regex_same_Hs
        inds_self_coupling = selFeatsRegexInds(feature_names_all,regexs)

        if len(inds_self_coupling):
            inds_non_self_coupling = set(range(len(feature_names_all) )) - set(inds_self_coupling)

            inds_bad_ = inds_non_self_coupling
            bad_inds.update( inds_bad_ )

            if verbose:
                remaining = set( range(len(feature_names_all) ) ) - bad_inds
                print(( f'self_couplings_only: after removing {len(inds_bad_)} '
                    f'remain {len(remaining) }') )
                if verbose > 1:
                    print('self_couplings_only: removing ',
                        featnames[ list(inds_bad_) ] )

                if verbose>2:
                    remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                    remaining_fis = list( sorted(remaining_fis) )
                    print('self_couplings_only: remaining ',  featnames[ list( remaining_fis) ] )

    #if len(fbands_to_use) < len(fband_names_fine_inc_HFO):
    if len(fbands_to_use) < len(fbands_def):
        fbnames_bad = set(fbands_def) - set(fbands_to_use)
        print('Removing bands ',fbnames_bad)
        regexs = []
        for fbname in fbnames_bad:
            regexs += [ '.*{}.*'.format(fbname)  ]
        inds_bad_fbands = selFeatsRegexInds(feature_names_all, regexs, unique=1)

        inds_bad_ = inds_bad_fbands
        bad_inds.update( inds_bad_ )

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'fbands_to_use: after removing {len(inds_bad_)} '
                f'remain {len(remaining) }') )
            if verbose > 1:
                print('fbands_to_use: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - inds_bad_
                remaining_fis = list( sorted(remaining_fis) )
                print('fbands_to_use: remaining ',  featnames[ list( remaining_fis) ] )

    if fbands_per_mod is not None and len(fbands_per_mod) and len(fbands_per_mod ) ==2 :
        inds_bad_ = getBadInds_doubleBandFeats(feature_names_all, \
            fbands_per_mod, fbands_def, \
            parsed_featnames=parsed_featnames)
        bad_inds.update( inds_bad_ )

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'fbands_per_mod: after removing {len(inds_bad_)} '
                f'remain {len(remaining) }') )

        if verbose>2:
            remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
            remaining_fis = list( sorted(remaining_fis) )
            print('fbands_per_mod: remaining ',  featnames[ list( remaining_fis) ] )

    # here 'bad' means nothing essentially, just something I want to remove
    if set(feat_types_all) != set(features_to_use):
        badfeats = set(feat_types_all) - set(features_to_use)
        print('Removing features ',badfeats)
        regexs = [ '{}.*'.format(feat_name) for feat_name in  badfeats]
        inds_badfeats = selFeatsRegexInds(feature_names_all, regexs, unique=1)

        inds_bad_ = inds_badfeats
        bad_inds.update( inds_bad_ )

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'features_to_use: after removing {len(inds_bad_)} '
                f'remain {len(remaining) }') )
            if verbose > 1:
                print('features_to_use: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('features_to_use: remaining ',  featnames[ list( remaining_fis) ] )

    if set(data_modalities_all) != set(data_modalities):
        badmod = list( set(data_modalities_all) - set(data_modalities) )
        print('Removing modalities ',badmod)
        assert len(badmod) == 1
        badmod = badmod[0]
        regexs = [ '.*{}.*'.format(badmod) ]
        inds_badmod = selFeatsRegexInds(feature_names_all, regexs, unique=1)

        inds_bad_ = inds_badmod
        bad_inds.update( inds_bad_ )

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'data_modalities: after removing {len(inds_bad_)} '
                f'remain {len(remaining) }') )
            if verbose > 1:
                print('data_modalities: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('data_modalities: remaining ',  featnames[ list( remaining_fis) ] )

    if use_main_LFP_chan:
        # take mainLFPchan, extract <side><number>,  select names where there
        # is LFP with other <side><number>
        # similar with msrc_inds

        #getFeatIndsRelToLFPchan(feature_names_all, chnpart=mainLFPchan,
        #                        verbose=False,

        chnames_bad_LFP = set(chnames_LFP) - set([mainLFPchan] )

        regexs = [ '.*{}.*'.format(chname) for chname in  chnames_bad_LFP]
        inds_bad_LFP = selFeatsRegexInds(feature_names_all, regexs, unique=1)

        if mainLFPchan_new_name_templ is not None:
            # replacing feature names in_place
            regexs = [ '.*{}.*'.format(mainLFPchan) ]
            inds_mainLFPchan_rel = selFeatsRegexInds(feature_names_all, regexs, unique=1)
            # TODO: if I reverse side forcefully, it should be done differently
            mainLFPchan_sidelet = mainLFPchan[3]
            assert mainLFPchan_sidelet in ['R', 'L']
            for ind in inds_mainLFPchan_rel:
                s = feature_names_all[ind].replace(mainLFPchan,
                    mainLFPchan_new_name_templ.format(mainLFPchan_sidelet) )
                feature_names_all[ind] = s

        print('Removing non-main LFPs ',chnames_bad_LFP)

        inds_bad_ = inds_bad_LFP
        bad_inds.update( inds_bad_ )

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'use_main_LFP_chan: after removing {len(inds_bad_)} '
                f'remain {len(remaining) }') )
            if verbose > 1:
                print('use_main_LFP_chan: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('use_main_LFP_chan: remaining ',  featnames[ list( remaining_fis) ] )

    # removing HFO-related feats if needed
    if not use_lfp_HFO:
        regexs = [ '.*HFO.*' ]
        inds_HFO = selFeatsRegexInds(feature_names_all, regexs, unique=1)

        inds_bad_ = inds_HFO
        bad_inds.update( inds_bad_ )

        if verbose:
            remaining = set( range(len(feature_names_all) ) ) - bad_inds
            print(( f'use_lfp_HFO: after removing {len(inds_bad_)} '
                f'remain {len(remaining) }') )
            if verbose > 1:
                print('use_lfp_HFO: removing ',
                    featnames[ list(inds_bad_) ] )

            if verbose>2:
                remaining_fis = set( range(len(feature_names_all) ) ) - bad_inds
                remaining_fis = list( sorted(remaining_fis) )
                print('use_lfp_HFO: remaining ',  featnames[ list( remaining_fis) ] )

    ############# for HisrchPt2011 only
    # collecting indices of all msrc that we have used
    regex = 'msrc._([0-9]+)'
    res = []
    for fn in feature_names_all:
        r = re.findall(regex,fn)
        res += r
        #print(r)
    #     if r is not None:
    #         print(fn, r.groups() )
    res = list( map(int,res) )
    res = np.unique(res)

    msrc_inds_undesired = np.setdiff1d( msrc_inds, res)
    if len(msrc_inds_undesired):
        #chnames_bad_src =

        regexs = [ '.*msrc.{}.*'.format(ind) for ind in msrc_inds_undesired]
        inds_bad_src = selFeatsRegexInds(feature_names_all, regexs, unique=1)
        bad_inds.update(inds_bad_src)

        if verbose:
            print(  featnames[ inds_bad_src ] )
    #################### end


    # keeping good features
    print('Removing {} features out of {}'.format( len(bad_inds) , len(feature_names_all) ) )
    selected_feat_inds = set( range(len(feature_names_all) ) ) - bad_inds
    selected_feat_inds = list( sorted(selected_feat_inds) )

    return selected_feat_inds

def genFeatnamesGroupCodes():
    return

def getFeatIndsRelToOnlyOneLFPchan(featnames, chn,
                            new_channel_name_templ = None, chnames_LFP=None,
                                  mainLFPchan = None ):
    '''
    replaces in-place
    keeps non-LFP-related channels as remaining (does not remove them)
    '''
    if chnames_LFP is None:
        chnames_LFP = getChnamesFromFeatlist(featnames, mod='LFP')
    chnames_bad_LFP = set(chnames_LFP) - set([chn] )

    #select features where appear some LFPs
    regexs = [ '.*{}.*'.format('LFP') for chname in  chnames_bad_LFP]
    inds_any_LFP = selFeatsRegexInds(featnames, regexs, unique=1)
    inds_no_LFP = set( range(len(featnames) ) ) - set(inds_any_LFP)

    #select features where appear other LFPs
    regexs = [ '.*{}.*'.format(chname) for chname in  chnames_bad_LFP]
    inds_bad_LFP = selFeatsRegexInds(featnames, regexs, unique=1)

    # TODO: if I reverse side forcefully, it should be done differently
    if new_channel_name_templ is not None:
        regexs = [ '.*{}.*'.format(mainLFPchan) ]
        inds_mainLFPchan_rel = selFeatsRegexInds(featnames, regexs, unique=1)
        mainLFPchan_sidelet = mainLFPchan[3]
        assert mainLFPchan_sidelet in ['R', 'L']
        for ind in inds_mainLFPchan_rel:
            s = featnames[ind].replace(mainLFPchan,
                new_channel_name_templ.format(mainLFPchan_sidelet) )
            featnames[ind] = s

    #print('Removing non-main LFPs ',chnames_bad_LFP)


    remaining = set( range(len(featnames) ) ) - set(inds_bad_LFP)
    remaining = list(sorted(remaining) )

    inds_bad_LFP = set(inds_bad_LFP) | set(inds_no_LFP)
    inds_bad_LFP = list(sorted(inds_bad_LFP ) )

    #assert len(remaining) < len(featnames)
    return remaining, inds_bad_LFP


def replaceMEGsrcChnamesParams(chns,old_group=None,new_group=None,old_comp=None,new_comp=None):
    import re
    assert not ((old_group is None) and (new_group is None) and (old_comp is None) and (new_comp is None))
    newchns = []
    sides,groups,pcs,comps = utils.parseMEGsrcChnamesShortList(chns)
    for i,chn in enumerate(chns):
        if not np.isnan(groups[i] ):
            g = groups[i]
            c = comps[i]
            if re.match( str(old_group) , str( g ) ):
                g = new_group
            if re.match( str(old_comp) , str( c ) ):
                c = new_comp
            newchn = f'msrc{sides[i]}_{g}_{pcs[i]}_c{c}'
        else:
            newchn = chn

        newchns += [newchn]
    return newchns

# select feature names related to current sanity check
def selectFeatNames(fts,relevant_freqs,desired_chns, feature_names_all,
                    fband_names = None, bands_acc = 'crude'):
    featnames_parse_res = parseFeatNames(feature_names_all)
#     for loc,v in locals().items():
#         if len(v) < 40:
#             print(loc,v)

#     if bands_acc == 'crude':
#         fbl = gv.fband_names_crude
#     else:
#         fbl = gv.fband_names_fine
#     desired_fbands = []
#     for rf in relevant_freqs:
#         for fbname in fbl:
#             fb = gv.fbands[fbname]
#             t = (rf >= fb[0]) and (rf <= fb[1])
#             if t:
#                 desired_fbands += [fbname]
    if relevant_freqs is not None:
        desired_fbands = utils.freqs2relevantBands(relevant_freqs,fband_names,bands_acc)
    else:
        desired_fbands = None
    #print(desired_fbands)

    fis = []
    for fi in range(len(feature_names_all)):
    #     if feature_names_all[fi] != 'con_tremor_LFPR092,msrcR_9_3_c0':
    #         continue
        if fts is not None:
            cond_ft = featnames_parse_res['ftype'][fi] in fts
        else:
            cond_ft = True

        chns_cur = [ featnames_parse_res['ch1'][fi] , featnames_parse_res['ch2'][fi] ]
        cond_chns = True
    #     for chn in desired_chns:
    #         cond_chns &= chn in chns_cur
    #         print(chns_cur   cond_chns,chn)
        desired_chns_processed = replaceMEGsrcChnamesParams(desired_chns, 0,9, '.*', 0)
        cond_chns = set(chns_cur) <= set(desired_chns_processed)
        #print(set(chns_cur), set(desired_chns_processed) )

        cond_bands = True


        fbs_cur = [ featnames_parse_res['fb1'][fi] , featnames_parse_res['fb2'][fi] ]



#         click = 0
#     #     if tuple(fbs_cur) == ('tremor','gamma'):
#     #         print('  fd')
#         for fb in fbs_cur:
#             for rf in relevant_freqs:
#                 if (fb is not None):
#                     t = (rf >= gv.fbands[fb][0]) and (rf <= gv.fbands[fb][1])
#                     click += int(t)
#     #                 if tuple(fbs_cur) == ('tremor','gamma'):
#     #                     print(fb,rf, t, click )
#                     #cond_bands &=  (rf >= gv.fbands[fb][0]) and (rf <= gv.fbands[fb][1])
#         cond_bands = (click == len(relevant_freqs))
        if desired_fbands is not None:
            cond_bands =  set(fbs_cur) <= set(desired_fbands)
        else:
            cond_bands = True

        if cond_ft and cond_bands and cond_chns:
            fis += [fi]

        #print(feature_names_all[fi],fbs_cur,chns_cur, fts,relevant_freqs,cond_ft,cond_bands,cond_chns)
        #print(feature_names_all[fi],desired_fbands,fbs_cur,cond_bands)
    return [feature_names_all[fi] for fi in fis]

def getChnamesFromFeatlist(featnames, mod='LFP'):
    from featlist import  parseFeatNames
    r = parseFeatNames(featnames)
    chnames = [chn for chn in r['ch1'] if chn.find(mod) >= 0]
    chnames += [chn for chn in r['ch2'] if \
                (chn is not None and chn.find(mod) >= 0) ]

    chnames = list(sorted(set(chnames)))
    return chnames


def sortFeats(featnames, desired_feature_order=None):
    if desired_feature_order is None:
        import globvars as gv
        desired_feature_order = gv.desired_feature_order
    pfn = parseFeatNames(featnames)
    regex_same_LFP = r'.?.?con.*(LFP.[0-9]+),.*\1.*'
    regex_same_src = r'.?.?con.*(msrc._[0-9]+_[0-9]+_c[0-9]+),.*\1.*'
    regexs = [regex_same_LFP, regex_same_src]
    inds_self_coupling = selFeatsRegexInds(featnames,regexs)
    #print(inds_self_coupling,featnames[inds_self_coupling] )

    # I want same-channel info come first
    ordinds = [desired_feature_order.index(ft) + 0.1 * (fti not in inds_self_coupling) \
               for fti,ft in enumerate(pfn['ftype']) ]
    return np.argsort(ordinds)
