def test_groupings():
    import os
    import numpy as np
    import globvars as gv
    rncur = 'S01_off_hold'
    sources_type='parcel_aal'
    src_file_grouping_ind = 10
    src_rec_info_fn = '{}_{}_grp{}_src_rec_info'.format(rncur,
                                                        sources_type,src_file_grouping_ind)
    src_rec_info_fn_full = os.path.join(gv.data_dir, src_rec_info_fn + '.npz')
    rec_info = np.load(src_rec_info_fn_full, allow_pickle=True)


    print( list(rec_info.keys()) )

    labels_dict = rec_info['label_groups_dict'][()]

    roi_labels = labels_dict['all_raw']

    # tests if labels in groupings are consistent with parcels
    # TODO: load roi label (and get list from the dict)
    roi_labels_noside = list( set([rl[:-2] for rl in roi_labels] ) )
    labels_marked = set()
    for ol in gv.old_labels_MATLAB_mod:
        for o in ol:
            try:
                #effo = o + '_R'
                ind = roi_labels_noside.index(o)
                labels_marked |= set([roi_labels_noside[ind]])
            except ValueError as e:
                print(ind,str(e))
    labels_unmarked = set(roi_labels_noside) - labels_marked
    assert len(labels_unmarked) == 1 and labels_unmarked == 'unlabel'  #becaue 'ed' was eaten off
