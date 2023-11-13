def test_parseFeatNames():
    #import os, sys
    #currentdir = os.path.dirname(os.path.realpath(__file__))
    #parentdir = os.path.dirname(currentdir)
    #parentdir = os.path.dirname(parentdir)
    #sys.path.append(parentdir)

    #import data_proc
    #import data_proc.featlist

    #from data_proc import featsel
    from featlist import parseFeatNames
    from numpy import array
    test_input1 = array(['bpcorr_tremor_msrcR_9_18_c0,HFO_LFPR01',
            'bpcorr_gamma_msrcR_9_42_c0,tremor_LFPR12',
            'bpcorr_gamma_msrcR_9_20_c0,beta_LFPR23'], dtype='<U48')
    test_output1 = {'ftype': ['bpcorr', 'bpcorr', 'bpcorr'],
    'fb1': ['tremor', 'gamma', 'gamma'],
    'fb2': ['HFO', 'tremor', 'beta'],
    'ch1': ['msrcR_9_18_c0', 'msrcR_9_42_c0', 'msrcR_9_20_c0'],
    'ch2': ['LFPR01', 'LFPR12', 'LFPR23'],
    'mod1': ['msrc', 'msrc', 'msrc'],
    'mod2': ['LFP', 'LFP', 'LFP']}

    test_input2= array(['H_compl_msrcR_9_58_c0',
        'bpcorr_tremor_msrcR_9_32_c0,gamma_msrcR_9_32_c0',
        'bpcorr_tremor_msrcR_9_16_c0,beta_LFPR01'], dtype='<U48')
    test_output2= {'ftype': ['H_compl', 'bpcorr', 'bpcorr'],
    'fb1': [None, 'tremor', 'tremor'],
    'fb2': [None, 'gamma', 'beta'],
    'ch1': ['msrcR_9_58_c0', 'msrcR_9_32_c0', 'msrcR_9_16_c0'],
    'ch2': [None, 'msrcR_9_32_c0', 'LFPR01'],
    'mod1': ['msrc', 'msrc', 'msrc'],
    'mod2': [None, 'msrc', 'LFP']}

    test_input3=array(['bpcorr_beta_msrcR_9_8_c0,gamma_msrcR_9_8_c0',
        'con_tremor_LFPR01,msrcR_9_38_c0',
        'rbcorr_gamma_msrcR_9_46_c0,gamma_msrcR_9_46_c0'], dtype='<U48')
    test_output3={'ftype': ['bpcorr', 'con', 'rbcorr'],
    'fb1': ['beta', 'tremor', 'gamma'],
    'fb2': ['gamma', 'tremor', 'gamma'],
    'ch1': ['msrcR_9_8_c0', 'LFPR01', 'msrcR_9_46_c0'],
    'ch2': ['msrcR_9_8_c0', 'msrcR_9_38_c0', 'msrcR_9_46_c0'],
    'mod1': ['msrc', 'LFP', 'msrc'],
    'mod2': ['msrc', 'msrc', 'msrc']}

    test_inputs = [test_input1, test_input2,test_input3]
    test_outputs = [test_output1, test_output2, test_output3]

    for i,ti in enumerate(test_inputs):
        out = parseFeatNames(ti)
        to = test_outputs[i]
        assert out == to, f'Test example {i} failed'

# pytest <fname>
