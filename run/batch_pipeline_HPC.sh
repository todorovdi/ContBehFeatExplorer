GENFEATS_PARAM_FILE=genfeats_HPC.ini

#ML_PARAM_FILE=ML_HPC.ini
#ML_PARAM_FILE=ML_HPC_nointerp.ini
#ML_PARAM_FILE=ML_per_subj_one_LFP_HPC.ini
#rawstrs_type="per_subject"

#ML_PARAM_FILE=ML_joint2_ICA_one_LFP_HPC.ini
#ML_PARAM_FILE=ML_joint2_one_LFP_HPC.ini
#rawstrs_type="together"

#ML_PARAM_FILE=ML_joint_one_LFP_HPC.ini
#rawstrs_type="per_medcond"

#ML_PARAM_FILE=ML_joint_ICA_one_LFP_HPC.ini
ML_PARAM_FILE=ML_joint_one_LFP_HPC.ini
rawstrs_type="per_medcond"

#ML_PARAM_FILE=ML_medcondsep_one_LFP_HPC.ini
#rawstrs_type="per_subject_per_medcond"

#--groupings_to_use
# using spaces inatead of commas kills compatibilitiy with prev versions (when I could run muliple groupings in one file)
# but otherwise it does not work on jusuf -- it has too old bash..
#GROUPINGS_TO_USE="merge_all_not_trem merge_movements merge_nothing"
#GROUPINGS_TO_USE="merge_nothing merge_all_not_trem"
#GROUPINGS_TO_USE="merge_nothing"
GROUPINGS_TO_USE="merge_movements merge_all_not_trem"
#GROUPINGS_TO_USE="merge_movements"

#int_types_to_use = gp.int_types_to_include
#INT_SETS_TO_USE="basic trem_vs_quiet"
#INT_SETS_TO_USE="basic"
#INT_SETS_TO_USE="trem_vs_quiet"
INT_SETS_TO_USE="basic trem_vs_hold&move"
#INT_SETS_TO_USE="trem_vs_hold&move"

NLPROJ_PARAM_FILE=nlproj_HPC.ini
source batch_pipeline.sh
