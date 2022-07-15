GENFEATS_PARAM_FILE=genfeats_HPC.ini
MI_PARAM_FILE=ML_searchLFP_HPC.ini
NLPROJ_PARAM_FILE=nlproj_HPC.ini
#$1=0
#$2=1
#$3=0
#$4=1
#$5=1

#do_genfeats=$1
#do_ML=$2
#do_nlproj=$3
#MULTI_RAW_MODE=$4  # run with arbitrary number of rawnames
#SAVE_RUNSTR_MODE=$5

source batch_pipeline.sh 0 1 0 1 1
