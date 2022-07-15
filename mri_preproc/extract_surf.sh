. $FREESURFER_HOME/SetUpFreeSurfer.sh

#export SUBJECTS_DIR=$HOME/mysubs
#export SUBJECTS_DIR_DEF=$HOME/raw_dacq
SUBJECTS_DIR_DEF=$HOME/data/MRI

SUBJ_IDS=( S10 )
SUBJ_IDS=( Dima_SERIES9 )
#/sub-013-newtry-newdicoms
#SUBJ_IDS=( sub-011 sub-012 sub-013 sub-015 sub-018 sub-020 sub-021 sub-022 )
#SUBJ_IDS=( S01 S02 S03 S04 S05 S07 S08 S09 S10 )
SUBJ_IDS=( S01 S02 )
SUBJ_IDS=( S01 )
SUBJ_IDS=( S07 )
SUBJ_IDS=( S02 )
nids=${#SUBJ_IDS[*]}
for (( i=0; i<=$nids; i++ )); do
	SUBJ_ID=${SUBJ_IDS[$i]}
	if [ -z $SUBJ_ID ]; then
		continue
	fi

	subject_folder_out="$SUBJECTS_DIR"/"$SUBJ_ID"
	my_NIFTI=$SUBJECTS_DIR_DEF/"$SUBJ_ID"_anat_t1.nii
	my_NIFTI2=$SUBJECTS_DIR_DEF/"$SUBJ_ID"_anat_t2.nii
	echo "$my_NIFTI $subject_folder_out"

  #cmd1="$FREESURFER_HOME/bin/recon-all -i $my_NIFTI -s $subject_folder_out -all"
  cmd1="$FREESURFER_HOME/bin/recon-all -s $subject_folder_out -T2 $my_NIFTI2 -T2pial -autorecon3"
  #-sd $SUBJECTS_DIR
  cmd2="mne watershed_bem -s $subject_folder_out"

  #terminator -e "$cmd1; $cmd2" &
  #terminator -e "$cmd1; $cmd2"  # does not work 
  #$cmd1  && $cmd 2  # does not work
  $cmd1
  #$cmd2
done

#An example of running a subject through Freesurfer with a T2 image is:
#
#recon-all -subject subjectname -i /path/to/input_volume -T2 /path/to/T2_volume -T2pial -all
#
#T2 or FLAIR images can also be used with Freesurfer subjects that have already been processed without them. Note that autorecon3 should also be re-ran to compute statistics based on the new surfaces. For example:
#
#recon-all -subject subjectname -T2 /path/to/T2_volume -T2pial -autorecon3

#SUBJ_ID=sub-013
#my_NIFTI=$SUBJECTS_DIR_DEF/"$SUBJ_ID"-newtry-newdicoms/anat_t1.nii
#subject_folder_out="$SUBJ_ID"
#echo "$my_NIFTI $subject_folder_out"
#recon-all -i $my_NIFTI -s $subject_folder_out -all -sd $HOME/mysubs

#slice_dir=$SUBJECTS_DIR/$SUBJ_ID/ses-01/mri/t1/slices
#my_NIFTI=$slice_dir/`ls -t $slice_dir | head -n 1`
#my_NIFTI=ses-01/$SUBJ_ID/anat_t1.nii
#my_NIFTI=$SUBJECTS_DIR/"$SUBJ_ID"_anat_t1.nii
#recon-all -i $my_NIFTI -s $my_subject -all -dontrun -sd $HOME/mysubs
#recon-all -s $my_subject -all -sd $HOME/mysubs

