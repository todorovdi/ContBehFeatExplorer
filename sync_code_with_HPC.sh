#rsync -avz --progress -e ssh $HOME/osccode/data_proc/* jusuf:data_proc_code/
#JUSUF="jusuf:data_proc_code"

ZBOOK_DIR="/home/demitau/osccode/data_proc"
FLAGS="-rtvz --progress"
# for dry run
#FLAGS="-rtvzn --progress"; echo "DRY RUN!!"

DIRECT_SSH=0
if [ $DIRECT_SSH -ne 0 ]; then
  echo "Using rsync -e ssh"
  SSH_FLAG="-e ssh"
  JUSUF="judac:data_proc_code"
  SLEEP="sleep 1s"
else
  echo "Using mounted sshfs"
  SSH_FLAG=""
  JUSUF="$HOME/ju_oscbagdis/data_proc_code"
  SLEEP=""
fi

echo "  rsync req"
rsync $FLAGS $SSH_FLAG "$ZBOOK_DIR/requirements.txt" $JUSUF/
$SLEEP
#FNS=`ls -L *.{py,sh}`
#rsync $FLAGS --progress $SSH_FLAG $FNS jusuf:data_proc_code/
echo "  rsync souce code"
rsync $FLAGS $SSH_FLAG --exclude="*HPC.py" --exclude="sync*HPC.sh"  $ZBOOK_DIR/*.{py,sh,m}  $JUSUF/
$SLEEP
echo "  rsync json"
rsync $FLAGS $SSH_FLAG --exclude="*HPC" $ZBOOK_DIR/*.json  $JUSUF/
$SLEEP
subdir=run
echo "  rsync run files"
rsync $FLAGS $SSH_FLAG --exclude="*HPC.sh" --exclude=srun_pipeline.sh --exclude=srun_exec_runstr.sh $ZBOOK_DIR/$subdir/*.{py,sh}  $JUSUF/$subdir/
$SLEEP
echo "  rsync params"
rsync $FLAGS $SSH_FLAG $ZBOOK_DIR/params/*.ini $JUSUF/params/
