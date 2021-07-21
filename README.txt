resave.py

jupyter LFP extraction
jupyter EMG labeling and extraction
jupyter MEG processing (removal of some ICA's perhaps)

for MEG sources
  first one prepares source coordinates  with run_prepSourceCoords.sh (it runs certain Matlab script)
  then one runs source reconstruction with run_ftscr
  then one semi-manually does Load FTsources

then run_pipeline_multi.sh 1 1 1
