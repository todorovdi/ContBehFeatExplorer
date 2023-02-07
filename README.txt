resave.py

jupyter LFP extraction
jupyter EMG labeling and extraction
jupyter MEG processing (removal of some ICA's perhaps)

for MEG sources
  first one prepares source coordinates  with run_prepSourceCoords.sh (it runs certain Matlab script)
  then one runs source reconstruction with run_ftscr
  then one semi-manually does Load FTsources

then run_pipeline_multi.sh 1 1 1


#########

one step is something that has to be done for all subjects in order to proceed to next step (though in general I could do for single as well (except gathering stats) but copying hence and forth is very painful anyway

resave with zero knowledge of artif -- creates plots
adjust thresholds for artif
resave again to gen artif
resave (on workstaton) to gen resample 
resave (on jusuf or workstation) to gen ICA
resave (on workstation) generating MATLAB 
resave (on workstation) processFTsources
copy to right dir, care abour rec info files

run_prep_dat
   gen runstrings for run_genfeats
run_genfeats
   gen runstrings for serach LFP
run_ML serach LFP
  collect serachLFP and save json (and maybe pandas dataframe)
  gen runstrings for main run_ML
run_ML 
  collect and save pandas
  make 3D plots on workstation  (care about recinfo, modcoord and headsurf files)



