#!/bin/bash

# feats pca tSNE
do_genfeats=$1
do_PCA=$2
do_tSNE=$3

raw=S01_off_hold
raw_compl=S01_off_move

#raw=S01_on_hold
#raw_compl=S01_on_move

interactive=""
#interactive="-i"

DESIRED_PCA_EXPLAIN=0.95
DESIRED_DISCARD=0.01
PCA_PLOTS=1

if [ $do_genfeats -gt 0 ]; then
  ipython3 $interactive run_genfeats.py -- -r $raw
  ipython3 $interactive run_genfeats.py -- -r $raw_compl
fi

if [ $do_PCA -gt 0 ]; then
  ipython3 $interactive run_PCA.py  -- -r $raw,$raw_compl

  COMMON_PART="-r $raw,$raw_compl --pcexpl $DESIRED_PCA_EXPLAIN --discard $DESIRED_DISCARD --show_plots $PCA_PLOTS"

  # use everything (from main trem side)
  ipython3 $interactive run_PCA.py -- $COMMON_PART --prefix all

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --mods LFP --prefix onlyLFP

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --mods msrc  --prefix src 

  # only time-domain
  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types H_act,H_mob,H_compl,rbcorr --prefix onlyTD

  # only freq-dmain
  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types con,bpcorr --prefix onlyFD

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types con,H_act,H_mob,H_compl --prefix conH

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types H_act,H_mob,H_compl --prefix onlyH

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types bpcorr --prefix onlyBpcorr

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types bpcorr --use_HFO 0 --prefix onlyBpcorrNoHFO

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types rbcorr --prefix onlyRbcorr

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types con --prefix onlyCon

  ipython3 $interactive run_PCA.py -- $COMMON_PART \
  --feat_types con --use_HFO 0 --prefix onlyConNoHFO

fi

if [ $do_tSNE -gt 0 ]; then
  #ipython3 $interactive run_tSNE.py --     -r $raw
 
dim_inp_tSNE=60
SUBSKIP=4
TSNE_PLOTS=1
#raws=$raw,$raw_compl
LOAD_TSNE=0

  raws=$raw

  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyTD          --n_feats_PCA 131 --dim_PCA 76  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyLFP         --n_feats_PCA 48  --dim_PCA 31  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyRbcorr      --n_feats_PCA 110 --dim_PCA 71  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix src             --n_feats_PCA 240 --dim_PCA 164 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyH           --n_feats_PCA 21  --dim_PCA 7   --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyFD          --n_feats_PCA 469 --dim_PCA 266 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix conH            --n_feats_PCA 170 --dim_PCA 47  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix all             --n_feats_PCA 600 --dim_PCA 329 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyCon         --n_feats_PCA 149 --dim_PCA 42  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyBpcorrNoHFO --n_feats_PCA 320 --dim_PCA 227 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyConNoHFO    --n_feats_PCA 140 --dim_PCA 33  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyBpcorr      --n_feats_PCA 320 --dim_PCA 227 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE

  raws=$raw_compl

  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyTD          --n_feats_PCA 131 --dim_PCA 76  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyLFP         --n_feats_PCA 48  --dim_PCA 31  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyRbcorr      --n_feats_PCA 110 --dim_PCA 71  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix src             --n_feats_PCA 240 --dim_PCA 164 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyH           --n_feats_PCA 21  --dim_PCA 7   --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyFD          --n_feats_PCA 469 --dim_PCA 266 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix conH            --n_feats_PCA 170 --dim_PCA 47  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix all             --n_feats_PCA 600 --dim_PCA 329 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyCon         --n_feats_PCA 149 --dim_PCA 42  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyBpcorrNoHFO --n_feats_PCA 320 --dim_PCA 227 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyConNoHFO    --n_feats_PCA 140 --dim_PCA 33  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyBpcorr      --n_feats_PCA 320 --dim_PCA 227 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE

  raws=$raw,$raw_compl

  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyTD          --n_feats_PCA 131 --dim_PCA 76  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyLFP         --n_feats_PCA 48  --dim_PCA 31  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyRbcorr      --n_feats_PCA 110 --dim_PCA 71  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix src             --n_feats_PCA 240 --dim_PCA 164 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyH           --n_feats_PCA 21  --dim_PCA 7   --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyFD          --n_feats_PCA 469 --dim_PCA 266 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix conH            --n_feats_PCA 170 --dim_PCA 47  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix all             --n_feats_PCA 600 --dim_PCA 329 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyCon         --n_feats_PCA 149 --dim_PCA 42  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyBpcorrNoHFO --n_feats_PCA 320 --dim_PCA 227 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyConNoHFO    --n_feats_PCA 140 --dim_PCA 33  --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE
  ipython3 $interactive run_tSNE.py -- -r $raws --prefix onlyBpcorr      --n_feats_PCA 320 --dim_PCA 227 --subskip $SUBSKIP --dim_inp_tSNE $dim_inp_tSNE --show_plots $TSNE_PLOTS --load_tSNE $LOAD_TSNE

fi
