data_dir = getenv("DATA_DUSS");

load('head_scalemat');
afni = ft_read_atlas('~/soft/fieldtrip-20190716/template/atlas/afni/TTatlas+tlrc.HEAD');  % goes deeper in the sulcus
srs = load('~/soft/fieldtrip-20190716/template/sourcemodel/standard_sourcemodel3d5mm');
atlas = ft_convert_units(afni,'cm'); % ftrop and our sourcemodels have cm units

roi = {"HirschPt2011"};

cfg_vlookup = [];
cfg_vlookup.atlas = atlas;
cfg_vlookup.roi = roi;
%cfg_vlookup.roi = atlas.tissuelabel;
cfg_vlookup.inputcoord = 'mni';  % coord of the source
%cfg_vlookup.inputcoord = 'tal';
mask = ft_volumelookup(cfg_vlookup,srs.sourcemodel);  % selecting only a subset

load(strcat(data_dir,"/Info.mat") );  % general info about subjects

%%%%%%%%%%%%%%%%%%%%%%  subject specific %%%%%%%%%%%%%%%%%%

subjstr = "S01"
medstr = "off";
typestr="hold";

%basename = sprintf('/%s_%s_%s_resample_raw.fif',subjstr,medstr,typestr);
basename = sprintf('/%s_%s_%s_resample_afterICA_raw.fif',subjstr,medstr,typestr);
fname = strcat(data_dir, basename );

fprintf(" current subjstr=%s\n",subjstr)
basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
fname_head = strcat(data_dir, basename_head );
hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid

bads = Info.(subjstr).bad_channels.(medstr).(typestr);

cfgload = [];
cfgload.dataset = fname;          % fname should be char array (single quotes), not string (double quotes)
cfgload.chantype = {'meg'};
datall_ = ft_preprocessing(cfgload);   

if length(bads) > 0
  chanselarr = {'meg'};
  for chani = 1:length(bads)
    chanselarr{chani + 1} = sprintf('-%s',bads{chani});   % marks channel for removal from seelction
  end
  selchan = ft_channelselection(chanselarr, datall_.label);
  bads = {};
  % I could also do ft_repairchannel
  
  cfgsel = [];
  cfgsel.channel = selchan;
  datall  = ft_selectdata(cfgsel, datall_);
else
  datall  = datall_;
end

%%%%%%%%%%%%%%%%%%%%%%%
crop_to_short = 0;
if crop_to_short
  cfg_crop = [];
  cfg_crop.toilim = [0 20];
  datall = ft_redefinetrial(cfg_crop, datall);
end
%%%%%%%%%%%%%%%%%%%%%%%


roicur = roi{1};
S = scalemat(subjstr);

use_DICS = 0;

%basename_srcd = sprintf("/srcd_%s_%s_%s_%s_test.mat",subjstr,medstr,typestr,roicur);
basename_srcd = sprintf("/srcd_%s_%s_%s_%s_test.mat",subjstr,medstr,typestr,roicur);
source_data = srcrec(subjstr,datall,hdmf,{roicur},bads,S,srs,mask, use_DICS);
fname_srcd = strcat(data_dir, basename_srcd );
save(fname_srcd,"source_data","-v7.3");
