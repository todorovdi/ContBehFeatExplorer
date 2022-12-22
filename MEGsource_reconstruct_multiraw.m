data_dir = getenv("DATA_DUSS");
%subjstr = "S01";
%taskstr = "move";
%medstr  = "off";

if ~exist("subjstrs")
  subjstrs = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10" ];
  fprintf("Starting source reconstruction\n subjstrs=%a",subjstrs)
end

if ~exist("medconds")
  medconds = ["off", "on"];
end

if ~exist("tasks")
  tasks = ["rest", "move", "hold"];
end

if ~exist("rawnames")
  rawnames = ["S01_off_hold"];
end

if ~exist("use_data_afterICA")
  use_data_afterICA = 1;
end

if ~exist("input_subdir")
  input_subdir = "";
end
if ~exist("output_subdir")
  output_subdir = "";
end

% double quotes are string array
%input_rawname_type = ["resample", "afterICA"]
%input_rawname_type = ["SSS", "notch", "highpass", "resample", "afterICA"]

if ~exist("roi")
  roi = {"parcel_aal_surf"};
end

% it would be ideal to load raw data but there are two problems: 
% in fieldtrip when one does filtering it is not possible to care about annotations,
% so it is better doing it in MNE
if ~exist("input_rawname_type")
  input_rawname_type = ["resample", "notch", "highpass"]  %this is default
end

if use_data_afterICA
  input_rawname_type = [input_rawname_type "afterICA"]
end
rawname_suffix = '';
for i=1:length(input_rawname_type)
  rawname_suffix = sprintf("%s_%s",rawname_suffix,input_rawname_type(i) );
end
rawname_suffix = sprintf("%s_%s",rawname_suffix,"raw" );

resname_add_str = '';
%if any(contains(input_rawname_type, "SSS" ) ) 
%  if any(strcmp(input_rawname_type, "afterICA" ) )   
if contains(rawname_suffix, "SSS")
  remove_bad_channels = 0;
  % I better control it with setting subfolder names
  %resname_add_str = "SSS"
  %if contains(rawname_suffix, "afterICA")
  %  resname_add_str = strcat(resname_add_str, "_afterICA")
  %end
else
 remove_bad_channels = 1
end

% Brodmann 4 -- M1,   Brodmann 6 -- PMC
%roi = {"Brodmann area 4"};
%roi = {"Brodmann area 6"};
%roi = {"HirschPt2011,2013"}
%roi = {"HirschPt2011,2013","Thalamus"}
%
%roi = {"Brodmann area 4","Brodmann area 6"};
%roi = {"HirschPt2011"};
save_srcrec          = 1;
if exist("TEST_SRCREC")
  save_srcrec          = 0;
end

use_DICS = 0;

% Load Jan-provided info
if 1 ~= exist("Info")
  load(strcat(data_dir,"/Info.mat") );
end

highpass_freq = 1.5;

f_scalemat = load('head_scalemat');


atlas_type = "aal"
if strcmp(atlas_type, "aal")
  aal = ft_read_atlas('~/soft/fieldtrip/template/atlas/aal/ROI_MNI_V4.nii');
  atlas = aal;
elseif strcmp(atlas_type, "afni")  % has Brodmann areas, no sides
  afni = ft_read_atlas('~/soft/fieldtrip/template/atlas/afni/TTatlas+tlrc.HEAD');  % goes deeper in the sulcus
  atlas = afni;
end
%singleshell = load('~/soft/fieldtrip/template/headmodel/standard_singleshell');
source_grid = load('~/soft/fieldtrip/template/sourcemodel/standard_sourcemodel3d5mm');
source_grid.sourcemodel.coordsys = 'mni';
% pts_converted = mni2icbm_spm( pts )
% atlas.  pts_converted = mni2icbm_spm( pts )
atlas = ft_convert_units(atlas,'cm'); % ftrop and our sourcemodels have cm units
atlas.coordsys = 'mni';

% if roi is found in the atlas. Here I assume that if one is from the atlas than other rois too
if isfield(atlas, "tissuelabel")
  tlarr = atlas.tissuelabel;
elseif isfield(atlas, "brick0label")  
  tlarr = cat ( 1, {atlas.brick0label{:},atlas.brick1label{:} } );
end

read_upd_bads = 0;

%TODO: first read and do reject artifact, then append data then call source reconstruction
%cfg = [];
%data_merged =  ft_appenddata(cfg, data1_resampled, data2);

num_prcessed = 0;
for rawi = 1:length(rawnames)
  r = split(rawnames(rawi),'_');
  subjstr = r(1); medstr=r(2); taskstr=r(3);

%for subji = 1:length(subjstrs)
%  subjstr = subjstrs(subji);
  fprintf(" current subjstr=%s\n",subjstr)

  basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
  fname_head = strcat(data_dir, basename_head );
  hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid

%  for medcondi = 1:length(medconds)
%    medstr = medconds(medcondi);
%    for taski = 1:length(tasks)
%      taskstr = tasks(taski);

      % rawnames_suffix starts with '_'
      basename = sprintf('/%s_%s_%s%s.fif',subjstr,medstr,taskstr,rawname_suffix);

      %if use_data_afterICA
      %  basename = sprintf('/%s_%s_%s_resample_afterICA_raw.fif',subjstr,medstr,taskstr);
      %else
      %  % resample was notched but not highpassed
      %  %basename = sprintf('/%s_%s_%s_resample_raw.fif',subjstr,medstr,taskstr);
      %  basename = sprintf('/%s_%s_%s_resample_notch_highpass_raw.fif',subjstr,medstr,taskstr);
      %end
      fprintf(" Using basename=%s\n",basename)
      %basename = sprintf('/%s_%s_%s_resample_maxwell_raw.fif',subjstr,medstr,taskstr);
      fname = fullfile(data_dir, input_subdir, basename );
      if isfile(fname)   % if file exists
        fprintf("%s\n",fname)

        cfgload = [];
        cfgload.dataset = char(fname);          % fname should be char array (single quotes), not string (double quotes)
        cfgload.chantype = {'meg'};
        cfgload.coilaccuracy = 1;   % 0 1 or 2, idk what each of them means -- it is about combining grad and mag

        %cfgload.hpfilter = 'yes';
        %cfgload.hpfreq = highpass_freq;  % NO, I don't want to do it in FT because it will not take care of artifacts

        datall_ = ft_preprocessing(cfgload);   

        bads = {};
        if isfield(Info, subjstr)
          if isfield(Info.(subjstr).bad_channels, medstr) &&  isfield(Info.(subjstr).bad_channels.(medstr), taskstr)
            bads = Info.(subjstr).bad_channels.(medstr).(taskstr);
          end

          if read_upd_bads;
              fname_bads_mat = sprintf('%s_MEGch_bads_upd.mat',fname_noext);
              fname_bads_mat_full = fullfile( data_dir, fname_bads_mat);
              bads_upd = load(fname_bads_mat_full).bads;
              bads = bads_upd;
          end
        else
          fprintf('!!!! Info.mat does not contain %s field, assuming no bad channels',subjstr)
        end

        chanselarr = {'meg'};
        if remove_bad_channels && length(bads) > 0
          for chani = 1:length(bads)
            chanselarr{chani + 1} = sprintf('-%s',bads{chani});   % marks channel for removal from seelction
          end
        end

        selchan = ft_channelselection(chanselarr, datall_.label);
        bads = {};
        % I could also do ft_repairchannel
        
        cfgsel = [];
        cfgsel.channel = selchan;
        datall  = ft_selectdata(cfgsel, datall_);

        %else
        %  datall  = datall_;
        %end

        %if use_data_afterICA  % because we also do SSS which restores bad channels
        %  bads = {}
        %end
        %else
        %  datall  = datall_;
        %end


        %deal with artifacts
        fname_srcrec_exclude = sprintf('/%s_%s_%s_ann_srcrec_exclude.txt', subjstr,medstr,taskstr);
        filename      = fullfile(data_dir,fname_srcrec_exclude);
        FID = fopen(filename);
        fgets(FID); fgets(FID); % skip first 2 lines
        form='%f,%f,%s'; % we have 7 columns, then use 7 %f
        out = textscan(FID, form);
        times_artif = [ out{1}, out{1}+out{2} ];  %third column is interval type, which I don't need
        bins_artif = datall_.fsample * times_artif;  
        bins_artif = 1 + round(bins_artif);
        %bins_artif = times_artif;
        fclose(FID);


        cfg_ar = [];
        cfg_ar.artfctdef.imported.artifact = bins_artif;
        %cfg_ar.artfctdef.muscle.artifact = bins_artif;

        cfg_ar.artfctdef.reject          = 'partial';
        %cfg_ar.artfctdef.reject          = 'nan';
        %cfg_ar.artfctdef.reject          = 'value';
        %cfg_ar.artfctdef.value           = 1e-2

        cfg_ar.artfctdef.minaccepttim    = 1; %min length of remaining trial in seconds
        %   cfg.artfctdef.feedback        = 'yes' or 'no' (default = 'no')
        %   cfg.artfctdef.invert          = 'yes' or 'no' (default = 'no') -- invert artifact selection
        
        % unfortunately it cannot reduce trial size, just throw away it 
        % completely, so I do some dirty work later        
        data_cleaned = ft_rejectartifact(cfg_ar,datall) 
        %return

        for roii = 1:length(roi) 
          roicur = roi{roii};

          % scaling matrix, no longer used, kept for compatibility
          if isKey(f_scalemat.scalemat,subjstr)
            S = f_scalemat.scalemat(subjstr);
          else
            S = eye(3);
          end

          % do we find roi in the atlas?
          roimask = contains(cellstr(tlarr), roicur );
          if sum(roimask) == 1 
            cfg_vlookup = [];
            cfg_vlookup.atlas = atlas;
            cfg_vlookup.roi = roi;
            %cfg_vlookup.roi = atlas.tissuelabel; % this would lookup for all possible
            cfg_vlookup.inputcoord = 'mni';  % coord of the source
            %cfg_vlookup.inputcoord = 'tal';
            source_grid_mask = ft_volumelookup(cfg_vlookup,source_grid.sourcemodel);  % selecting only a subset
            fprintf("%s: Num of sources = %d\n",roicur,sum(source_grid_mask) )
          else
            fprintf("%s not found in atlas\n",roicur )
            source_grid_mask = [];
          end

          source_data = srcrec(subjstr,datall,data_cleaned,hdmf,{roicur},bads,S,source_grid,source_grid_mask,use_DICS);

          for fbi = 1:length(source_data) 
            source_data{fbi}.source_data.roi = {roicur};
          end

          if save_srcrec
            basename_srcd = sprintf("/srcd_%s_%s_%s_%s%s.mat",subjstr,medstr,taskstr,roicur,resname_add_str);
            %data_dir_out = strcat(
            fname_srcd = fullfile(data_dir, output_subdir, basename_srcd );
            fprintf("Saving to %s\n",fname_srcd)
            save(fname_srcd,"source_data","-v7.3");
          end
        end
      else
        fprintf(" file not found! %s",fname)
      end
%    end
%  end
 
  num_prcessed = num_prcessed + 1;
end

if num_prcessed == 0
  fprintf('AAAAAA')
end
