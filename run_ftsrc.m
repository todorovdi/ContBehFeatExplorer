data_dir = getenv("DATA_DUSS");
%subjstr = "S01";
%typestr = "move";
%medstr  = "off";

if ~exist("subjstrs")
  subjstrs = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10" ];
  fprintf("Starting source reconstruction\n subjstrs=%a",subjstrs)
end
%subjstrs = ["S01" ] 
medconds = ["off", "on"];
tasks = ["rest", "move", "hold"];

% Brodmann 4 -- M1,   Brodmann 6 -- PMC
%roi = {"Brodmann area 4","Brodmann area 6"};
roi = {"Brodmann area 6"};
save_srcrec          = 1;
remove_bad_channels = 1;

if 1 ~= exist("Info")
  load(strcat(data_dir,"/Info.mat") );
end


load('head_scalemat')

for subji = 1:length(subjstrs)
  subjstr = subjstrs(subji);
  fprintf(" current subjstr=%s\n",subjstr)

  basename_head = sprintf('/headmodel_grid_%s.mat',subjstr);
  fname_head = strcat(data_dir, basename_head );
  hdmf = load(fname_head);   %hdmf.hdm, hdmf.mni_aligned_grid

  for medcondi = 1:length(medconds)
    medstr = medconds(medcondi);
    for taski = 1:length(tasks)
      typestr = tasks(taski);

      basename = sprintf('/%s_%s_%s_resample_raw.fif',subjstr,medstr,typestr);
      fname = strcat(data_dir, basename );
      if isfile(fname)
        fprintf("%s\n",fname)

        cfgload = [];
        cfgload.dataset = fname;          % fname should be char array (single quotes), not string (double quotes)
        cfgload.chantype = {'meg'};
        datall_ = ft_preprocessing(cfgload);   

        bads = {};
        if isfield(Info.(subjstr).bad_channels, medstr) &&  isfield(Info.(subjstr).bad_channels.(medstr), typestr)
          bads = Info.(subjstr).bad_channels.(medstr).(typestr);
          if remove_bad_channels && length(bads) > 0
            chanselarr = {'meg'};
            for chani = 1:length(bads)
              chanselarr{chani + 1} = sprintf('-%s',bads{chani});
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
        else
          datall  = datall_;
        end


        for roii = 1:length(roi) 
          roicur = roi{roii};

          S = scalemat(subjstr)
          source_data = srcrec(datall,hdmf,{roicur},bads,S);
          source_data.roi = {roicur};

          if save_srcrec
            basename_srcd = sprintf("/srcd_%s_%s_%s_%s.mat",subjstr,medstr,typestr,roicur);
            fname_srcd = strcat(data_dir, basename_srcd );
            fprintf("Saving to %s\n",fname_srcd)
            save(fname_srcd,"source_data","-v7.3");
          end
        end

      end
    end
  end
end