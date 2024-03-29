function output = srcrec(subjstr,datall,data_cleaned,hdmf,roi_type,bads,S,srs,mask,use_DICS)
  % todo allo dataall and data_cleaned to be cell arrays that I could join to get common cov
  % subjstr in needed to load actual coords
  % roi_type is a string
  % mask argument only used if roi_type is of special type 
  %cd 

  data_dir = getenv("DATA_DUSS");

  do_load_only_ifnew   = 1;
  do_lookup_only_ifnew = 1;
  do_srcrec            = 1;

  %tstart = 300;
  %tend = 400;

  % merge all trials into one before computing covariance matrix
  % here a trial is just a segment of data, different trials are not necessarily of fixed size
  ntrials = size( data_cleaned.trial, 2 )  %number of segments after separations by artifact placements
  if ntrials > 1
    data = [];
    data.hdr = data_cleaned.hdr;
    data.fsample = data_cleaned.fsample;
    data.cfg = data_cleaned.cfg;
    data.label = data_cleaned.label;
    data.grad = data_cleaned.grad;
    nbinstot = 0 ;
    for nt = 1:ntrials
      nbinstot = nbinstot + size(data_cleaned.trial{nt}, 2);
    end

    data_concat = zeros(size(data_cleaned.trial{nt}, 1), nbinstot );
    time_concat = zeros(1, nbinstot );

    %initialize matrix data_concat of size channels X total time points
    %initialize matrix time_concat of size 1 X total time points

    shift_bin = 1;
    last_time = -1;
    for nt = 1:ntrials
      nbins_cur = size(data_cleaned.trial{nt}, 2);
      data_concat(:,shift_bin:shift_bin+nbins_cur-1) = data_cleaned.trial{nt};
      time_shift = 0;
      if last_time >= 0
        time_shift = last_time - time_concat.trial{nt};
      end
      time_concat(:,shift_bin:shift_bin+nbins_cur-1) = data_cleaned.time{nt} + time_shift;  % maybe I need to shift time as well
      shift_bin = shift_bin + (nbins_cur - 1);
    end

    %loop over data.trial and data.time (have the same size)
    %   data_concat(:,appropriate time indices) = data.trial{k}(:,:);
    %   time_concat(1,appropriate time indices) = data.time{k}(:);
    %end loop

    data.hdr.nSamples = nbinstot;
    data.trial = {data_concat};
    data.time = {time_concat};
    data.sampleinfo = [[1 nbinstot ] ];

    data_cleaned_concat = data;
  else
    fprintf("only one trial");
    data_cleaned_concat = data_cleaned;
  end
  data_cleaned_concat
  
  
  %data_cleaned_concat2 = ft_appenddata([], data_cleaned)  % does not merge trials :(

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  %save(fname_resampled,'datall')

  %cfg_sel = [];
  %cfg_sel.lpfilter = 'yes'
  %cfg_sel.lpfreq = 256
  %cfg_sel.channel = meg_channels_toinclude;
  %cfg_sel.latency = [tstart tend];
  %seldat = ft_preprocessing(cfg_sel,data)

  if 1~= exist("srcpos") || (1==exist("srcpos") && ~do_lookup_only_ifnew)
    %m1 = ft_read_atlas('HMAT_Right_M1.nii')  % goes deeper in the sulcus


    if strcmp(roi_type{1} , "HirschPt2011,2013" ) == 1 || strcmp(roi_type{1} , "HirschPt2011" ) == 1 

      % places where cortico-muscular coherence (at trem freq) changed during tremor
      %
      %They were located in the primary motor cortex (MNI coordinates:
      % 60, 15, 50), premotor cortex (MNI coordinates:  30, 10, 70)
      %and posterior parietal cortex (MNI coordinates:  20, 75, 50).
      fn = strcat( subjstr, '_modcoord_HirschPt.mat');
      if exist(fn,'file') > 0
        sprintf('Loading %s',fn)
        load(fn);
      else
        sprintf('CoordFile %s does not exist, exiting',fn)
        exit(0)

        load('coordsJan.mat')

        % compute transformation matrix
        vecinds = [1 200 3000 10000];
        vecinds = [8157       16509          99       22893];
        preX = srs.sourcemodel.pos;   % in each ROW -- x,y,z
        preY = hdmf.mni_aligned_grid.pos;
        % M * X = Y 
        X0 = preX( vecinds, : );  %3x3
        Y0 = preY( vecinds, : );  %3x3

        X1 = transpose(X0);
        Y1 = transpose(Y0);

        %X = [X1; [0 0 0 1] ];
        %Y = [Y1; [0 0 0 1] ];
        X = [X1; [1 1 1 1] ];
        Y = [Y1; [1 1 1 1] ];

        d = det(Y);
        if abs(d) < 1e-10
          printf("Bad selection of vectors")
          return
        end

        M = Y * inv(X);

        %coords_tremCohTremFreq_M1 = [ [ 60, 15, 50]; [-60,16,50] ];
        %coords_tremCohTremFreq_PMC = [ [ 30 10 70]; [ -30 10 70]  ];
        %coords_betaCohMax = [ [33 -22 57] ; [-33 -22 57] ] ;  % spatial maxima where where beta coherence between PreCG (putative M1) and LFPs are largest 

        %coords_Jan_ = [coords_betaCohMax; coords_tremCohTremFreq_M1; coords_tremCohTremFreq_PMC]
        coords_Jan_ = coords;
        coords_Jan_MNI =  transpose( coords_Jan_ ) ;
        %size(coords_Jan_MNI)
        coords_Jan_MNI_t = [ coords_Jan_MNI; ones(1, size(coords_Jan_MNI, 2) )  ];
        yy = M * coords_Jan_MNI_t;
        coords_Jan_actual = transpose( yy(1:3,:)  );
      end


      srcpos = coords_Jan_actual;
      srcpos = surroundPts;

    elseif strcmp(roi_type{1}, "parcel_aal_surf") == 1
      
      fn = strcat( data_dir , '/modcoord/', subjstr, '_modcoord_parcel_aal.mat');
      sprintf('Loading %s',fn)
      load(fn);
      srcpos = coords_Jan_actual;
    else
      srcpos_ = hdmf.mni_aligned_grid.pos(mask,:);
      srcpos = zeros(size(srcpos_));
      for i=1:length(srcpos)
        srcpos(i,:) = S * transpose( srcpos_(i,:) );
      end
    end
    %hdmf.mni_aligned_grid.inside = tmp;  % this one will be used for source reconstruction
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%  Plotting end
  %cfg                 = [];
  %cfg.channel         = data.label; % ensure that rejected sensors are not present
  %cfg.grad            = data.grad;
  %cfg.headmodel       = hdmf.hdm;
  %cfg.lcmv.reducerank = 2; % default for MEG is 2, for EEG is 3
  %cfg.grid = sourcemodel;
  %[grid] = ft_prepare_leadfield(cfg);

  res = {};

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  fbands = [ [2 10];   [10.5 30];    [30.5 100]  ];
  fbands = [ [4 10];   [10 30];    [30 -1];  [-1 -1]  ];
  fbands = [ [-1 -1]  ];
  %fbands = [ [2 10];  ];
  %fbands = [ [10.5 30];  ];
    
  [nfreqBands, zz] = size(fbands);
  if use_DICS
    endtime = datall.time{1}(end);
    step_sec = 0.5;
    windowsz_sec = 1.;

    cfg = [];
    %cfg.method    = 'mtmfft';
    cfg.method    = 'mtmconvol';
    cfg.output    = 'powandcsd';
    cfg.tapsmofrq = 2;
    %cfg.foilim    = [2 100];
    %cfg.channelcmb = {'all' 'all'}
    cfg.foi = 2:1:100;
    cfg.t_ftimwin = ones(length(cfg.foi), 1) .* windowsz_sec;
    cfg.toi = step_sec:step_sec:(endtime-step_sec - 1e-8);
    cfg.pad = 'nextpow2';

    if do_srcrec
      load_prev_freqdata = 0;

      fn = strcat(data_dir, '/tmp_freq_data.mat' ); 
      fprintf(fn);

      if load_prev_freqdata
        f = load(fn);
        freqdata = f.freqdata
        fprintf("Freq analysis loaded")
      else
        freqdata = ft_freqanalysis(cfg, datall);
        save(fn,"freqdata","-v7.3");
        fprintf("!! Freq analysis finished and saved")
      end


      %cfg                 = [];
      %cfg.grad            = freqPost.grad;
      %cfg.headmodel       = hdmf.hdm;
      %cfg.reducerank      = 2;
      %cfg.channel         = {'MEG','-MLP31', '-MLO12'};
      %cfg.resolution = 1;   % use a 3-D grid with a 1 cm resolution
      %cfg.sourcemodel.unit       = 'cm';
      %grid = ft_prepare_leadfield(cfg);


      for fbi =1:nfreqBands
        fband_cur = fbands(fbi,:); 

        cfg_srcrec              = [];
        cfg_srcrec.method       = 'dics';
        cfg_srcrec.frequency    = fband_cur;
        %cf_srcrecg.sourcemodel  = grid;
        cfg_srcrec.sourcemodel = [];
        cfg_srcrec.sourcemodel.pos = srcpos;
        cfg_srcrec.headmodel    = hdmf.hdm;
        cfg_srcrec.dics.projectnoise = 'yes';
        cfg_srcrec.dics.lambda       = 0;
        cfg_srcrec.keepfilter = 'yes'

        source_data_cur = ft_sourceanalysis(cfg_srcrec,freqdata);


        resEntry = [];
        resEntry.source_data = source_data_cur;
        resEntry.bpfreq = fband_cur;
        res = {res; resEntry};

        %if length(source_data)
        %  cfg_append = []
        %  cfg_append.parameter = 'freq'
        %  source_data = ft_appendsource(cfg_append, source_data, source_data_cur)
        %else
        %  source_data = source_data_cur
        %end
      end
    end
  else
    %sourcePost_nocon = ft_sourceanalysis(cfg, freqPre);

    for fbi =1:nfreqBands
      fband_cur = fbands(fbi,:); 

      if fband_cur(1) > 0
        cfg_bp = [];
        if fband_cur(2) > 0
          cfg_bp.bpfilter = 'yes';
          cfg_bp.bpfreq = fband_cur;
          %cfg_bp.bpfiltord
        else
          cfg_bp.hpfilter = 'yes';
          cfg_bp.hpfreq = fband_cur(1);
          %cfg_bp.hpfiltord
        end
        datall_cleaned_cur = ft_preprocessing(cfg_bp, data_cleaned_concat);
        datall_cur = ft_preprocessing(cfg_bp, datall);
      else
        datall_cleaned_cur = data_cleaned_concat;
        datall_cur = datall;
      end


      cfg_tla = [];
      cfg_tla.covariance = 'yes';
      avg = ft_timelockanalysis(cfg_tla,datall_cleaned_cur);
      eigval = eig(avg.cov);
      lambda = max(eigval) * 0.001;


      % for lcmv
      cfg_srcrec=[];
      cfg_srcrec.method='lcmv';
      cfg_srcrec.headmodel=hdmf.hdm;
      %cfg_srcrec.grid=mni_aligned_grid;
      cfg_srcrec.sourcemodel = [];
      cfg_srcrec.sourcemodel.pos = srcpos;
      %cfg_srcrec.grid = [];
      %cfg_srcrec.grid.pos = srcpos;
      cfg_srcrec.supchan = bads;

      %cfg_srcrec.lcmv.lambda='5%';
      cfg_srcrec.lcmv.lamba = [num2str(lambda),'%']; %regularization value. Can be changed by inspection

      cfg_srcrec.lcmv.keepfilter = 'yes';
      cfg_srcrec.lcmv.projectmom='yes';
      cfg_srcrec.reducerank=2;  % always like that for MEG

      %%%% MNE, not fully implemented (not computed). Why -- see explanation later
      %cfg_fwd = [];
      %cfg_fwd.grad = avg.grad;
      %cfg_fwd.channel = avg.label;
      %cfg_fwd.grid = [];
      %cfg_fwd.grid.pos = srcpos;
      %cfg_fwd.headmodel = hdmf.hdm;
      %leadfield = ft_prepare_leadfield(cfg_fwd);
      %
      %cfg_srcrec2=[];
      %cfg_srcrec2.method='mne';
      %cfg_srcrec2.headmodel=hdmf.hdm;
      %cfg_srcrec2.grid = leadfield;
      %%cfg_srcrec2.sourcemodel = [];
      %%cfg_srcrec2.sourcemodel.pos = srcpos;
      %%cfg_srcrec2.grid = [];
      %%cfg_srcrec2.grid.pos = srcpos;
      %cfg_srcrec2.supchan = bads;

      %cfg_srcrec2.mne = [];
      %cfg_srcrec2.mne.prewhiten = 'yes';
      %cfg_srcrec2.mne.lambda= 3;
      %cfg_srcrec2.mne.scalesourcecov= 'yes';

      %cfg_srcrec2.reducerank=2;  % always like that for MEG

      % for LCMV we don't run src rec immediately, we first compute spatial filter
      % (on cleaned data) then apply it to full (uncleaned) data
      % this is important becase we want to conserve the data dimensions
      % I cannot do it with MNE, so I'd have to either remove the data (and later have problems with data annotations) or apply it to entire data (and have problems due to artifacts affecting estimations) or apply it to epoched data with artif rejection, but this changes entire pipeline logic
      if do_srcrec
        %source_data_cur = ft_sourceanalysis(cfg_srcrec,datall_cur);  

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%   HERE IS WHERE STUFF HAPPENS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        source_time_tmp = ft_sourceanalysis(cfg_srcrec,avg);  
        %source_time_tmp = ft_sourceanalysis(cfg_srcrec,datall_cleaned_cur);  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        spatial_filt = cell2mat(source_time_tmp.avg.filter);

        if size(datall.trial{1},1) ~= size(spatial_filt,2)
          error('Wrong sizes')
        end

        %preallocate variable that contains the source activity per trial
        trial_source = zeros( size(spatial_filt,1), size(datall_cur.trial{1},2), length(datall_cur.trial) ); %Sources X Samples X Trials

        %get the data [with all trials] projected into source space with the individual spatial filter
        for j = 1:length(datall_cur.trial)
            trial_source(:,:,j) = spatial_filt * datall_cur.trial{j}; %Sources X Timepoint X Trails
        end
        
        %line trials up as a continuous (the parcel calculation is much faster then. Later I go back to trials)
        trial_source = reshape( trial_source, [size(trial_source,1),size(trial_source,2) * size(trial_source,3)] );

        source_data_cur = source_time_tmp;
        source_data_cur.avg.mom = trial_source;
        source_data_cur.time = datall_cur.time{1};

        resEntry = [];
        resEntry.source_data = source_data_cur;
        resEntry.bpfreq = fband_cur;
        resEntry.spatial_filt = spatial_filt;
        res{fbi} = resEntry;
        %%source_data = ft_sourceanalysis(cfg_srcrec,output_of_ft_timelockanalysis);
        %
      end
  end

  %%% DICS -- 
  %% ft_freqanalysis   mtmfft, output powandcsd (or fourier)  ,, maybe hanning
  %% foilim -- only if have low point power, better foi,  0.5 Hz

  output = res;
  %source_data.inside -- 3D array of  0 or 1
  %source_data.avg.pow -- 3D array of  floats (or NaN s for outside)
  %source_data.avg.mom -- 3D array of  floats (or NaN s for outside)

  % even sabasmpled to 256 Hz with only 100 second duration, source reconstructed weights 4 Gb single subject. Too dense grid
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% do_resample          = 0;
%  if do_resample
%    basename = sprintf('/%s_%s_%s.mat',subjstr,medstr,typestr);
%    fname = strcat(data_dir, basename );
%
%    basename_resampled = sprintf('/%s_%s_%s_resampled.mat',subjstr,medstr,typestr);
%    fname_resampled = strcat(data_dir, basename_resampled );
%    doload = 1;
%    if 1 ~= exist("data") && doload
%      load(fname );
%    end
%
%    MEGonly = 0
%
%
%    cfg1                     = [];
%    %cfg1.dataset             = [ fname];
%    cfg1.lpfreq = 256
%    cfg1.lpfilter = 'yes'
%    cfg1.latency = [tstart tend];
%
%    cfg2 = [];
%    cfg2.toilim = [tstart tend];
%
%    %cfgr.dataset = [ fname ];
%    cfgr            = [];
%    cfgr.resamplefs = 256;
%    %cfgr.baselinewindow = [tstart tend]
%
%    desiredChNum = 6
%
%    if exist('data') %&& ! exist( )
%      labels = data.label;
%      labels = data.label(1:desiredChNum);
%
%      meg_channels_toinclude = [];
%      nch = length(labels);
%      nchbad = length(bads);
%      for indch = 1:nch 
%        isbad = 0;
%        for indbad = 1:nchbad
%          if strcmp( bads(indbad) , labels{indch} )  == 1
%            isbad = 1;
%            break;
%          end
%          if MEGonly && length(strfind(labels{indch}, 'MEG' ) ) == 0
%            isbad = 1;
%            break
%          end
%        end
%        if isbad == 0
%          %meg_channels_toinclude = [ meg_channels_toinclude data.label(indch) ];
%          meg_channels_toinclude = [ meg_channels_toinclude indch ];
%        end
%      end
%    else
%      nch = desiredChNum
%    end
%
%    singlech = cell(nch,1);
%    %parfor indch = 1:nch
%    for indch = 1:nch
%      fprintf('Starting analysis of channel %i of %i',indch, nch)
%      cfg_ = cfg1;
%      cfg_.channel = indch;
%      tmpdat1 = ft_preprocessing(cfg_,data);
%      tmpdat2 = ft_redefinetrial(cfg2,tmpdat1);
%      clear tmpdat1;
%      cfgr.detrend = 'yes'
%      singlech{indch} = ft_resampledata(cfgr,tmpdat2);
%      clear tmpdat2;
%    end
%
%
%    cfgmerge=[];
%    datall = ft_appenddata(cfgmerge,singlech{:});
%    clear singlech
%  else
